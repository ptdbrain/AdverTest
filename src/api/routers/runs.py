import hashlib
import hmac
import json
import time

from fastapi import APIRouter, Depends, HTTPException, Request

from src.api.dependencies import get_runner, get_store, get_worker
from src.api.jobs import LocalRunWorker, SqliteRunStore
from src.api.routes import _resolve_run_config
from src.api.schemas.run import CostEstimateOut, PreflightOut, RunJobOut, RunReportOut
from src.config import get_settings
from src.pipeline.runner import RunConfig, TestRunner

router = APIRouter(prefix="/runs", tags=["Runs"])


def _job_out(record: dict) -> RunJobOut:
    return RunJobOut(**record)


def _generate_estimate_token(config: RunConfig) -> str:
    settings = get_settings()
    config_dict = {
        "model": config.model,
        "dataset": config.dataset,
        "attacks": sorted(config.attacks),
        "severities": sorted(config.severities),
        "limit": config.limit,
    }
    payload = json.dumps(config_dict, sort_keys=True)
    exp = int(time.time()) + 3600
    sig = hmac.new(settings.jwt_secret.encode(), f"{payload}:{exp}".encode(), hashlib.sha256).hexdigest()
    return f"{exp}:{sig}"


def _verify_estimate_token(config: RunConfig, token: str) -> bool:
    try:
        settings = get_settings()
        exp_str, sig = token.split(":", 1)
        if int(exp_str) < int(time.time()):
            return False
        config_dict = {
            "model": config.model,
            "dataset": config.dataset,
            "attacks": sorted(config.attacks),
            "severities": sorted(config.severities),
            "limit": config.limit,
        }
        payload = json.dumps(config_dict, sort_keys=True)
        expected_sig = hmac.new(
            settings.jwt_secret.encode(), f"{payload}:{exp_str}".encode(), hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(sig, expected_sig)
    except Exception:
        return False


@router.post("/estimate", response_model=CostEstimateOut)
async def estimate_run(config: RunConfig, runner: TestRunner = Depends(get_runner)) -> CostEstimateOut:
    est = runner.estimate(config).as_dict()
    # Add a custom warning if YOLO model requires download
    warnings = []
    if config.model.startswith("yolov8"):
        from pathlib import Path

        if not Path(f"{config.model}.pt").is_file():
            warnings.append(
                f"Model '{config.model}' requires downloading weights. Estimated time does not include download time."
            )
    token = _generate_estimate_token(config)
    return CostEstimateOut(
        **est,
        estimate_token=token,
        artifact_storage_estimate_bytes=est.get("n_samples", 0) * max(1, len(config.attacks)) * 50_000,
        gpu_cpu_cost_estimate=est.get("cost_units", 0.0) * 0.05,
        warnings=warnings,
    )


@router.post("/preflight", response_model=PreflightOut)
async def preflight_run(config: RunConfig, runner: TestRunner = Depends(get_runner)) -> PreflightOut:
    return PreflightOut(**runner.preflight(_resolve_run_config(config)).as_dict())


@router.post("", status_code=202, response_model=RunJobOut)
async def create_run(
    request: Request,
    config: RunConfig,
    runner: TestRunner = Depends(get_runner),
    store: SqliteRunStore = Depends(get_store),
    worker: LocalRunWorker = Depends(get_worker),
) -> RunJobOut:
    """Persist and enqueue a run. Heavy model work never runs in the request."""
    # Reject empty payloads without explicit confirmation or configuration
    body = await request.body()
    try:
        raw_json = json.loads(body.decode()) if body else {}
    except Exception:
        raw_json = {}

    is_empty_payload = not raw_json or raw_json == {}
    if is_empty_payload and not config.confirmed:
        raise HTTPException(
            status_code=422,
            detail="CONFIRMATION_REQUIRED: Empty or unconfirmed run payload cannot start background workload. Run /runs/estimate first or provide explicit configuration with confirmed=true.",
        )

    if config.estimate_token:
        if not _verify_estimate_token(config, config.estimate_token):
            raise HTTPException(
                status_code=409,
                detail="ESTIMATE_TOKEN_MISMATCH: Estimate token is invalid, expired, or configuration was modified after estimation.",
            )

    config = _resolve_run_config(config)
    preflight = runner.preflight(config)
    if preflight.fatal_errors:
        raise HTTPException(status_code=422, detail={"fatal_errors": list(preflight.fatal_errors)})
    run_id = store.create(config)
    worker.enqueue(run_id, config)
    return _job_out(store.get(run_id))


@router.get("", response_model=list[RunJobOut])
async def list_runs(store: SqliteRunStore = Depends(get_store)) -> list[RunJobOut]:
    return [_job_out(r) for r in store.list()]


@router.get("/{run_id}", response_model=RunJobOut)
async def get_run(run_id: str, store: SqliteRunStore = Depends(get_store)) -> RunJobOut:
    record = store.get(run_id)
    if not record:
        raise HTTPException(status_code=404, detail="Run not found")
    return _job_out(record)


@router.get("/{run_id}/report", response_model=RunReportOut)
async def get_run_report(run_id: str, store: SqliteRunStore = Depends(get_store)) -> RunReportOut:
    record = store.get(run_id)
    if not record:
        raise HTTPException(status_code=404)
    if "report" not in record:
        raise HTTPException(status_code=404, detail="Report not ready")
    return RunReportOut(**record["report"])


@router.get("/{run_id}/download-zip")
@router.get("/{run_id}/zip")
async def download_run_artifacts_zip(
    run_id: str,
    store: SqliteRunStore = Depends(get_store),
):
    """Bundle report, config, sample metrics and artifacts into a single downloadable zip file."""
    import io
    import json
    import zipfile
    from pathlib import Path

    from fastapi.responses import Response

    from src.config import get_settings

    record = store.get(run_id)
    if not record:
        raise HTTPException(status_code=404, detail="Run not found")

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        report = record.get("report") or {}
        if report:
            zf.writestr("metrics_report.json", json.dumps(report, indent=2))
            model_name = report.get("model", "unknown")
            dataset_name = report.get("dataset", "unknown")
            ap_clean = report.get("ap_clean", 0.0)
            csv_lines = [
                "run_id,model,dataset,ap_clean,status",
                f"{run_id},{model_name},{dataset_name},{ap_clean},{record.get('status', 'COMPLETED')}",
            ]
            zf.writestr("summary.csv", "\n".join(csv_lines) + "\n")

        config = record.get("config") or {}
        if config:
            zf.writestr("run_config.json", json.dumps(config, indent=2))

        events = store.events(run_id)
        if events:
            zf.writestr("events.json", json.dumps(events, indent=2))

        cells = report.get("cells", [])
        if cells:
            zf.writestr("samples_diagnostics.json", json.dumps(cells, indent=2))

        settings = get_settings()
        for root_candidate in (Path(settings.artifact_root) / run_id, Path(settings.runs_root) / run_id):
            if root_candidate.is_dir():
                for file_path in root_candidate.glob("**/*"):
                    if file_path.is_file():
                        rel_name = f"artifacts/{file_path.relative_to(root_candidate)}"
                        zf.write(file_path, arcname=rel_name)

    zip_bytes = zip_buffer.getvalue()
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=adversai_run_{run_id}.zip"},
    )


@router.get("/{run_id}/download-pdf")
@router.get("/{run_id}/pdf")
async def download_run_report_pdf(
    run_id: str,
    store: SqliteRunStore = Depends(get_store),
):
    """Generate and stream a professional PDF evaluation report for the run."""
    from fastapi.responses import Response

    from src.evaluation.pdf_export import generate_run_report_pdf

    record = store.get(run_id)
    if not record:
        raise HTTPException(status_code=404, detail="Run not found")

    report = record.get("report")
    if not report:
        raise HTTPException(status_code=404, detail="Report not ready")

    pdf_bytes = generate_run_report_pdf(report, run_id=run_id)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=advertest_report_{run_id}.pdf"},
    )


@router.post("/{run_id}/cancel", response_model=RunJobOut)
async def cancel_run(
    run_id: str,
    store: SqliteRunStore = Depends(get_store),
) -> RunJobOut:
    record = store.get(run_id)
    if not record:
        raise HTTPException(status_code=404, detail="Run not found")
    if record["status"] not in ("COMPLETED", "FAILED", "CANCELLED"):
        store.request_cancel(run_id)
        if record["status"] == "QUEUED":
            store.fail(run_id, "Cancelled by user", cancelled=True)
    return _job_out(store.get(run_id))
