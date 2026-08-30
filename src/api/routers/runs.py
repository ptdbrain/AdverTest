import hashlib
import hmac
import json
import time

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from src.api.dependencies import get_runner, get_store, get_worker
from src.api.jobs import LocalRunWorker, SqliteRunStore
from src.api.platform_dependencies import get_platform_artifacts, require_project_member
from src.api.routes import _resolve_run_config
from src.api.schemas.run import CostEstimateOut, PreflightOut, RunJobOut, RunReportOut
from src.config import get_settings
from src.pipeline.runner import RunConfig, TestRunner
from src.storage.service import ArtifactService

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
    project_id: str,
    actor_id: str = Depends(require_project_member),
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
    run_id = store.create(config, project_id=project_id, owner_user_id=actor_id)
    worker.enqueue(run_id, config)
    return _job_out(store.get(run_id))


@router.get("", response_model=list[RunJobOut])
async def list_runs(
    project_id: str,
    actor_id: str = Depends(require_project_member),
    store: SqliteRunStore = Depends(get_store),
) -> list[RunJobOut]:
    del actor_id
    return [_job_out(r) for r in store.list_scoped(project_id=project_id)]


@router.get("/{run_id}", response_model=RunJobOut)
async def get_run(
    run_id: str,
    project_id: str,
    actor_id: str = Depends(require_project_member),
    store: SqliteRunStore = Depends(get_store),
) -> RunJobOut:
    del actor_id
    record = store.get_scoped(run_id, project_id=project_id)
    if not record:
        raise HTTPException(status_code=404, detail="Run not found")
    return _job_out(record)


@router.get("/{run_id}/report", response_model=RunReportOut)
async def get_run_report(
    run_id: str,
    project_id: str,
    actor_id: str = Depends(require_project_member),
    store: SqliteRunStore = Depends(get_store),
) -> RunReportOut:
    del actor_id
    record = store.get_scoped(run_id, project_id=project_id)
    if not record:
        raise HTTPException(status_code=404)
    if "report" not in record:
        raise HTTPException(status_code=404, detail="Report not ready")
    return RunReportOut(**record["report"])


@router.get("/{run_id}/download-zip")
@router.get("/{run_id}/zip")
async def download_run_artifacts_zip(
    run_id: str,
    project_id: str,
    actor_id: str = Depends(require_project_member),
    store: SqliteRunStore = Depends(get_store),
    artifacts: ArtifactService = Depends(get_platform_artifacts),
):
    """Bundle report, config, sample metrics and artifacts into a single downloadable zip file."""
    import io
    import json
    import zipfile

    from fastapi.responses import Response

    record = store.get_scoped(run_id, project_id=project_id)
    if not record:
        raise HTTPException(status_code=404, detail="Run not found")

    report = record.get("report")
    if not report:
        raise HTTPException(status_code=409, detail={"code": "REPORT_NOT_READY"})
    evidence = _report_evidence(report)
    if evidence["status"] != "VERIFIED":
        raise HTTPException(
            status_code=409,
            detail={"code": "NOT_ELIGIBLE_FOR_CONCLUSION_EXPORT", "evidence": evidence},
        )

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("metrics_report.json", json.dumps(report, indent=2))
        model_name = report.get("model", "unknown")
        dataset_name = report.get("dataset", "unknown")
        ap_clean = report.get("ap_clean", "")
        csv_lines = [
            "run_id,model,dataset,ap_clean,status,evidence_status",
            f"{run_id},{model_name},{dataset_name},{ap_clean},{record.get('status', 'COMPLETED')},VERIFIED",
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

        for artifact in artifacts.list_for_run(project_id, run_id, actor_id=actor_id):
            try:
                content = artifacts.read_bytes(project_id, artifact["id"], actor_id=actor_id)
            except (FileNotFoundError, KeyError):
                continue
            filename = _safe_zip_filename(artifact["original_filename"])
            zf.writestr(f"artifacts/{artifact['id']}_{filename}", content)

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
    project_id: str,
    actor_id: str = Depends(require_project_member),
    store: SqliteRunStore = Depends(get_store),
):
    """Generate and stream a professional PDF evaluation report for the run."""
    from fastapi.responses import Response

    from src.evaluation.pdf_export import generate_run_report_pdf

    del actor_id
    record = store.get_scoped(run_id, project_id=project_id)
    if not record:
        raise HTTPException(status_code=404, detail="Run not found")

    report = record.get("report")
    if not report:
        raise HTTPException(status_code=404, detail="Report not ready")
    evidence = _report_evidence(report)
    if evidence["status"] != "VERIFIED":
        raise HTTPException(
            status_code=409,
            detail={"code": "NOT_ELIGIBLE_FOR_CONCLUSION_EXPORT", "evidence": evidence},
        )

    pdf_bytes = generate_run_report_pdf(report, run_id=run_id)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=advertest_report_{run_id}.pdf"},
    )


@router.get("/{run_id}/export")
async def export_run_report_data(
    run_id: str,
    format: str,
    project_id: str,
    actor_id: str = Depends(require_project_member),
    store: SqliteRunStore = Depends(get_store),
) -> Response:
    """Export canonical JSON or CSV only when the server verifies evidence."""
    del actor_id
    record = store.get_scoped(run_id, project_id=project_id)
    if not record:
        raise HTTPException(status_code=404, detail="Run not found")
    report = record.get("report")
    if not report:
        raise HTTPException(status_code=409, detail={"code": "REPORT_NOT_READY"})
    evidence = _report_evidence(report)
    if evidence["status"] != "VERIFIED":
        raise HTTPException(
            status_code=409,
            detail={"code": "NOT_ELIGIBLE_FOR_CONCLUSION_EXPORT", "evidence": evidence},
        )
    if format == "json":
        return Response(
            json.dumps(report, indent=2),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=advertest_report_{run_id}.json"},
        )
    if format == "csv":
        model_name = report.get("model", "unknown")
        dataset_name = report.get("dataset", "unknown")
        ap_clean = report.get("ap_clean", "")
        csv_payload = "\n".join(
            [
                "run_id,model,dataset,ap_clean,status,evidence_status",
                f"{run_id},{model_name},{dataset_name},{ap_clean},{record.get('status', 'COMPLETED')},VERIFIED",
                "",
            ]
        )
        return Response(
            csv_payload,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=advertest_report_{run_id}.csv"},
        )
    raise HTTPException(status_code=422, detail={"code": "EXPORT_FORMAT_UNSUPPORTED", "format": format})


@router.post("/{run_id}/promote", status_code=201)
async def promote_run(
    run_id: str,
    project_id: str,
    actor_id: str = Depends(require_project_member),
    store: SqliteRunStore = Depends(get_store),
) -> dict:
    """Allow a promotion request only after the report's shared evidence gate passes."""
    del actor_id
    record = store.get_scoped(run_id, project_id=project_id)
    if not record:
        raise HTTPException(status_code=404, detail="Run not found")
    report = record.get("report")
    if not report:
        raise HTTPException(status_code=409, detail={"code": "REPORT_NOT_READY"})
    evidence = _report_evidence(report)
    if evidence["status"] != "VERIFIED":
        raise HTTPException(status_code=409, detail={"code": "NOT_ELIGIBLE", "evidence": evidence})
    return {"run_id": run_id, "project_id": project_id, "status": "PROMOTION_ELIGIBLE", "evidence": evidence}


@router.post("/{run_id}/cancel", response_model=RunJobOut)
async def cancel_run(
    run_id: str,
    project_id: str,
    actor_id: str = Depends(require_project_member),
    store: SqliteRunStore = Depends(get_store),
) -> RunJobOut:
    del actor_id
    record = store.get_scoped(run_id, project_id=project_id)
    if not record:
        raise HTTPException(status_code=404, detail="Run not found")
    if record["status"] not in ("COMPLETED", "FAILED", "CANCELLED"):
        store.request_cancel(run_id)
        if record["status"] == "QUEUED":
            store.fail(run_id, "Cancelled by user", cancelled=True)
    return _job_out(store.get(run_id))


def _report_evidence(report: dict) -> dict:
    """Recompute evidence instead of trusting a persisted or browser-provided label."""
    from src.evaluation.evidence import evaluate_evidence

    return evaluate_evidence(
        report.get("provenance") or {},
        simulation_only=bool(report.get("simulation_only", True)),
    ).as_dict()




def _safe_zip_filename(value: object) -> str:
    from pathlib import PurePath

    filename = PurePath(str(value or "artifact")).name
    return filename or "artifact"
