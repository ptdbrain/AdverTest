
from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_runner, get_store, get_worker
from src.api.jobs import LocalRunWorker, SqliteRunStore
from src.api.platform_dependencies import get_platform_compute, get_platform_jobs
from src.api.routes import _resolve_run_config, _sample_with_artifact_urls
from src.api.schemas.run import CostEstimateOut, PreflightOut, RunJobOut, RunReportOut
from src.compute.backends import ComputeBackend
from src.config import get_settings
from src.jobs.service import PlatformJobService
from src.pipeline.runner import RunConfig, TestRunner

router = APIRouter(prefix="/runs", tags=["Runs"])

def _job_out(record: dict) -> RunJobOut:
    return RunJobOut(**record)


def _platform_job_out(job: dict) -> RunJobOut:
    status = job["stage"] if job["status"] == "RUNNING" or job["stage"] == "GPU_STARTING" else job["status"]
    detail = {"platform_job": True, "stage": job["stage"]}
    return RunJobOut(
        run_id=job["id"],
        status=status,
        progress=(job["completed_units"] / job["total_units"] if job["total_units"] else 0.0),
        detail=detail,
        report=job["result"],
        error=job["error_message"],
    )


def _remote_enabled() -> bool:
    return get_settings().run_execution_backend == "platform"

@router.post("/estimate", response_model=CostEstimateOut)
async def estimate_run(
    config: RunConfig,
    runner: TestRunner = Depends(get_runner)
) -> CostEstimateOut:
    est = runner.estimate(config).as_dict()
    # Add a custom warning if YOLO model requires download
    warnings = []
    if config.model.startswith("yolov8"):
        from pathlib import Path
        if not Path(f"{config.model}.pt").is_file():
            warnings.append(f"Model '{config.model}' requires downloading weights. Estimated time does not include download time.")
    return CostEstimateOut(**est, warnings=warnings)

@router.post("/preflight", response_model=PreflightOut)
async def preflight_run(
    config: RunConfig,
    runner: TestRunner = Depends(get_runner)
) -> PreflightOut:
    return PreflightOut(**runner.preflight(_resolve_run_config(config)).as_dict())

@router.post("", status_code=202, response_model=RunJobOut)
async def create_run(
    config: RunConfig,
    runner: TestRunner = Depends(get_runner),
    store: SqliteRunStore = Depends(get_store),
    worker: LocalRunWorker = Depends(get_worker),
    platform_jobs: PlatformJobService = Depends(get_platform_jobs),
    compute: ComputeBackend = Depends(get_platform_compute),
) -> RunJobOut:
    """Persist and enqueue a run. Heavy model work never runs in the request."""
    config = _resolve_run_config(config)
    preflight = runner.preflight(config)
    if preflight.fatal_errors:
        raise HTTPException(status_code=422, detail={"fatal_errors": list(preflight.fatal_errors)})
    if _remote_enabled():
        settings = get_settings()
        job = platform_jobs.create(
            project_id=settings.platform_default_project_id,
            owner_user_id=settings.platform_default_user_id,
            job_type="benchmark_run",
            request=config.model_dump(mode="json"),
            total_units=max(1, runner.estimate(config).n_cells),
        )
        # Keep the durable status QUEUED until the GCE worker claims it, while
        # giving the UI a truthful cold-start status immediately.
        platform_jobs.waiting_for_gpu(job["id"], "GPU đang khởi động, job sẽ tự chạy sau khi worker sẵn sàng...")
        compute.dispatch(job["id"])
        return _platform_job_out(platform_jobs.get(settings.platform_default_project_id, job["id"]) or job)
    run_id = store.create(config)
    worker.enqueue(run_id, config)
    return _job_out(store.get(run_id))

@router.get("", response_model=list[RunJobOut])
async def list_runs(
    store: SqliteRunStore = Depends(get_store)
) -> list[RunJobOut]:
    if _remote_enabled():
        settings = get_settings()
        from src.api.platform_dependencies import get_platform_jobs as _get_platform_jobs

        return [_platform_job_out(job) for job in _get_platform_jobs().list(settings.platform_default_project_id, job_type="benchmark_run")]
    return [_job_out(r) for r in store.list()]

@router.get("/{run_id}", response_model=RunJobOut)
async def get_run(
    run_id: str,
    store: SqliteRunStore = Depends(get_store)
) -> RunJobOut:
    record = store.get(run_id)
    if record is None and _remote_enabled():
        settings = get_settings()
        from src.api.platform_dependencies import get_platform_jobs as _get_platform_jobs

        job = _get_platform_jobs().get(settings.platform_default_project_id, run_id)
        if job is not None and job["type"] == "benchmark_run":
            return _platform_job_out(job)
    if not record:
        raise HTTPException(status_code=404, detail="Run not found")
    return _job_out(record)

@router.get("/{run_id}/report", response_model=RunReportOut)
async def get_run_report(
    run_id: str,
    store: SqliteRunStore = Depends(get_store)
) -> RunReportOut:
    record = store.get(run_id)
    if record is None and _remote_enabled():
        settings = get_settings()
        from src.api.platform_dependencies import get_platform_jobs as _get_platform_jobs

        job = _get_platform_jobs().get(settings.platform_default_project_id, run_id)
        if job is not None and job["type"] == "benchmark_run" and job["result"] is not None:
            return RunReportOut(**job["result"])
    if not record:
        raise HTTPException(status_code=404)
    if "report" not in record:
        raise HTTPException(status_code=404, detail="Report not ready")
    return RunReportOut(**record["report"])

@router.get("/{run_id}/samples")
async def get_run_samples(
    run_id: str,
    attack: str | None = None,
    severity: int | None = None,
    store: SqliteRunStore = Depends(get_store),
) -> list[dict]:
    """Return evidence from the same durable store as platform run status."""
    record = store.get(run_id)
    if record is None and _remote_enabled():
        settings = get_settings()
        record = get_platform_jobs().get(settings.platform_default_project_id, run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"unknown run {run_id!r}")
    report = record.get("report") if "report" in record else record.get("result")
    if report is None:
        raise HTTPException(status_code=409, detail="sample evidence is not available until the run completes")
    return [
        _sample_with_artifact_urls(sample)
        for sample in report.get("sample_results", [])
        if (attack is None or sample["attack"] == attack)
        and (severity is None or sample["severity"] == severity)
    ]

@router.get("/{run_id}/download-zip")
async def download_run_artifacts_zip(
    run_id: str,
    store: SqliteRunStore = Depends(get_store),
):
    """Bundle report, config, sample metrics and artifacts into a single downloadable zip file."""
    import io
    import json
    import zipfile
    from pathlib import Path

    from fastapi.responses import StreamingResponse

    from src.config import get_settings

    record = store.get(run_id)
    if not record:
        raise HTTPException(status_code=404, detail="Run not found")

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        if "report" in record and record["report"]:
            zf.writestr("metrics_report.json", json.dumps(record["report"], indent=2))

        if "config" in record:
            zf.writestr("run_config.json", json.dumps(record["config"], indent=2))

        samples = store.list_samples(run_id)
        if samples:
            zf.writestr("samples_diagnostics.json", json.dumps(samples, indent=2))

        settings = get_settings()
        artifacts_dir = Path(settings.artifact_root) / run_id
        if artifacts_dir.is_dir():
            for file_path in artifacts_dir.glob("**/*"):
                if file_path.is_file():
                    rel_name = f"artifacts/{file_path.relative_to(artifacts_dir)}"
                    zf.write(file_path, arcname=rel_name)

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=adversai_run_{run_id}.zip"}
    )

@router.post("/{run_id}/cancel", response_model=RunJobOut)
async def cancel_run(
    run_id: str,
    store: SqliteRunStore = Depends(get_store),
) -> RunJobOut:
    record = store.get(run_id)
    if record is None and _remote_enabled():
        settings = get_settings()
        from src.api.platform_dependencies import get_platform_jobs as _get_platform_jobs

        jobs = _get_platform_jobs()
        job = jobs.get(settings.platform_default_project_id, run_id)
        if job is not None and job["type"] == "benchmark_run":
            jobs.cancel(settings.platform_default_project_id, run_id)
            return _platform_job_out(jobs.get(settings.platform_default_project_id, run_id))
    if not record:
        raise HTTPException(status_code=404, detail="Run not found")
    if record["status"] not in ("COMPLETED", "FAILED", "CANCELLED"):
        store.request_cancel(run_id)
        if record["status"] == "QUEUED":
            store.fail(run_id, "Cancelled by user", cancelled=True)
    return _job_out(store.get(run_id))
