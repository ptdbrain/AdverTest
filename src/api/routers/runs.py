
from pathlib import Path
from threading import Lock

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.dependencies import get_runner, get_store, get_worker
from src.api.jobs import LocalRunWorker, SqliteRunStore
from src.api.platform_dependencies import (
    get_platform_compute,
    get_platform_jobs,
    require_run_project_member,
)
from src.api.routes import _resolve_run_config, _sample_with_artifact_urls
from src.api.schemas.run import CostEstimateOut, PreflightOut, RunJobOut, RunReportOut
from src.compute.backends import ComputeBackend
from src.config import get_settings
from src.demo_bootstrap import ensure_drive_export_bundle
from src.jobs.service import PlatformJobService
from src.pipeline.runner import RunConfig, TestRunner

router = APIRouter(prefix="/runs", tags=["Runs"])

_CATALOG_HYDRATION_LOCK = Lock()
_DRIVE_CATALOG_BUNDLES = {
    "kitti": ("kitti2d-100", "drive_export_kitti2d_storage_prefix"),
    "cityscapes_segmentation": ("cityscapes-instance-100", "drive_export_cityscapes_storage_prefix"),
    "nuscenes": ("nuscenes-mini-100", "drive_export_nuscenes_storage_prefix"),
}


def _hydrate_selected_catalog_bundle(config: RunConfig) -> None:
    """Hydrate only the reviewed bundle selected by this run.

    The control plane must validate the exact same anonymisation manifest as
    the worker, but downloading every catalog during Render startup delays
    readiness.  Only catalog-owned roots are eligible; user-provided paths
    never trigger storage reads here.
    """
    selection = _DRIVE_CATALOG_BUNDLES.get(config.dataset)
    if selection is None:
        return
    settings = get_settings()
    bundle_name, prefix_attribute = selection
    expected_root = (Path(settings.data_root).expanduser().resolve() / "catalog" / bundle_name)
    requested_root = config.dataset_params.get("root")
    if requested_root is None or Path(str(requested_root)).expanduser().resolve() != expected_root:
        return
    if (expected_root / "dataset.json").is_file() and (expected_root / "manifest.jsonl").is_file():
        return
    with _CATALOG_HYDRATION_LOCK:
        if (expected_root / "dataset.json").is_file() and (expected_root / "manifest.jsonl").is_file():
            return
        from src.api.platform_dependencies import get_platform_storage

        ensure_drive_export_bundle(
            storage=get_platform_storage(),
            storage_prefix=getattr(settings, prefix_attribute),
            data_root=settings.data_root,
            bundle_name=bundle_name,
        )

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


def _project_scope(project_id: str | None, actor_id: str | None) -> str:
    """Resolve the only project a platform run request may access.

    Product requests must explicitly select a project and prove current
    membership through the HttpOnly session.  The local runner intentionally
    retains its lightweight, unauthenticated test contract.
    """
    if not _remote_enabled():
        return get_settings().platform_default_project_id
    if not actor_id:
        raise HTTPException(status_code=401, detail="AUTHENTICATION_REQUIRED: Sign in before accessing project runs.")
    # ``require_run_project_member`` has already validated the authenticated
    # actor against this exact query project_id before the route executes.
    return project_id

@router.post("/estimate", response_model=CostEstimateOut)
async def estimate_run(
    config: RunConfig,
    runner: TestRunner = Depends(get_runner)
) -> CostEstimateOut:
    _hydrate_selected_catalog_bundle(config)
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
    config = _resolve_run_config(config)
    _hydrate_selected_catalog_bundle(config)
    return PreflightOut(**runner.preflight(config).as_dict())

@router.post("", status_code=202, response_model=RunJobOut)
async def create_run(
    config: RunConfig,
    project_id: str | None = Query(default=None),
    actor_id: str | None = Depends(require_run_project_member),
    runner: TestRunner = Depends(get_runner),
    store: SqliteRunStore = Depends(get_store),
    worker: LocalRunWorker = Depends(get_worker),
    platform_jobs: PlatformJobService = Depends(get_platform_jobs),
    compute: ComputeBackend = Depends(get_platform_compute),
) -> RunJobOut:
    """Persist and enqueue a run. Heavy model work never runs in the request."""
    config = _resolve_run_config(config)
    _hydrate_selected_catalog_bundle(config)
    preflight = runner.preflight(config)
    if preflight.fatal_errors:
        raise HTTPException(status_code=422, detail={"fatal_errors": list(preflight.fatal_errors)})
    if _remote_enabled():
        scoped_project_id = _project_scope(project_id, actor_id)
        job = platform_jobs.create(
            project_id=scoped_project_id,
            owner_user_id=actor_id,
            job_type="benchmark_run",
            request=config.model_dump(mode="json"),
            total_units=max(1, runner.estimate(config).n_cells),
        )
        # Keep the durable status QUEUED until the GCE worker claims it, while
        # giving the UI a truthful cold-start status immediately.
        platform_jobs.waiting_for_gpu(job["id"], "GPU đang khởi động, job sẽ tự chạy sau khi worker sẵn sàng...")
        compute.dispatch(job["id"])
        return _platform_job_out(platform_jobs.get(scoped_project_id, job["id"]) or job)
    run_id = store.create(config)
    worker.enqueue(run_id, config)
    return _job_out(store.get(run_id))

@router.get("", response_model=list[RunJobOut])
async def list_runs(
    project_id: str | None = Query(default=None),
    actor_id: str | None = Depends(require_run_project_member),
    store: SqliteRunStore = Depends(get_store),
) -> list[RunJobOut]:
    if _remote_enabled():
        scoped_project_id = _project_scope(project_id, actor_id)
        from src.api.platform_dependencies import get_platform_jobs as _get_platform_jobs

        return [_platform_job_out(job) for job in _get_platform_jobs().list(scoped_project_id, job_type="benchmark_run")]
    return [_job_out(r) for r in store.list()]

@router.get("/{run_id}", response_model=RunJobOut)
async def get_run(
    run_id: str,
    project_id: str | None = Query(default=None),
    actor_id: str | None = Depends(require_run_project_member),
    store: SqliteRunStore = Depends(get_store)
) -> RunJobOut:
    record = store.get(run_id)
    if record is None and _remote_enabled():
        scoped_project_id = _project_scope(project_id, actor_id)
        from src.api.platform_dependencies import get_platform_jobs as _get_platform_jobs

        job = _get_platform_jobs().get(scoped_project_id, run_id)
        if job is not None and job["type"] == "benchmark_run":
            return _platform_job_out(job)
    if not record:
        raise HTTPException(status_code=404, detail="Run not found")
    return _job_out(record)

@router.get("/{run_id}/report", response_model=RunReportOut)
async def get_run_report(
    run_id: str,
    project_id: str | None = Query(default=None),
    actor_id: str | None = Depends(require_run_project_member),
    store: SqliteRunStore = Depends(get_store)
) -> RunReportOut:
    record = store.get(run_id)
    if record is None and _remote_enabled():
        scoped_project_id = _project_scope(project_id, actor_id)
        from src.api.platform_dependencies import get_platform_jobs as _get_platform_jobs

        job = _get_platform_jobs().get(scoped_project_id, run_id)
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
    project_id: str | None = Query(default=None),
    actor_id: str | None = Depends(require_run_project_member),
    store: SqliteRunStore = Depends(get_store),
) -> list[dict]:
    """Return evidence from the same durable store as platform run status."""
    record = store.get(run_id)
    if record is None and _remote_enabled():
        scoped_project_id = _project_scope(project_id, actor_id)
        record = get_platform_jobs().get(scoped_project_id, run_id)
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

@router.post("/{run_id}/cancel", response_model=RunJobOut)
async def cancel_run(
    run_id: str,
    project_id: str | None = Query(default=None),
    actor_id: str | None = Depends(require_run_project_member),
    store: SqliteRunStore = Depends(get_store),
) -> RunJobOut:
    record = store.get(run_id)
    if record is None and _remote_enabled():
        scoped_project_id = _project_scope(project_id, actor_id)
        from src.api.platform_dependencies import get_platform_jobs as _get_platform_jobs

        jobs = _get_platform_jobs()
        job = jobs.get(scoped_project_id, run_id)
        if job is not None and job["type"] == "benchmark_run":
            jobs.cancel(scoped_project_id, run_id)
            return _platform_job_out(jobs.get(scoped_project_id, run_id))
    if not record:
        raise HTTPException(status_code=404, detail="Run not found")
    if record["status"] not in ("COMPLETED", "FAILED", "CANCELLED"):
        store.request_cancel(run_id)
        if record["status"] == "QUEUED":
            store.fail(run_id, "Cancelled by user", cancelled=True)
    return _job_out(store.get(run_id))
