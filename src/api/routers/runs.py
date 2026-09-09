
import hashlib
import hmac
import json
import secrets
import uuid
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.catalog_assets import resolve_catalog_dataset
from src.api.dependencies import get_runner, get_store, get_worker
from src.api.helpers import resolve_run_config as _resolve_run_config
from src.api.helpers import sample_with_artifact_urls as _sample_with_artifact_urls
from src.api.jobs import LocalRunWorker, SqliteRunStore
from src.api.platform_dependencies import (
    get_platform_compute,
    get_platform_jobs,
    require_run_project_member,
)
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
_ESTIMATE_TOKEN_SECRET = secrets.token_bytes(32)


def _estimate_payload(config: RunConfig) -> bytes:
    """Canonical workload fields bound to an estimate confirmation token."""
    payload = config.model_dump(mode="json", exclude={"estimate_token", "confirmed"})
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _estimate_token(config: RunConfig) -> str:
    return hmac.new(_ESTIMATE_TOKEN_SECRET, _estimate_payload(config), hashlib.sha256).hexdigest()


def _verify_estimate_token(config: RunConfig) -> None:
    if not config.estimate_token:
        return
    if not hmac.compare_digest(config.estimate_token, _estimate_token(config)):
        raise HTTPException(status_code=409, detail="ESTIMATE_MISMATCH: request differs from the reviewed workload estimate.")


def _worker_request_payload(config: RunConfig) -> dict:
    """Return only scientific run fields for workers.

    ``confirmed`` and ``estimate_token`` are control-plane metadata used to
    gate submission.  They are intentionally not part of the worker contract:
    older GPU workers reject unknown fields with Pydantic's ``extra=forbid``.
    Keeping them out of the durable request also prevents a token from being
    persisted alongside executable workload configuration.
    """
    return config.model_dump(mode="json", exclude={"estimate_token", "confirmed"})


def _catalog_runtime_params(config: RunConfig, runtime: Any) -> dict[str, Any]:
    """Merge catalog paths with bounded loader options supplied by the caller."""
    params = dict(runtime.as_params())
    if "sample_offset" in config.dataset_params:
        params["sample_offset"] = config.dataset_params["sample_offset"]
    return params


def _hydrate_selected_catalog_bundle(config: RunConfig) -> RunConfig:
    """Hydrate only the reviewed bundle selected by this run.

    The control plane must validate the exact same anonymisation manifest as
    the worker, but downloading every catalog during Render startup delays
    readiness.  Only catalog-owned roots are eligible; user-provided paths
    never trigger storage reads here.
    """
    selection = _DRIVE_CATALOG_BUNDLES.get(config.dataset)
    if selection is None:
        return config
    settings = get_settings()
    bundle_name, prefix_attribute = selection
    expected_root = (Path(settings.data_root).expanduser().resolve() / "catalog" / bundle_name)
    requested_root = config.dataset_params.get("root")
    runtime = resolve_catalog_dataset(config.dataset, settings)
    if runtime is None:
        return config
    local_root = runtime.root
    legacy_root = Path("/app/data/catalog") / bundle_name
    requested = Path(str(requested_root)).expanduser().resolve() if requested_root else None
    allowed = {expected_root, local_root, legacy_root.resolve()}
    if requested is not None and requested not in allowed:
        return config
    if runtime.root == local_root and runtime.runnable:
        return config.model_copy(update={"dataset_params": _catalog_runtime_params(config, runtime)})
    if (expected_root / "dataset.json").is_file() and (expected_root / "manifest.jsonl").is_file():
        return config.model_copy(update={"dataset_params": _catalog_runtime_params(config, runtime)})
    with _CATALOG_HYDRATION_LOCK:
        if (expected_root / "dataset.json").is_file() and (expected_root / "manifest.jsonl").is_file():
            return config.model_copy(update={"dataset_params": _catalog_runtime_params(config, runtime)})
        from src.api.platform_dependencies import get_platform_storage

        ensure_drive_export_bundle(
            storage=get_platform_storage(),
            storage_prefix=getattr(settings, prefix_attribute),
            data_root=settings.data_root,
            bundle_name=bundle_name,
        )
    refreshed = resolve_catalog_dataset(config.dataset, settings)
    return config.model_copy(
        update={"dataset_params": _catalog_runtime_params(config, refreshed) if refreshed else config.dataset_params}
    )

def _job_out(record: dict) -> RunJobOut:
    return RunJobOut(**record)


def _run_project_id(record: dict) -> str | None:
    config = record.get("config")
    return config.get("project_id") if isinstance(config, dict) else None


def _ensure_local_run_scope(record: dict, project_id: str | None) -> None:
    run_proj = _run_project_id(record)
    if project_id and run_proj and run_proj != project_id:
        raise HTTPException(status_code=404, detail="Run not found")


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
    config = _hydrate_selected_catalog_bundle(config)
    est = runner.estimate(config).as_dict()
    # Add a custom warning if YOLO model requires download
    warnings = []
    if config.model.startswith("yolov8"):
        from pathlib import Path
        if not Path(f"{config.model}.pt").is_file():
            warnings.append(f"Model '{config.model}' requires downloading weights. Estimated time does not include download time.")
    return CostEstimateOut(**est, estimate_token=_estimate_token(config), warnings=warnings)

@router.post("/preflight", response_model=PreflightOut)
async def preflight_run(
    config: RunConfig,
    project_id: str | None = Query(default=None),
    actor_id: str | None = Depends(require_run_project_member),
    runner: TestRunner = Depends(get_runner),
) -> PreflightOut:
    if _remote_enabled():
        scoped_project_id = _project_scope(project_id, actor_id)
        if config.project_id and config.project_id != scoped_project_id:
            raise HTTPException(status_code=403, detail="PROJECT_ASSET_SCOPE_MISMATCH")
        config = config.model_copy(update={"project_id": scoped_project_id})
    config = _resolve_run_config(config)
    config = _hydrate_selected_catalog_bundle(config)
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
    if not config.model_fields_set:
        raise HTTPException(status_code=422, detail="CONFIRMATION_REQUIRED: request an estimate and submit an explicit workload.")
    _verify_estimate_token(config)
    if not config.evidence_dir:
        evidence_dir = (
            Path(get_settings().data_root).expanduser().resolve()
            / "runs"
            / "evidence"
            / uuid.uuid4().hex
        )
        config = config.model_copy(update={"evidence_dir": str(evidence_dir)})
    scoped_project_id = None
    if _remote_enabled():
        scoped_project_id = _project_scope(project_id, actor_id)
        if config.project_id and config.project_id != scoped_project_id:
            raise HTTPException(status_code=403, detail="PROJECT_ASSET_SCOPE_MISMATCH")
        config = config.model_copy(update={"project_id": scoped_project_id})
    else:
        effective_project_id = project_id or config.project_id
        if effective_project_id:
            config = config.model_copy(update={"project_id": effective_project_id})
    config = _resolve_run_config(config)
    config = _hydrate_selected_catalog_bundle(config)
    preflight = runner.preflight(config)
    if preflight.fatal_errors:
        raise HTTPException(status_code=422, detail={"fatal_errors": list(preflight.fatal_errors)})
    if _remote_enabled():
        job = platform_jobs.create(
            project_id=scoped_project_id,
            owner_user_id=actor_id,
            job_type="benchmark_run",
            request=_worker_request_payload(config),
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
def list_runs(
    project_id: str | None = Query(default=None),
    actor_id: str | None = Depends(require_run_project_member),
    store: SqliteRunStore = Depends(get_store),
) -> list[RunJobOut]:
    if _remote_enabled():
        scoped_project_id = _project_scope(project_id, actor_id)
        from src.api.platform_dependencies import get_platform_jobs as _get_platform_jobs

        return [_platform_job_out(job) for job in _get_platform_jobs().list(scoped_project_id, job_type="benchmark_run")]
    return [
        _job_out(r)
        for r in store.list()
        if not project_id or _run_project_id(r) == project_id or _run_project_id(r) is None
    ]

@router.get("/{run_id}", response_model=RunJobOut)
def get_run(
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
    if not _remote_enabled():
        _ensure_local_run_scope(record, project_id)
    return _job_out(record)

@router.get("/{run_id}/report", response_model=RunReportOut)
def get_run_report(
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
    if not _remote_enabled():
        _ensure_local_run_scope(record, project_id)
    if "report" not in record:
        raise HTTPException(status_code=404, detail="Report not ready")
    return RunReportOut(**record["report"])


@router.get("/{run_id}/samples")
def get_run_samples(
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
    if not _remote_enabled():
        _ensure_local_run_scope(record, project_id)
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
def cancel_run(
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
    if not _remote_enabled():
        _ensure_local_run_scope(record, project_id)
    if record["status"] not in ("COMPLETED", "FAILED", "CANCELLED"):
        store.request_cancel(run_id)
        if record["status"] == "QUEUED":
            store.fail(run_id, "Cancelled by user", cancelled=True)
    return _job_out(store.get(run_id))
