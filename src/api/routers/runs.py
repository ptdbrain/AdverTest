from fastapi import APIRouter, Depends, HTTPException
from src.pipeline.runner import TestRunner, RunConfig
from src.api.jobs import SqliteRunStore, LocalRunWorker
from src.api.dependencies import get_runner, get_store, get_worker
from src.api.schemas.run import CostEstimateOut, PreflightOut, RunJobOut, RunReportOut
import copy

from src.api.routes import _resolve_run_config

router = APIRouter(prefix="/runs", tags=["Runs"])

def _job_out(record: dict) -> RunJobOut:
    return RunJobOut(**record)

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
    worker: LocalRunWorker = Depends(get_worker)
) -> RunJobOut:
    """Persist and enqueue a run. Heavy model work never runs in the request."""
    config = _resolve_run_config(config)
    preflight = runner.preflight(config)
    if preflight.fatal_errors:
        raise HTTPException(status_code=422, detail={"fatal_errors": list(preflight.fatal_errors)})
    run_id = store.create(config)
    worker.enqueue(run_id, config)
    return _job_out(store.get(run_id))

@router.get("", response_model=list[RunJobOut])
async def list_runs(
    store: SqliteRunStore = Depends(get_store)
) -> list[RunJobOut]:
    return [_job_out(r) for r in store.list()]

@router.get("/{run_id}", response_model=RunJobOut)
async def get_run(
    run_id: str,
    store: SqliteRunStore = Depends(get_store)
) -> RunJobOut:
    record = store.get(run_id)
    if not record:
        raise HTTPException(status_code=404, detail="Run not found")
    return _job_out(record)

@router.get("/{run_id}/report", response_model=RunReportOut)
async def get_run_report(
    run_id: str,
    store: SqliteRunStore = Depends(get_store)
) -> RunReportOut:
    record = store.get(run_id)
    if not record:
        raise HTTPException(status_code=404)
    if "report" not in record:
        raise HTTPException(status_code=404, detail="Report not ready")
    return RunReportOut(**record["report"])

@router.post("/{run_id}/cancel", response_model=RunJobOut)
async def cancel_run(
    run_id: str,
    store: SqliteRunStore = Depends(get_store),
    worker: LocalRunWorker = Depends(get_worker)
) -> RunJobOut:
    record = store.get(run_id)
    if not record:
        raise HTTPException(status_code=404, detail="Run not found")
    if record["status"] in ("PENDING", "RUNNING"):
        worker.cancel(run_id)
        store.update_status(run_id, "CANCELLED")
    return _job_out(store.get(run_id))
