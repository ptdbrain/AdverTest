"""AdverTest API: catalog plus durable, asynchronous test-run jobs."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect

from src.adapters import load_adapters
from src.api.jobs import LocalRunWorker, SqliteRunStore
from src.api.schemas import (
    AttackCatalogItem,
    CostEstimateOut,
    CreateReviewIn,
    DatasetCatalogItem,
    ModelCatalogItem,
    PreflightOut,
    ResolveReviewIn,
    ReviewOut,
    RunJobOut,
    RunReportOut,
)
from src.attacks import load_attacks
from src.config import get_settings
from src.datasets import load_datasets
from src.pipeline import RunConfig, TestRunner

router = APIRouter()
_runner = TestRunner()
_store = SqliteRunStore(get_settings().database_url)
_worker = LocalRunWorker(_store, max_workers=get_settings().worker_max_concurrency)
for _run_id, _config in _store.recoverable():
    _worker.enqueue(_run_id, _config)


@router.get("/catalog/attacks", response_model=list[AttackCatalogItem])
async def list_attacks(
    group: str | None = Query(default=None, min_length=1, max_length=1),
    cost_class: str | None = None,
    modality: str | None = None,
) -> list[AttackCatalogItem]:
    items = [attack.describe() for attack in load_attacks().values()]
    for field, wanted in (("group", group), ("cost_class", cost_class), ("modality", modality)):
        if wanted is not None:
            items = [item for item in items if item[field] == wanted]
    return [AttackCatalogItem(**item) for item in items]


@router.get("/catalog/models", response_model=list[ModelCatalogItem])
async def list_models() -> list[ModelCatalogItem]:
    return [ModelCatalogItem(**adapter.describe()) for adapter in load_adapters().values()]


@router.get("/catalog/datasets", response_model=list[DatasetCatalogItem])
async def list_datasets() -> list[DatasetCatalogItem]:
    return [DatasetCatalogItem(**dataset.describe()) for dataset in load_datasets().values()]


@router.post("/runs/estimate", response_model=CostEstimateOut)
async def estimate_run(config: RunConfig) -> CostEstimateOut:
    return CostEstimateOut(**_runner.estimate(config).as_dict())


@router.post("/runs/preflight", response_model=PreflightOut)
async def preflight_run(config: RunConfig) -> PreflightOut:
    return PreflightOut(**_runner.preflight(config).as_dict())


@router.post("/runs", status_code=202, response_model=RunJobOut)
async def create_run(config: RunConfig) -> RunJobOut:
    """Persist and enqueue a run. Heavy model work never runs in the request."""
    preflight = _runner.preflight(config)
    if preflight.fatal_errors:
        raise HTTPException(status_code=422, detail={"fatal_errors": list(preflight.fatal_errors)})
    run_id = _store.create(config)
    _worker.enqueue(run_id, config)
    return _job_out(_store.get(run_id))


@router.get("/runs", response_model=list[RunJobOut])
async def list_runs() -> list[RunJobOut]:
    return [_job_out(item) for item in _store.list()]


@router.get("/runs/{run_id}", response_model=RunJobOut)
async def get_run(run_id: str) -> RunJobOut:
    return _job_out(_require_run(run_id))


@router.get("/runs/{run_id}/report", response_model=RunReportOut)
async def get_report(run_id: str) -> RunReportOut:
    item = _require_run(run_id)
    if item["report"] is None:
        raise HTTPException(status_code=409, detail=f"run {run_id!r} is {item['status'].lower()}")
    return RunReportOut(**item["report"])


@router.get("/runs/{run_id}/samples")
async def list_samples(
    run_id: str,
    attack: str | None = None,
    severity: int | None = None,
) -> list[dict[str, Any]]:
    item = _require_run(run_id)
    report = item["report"]
    if report is None:
        raise HTTPException(status_code=409, detail="sample evidence is not available until the run completes")
    return [
        sample
        for sample in report.get("sample_results", [])
        if (attack is None or sample["attack"] == attack) and (severity is None or sample["severity"] == severity)
    ]


@router.post("/runs/{run_id}/cancel", response_model=RunJobOut)
async def cancel_run(run_id: str) -> RunJobOut:
    if not _store.request_cancel(run_id):
        raise HTTPException(status_code=404, detail=f"unknown run {run_id!r}")
    return _job_out(_require_run(run_id))


@router.post("/runs/{run_id}/flag-reviews")
async def auto_flag_reviews(run_id: str, threshold: float = Query(default=30.0)) -> dict[str, Any]:
    """Auto-create review items for cells exceeding degradation threshold."""
    _require_run(run_id)
    created = _store.auto_flag_reviews(run_id, threshold=threshold)
    return {"run_id": run_id, "created_reviews": created, "count": len(created)}


@router.websocket("/runs/{run_id}/events/ws")
async def run_events(run_id: str, websocket: WebSocket) -> None:
    await websocket.accept()
    if _store.get(run_id) is None:
        await websocket.send_json({"error": "unknown run"})
        await websocket.close(code=4404)
        return
    cursor = 0
    try:
        while True:
            for event in _store.events(run_id, cursor):
                cursor = event["event_id"]
                await websocket.send_json(event)
            status = _store.get(run_id)
            if status and status["status"] in {"COMPLETED", "FAILED", "CANCELLED"}:
                return
            await asyncio.sleep(0.25)
    except WebSocketDisconnect:
        return


# ---- Review endpoints ----

@router.get("/reviews", response_model=list[ReviewOut])
async def list_reviews(status: str | None = Query(default=None)) -> list[ReviewOut]:
    rows = _store.list_reviews(status=status)
    return [ReviewOut(**row) for row in rows]


@router.get("/reviews/{review_id}", response_model=ReviewOut)
async def get_review(review_id: str) -> ReviewOut:
    row = _store.get_review(review_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"unknown review {review_id!r}")
    return ReviewOut(**row)


@router.post("/reviews", status_code=201, response_model=ReviewOut)
async def create_review(body: CreateReviewIn) -> ReviewOut:
    review_id = _store.create_review(
        run_id=body.run_id,
        attack=body.attack,
        severity=body.severity,
        degradation=body.degradation,
        dataset=body.dataset,
        model=body.model,
        flagged_by=body.flagged_by,
        notes=body.notes,
    )
    row = _store.get_review(review_id)
    return ReviewOut(**row)


@router.patch("/reviews/{review_id}", response_model=ReviewOut)
async def resolve_review(review_id: str, body: ResolveReviewIn) -> ReviewOut:
    success = _store.resolve_review(review_id, body.decision, body.decision_note, body.resolved_by)
    if not success:
        raise HTTPException(status_code=404, detail=f"unknown review {review_id!r}")
    row = _store.get_review(review_id)
    return ReviewOut(**row)


# ---- Helpers ----

def _require_run(run_id: str) -> dict[str, Any]:
    item = _store.get(run_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"unknown run {run_id!r}")
    return item


def _job_out(item: dict[str, Any] | None) -> RunJobOut:
    if item is None:
        raise HTTPException(status_code=404, detail="unknown run")
    report = RunReportOut(**item["report"]) if item["report"] else None
    return RunJobOut(
        run_id=item["run_id"],
        status=item["status"],
        progress=item["progress"],
        detail=item["detail"],
        report=report,
        error=item["error"],
    )
