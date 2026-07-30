"""AdverTest HTTP API: plugin catalog + test runs.

Runs execute synchronously in-process, which is fine for the reference dataset
and keeps the template dependency-free. Plan §4 replaces this with Redis +
Celery workers and PostgreSQL storage; the route contract is meant to survive
that move unchanged.

Open slots (plan §7, §8): RBAC for the Engineer / Reviewer roles, the review
queue with its four mandatory decisions, the audit log, WebSocket progress.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from src.adapters import load_adapters
from src.api.schemas import (
    AttackCatalogItem,
    CostEstimateOut,
    DatasetCatalogItem,
    ModelCatalogItem,
    RunReportOut,
    RunSummaryOut,
)
from src.attacks import load_attacks
from src.config import get_settings
from src.datasets import load_datasets
from src.evaluation.report import RunReport
from src.pipeline import RunConfig, TestRunner

router = APIRouter()

#: Shared so the prediction cache is reused across requests (plan §5).
_runner = TestRunner()
#: In-memory run store; swap for PostgreSQL per plan §4.
_runs: dict[str, RunReport] = {}


@router.get("/catalog/attacks", response_model=list[AttackCatalogItem])
async def list_attacks(
    group: str | None = Query(default=None, min_length=1, max_length=1),
    cost_class: str | None = None,
    modality: str | None = None,
) -> list[AttackCatalogItem]:
    """Every registered attack, with its parameter schema and owner."""
    items = [attack.describe() for attack in load_attacks().values()]
    for field, wanted in (("group", group), ("cost_class", cost_class), ("modality", modality)):
        if wanted is not None:
            items = [item for item in items if item[field] == wanted]
    return [AttackCatalogItem(**item) for item in items]


@router.get("/catalog/models", response_model=list[ModelCatalogItem])
async def list_models() -> list[ModelCatalogItem]:
    """Model adapters available for testing."""
    return [ModelCatalogItem(**adapter.describe()) for adapter in load_adapters().values()]


@router.get("/catalog/datasets", response_model=list[DatasetCatalogItem])
async def list_datasets() -> list[DatasetCatalogItem]:
    """Datasets; ``anonymized=false`` cannot enter a run (plan §6)."""
    return [DatasetCatalogItem(**dataset.describe()) for dataset in load_datasets().values()]


@router.post("/runs/estimate", response_model=CostEstimateOut)
async def estimate_run(config: RunConfig) -> CostEstimateOut:
    """Cost of a run, computed before anything is executed."""
    return CostEstimateOut(**_runner.estimate(config).as_dict())


@router.post("/runs", response_model=RunReportOut)
async def create_run(config: RunConfig) -> RunReportOut:
    """Execute a test run and return its report."""
    report = _runner.run(config)
    _runs[report.run_id] = report
    return RunReportOut(**report.as_dict())


@router.get("/runs", response_model=list[RunSummaryOut])
async def list_runs() -> list[RunSummaryOut]:
    """Compact list of the runs this process has executed."""
    return [_summarize(report) for report in _runs.values()]


@router.get("/runs/{run_id}", response_model=RunReportOut)
async def get_run(run_id: str) -> RunReportOut:
    """Full report of a previous run."""
    report = _runs.get(run_id)
    if report is None:
        raise HTTPException(status_code=404, detail=f"unknown run {run_id!r}")
    return RunReportOut(**report.as_dict())


def _summarize(report: RunReport) -> RunSummaryOut:
    """Row for the run list, including the human-in-the-loop flag."""
    worst = max((report.degradation(cell) for cell in report.cells), default=0.0)
    return RunSummaryOut(
        run_id=report.run_id,
        model=report.model,
        dataset=report.dataset,
        ap_clean=round(report.ap_clean, 4),
        n_cells=len(report.cells),
        worst_degradation=round(worst, 4),
        needs_review=worst >= get_settings().review_degradation_threshold,
    )
