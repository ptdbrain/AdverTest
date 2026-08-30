"""Analytics router: scientific evaluation, robustness curves, and paired comparison metrics."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from src.analytics.run_analytics import (
    compute_run_attacks_breakdown,
    compute_run_classes_breakdown,
    compute_run_distance_breakdown,
    compute_run_samples_breakdown,
    compute_run_severity_breakdown,
    compute_run_summary,
)
from src.api.defense_scope import require_scoped_record, require_scoped_run
from src.api.dependencies import get_store
from src.api.jobs import SqliteRunStore
from src.api.platform_dependencies import require_project_member

router = APIRouter(prefix="/analytics", tags=["Analytics"])


def _require_completed_run(store: SqliteRunStore, run_id: str, *, project_id: str) -> dict[str, Any]:
    item = require_scoped_run(store, run_id=run_id, project_id=project_id)
    if item.get("report") is None:
        raise HTTPException(
            status_code=409,
            detail=f"run {run_id!r} is {str(item.get('status', 'PENDING')).lower()}; report not ready",
        )
    evidence = item["report"].get("evidence") or {}
    if evidence.get("status") != "VERIFIED":
        raise HTTPException(
            status_code=409,
            detail={
                "code": "NOT_ELIGIBLE",
                "run_id": run_id,
                "evidence": evidence or {"status": "NOT_ELIGIBLE"},
            },
        )
    return item


def _require_comparison(store: SqliteRunStore, comparison_id: str, *, project_id: str) -> dict[str, Any]:
    return require_scoped_record(
        store,
        record_type="model_comparison",
        record_id=comparison_id,
        project_id=project_id,
    )


# ---- Single Run Analytics ----


@router.get("/runs/{run_id}/summary")
async def get_run_analytics_summary(
    run_id: str,
    project_id: str = Query(..., min_length=1),
    actor_id: str = Depends(require_project_member),
    store: SqliteRunStore = Depends(get_store),
) -> dict[str, Any]:
    """Get high-level executive robustness analytics for a completed run."""
    del actor_id
    run_item = _require_completed_run(store, run_id, project_id=project_id)
    return compute_run_summary(run_item["report"])


@router.get("/runs/{run_id}/attacks")
async def get_run_analytics_attacks(
    run_id: str,
    project_id: str = Query(..., min_length=1),
    actor_id: str = Depends(require_project_member),
    store: SqliteRunStore = Depends(get_store),
) -> list[dict[str, Any]]:
    """Get per-attack performance degradation breakdown across severities."""
    del actor_id
    run_item = _require_completed_run(store, run_id, project_id=project_id)
    return compute_run_attacks_breakdown(run_item["report"])


@router.get("/runs/{run_id}/severity")
@router.get("/runs/{run_id}/severities")
async def get_run_analytics_severity(
    run_id: str,
    project_id: str = Query(..., min_length=1),
    actor_id: str = Depends(require_project_member),
    store: SqliteRunStore = Depends(get_store),
) -> list[dict[str, Any]]:
    """Get robustness performance breakdown aggregated by severity ladder (1..5)."""
    del actor_id
    run_item = _require_completed_run(store, run_id, project_id=project_id)
    return compute_run_severity_breakdown(run_item["report"])


@router.get("/runs/{run_id}/classes")
async def get_run_analytics_classes(
    run_id: str,
    project_id: str = Query(..., min_length=1),
    actor_id: str = Depends(require_project_member),
    store: SqliteRunStore = Depends(get_store),
) -> list[dict[str, Any]]:
    """Get class-level detection rates and vulnerability breakdown."""
    del actor_id
    run_item = _require_completed_run(store, run_id, project_id=project_id)
    return compute_run_classes_breakdown(run_item["report"])


@router.get("/runs/{run_id}/samples")
@router.get("/runs/{run_id}/analytics/samples")
async def get_run_analytics_samples(
    run_id: str,
    attack: str | None = Query(default=None, description="Filter by attack name"),
    severity: int | None = Query(default=None, ge=1, le=5, description="Filter by severity"),
    limit: int | None = Query(default=50, ge=1, le=500, description="Max samples to return"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    project_id: str = Query(..., min_length=1),
    actor_id: str = Depends(require_project_member),
    store: SqliteRunStore = Depends(get_store),
) -> dict[str, Any]:
    """Get granular sample-level evidence and per-object failure diagnostics."""
    del actor_id
    run_item = _require_completed_run(store, run_id, project_id=project_id)
    return compute_run_samples_breakdown(
        run_item["report"],
        attack=attack,
        severity=severity,
        limit=limit,
        offset=offset,
    )


@router.get("/runs/{run_id}/distance")
async def get_run_analytics_distance(
    run_id: str,
    project_id: str = Query(..., min_length=1),
    actor_id: str = Depends(require_project_member),
    store: SqliteRunStore = Depends(get_store),
) -> dict[str, Any]:
    """Get 3D perception performance and vulnerability breakdown across distance buckets (near/medium/far)."""
    del actor_id
    run_item = _require_completed_run(store, run_id, project_id=project_id)
    return compute_run_distance_breakdown(run_item["report"])


# ---- Paired Model Comparison Analytics ----


@router.get("/comparisons/{comparison_id}/summary")
async def get_comparison_analytics_summary(
    comparison_id: str,
    project_id: str = Query(..., min_length=1),
    store: SqliteRunStore = Depends(get_store),
) -> dict[str, Any]:
    """Get executive comparison summary, score deltas, and promotion gate verdict."""
    comparison = _require_comparison(store, comparison_id, project_id=project_id)
    return {
        "comparison_id": comparison_id,
        "eligibility": comparison.get("eligibility"),
        "decision": comparison.get("decision"),
        "metric_deltas": comparison.get("metric_deltas", []),
        "recovery": comparison.get("recovery"),
        "failures": comparison.get("failures"),
    }


@router.get("/comparisons/{comparison_id}/recovery")
async def get_comparison_analytics_recovery(
    comparison_id: str,
    project_id: str = Query(..., min_length=1),
    store: SqliteRunStore = Depends(get_store),
) -> dict[str, Any]:
    """Get granular recovery curves broken down per attack and severity level."""
    comparison = _require_comparison(store, comparison_id, project_id=project_id)
    return comparison.get("recovery", {"reason": "REPORT_MISSING"})


@router.get("/comparisons/{comparison_id}/classes")
async def get_comparison_analytics_classes(
    comparison_id: str,
    project_id: str = Query(..., min_length=1),
    store: SqliteRunStore = Depends(get_store),
) -> list[dict[str, Any]]:
    """Get class-level accuracy shifts and per-class recovery comparisons."""
    _require_comparison(store, comparison_id, project_id=project_id)
    return []


@router.get("/comparisons/{comparison_id}/failures")
async def get_comparison_analytics_failures(
    comparison_id: str,
    project_id: str = Query(..., min_length=1),
    store: SqliteRunStore = Depends(get_store),
) -> dict[str, Any]:
    """Get failure transitions (FAILED->RECOVERED, FAILED->STILL_FAILED, CORRECT->FAILED)."""
    comparison = _require_comparison(store, comparison_id, project_id=project_id)
    return comparison.get(
        "failures",
        {"baseline_count": 0, "candidate_count": 0, "recovered_count": 0, "regressed_count": 0, "transitions": []},
    )
