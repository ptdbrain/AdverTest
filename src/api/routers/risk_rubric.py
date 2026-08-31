"""Risk rubric API: risk assessment, rubric table, and session risk summaries."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.jobs import SqliteRunStore
from src.evaluation.risk_rubric import (
    RISK_RUBRIC_TABLE,
    assess_risk,
    generate_risk_summary,
)

router = APIRouter(prefix="/risk-rubric", tags=["Risk Rubric"])


def _get_current_store() -> SqliteRunStore:
    """Resolve the active store after application/test dependency reloads."""
    from src.api.dependencies import get_store

    return get_store()


@router.get("")
async def get_risk_rubric() -> list[dict[str, Any]]:
    """Return the standard risk classification rubric table."""
    return list(RISK_RUBRIC_TABLE)


@router.post("/assess")
async def assess_review_risk(
    review_id: str = Query(...),
    store: SqliteRunStore = Depends(_get_current_store),
) -> dict[str, Any]:
    """Compute a risk assessment for a specific review item."""
    review = store.get_review(review_id)
    if review is None:
        raise HTTPException(status_code=404, detail=f"unknown review {review_id!r}")

    run_id = review.get("run_id", "")
    run = store.get(run_id) if run_id else None
    report = run.get("report") if run else None

    assessment = assess_risk(review, report=report)

    return {
        "review_id": review_id,
        "risk_level": assessment.risk_level,
        "risk_category": assessment.risk_category,
        "recommended_decision": assessment.recommended_decision,
        "recommended_decision_label": assessment.recommended_decision_label,
        "justification": assessment.justification,
        "downstream_actions": list(assessment.downstream_actions),
        "confidence": assessment.confidence,
        "contributing_factors": list(assessment.contributing_factors),
    }


@router.get("/session-summary")
async def get_risk_session_summary(
    run_id: str = Query(default=None),
    status: str = Query(default="PENDING"),
    store: SqliteRunStore = Depends(_get_current_store),
) -> dict[str, Any]:
    """Aggregate risk summary across reviews, optionally filtered by run or status."""
    reviews = store.list_reviews(status=status if status != "ALL" else None)

    if run_id:
        reviews = [r for r in reviews if r.get("run_id") == run_id]

    summary = generate_risk_summary(reviews)

    return {
        "total_reviews": summary.total_reviews,
        "critical_count": summary.critical_count,
        "high_count": summary.high_count,
        "medium_count": summary.medium_count,
        "low_count": summary.low_count,
        "auto_pass_count": summary.auto_pass_count,
        "dominant_risk_level": summary.dominant_risk_level,
        "overall_recommendation": summary.overall_recommendation,
        "top_affected_classes": list(summary.top_affected_classes),
        "top_attack_groups": list(summary.top_attack_groups),
    }
