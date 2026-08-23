"""Advisor router: proactive next-action recommendations and evidence-based guidance."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from src.agents.advisor_service import AIAdvisorService
from src.agents.contracts import DismissRecommendationIn, RecommendationListOut
from src.api.dependencies import get_advisor_service

router = APIRouter(prefix="/advisor", tags=["AI Advisor"])


@router.get("/recommendations", response_model=RecommendationListOut)
async def list_recommendations(
    project_id: str | None = Query(default=None, description="Filter recommendations by project"),
    run_id: str | None = Query(default=None, description="Filter recommendations by specific run"),
    advisor: AIAdvisorService = Depends(get_advisor_service),
) -> RecommendationListOut:
    """Get active, rule-evaluated AI recommendations for next actions (benchmarking, defenses, repair)."""
    return advisor.get_recommendations(project_id=project_id, run_id=run_id)


@router.post("/dismiss")
async def dismiss_recommendation(
    payload: DismissRecommendationIn,
    advisor: AIAdvisorService = Depends(get_advisor_service),
) -> dict[str, str]:
    """Dismiss a specific recommendation from future evaluation in the current session."""
    advisor.dismiss(payload.recommendation_id)
    return {"status": "dismissed", "recommendation_id": payload.recommendation_id}
