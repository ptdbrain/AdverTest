"""Contracts and schemas for the AI Next-Action Advisor."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

ActionType = Literal[
    "LABEL_DATASET",
    "RUN_CLEAN_BASELINE",
    "RUN_ATTACK",
    "RUN_HELDOUT_ATTACK",
    "GENERATE_DEFENCE_DATASET",
    "FINE_TUNE",
    "RE_EVALUATE",
    "REVIEW_CLASS_MAPPING",
    "UPLOAD_CHECKPOINT",
    "IMPORT_PRETRAINED",
    "ADJUST_TRAINING",
    "INVESTIGATE_FAR_RANGE_FAILURES",
]

Priority = Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]


class Recommendation(BaseModel):
    """Structured action recommendation generated from deterministic rules and evaluation evidence."""

    id: str
    priority: Priority
    action_type: ActionType
    title: str
    reason: str
    evidence: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    suggested_parameters: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class RecommendationListOut(BaseModel):
    """API response envelope for advisor recommendations."""

    recommendations: list[Recommendation]
    total_count: int
    project_id: str | None = None
    evaluated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class DismissRecommendationIn(BaseModel):
    """Input payload for dismissing an active recommendation."""

    recommendation_id: str
    reason: str | None = None
