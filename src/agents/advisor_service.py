"""Advisor service providing high-level AI next-action recommendations with safety constraints."""

from __future__ import annotations

from src.agents.contracts import RecommendationListOut
from src.agents.rule_engine import evaluate_rules
from src.api.jobs import SqliteRunStore


class AIAdvisorService:
    """Service providing proactive next-action recommendations based on deterministic rules."""

    _dismissed_ids: set[str] = set()

    def __init__(self, store: SqliteRunStore) -> None:
        self._store = store

    def get_recommendations(
        self,
        *,
        project_id: str | None = None,
        run_id: str | None = None,
    ) -> RecommendationListOut:
        """Evaluate project state and return non-dismissed recommendations."""
        all_recs = evaluate_rules(self._store, project_id=project_id, run_id=run_id)
        active_recs = [r for r in all_recs if r.id not in self._dismissed_ids]

        return RecommendationListOut(
            recommendations=active_recs,
            total_count=len(active_recs),
            project_id=project_id,
        )

    def dismiss(self, recommendation_id: str) -> None:
        """Mark a recommendation as dismissed for the active session."""
        self._dismissed_ids.add(recommendation_id)

    def clear_dismissed(self) -> None:
        """Clear all dismissed recommendation IDs for session reset."""
        self._dismissed_ids.clear()
