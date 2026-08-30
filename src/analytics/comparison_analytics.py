"""Read-only views of the canonical, evidence-gated defence report.

This module intentionally does not derive metrics from raw run reports. A
comparison can only be interpreted by ``src.evaluation.defense_report``,
which has already verified the paired protocol and provenance.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _canonical(comparison: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a persisted canonical report or a clear diagnostic shell."""
    if comparison.get("eligibility") and comparison.get("decision"):
        return comparison
    return {
        "comparison_id": comparison.get("comparison_id", ""),
        "eligibility": {
            "status": "NOT_ELIGIBLE",
            "reasons": ["CANONICAL_REPORT_MISSING"],
            "next_action": "Create an evidence-gated comparison before reading analytics.",
        },
        "decision": {
            "status": "NOT_ELIGIBLE",
            "summary": "No canonical comparison report is available.",
            "next_action": "Create an evidence-gated comparison before reading analytics.",
        },
    }


def compute_comparison_summary(
    comparison: Mapping[str, Any],
    baseline_run: Mapping[str, Any] | None = None,
    candidate_run: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Expose persisted deltas and decision; raw runs are deliberately ignored."""
    del baseline_run, candidate_run
    report = _canonical(comparison)
    return {
        "comparison_id": str(report.get("comparison_id", "")),
        "eligibility": report["eligibility"],
        "decision": report["decision"],
        "metric_deltas": list(report.get("metric_deltas", [])),
        "recovery": report.get("recovery"),
        "failures": report.get("failures"),
    }


def compute_comparison_recovery(
    comparison: Mapping[str, Any],
    baseline_run: Mapping[str, Any] | None = None,
    candidate_run: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return only the persisted recovery calculation, never a fallback value."""
    del baseline_run, candidate_run
    recovery = _canonical(comparison).get("recovery")
    if isinstance(recovery, Mapping):
        return dict(recovery)
    return {"reason": "CANONICAL_RECOVERY_MISSING"}


def compute_comparison_classes(
    comparison: Mapping[str, Any],
    baseline_run: Mapping[str, Any] | None = None,
    candidate_run: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Class values are displayed only when the canonical report stores them."""
    del baseline_run, candidate_run
    entries = _canonical(comparison).get("class_deltas", [])
    return [dict(item) for item in entries if isinstance(item, Mapping)]


def compute_comparison_failures(
    comparison: Mapping[str, Any],
    baseline_run: Mapping[str, Any] | None = None,
    candidate_run: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return persisted failure transitions without inferring them from hints."""
    del baseline_run, candidate_run
    failures = _canonical(comparison).get("failures")
    if isinstance(failures, Mapping):
        return dict(failures)
    return {"reason": "CANONICAL_FAILURE_SUMMARY_MISSING"}
