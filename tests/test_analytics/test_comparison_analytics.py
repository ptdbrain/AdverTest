"""Regression tests: comparison analytics may not recreate scientific metrics."""

from __future__ import annotations

from src.analytics.comparison_analytics import (
    compute_comparison_classes,
    compute_comparison_failures,
    compute_comparison_recovery,
    compute_comparison_summary,
)


def test_comparison_analytics_reads_canonical_report_only() -> None:
    comparison = {
        "comparison_id": "comparison-abc-123",
        "eligibility": {"status": "ELIGIBLE", "reasons": [], "next_action": "Human review."},
        "decision": {"status": "READY_FOR_REVIEW", "summary": "Evidence complete.", "next_action": "Review."},
        "metric_deltas": [{"key": "map", "attacked_delta": 0.1}],
        "recovery": {"metric_key": "map", "value": 0.5, "unbounded": True},
        "failures": {"baseline_count": 1, "candidate_count": 0, "recovered_count": 1, "regressed_count": 0},
    }
    raw_run_with_conflicting_numbers = {"report": {"ap_clean": 999, "cells": [{"ap": 999}]}}

    summary = compute_comparison_summary(comparison, raw_run_with_conflicting_numbers, raw_run_with_conflicting_numbers)
    assert summary["metric_deltas"] == comparison["metric_deltas"]
    assert compute_comparison_recovery(comparison)["value"] == 0.5
    assert compute_comparison_failures(comparison)["recovered_count"] == 1
    assert compute_comparison_classes(comparison) == []


def test_missing_canonical_report_is_not_converted_to_zero_metrics() -> None:
    summary = compute_comparison_summary({"comparison_id": "legacy"}, {"report": {"ap_clean": 1.0}})
    assert summary["eligibility"]["status"] == "NOT_ELIGIBLE"
    assert summary["metric_deltas"] == []
    assert compute_comparison_recovery({"comparison_id": "legacy"}) == {"reason": "CANONICAL_RECOVERY_MISSING"}
    assert compute_comparison_failures({"comparison_id": "legacy"}) == {"reason": "CANONICAL_FAILURE_SUMMARY_MISSING"}
