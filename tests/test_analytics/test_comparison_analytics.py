"""Unit tests for paired model comparison analytics."""

from __future__ import annotations

import pytest

from src.analytics.comparison_analytics import (
    compute_comparison_classes,
    compute_comparison_failures,
    compute_comparison_recovery,
    compute_comparison_summary,
)


@pytest.fixture
def mock_comparison_fixture() -> tuple[dict, dict, dict]:
    comparison = {
        "comparison_id": "comparison-abc-123",
        "baseline_run_id": "run-base-1",
        "candidate_run_id": "run-cand-2",
        "paired": True,
        "incompatibilities": [],
        "metric_deltas": {
            "clean_detection_score": {"value": 0.01, "unit": "ratio"},
        },
        "recovery_report": {
            "baseline_clean": 0.80,
            "candidate_clean": 0.81,
            "recovery_rate": {"ratio_value": 0.40, "percent_value": 40.0, "unit": "percent"},
        },
    }

    base_run = {
        "run_id": "run-base-1",
        "report": {
            "ap_clean": 0.80,
            "cells": [
                {"attack": "gaussian_noise", "severity": 1, "ap": 0.50, "group": "B"},
                {"attack": "fog", "severity": 2, "ap": 0.40, "group": "A"},
            ],
            "sample_results": [
                {
                    "sample_id": "img_01",
                    "attack": "gaussian_noise",
                    "severity": 1,
                    "degradation_hint": 0.30,
                    "failed": True,
                    "object_evidence": [
                        {"label": "car", "clean_detected": True, "attacked_detected": False}
                    ],
                },
                {
                    "sample_id": "img_02",
                    "attack": "fog",
                    "severity": 2,
                    "degradation_hint": 0.45,
                    "failed": True,
                    "object_evidence": [
                        {"label": "bus", "clean_detected": True, "attacked_detected": False}
                    ],
                },
            ],
        },
    }

    cand_run = {
        "run_id": "run-cand-2",
        "report": {
            "ap_clean": 0.81,
            "cells": [
                {"attack": "gaussian_noise", "severity": 1, "ap": 0.70, "group": "B"},
                {"attack": "fog", "severity": 2, "ap": 0.45, "group": "A"},
            ],
            "sample_results": [
                {
                    "sample_id": "img_01",
                    "attack": "gaussian_noise",
                    "severity": 1,
                    "degradation_hint": 0.0,
                    "failed": False,
                    "object_evidence": [
                        {"label": "car", "clean_detected": True, "attacked_detected": True}
                    ],
                },
                {
                    "sample_id": "img_02",
                    "attack": "fog",
                    "severity": 2,
                    "degradation_hint": 0.40,
                    "failed": True,
                    "object_evidence": [
                        {"label": "bus", "clean_detected": True, "attacked_detected": False}
                    ],
                },
            ],
        },
    }

    return comparison, base_run, cand_run


def test_compute_comparison_summary(mock_comparison_fixture) -> None:
    comparison, base_run, cand_run = mock_comparison_fixture
    summary = compute_comparison_summary(comparison, base_run, cand_run)

    assert summary["comparison_id"] == "comparison-abc-123"
    assert summary["paired"] is True
    assert summary["baseline_clean_ap"] == 0.80
    assert summary["candidate_clean_ap"] == 0.81
    assert summary["clean_ap_delta"] == 0.01
    assert summary["attacked_ap_delta"] > 0.0
    assert summary["verdict"] == "PROMOTABLE"


def test_compute_comparison_recovery(mock_comparison_fixture) -> None:
    comparison, base_run, cand_run = mock_comparison_fixture
    recovery = compute_comparison_recovery(comparison, base_run, cand_run)

    assert recovery["overall_recovery"]["ratio_value"] is not None
    assert len(recovery["per_attack_recovery"]) == 2

    noise_rec = next(item for item in recovery["per_attack_recovery"] if item["attack"] == "gaussian_noise")
    assert noise_rec["delta_ap"] == 0.20  # 0.70 - 0.50
    assert noise_rec["status"] == "PARTIAL_RECOVERY" or noise_rec["status"] == "RECOVERED"

    assert recovery["clean_retention"]["retained_percent"] >= 100.0


def test_compute_comparison_classes(mock_comparison_fixture) -> None:
    comparison, base_run, cand_run = mock_comparison_fixture
    classes = compute_comparison_classes(comparison, base_run, cand_run)

    assert len(classes) == 2
    car_item = next(item for item in classes if item["class_name"] == "car")
    assert car_item["baseline_attacked_rate"] == 0.0
    assert car_item["candidate_attacked_rate"] == 1.0
    assert car_item["status"] == "IMPROVED"


def test_compute_comparison_failures(mock_comparison_fixture) -> None:
    comparison, base_run, cand_run = mock_comparison_fixture
    failures = compute_comparison_failures(comparison, base_run, cand_run)

    assert failures["total_baseline_failures"] == 2
    assert failures["total_candidate_failures"] == 1
    assert failures["recovered_count"] == 1
    assert failures["still_failed_count"] == 1
    assert failures["regressed_count"] == 0
    assert failures["net_failure_reduction"] == 1

    assert len(failures["recovered_failures"]) == 1
    assert failures["recovered_failures"][0]["sample_id"] == "img_01"
    assert failures["recovered_failures"][0]["transition"] == "FAILED->RECOVERED"
