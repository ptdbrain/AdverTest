"""Unit tests for single-run analytics aggregations."""

from __future__ import annotations

import pytest

from src.analytics.run_analytics import (
    compute_run_attacks_breakdown,
    compute_run_classes_breakdown,
    compute_run_samples_breakdown,
    compute_run_severity_breakdown,
    compute_run_summary,
)


@pytest.fixture
def mock_run_report() -> dict:
    return {
        "run_id": "run-test-123",
        "model": "yolo11s",
        "model_version": "v1.0.0",
        "dataset": "synthetic_shapes",
        "n_samples": 20,
        "ap_clean": 0.85,
        "cells": [
            {
                "attack": "gaussian_noise",
                "group": "B",
                "severity": 1,
                "ap": 0.80,
                "n_samples": 20,
                "seconds": 1.2,
                "category": "noise",
            },
            {
                "attack": "gaussian_noise",
                "group": "B",
                "severity": 3,
                "ap": 0.60,
                "n_samples": 20,
                "seconds": 1.5,
                "category": "noise",
            },
            {
                "attack": "fog",
                "group": "A",
                "severity": 2,
                "ap": 0.70,
                "n_samples": 20,
                "seconds": 2.0,
                "category": "weather",
            },
            {
                "attack": "fog",
                "group": "A",
                "severity": 4,
                "ap": 0.40,
                "n_samples": 20,
                "seconds": 2.5,
                "category": "weather",
            },
        ],
        "skipped": [{"attack": "snow", "reason": "UNSUPPORTED_MODALITY"}],
        "sample_results": [
            {
                "sample_id": "img_001",
                "attack": "gaussian_noise",
                "severity": 1,
                "clean_prediction": {"boxes": [[0, 0, 10, 10]], "labels": ["car"]},
                "attacked_prediction": {"boxes": [[0, 0, 10, 10]], "labels": ["car"]},
                "object_evidence": [
                    {
                        "ground_truth_label": "car",
                        "clean_detected": True,
                        "attacked_detected": True,
                        "transition": "CORRECT->CORRECT",
                    }
                ],
                "degradation_hint": 0.0,
            },
            {
                "sample_id": "img_002",
                "attack": "fog",
                "severity": 4,
                "clean_prediction": {"boxes": [[0, 0, 20, 20]], "labels": ["pedestrian"]},
                "attacked_prediction": {"boxes": [], "labels": []},
                "object_evidence": [
                    {
                        "ground_truth_label": "pedestrian",
                        "clean_detected": True,
                        "attacked_detected": False,
                        "transition": "CORRECT->FAILED",
                    }
                ],
                "degradation_hint": 0.55,
            },
        ],
        "metrics": {"clean": {"ap50": 0.85, "ap75": 0.75, "map50_95": 0.65}},
        "seconds": 7.2,
    }


def test_compute_run_summary_happy_path(mock_run_report: dict) -> None:
    summary = compute_run_summary(mock_run_report)

    assert summary["run_id"] == "run-test-123"
    assert summary["ap_clean"] == 0.85
    assert summary["total_cells_evaluated"] == 4
    assert summary["total_attacks_evaluated"] == 2
    assert summary["total_skipped_attacks"] == 1
    assert summary["worst_attack"]["attack"] == "fog"
    assert summary["most_resilient_attack"]["attack"] == "gaussian_noise"
    assert "A" in summary["group_summaries"]
    assert "B" in summary["group_summaries"]
    assert summary["robust_score"] > 0.0


def test_compute_run_summary_empty_report() -> None:
    empty_report = {
        "run_id": "empty-1",
        "ap_clean": 0.0,
        "cells": [],
        "sample_results": [],
    }
    summary = compute_run_summary(empty_report)

    assert summary["run_id"] == "empty-1"
    assert summary["total_cells_evaluated"] == 0
    assert summary["worst_attack"] is None
    assert summary["group_summaries"] == {}


def test_compute_run_attacks_breakdown(mock_run_report: dict) -> None:
    breakdown = compute_run_attacks_breakdown(mock_run_report)

    assert len(breakdown) == 2
    # fog had lower AP (0.70 & 0.40 -> mean 0.55) so higher degradation than gaussian_noise (0.80 & 0.60 -> mean 0.70)
    assert breakdown[0]["attack"] == "fog"
    assert breakdown[0]["mean_ap"] == 0.55
    assert breakdown[0]["worst_severity"] == 4
    assert breakdown[0]["failure_sample_count"] == 1

    assert breakdown[1]["attack"] == "gaussian_noise"
    assert breakdown[1]["mean_ap"] == 0.70
    assert breakdown[1]["worst_severity"] == 3


def test_compute_run_severity_breakdown(mock_run_report: dict) -> None:
    breakdown = compute_run_severity_breakdown(mock_run_report)

    assert len(breakdown) == 4
    severities = [item["severity"] for item in breakdown]
    assert severities == [1, 2, 3, 4]

    sev4 = next(item for item in breakdown if item["severity"] == 4)
    assert sev4["worst_attack"] == "fog"
    assert sev4["mean_ap"] == 0.40
    assert sev4["worst_attack_degradation_percent"] > 50.0


def test_compute_run_classes_breakdown(mock_run_report: dict) -> None:
    classes = compute_run_classes_breakdown(mock_run_report)

    assert len(classes) == 2
    car = next(item for item in classes if item["class_name"] == "car")
    ped = next(item for item in classes if item["class_name"] == "pedestrian")

    assert car["clean_detection_rate"] == 1.0
    assert car["attacked_detection_rate"] == 1.0
    assert car["lost_objects_count"] == 0
    assert car["unaffected_objects_count"] == 1

    assert ped["clean_detection_rate"] == 1.0
    assert ped["attacked_detection_rate"] == 0.0
    assert ped["lost_objects_count"] == 1


def test_compute_run_samples_breakdown(mock_run_report: dict) -> None:
    samples = compute_run_samples_breakdown(mock_run_report)

    assert samples["total_samples"] == 2
    assert samples["filtered_samples_count"] == 2
    # Highest degradation hint sorted first
    assert samples["items"][0]["sample_id"] == "img_002"
    assert samples["items"][0]["object_transitions"] == ["CORRECT->FAILED"]
    assert samples["items"][1]["sample_id"] == "img_001"

    # Test filtering by attack
    fog_samples = compute_run_samples_breakdown(mock_run_report, attack="fog")
    assert fog_samples["filtered_samples_count"] == 1
    assert fog_samples["items"][0]["sample_id"] == "img_002"
