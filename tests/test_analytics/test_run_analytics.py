"""Unit tests for single-run analytics aggregations."""

from __future__ import annotations

import pytest

from src.analytics.run_analytics import (
    compute_run_attacks_breakdown,
    compute_run_classes_breakdown,
    compute_run_distance_breakdown,
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
                "metrics": {"map50_95": 0.80},
            },
            {
                "attack": "gaussian_noise",
                "group": "B",
                "severity": 3,
                "ap": 0.60,
                "n_samples": 20,
                "seconds": 1.5,
                "category": "noise",
                "metrics": {"map50_95": 0.60},
            },
            {
                "attack": "fog",
                "group": "A",
                "severity": 2,
                "ap": 0.70,
                "n_samples": 20,
                "seconds": 2.0,
                "category": "weather",
                "metrics": {"map50_95": 0.70},
            },
            {
                "attack": "fog",
                "group": "A",
                "severity": 4,
                "ap": 0.40,
                "n_samples": 20,
                "seconds": 2.5,
                "category": "weather",
                "metrics": {"map50_95": 0.40},
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
    assert summary["ap_clean"] == 0.65
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
    assert summary["primary_metric"]["value"] is None
    assert summary["robust_score"] is None
    assert summary["data_state"] == "NO_DATA"


@pytest.mark.parametrize(
    ("task_id", "clean_metrics", "cell_metrics", "metric_key", "label"),
    [
        ("detection2d", {"map50_95": 0.51}, {"map50_95": 0.31}, "map50_95", "mAP@50-95"),
        ("segmentation", {"miou": 0.68}, {"miou": 0.44}, "miou", "mIoU"),
        ("detection3d", {"kitti_3d_ap": 0.59}, {"kitti_3d_ap": 0.36}, "kitti_3d_ap", "KITTI 3D AP"),
    ],
)
def test_compute_run_summary_is_task_aware(
    task_id: str,
    clean_metrics: dict,
    cell_metrics: dict,
    metric_key: str,
    label: str,
) -> None:
    report = {
        "run_id": f"run-{task_id}",
        "ap_clean": next(iter(clean_metrics.values())),
        "n_samples": 10,
        "cells": [
            {
                "attack": "fog",
                "group": "A",
                "severity": 3,
                "ap": next(iter(cell_metrics.values())),
                "metrics": cell_metrics,
            }
        ],
        "metrics": {"clean": clean_metrics},
        "provenance": {
            "run_config": {"task_id": task_id, "benchmark_protocol_id": "locked-v1"},
            "dataset_version_id": "dataset-sha256:abc",
        },
        "simulation_only": False,
        "benchmark_metrics_available": True,
    }

    summary = compute_run_summary(report)

    assert summary["task_id"] == task_id
    assert summary["primary_metric"] == {
        "key": metric_key,
        "label": label,
        "unit": "ratio",
        "clean": next(iter(clean_metrics.values())),
        "attacked_mean": next(iter(cell_metrics.values())),
        "value": next(iter(cell_metrics.values())),
    }
    assert summary["data_state"] == "MEASURED"
    assert summary["protocol"]["benchmark_protocol_id"] == "locked-v1"


def test_compute_run_summary_marks_quick_inference_as_not_benchmarkable() -> None:
    summary = compute_run_summary(
        {
            "run_id": "quick-1",
            "ap_clean": 0.0,
            "cells": [],
            "metrics": {"benchmark_metrics_available": False},
            "benchmark_metrics_available": False,
            "simulation_only": True,
        }
    )

    assert summary["data_state"] == "NO_GROUND_TRUTH"
    assert summary["primary_metric"]["value"] is None
    assert "Ground-truth annotations" in summary["limitations"][0]


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
    assert sev4["worst_attack_degradation_percent"] > 30.0


def test_summary_does_not_mix_ap50_with_missing_map50_95_cells() -> None:
    summary = compute_run_summary(
        {
            "run_id": "mixed-metric-run",
            "ap_clean": 0.9,
            "metrics": {"clean": {"map50_95": 0.55}},
            "cells": [{"attack": "fog", "severity": 3, "ap": 0.4, "metrics": {"ap50": 0.4}}],
            "provenance": {"run_config": {"task_id": "detection2d"}},
            "benchmark_metrics_available": True,
        }
    )

    assert summary["data_state"] == "NO_DATA"
    assert summary["primary_metric"]["clean"] == 0.55
    assert summary["primary_metric"]["attacked_mean"] is None
    assert summary["robust_score"] is None


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


def test_class_breakdown_requires_object_level_ground_truth_evidence() -> None:
    report = {
        "sample_results": [
            {
                "clean_prediction": {"labels": ["car", "car"]},
                "attacked_prediction": {"labels": []},
            }
        ]
    }

    assert compute_run_classes_breakdown(report) == []


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


def test_distance_breakdown_does_not_invent_empty_3d_buckets() -> None:
    result = compute_run_distance_breakdown({"run_id": "2d-run", "metrics": {"clean": {}}, "cells": []})

    assert result["data_state"] == "NO_DATA"
    assert result["most_vulnerable_distance"] is None
    assert all(bucket["clean_ap"] is None for bucket in result["buckets"].values())
