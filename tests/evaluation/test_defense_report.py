from __future__ import annotations

from copy import deepcopy

from src.evaluation.defense_report import build_defense_report


def _benchmark_run(run_id: str, *, clean: float = 0.70, attacked: float = 0.20) -> dict[str, object]:
    return {
        "run_id": run_id,
        "status": "COMPLETED",
        "report": {
            "dataset": "kitti",
            "simulation_only": False,
            "metrics": {
                "clean": {
                    "car_3d_ap_0.7": clean,
                    "pedestrian_3d_ap_0.5": clean,
                    "cyclist_3d_ap_0.5": clean,
                    "bev_ap": clean,
                }
            },
            "cells": [
                {
                    "attack": "fog",
                    "severity": 3,
                    "recipe_hash": "fog-v1",
                    "metrics": {
                        "car_3d_ap_0.7": attacked,
                        "pedestrian_3d_ap_0.5": attacked,
                        "cyclist_3d_ap_0.5": attacked,
                        "bev_ap": attacked,
                    },
                }
            ],
            "sample_results": [{"sample_id": "000001"}],
            "provenance": {
                "task_id": "detection3d",
                "dataset_version_id": "kitti-3d-v1",
                "split_manifest_hash": "split-sha",
                "ground_truth_hash": "gt-sha",
                "checkpoint_sha256": f"checkpoint-{run_id}",
                "config_sha256": f"config-{run_id}",
                "artifact_hashes": {"prediction": f"artifact-{run_id}"},
                "benchmark_protocol_id": "protocol-kitti",
                "recipe_cells": [["fog", 3, "fog-v1"]],
                "seed": 195,
                "thresholds": {"iou": 0.7},
                "preprocessing_hash": "pre-sha",
                "class_mapping_hash": "classes-sha",
                "metric_protocol": "kitti_3d_ap",
                "metric_version": "kitti-3d-ap-v1",
            },
        },
    }


def test_missing_ground_truth_blocks_a_benchmark_conclusion() -> None:
    baseline = _benchmark_run("baseline")
    candidate = _benchmark_run("candidate")
    del candidate["report"]["provenance"]["ground_truth_hash"]

    report = build_defense_report(project_id="project-a", baseline_run=baseline, candidate_run=candidate)

    assert report.eligibility.status == "NOT_ELIGIBLE"
    assert "GROUND_TRUTH_HASH_MISSING" in report.eligibility.reasons
    assert report.decision.status == "NOT_ELIGIBLE"


def test_recovery_stays_unbounded_when_defence_exceeds_clean_baseline() -> None:
    baseline = _benchmark_run("baseline", clean=0.70, attacked=0.20)
    candidate = _benchmark_run("candidate", clean=0.90, attacked=0.90)

    report = build_defense_report(project_id="project-a", baseline_run=baseline, candidate_run=candidate)

    assert report.eligibility.status == "ELIGIBLE"
    assert report.recovery.value > 1.0
    assert report.recovery.percent_value > 100.0


def test_demo_or_unpaired_source_cannot_be_promoted() -> None:
    baseline = _benchmark_run("baseline")
    candidate = deepcopy(_benchmark_run("candidate"))
    candidate["report"]["source_kind"] = "demo"
    candidate["report"]["provenance"]["class_mapping_hash"] = "different-classes"

    report = build_defense_report(project_id="project-a", baseline_run=baseline, candidate_run=candidate)

    assert report.eligibility.status == "NOT_ELIGIBLE"
    assert "DEMO_SOURCE" in report.eligibility.reasons
    assert "UNPAIRED_PROTOCOL" in report.eligibility.reasons
    assert "class_mapping_hash" in report.incompatibilities
