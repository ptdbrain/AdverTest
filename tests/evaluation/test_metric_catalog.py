from __future__ import annotations

from src.evaluation.metric_catalog import (
    build_pairing_signature,
    metric_definitions_for,
    required_protocol_fields,
)


def _report() -> dict[str, object]:
    return {
        "dataset": "kitti",
        "sample_results": [{"sample_id": "000001"}],
        "cells": [{"attack": "fog", "severity": 3, "recipe_hash": "fog-v1"}],
        "provenance": {
            "task_id": "detection3d",
            "dataset_version_id": "kitti-3d-v1",
            "split_manifest_hash": "split-sha",
            "ground_truth_hash": "gt-sha",
            "benchmark_protocol_id": "protocol-kitti",
            "seed": 195,
            "thresholds": {"iou": 0.7},
            "preprocessing_hash": "pre-sha",
            "class_mapping_hash": "classes-sha",
            "metric_protocol": "kitti_3d_ap",
            "metric_version": "kitti-3d-ap-v1",
        },
    }


def test_nuscenes_requires_nds_before_metric_definitions_are_available() -> None:
    assert metric_definitions_for("detection3d", "nuscenes", metric_protocol=None) == ()
    assert "nds_protocol" in required_protocol_fields("detection3d", "nuscenes")


def test_kitti_3d_catalog_exposes_ap_and_bev_with_explicit_thresholds() -> None:
    definitions = metric_definitions_for("detection3d", "kitti", metric_protocol="kitti_3d_ap")

    assert {item.key for item in definitions} == {"car_3d_ap_0.7", "pedestrian_3d_ap_0.5", "cyclist_3d_ap_0.5", "bev_ap"}
    assert all(item.higher_is_better for item in definitions)


def test_pairing_signature_changes_for_class_mapping_and_attack_cell_identity() -> None:
    report = _report()
    changed_class_mapping = _report()
    changed_class_mapping["provenance"] = {**changed_class_mapping["provenance"], "class_mapping_hash": "other"}
    changed_cell = _report()
    changed_cell["cells"] = [{"attack": "fog", "severity": 4, "recipe_hash": "fog-v1"}]

    assert build_pairing_signature(report) != build_pairing_signature(changed_class_mapping)
    assert build_pairing_signature(report) != build_pairing_signature(changed_cell)
