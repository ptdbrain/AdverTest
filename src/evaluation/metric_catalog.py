"""Protocol-aware metric definitions and immutable comparison identities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal


MetricUnit = Literal["ratio", "percent", "count"]


@dataclass(frozen=True, slots=True)
class MetricDefinition:
    key: str
    label: str
    unit: MetricUnit
    higher_is_better: bool
    protocol: str


_DETECTION_2D = (
    MetricDefinition("ap50", "AP@0.50", "ratio", True, "coco_detection"),
    MetricDefinition("map50_95", "mAP@0.50:0.95", "ratio", True, "coco_detection"),
)
_SEGMENTATION = (MetricDefinition("miou", "mIoU", "ratio", True, "semantic_segmentation"),)
_KITTI_3D = (
    MetricDefinition("car_3d_ap_0.7", "Car 3D AP@0.70", "ratio", True, "kitti_3d_ap"),
    MetricDefinition("pedestrian_3d_ap_0.5", "Pedestrian 3D AP@0.50", "ratio", True, "kitti_3d_ap"),
    MetricDefinition("cyclist_3d_ap_0.5", "Cyclist 3D AP@0.50", "ratio", True, "kitti_3d_ap"),
    MetricDefinition("bev_ap", "BEV AP", "ratio", True, "kitti_3d_ap"),
)
_NUSCENES = (
    MetricDefinition("nds", "nuScenes Detection Score", "ratio", True, "nuscenes_nds"),
    MetricDefinition("map", "nuScenes mAP", "ratio", True, "nuscenes_nds"),
)


def metric_definitions_for(
    task_id: str | None,
    dataset_name: str | None,
    *,
    metric_protocol: str | None,
) -> tuple[MetricDefinition, ...]:
    """Return metrics only when the declared task/dataset protocol is supported."""
    task = (task_id or "").lower()
    dataset = (dataset_name or "").lower()
    protocol = (metric_protocol or "").lower()
    if task == "detection2d" and protocol in {"coco_detection", "coco_ap", ""}:
        return _DETECTION_2D
    if task == "segmentation" and protocol in {"semantic_segmentation", "miou", ""}:
        return _SEGMENTATION
    if task == "detection3d" and "nuscenes" in dataset:
        return _NUSCENES if protocol == "nuscenes_nds" else ()
    if task == "detection3d" and "kitti" in dataset:
        return _KITTI_3D if protocol == "kitti_3d_ap" else ()
    return ()


def required_protocol_fields(task_id: str | None, dataset_name: str | None) -> frozenset[str]:
    """Return provenance fields required before a task can support a conclusion."""
    fields = {
        "dataset_version_id",
        "split_manifest_hash",
        "ground_truth_hash",
        "checkpoint_sha256",
        "config_sha256",
        "artifact_hashes",
        "benchmark_protocol_id",
        "metric_protocol",
        "metric_version",
        "recipe_cells",
        "seed",
        "thresholds",
        "preprocessing_hash",
        "class_mapping_hash",
    }
    if (task_id or "").lower() == "detection3d" and "nuscenes" in (dataset_name or "").lower():
        fields.add("nds_protocol")
    return frozenset(fields)


def build_pairing_signature(report: Mapping[str, Any]) -> dict[str, Any]:
    """Capture every immutable factor that may alter a before/after metric."""
    provenance = _mapping(report.get("provenance"))
    config = _mapping(provenance.get("run_config"))
    cells = report.get("cells") if isinstance(report.get("cells"), list) else []
    recipe_cells = sorted(
        {
            (
                str(cell.get("attack", "")),
                cell.get("severity"),
                str(cell.get("recipe_hash") or _mapping(cell.get("metrics")).get("recipe_hash") or ""),
            )
            for cell in cells
            if isinstance(cell, Mapping)
        }
    )
    sample_ids = sorted(
        str(item.get("sample_id"))
        for item in report.get("sample_results", [])
        if isinstance(item, Mapping) and item.get("sample_id")
    )
    return {
        "task_id": provenance.get("task_id") or config.get("task_id"),
        "dataset_version_id": provenance.get("dataset_version_id") or config.get("dataset_version_id") or report.get("dataset"),
        "split_manifest_hash": provenance.get("split_manifest_hash"),
        "ground_truth_hash": provenance.get("ground_truth_hash"),
        "benchmark_protocol_id": provenance.get("benchmark_protocol_id") or config.get("benchmark_protocol_id"),
        "recipe_cells": provenance.get("recipe_cells") or recipe_cells,
        "sample_ids": sample_ids,
        "seed": provenance.get("seed") if provenance.get("seed") is not None else config.get("seed"),
        "thresholds": provenance.get("thresholds") or config.get("thresholds"),
        "preprocessing_hash": provenance.get("preprocessing_hash") or config.get("preprocessing_hash"),
        "class_mapping_hash": provenance.get("class_mapping_hash") or config.get("class_mapping_hash"),
        "metric_protocol": provenance.get("metric_protocol"),
        "metric_version": provenance.get("metric_version"),
    }


def differing_signature_keys(left: Mapping[str, Any], right: Mapping[str, Any]) -> list[str]:
    """Return stable, human-actionable mismatch names without numeric coercion."""
    return [key for key in sorted(set(left) | set(right)) if left.get(key) != right.get(key)]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}
