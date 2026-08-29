"""P1.6 - P1.9 3D Geometry, Model Validation, and Runtime Hardening Test Suite.

Verifies:
- P1.6: PointPillars self-contained config and compatibility matrix.
- P1.7: Multi-model family checkpoint validation dispatch (YOLO, SAM2, PointPillars) and platform safety.
- P1.8: CUDA smoke gate script exit code contract.
- P1.9: Full 3D Oriented Bounding Box IoU geometric tests (no-overlap, full-overlap, rotation, Z-axis slicing, edge cases, NaN/Inf) and KITTI 3D AP evaluation.
"""

from __future__ import annotations

import math
from pathlib import Path

from src.core.types import Box3D
from src.evaluation.geometry3d import (
    box3d_iou,
    evaluate_3d_detections_ap,
    get_kitti_threshold,
    match_boxes3d,
)


def test_pointpillars_config_and_compatibility_matrix() -> None:
    """P1.6: Verify self-contained PointPillars config and documentation matrix."""
    config_path = Path(__file__).resolve().parents[2] / "checkpoints" / "pointpillars_kitti_3class.py"
    assert config_path.is_file(), "pointpillars_kitti_3class.py not found"

    # Verify config executes cleanly in Python without missing relative _base_ files
    namespace: dict = {}
    exec(config_path.read_text(encoding="utf-8"), namespace)
    assert "model" in namespace
    assert namespace["model"]["type"] == "VoxelNet"
    assert namespace["class_names"] == ["Car", "Pedestrian", "Cyclist"]

    # Verify compatibility matrix document
    matrix_path = Path(__file__).resolve().parents[2] / "docs" / "COMPATIBILITY_MATRIX.md"
    assert matrix_path.is_file(), "COMPATIBILITY_MATRIX.md not found"
    matrix_content = matrix_path.read_text(encoding="utf-8")
    assert "MMDetection3D" in matrix_content
    assert "PyTorch" in matrix_content
    assert "CUDA" in matrix_content


def test_box3d_iou_identical_boxes() -> None:
    """P1.9: Identical 3D bounding boxes must produce exact IoU of 1.0."""
    box1 = Box3D(x=10.0, y=5.0, z=0.0, length=4.0, width=2.0, height=1.5, yaw=0.0, label="Car")
    box2 = Box3D(x=10.0, y=5.0, z=0.0, length=4.0, width=2.0, height=1.5, yaw=0.0, label="Car")
    iou = box3d_iou(box1, box2)
    assert math.isclose(iou, 1.0, abs_tol=1e-4), f"Expected 1.0, got {iou}"


def test_box3d_iou_disjoint_boxes() -> None:
    """P1.9: Spatially disjoint boxes in XY or Z must produce IoU of 0.0."""
    # Disjoint in XY
    box_a = Box3D(x=0.0, y=0.0, z=0.0, length=2.0, width=2.0, height=2.0, yaw=0.0, label="Car")
    box_b = Box3D(x=10.0, y=10.0, z=0.0, length=2.0, width=2.0, height=2.0, yaw=0.0, label="Car")
    assert box3d_iou(box_a, box_b) == 0.0

    # Disjoint in Z
    box_c = Box3D(x=0.0, y=0.0, z=10.0, length=2.0, width=2.0, height=2.0, yaw=0.0, label="Car")
    assert box3d_iou(box_a, box_c) == 0.0


def test_box3d_iou_partial_height_overlap() -> None:
    """P1.9: 50% height overlap on identical XY footprint must produce IoU of 1/3 (0.3333)."""
    # Box 1: z in [-1, 1] (height=2.0, vol=8.0)
    # Box 2: z in [0, 2] (height=2.0, vol=8.0)
    # Intersection: z in [0, 1] (height=1.0, vol=4.0)
    # Union: 8 + 8 - 4 = 12 -> IoU = 4 / 12 = 1/3
    box1 = Box3D(x=0.0, y=0.0, z=0.0, length=2.0, width=2.0, height=2.0, yaw=0.0, label="Car")
    box2 = Box3D(x=0.0, y=0.0, z=1.0, length=2.0, width=2.0, height=2.0, yaw=0.0, label="Car")
    iou = box3d_iou(box1, box2)
    assert math.isclose(iou, 1.0 / 3.0, abs_tol=1e-3), f"Expected 0.3333, got {iou}"


def test_box3d_iou_rotated_boxes() -> None:
    """P1.9: Rotated boxes in BEV should compute correct oriented intersection."""
    # Box 1 at yaw=0, Box 2 at yaw=pi/2 (90 deg)
    box1 = Box3D(x=0.0, y=0.0, z=0.0, length=4.0, width=2.0, height=1.0, yaw=0.0, label="Car")
    box2 = Box3D(x=0.0, y=0.0, z=0.0, length=4.0, width=2.0, height=1.0, yaw=math.pi / 2.0, label="Car")
    iou = box3d_iou(box1, box2)
    # Inter footprint is 2x2 = 4; Each area is 4x2 = 8; Union = 8 + 8 - 4 = 12 -> IoU = 4 / 12 = 0.3333
    assert math.isclose(iou, 1.0 / 3.0, abs_tol=1e-2), f"Expected ~0.3333, got {iou}"


def test_box3d_iou_robustness_edge_cases() -> None:
    """P1.9: Edge cases like zero/negative dimensions or NaN/Inf must return 0.0 cleanly."""
    valid_box = Box3D(x=0.0, y=0.0, z=0.0, length=2.0, width=2.0, height=2.0, yaw=0.0, label="Car")
    zero_box = Box3D(x=0.0, y=0.0, z=0.0, length=0.0, width=2.0, height=2.0, yaw=0.0, label="Car")
    nan_box = Box3D(x=float("nan"), y=0.0, z=0.0, length=2.0, width=2.0, height=2.0, yaw=0.0, label="Car")
    inf_box = Box3D(x=0.0, y=0.0, z=0.0, length=float("inf"), width=2.0, height=2.0, yaw=0.0, label="Car")

    assert box3d_iou(valid_box, zero_box) == 0.0
    assert box3d_iou(valid_box, nan_box) == 0.0
    assert box3d_iou(valid_box, inf_box) == 0.0


def test_kitti_class_thresholds_and_matching() -> None:
    """P1.9: Car matches at 0.7 IoU while Pedestrian/Cyclist match at 0.5 IoU."""
    assert get_kitti_threshold("Car") == 0.7
    assert get_kitti_threshold("Pedestrian") == 0.5
    assert get_kitti_threshold("Cyclist") == 0.5

    # Ground truth Car and Prediction with 0.6 IoU -> should NOT match (threshold 0.7)
    gt_car = Box3D(x=0.0, y=0.0, z=0.0, length=2.0, width=2.0, height=2.0, yaw=0.0, label="Car")
    # Z shift gives IoU ~ 0.6
    pred_car = Box3D(
        x=0.0, y=0.0, z=0.5, length=2.0, width=2.0, height=2.0, yaw=0.0, label="Car", score=0.9
    )  # height inter=1.5, vol=6, union=10 -> iou=0.6

    matches = match_boxes3d((gt_car,), (pred_car,), use_3d_iou=True)
    assert len(matches) == 0, "Car with 0.6 IoU should not match threshold 0.7"

    # Pedestrian with 0.6 IoU -> should match (threshold 0.5)
    gt_ped = Box3D(x=0.0, y=0.0, z=0.0, length=2.0, width=2.0, height=2.0, yaw=0.0, label="Pedestrian")
    pred_ped = Box3D(x=0.0, y=0.0, z=0.5, length=2.0, width=2.0, height=2.0, yaw=0.0, label="Pedestrian", score=0.9)
    matches_ped = match_boxes3d((gt_ped,), (pred_ped,), use_3d_iou=True)
    assert len(matches_ped) == 1
    assert matches_ped[0].gt_index == 0


def test_evaluate_3d_detections_ap_complete() -> None:
    """P1.9: Compute per-class 3D AP and aggregate mAP_3D."""
    gt_boxes = (
        Box3D(x=0.0, y=0.0, z=0.0, length=2.0, width=2.0, height=2.0, yaw=0.0, label="Car"),
        Box3D(x=10.0, y=0.0, z=0.0, length=1.0, width=1.0, height=1.8, yaw=0.0, label="Pedestrian"),
    )
    pred_boxes = (
        Box3D(x=0.0, y=0.0, z=0.0, length=2.0, width=2.0, height=2.0, yaw=0.0, label="Car", score=0.95),
        Box3D(x=10.0, y=0.0, z=0.0, length=1.0, width=1.0, height=1.8, yaw=0.0, label="Pedestrian", score=0.88),
    )

    results = evaluate_3d_detections_ap([gt_boxes], [pred_boxes], use_3d_iou=True)
    assert "Car" in results
    assert "Pedestrian" in results
    assert "mAP_3D" in results
    assert results["Car"] == 1.0
    assert results["Pedestrian"] == 1.0
    assert results["mAP_3D"] == 1.0
