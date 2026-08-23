"""Unit tests for 3D BEV geometry and matching."""

import math

import pytest

from src.core.types import Box3D
from src.evaluation.geometry3d import HAS_SHAPELY, Match3D, bev_iou, match_boxes3d


def test_identical_boxes_have_unit_iou():
    box = Box3D(x=0, y=0, z=0, length=4, width=2, height=1.5, yaw=0, label="Car", score=0.9)
    assert math.isclose(bev_iou(box, box), 1.0, rel_tol=1e-5)


def test_disjoint_boxes_have_zero_iou():
    box_a = Box3D(x=0, y=0, z=0, length=4, width=2, height=1.5, yaw=0, label="Car", score=0.9)
    box_b = Box3D(x=10, y=10, z=0, length=4, width=2, height=1.5, yaw=0, label="Car", score=0.9)
    assert bev_iou(box_a, box_b) == 0.0


def test_partial_overlap_iou_in_range():
    box_a = Box3D(x=0, y=0, z=0, length=4, width=2, height=1.5, yaw=0, label="Car", score=0.9)
    box_b = Box3D(x=2, y=0, z=0, length=4, width=2, height=1.5, yaw=0, label="Car", score=0.9)
    overlap = bev_iou(box_a, box_b)
    assert 0.0 < overlap < 1.0


def test_matching_respects_class():
    ground_truth = (Box3D(x=0, y=0, z=0, length=4, width=2, height=1.5, yaw=0, label="Car", score=1.0),)
    predictions = (Box3D(x=0, y=0, z=0, length=4, width=2, height=1.5, yaw=0, label="Pedestrian", score=0.9),)
    matches = match_boxes3d(ground_truth, predictions, iou_threshold=0.5)
    assert len(matches) == 0


def test_one_prediction_cannot_match_two_gt_boxes():
    ground_truth = (
        Box3D(x=0, y=0, z=0, length=4, width=2, height=1.5, yaw=0, label="Car", score=1.0),
        Box3D(x=0.1, y=0.1, z=0, length=4, width=2, height=1.5, yaw=0, label="Car", score=1.0),
    )
    predictions = (Box3D(x=0.05, y=0.05, z=0, length=4, width=2, height=1.5, yaw=0, label="Car", score=0.9),)
    matches = match_boxes3d(ground_truth, predictions, iou_threshold=0.5)
    assert len(matches) == 1


def test_matching_prefers_higher_score():
    ground_truth = (Box3D(x=0, y=0, z=0, length=4, width=2, height=1.5, yaw=0, label="Car", score=1.0),)
    predictions = (
        Box3D(x=0.1, y=0.1, z=0, length=4, width=2, height=1.5, yaw=0, label="Car", score=0.5),
        Box3D(x=0, y=0, z=0, length=4, width=2, height=1.5, yaw=0, label="Car", score=0.9),
    )
    matches = match_boxes3d(ground_truth, predictions, iou_threshold=0.5)
    assert len(matches) == 1
    assert matches[0].prediction_index == 1


@pytest.mark.skipif(not HAS_SHAPELY, reason="Shapely required for rotated IoU test")
def test_rotated_box_iou_differs_from_axis_aligned():
    box_a = Box3D(x=0, y=0, z=0, length=4, width=2, height=1.5, yaw=0, label="Car", score=0.9)
    box_b = Box3D(x=0, y=0, z=0, length=4, width=2, height=1.5, yaw=math.pi / 4, label="Car", score=0.9)
    overlap = bev_iou(box_a, box_b)
    assert 0.0 < overlap < 1.0
