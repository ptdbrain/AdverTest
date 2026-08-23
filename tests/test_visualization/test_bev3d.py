import numpy as np
import pytest
from src.core.types import Box3D
from src.visualization.bev3d import render_bev

@pytest.fixture
def sample_points():
    return np.array([
        [10.0, 0.0, 0.0],
        [20.0, -5.0, 0.0],
        [30.0, 5.0, 0.0],
    ], dtype=np.float32)

@pytest.fixture
def sample_gt_boxes():
    return [
        Box3D(x=10, y=0, z=0, length=4, width=2, height=1.5, yaw=0.0, label="Car"),
    ]

@pytest.fixture
def sample_pred_boxes():
    return [
        Box3D(x=10.5, y=0.2, z=0, length=4, width=2, height=1.5, yaw=0.1, label="Car"),
    ]

def test_bev_renderer_returns_rgb_image(sample_points):
    img = render_bev(points=sample_points)
    assert img.shape == (800, 800, 3)
    assert img.dtype == np.uint8

def test_bev_renderer_does_not_mutate_points(sample_points):
    points_copy = sample_points.copy()
    render_bev(points=sample_points)
    np.testing.assert_array_equal(sample_points, points_copy)

def test_bev_renderer_handles_empty_predictions(sample_points, sample_gt_boxes):
    img = render_bev(points=sample_points, ground_truth=sample_gt_boxes)
    assert img.shape == (800, 800, 3)

def test_bev_renderer_handles_empty_points(sample_gt_boxes):
    empty_points = np.zeros((0, 3), dtype=np.float32)
    img = render_bev(points=empty_points, ground_truth=sample_gt_boxes)
    assert img.shape == (800, 800, 3)

def test_bev_renderer_with_boxes_renders_without_error(sample_points, sample_gt_boxes, sample_pred_boxes):
    img = render_bev(points=sample_points, ground_truth=sample_gt_boxes, predictions=sample_pred_boxes)
    assert img.shape == (800, 800, 3)
