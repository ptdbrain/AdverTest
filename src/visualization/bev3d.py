from collections.abc import Sequence

import cv2
import numpy as np

from src.core.types import Box3D


def _box3d_corners(box: Box3D) -> np.ndarray:
    """Compute 8 corners of a 3D box."""
    width, length, height = box.width, box.length, box.height
    x_corners = [length / 2, length / 2, -length / 2, -length / 2, length / 2, length / 2, -length / 2, -length / 2]
    y_corners = [width / 2, -width / 2, -width / 2, width / 2, width / 2, -width / 2, -width / 2, width / 2]
    z_corners = [0, 0, 0, 0, height, height, height, height]
    corners = np.vstack([x_corners, y_corners, z_corners])

    rot_mat = np.array(
        [
            [np.cos(box.yaw), -np.sin(box.yaw), 0],
            [np.sin(box.yaw), np.cos(box.yaw), 0],
            [0, 0, 1],
        ]
    )

    corners = rot_mat @ corners
    corners[0, :] += box.x
    corners[1, :] += box.y
    corners[2, :] += box.z

    return corners.T


def render_bev(
    *,
    points: np.ndarray,
    ground_truth: Sequence[Box3D] = (),
    predictions: Sequence[Box3D] = (),
    x_range: tuple[float, float] = (0.0, 70.0),
    y_range: tuple[float, float] = (-40.0, 40.0),
    canvas_size: tuple[int, int] = (800, 800),
) -> np.ndarray:
    """Render a BEV projection of points and 3D boxes.

    Returns RGB uint8 image of shape (canvas_size[1], canvas_size[0], 3).
    Does NOT mutate input arrays.
    """
    canvas_w, canvas_h = canvas_size
    img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    if points.size > 0:
        valid_idx = (
            (points[:, 0] >= x_range[0])
            & (points[:, 0] <= x_range[1])
            & (points[:, 1] >= y_range[0])
            & (points[:, 1] <= y_range[1])
        )
        valid_pts = points[valid_idx]

        if len(valid_pts) > 0:
            px = ((valid_pts[:, 0] - x_range[0]) / (x_range[1] - x_range[0]) * (canvas_w - 1)).astype(np.int32)
            py = ((valid_pts[:, 1] - y_range[0]) / (y_range[1] - y_range[0]) * (canvas_h - 1)).astype(np.int32)
            py = (canvas_h - 1) - py
            img[py, px] = [200, 200, 200]

    def _draw_box(b: Box3D, color: tuple[int, int, int]):
        corners = _box3d_corners(b)[:4, :2]
        px = ((corners[:, 0] - x_range[0]) / (x_range[1] - x_range[0]) * (canvas_w - 1)).astype(np.int32)
        py = ((corners[:, 1] - y_range[0]) / (y_range[1] - y_range[0]) * (canvas_h - 1)).astype(np.int32)
        py = (canvas_h - 1) - py
        poly = np.stack([px, py], axis=-1)
        cv2.polylines(img, [poly], isClosed=True, color=color, thickness=2)

    for gt in ground_truth:
        _draw_box(gt, (0, 255, 0))

    for pred in predictions:
        _draw_box(pred, (0, 0, 255))

    return img
