from collections.abc import Sequence

import cv2
import numpy as np

from src.core.types import Box3D


def _box3d_corners(box: Box3D) -> np.ndarray:
    """Compute 8 corners of a 3D box."""
    width, length, height = box.width, box.length, box.height
    x_corners = [length / 2, length / 2, -length / 2, -length / 2] * 2
    y_corners = [width / 2, -width / 2, -width / 2, width / 2] * 2
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
    w, h = canvas_size
    img = np.zeros((h, w, 3), dtype=np.uint8)

    if points.size > 0:
        valid_idx = (
            (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1]) &
            (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1])
        )
        pts_valid = points[valid_idx]

        u = ((pts_valid[:, 1] - y_range[0]) / (y_range[1] - y_range[0]) * w).astype(np.int32)
        v = (h - (pts_valid[:, 0] - x_range[0]) / (x_range[1] - x_range[0]) * h).astype(np.int32)

        in_screen = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u = u[in_screen]
        v = v[in_screen]

        img[v, u] = (200, 200, 200)

    def draw_boxes(boxes: Sequence[Box3D], color: tuple[int, int, int]):
        for box in boxes:
            corners = _box3d_corners(box)
            bev_corners = corners[:4]
            u = ((bev_corners[:, 1] - y_range[0]) / (y_range[1] - y_range[0]) * w).astype(np.int32)
            v = (h - (bev_corners[:, 0] - x_range[0]) / (x_range[1] - x_range[0]) * h).astype(np.int32)
            pts = np.vstack([u, v]).T.reshape((-1, 1, 2))
            cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)

    if ground_truth:
        draw_boxes(ground_truth, (0, 255, 0))

    if predictions:
        draw_boxes(predictions, (0, 0, 255))

    return img
