"""Small numpy-only image helpers shared by attack plugins.

Attack authors should reuse these instead of re-implementing them, so every
attack behaves identically with respect to dtype, clipping, and rounding. No
OpenCV / PIL / torch dependency — the reference pipeline runs on numpy alone.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from src.core.types import Box

#: How occlusion-style attacks paint over a region.
FillMode = Literal["mean", "gray", "black", "noise"]

#: ITU-R BT.601 luma weights, used by the reference detector and its gradients.
LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def clip01(image: np.ndarray) -> np.ndarray:
    """Clip to the valid range and guarantee float32 output."""
    return np.clip(image, 0.0, 1.0).astype(np.float32, copy=False)


def luminance(image: np.ndarray) -> np.ndarray:
    """(H, W, 3) -> (H, W) luma channel."""
    return (image * LUMA_WEIGHTS).sum(axis=2).astype(np.float32)


def spread_to_channels(gray_grad: np.ndarray) -> np.ndarray:
    """Chain rule for :func:`luminance`: (H, W) gradient -> (H, W, 3)."""
    return (gray_grad[..., None] * LUMA_WEIGHTS[None, None, :]).astype(np.float32)


def nearest_resize(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Nearest-neighbour resize (used by pixelate-style corruptions)."""
    src_h, src_w = image.shape[:2]
    rows = np.clip((np.arange(height) * src_h) // max(height, 1), 0, src_h - 1)
    cols = np.clip((np.arange(width) * src_w) // max(width, 1), 0, src_w - 1)
    return image[rows][:, cols].astype(np.float32, copy=False)


def box_blur(image: np.ndarray, radius: int) -> np.ndarray:
    """Separable box blur via cumulative sums; ``radius=0`` is a no-op."""
    if radius <= 0:
        return image.astype(np.float32, copy=False)
    blurred = _blur_axis(image.astype(np.float32), radius, axis=0)
    return _blur_axis(blurred, radius, axis=1)


def _blur_axis(image: np.ndarray, radius: int, axis: int) -> np.ndarray:
    """One-dimensional moving average with edge padding."""
    padded = np.pad(image, _pad_width(image.ndim, axis, radius), mode="edge")
    window = 2 * radius + 1
    cumulative = np.cumsum(padded, axis=axis)
    zeros = np.zeros_like(np.take(cumulative, [0], axis=axis))
    cumulative = np.concatenate([zeros, cumulative], axis=axis)
    upper = np.take(cumulative, np.arange(window, window + image.shape[axis]), axis=axis)
    lower = np.take(cumulative, np.arange(0, image.shape[axis]), axis=axis)
    return ((upper - lower) / window).astype(np.float32)


def _pad_width(ndim: int, axis: int, radius: int) -> list[tuple[int, int]]:
    return [(radius, radius) if dim == axis else (0, 0) for dim in range(ndim)]


def linear_depth_prior(height: int, width: int, *, near: float = 5.0, far: float = 60.0) -> np.ndarray:
    """Fallback depth map in metres when a sample carries no LiDAR depth.

    Rows near the top of the frame are treated as far away, rows at the bottom
    as close — the usual geometry for a forward-facing driving camera.
    """
    column = np.linspace(far, near, num=height, dtype=np.float32)
    return np.repeat(column[:, None], width, axis=1)


def box_slice(box: Box, height: int, width: int) -> tuple[slice, slice] | None:
    """Clamp a box to the image grid and return ``(rows, cols)`` slices."""
    x1 = int(np.floor(max(0.0, box.x1)))
    y1 = int(np.floor(max(0.0, box.y1)))
    x2 = int(np.ceil(min(float(width), box.x2)))
    y2 = int(np.ceil(min(float(height), box.y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return slice(y1, y2), slice(x1, x2)


def paste(image: np.ndarray, region: tuple[slice, slice], values: np.ndarray | float) -> np.ndarray:
    """Return a copy of ``image`` with ``region`` overwritten by ``values``."""
    out = image.copy()
    out[region] = values
    return clip01(out)


def fill_values(
    mode: FillMode,
    image: np.ndarray,
    shape: tuple[int, ...],
    rng: np.random.Generator,
) -> np.ndarray:
    """Pixels used by occlusion-style attacks to cover a region."""
    if mode == "mean":
        return np.full(shape, float(image.mean()), dtype=np.float32)
    if mode == "gray":
        return np.full(shape, 0.5, dtype=np.float32)
    if mode == "black":
        return np.zeros(shape, dtype=np.float32)
    return rng.random(shape, dtype=np.float32)
