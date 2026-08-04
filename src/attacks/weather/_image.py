"""Depth-aware image weather primitives shared by Group B attacks."""

from __future__ import annotations

import numpy as np

from src.core.image_ops import clip01, linear_depth_prior
from src.core.types import CameraView, Sample


def depth_for(image: np.ndarray, depth: np.ndarray | None) -> np.ndarray:
    if depth is None or depth.shape != image.shape[:2] or not np.isfinite(depth).all():
        return linear_depth_prior(*image.shape[:2])
    return np.maximum(depth.astype(np.float32), 1e-3)


def views(sample: Sample) -> tuple[CameraView, ...]:
    if sample.camera_views:
        return sample.camera_views
    return (CameraView("CAM_FRONT", sample.image, sample.depth),)


def replace_views(sample: Sample, result: dict[str, np.ndarray]) -> Sample:
    if not sample.camera_views:
        return sample.with_image(result["CAM_FRONT"])
    updated = tuple(
        CameraView(
            name=view.name,
            image=result.get(view.name, view.image),
            depth=view.depth,
            intrinsic=view.intrinsic,
            sensor_to_ego=view.sensor_to_ego,
            previous_image=view.previous_image,
        )
        for view in sample.camera_views
    )
    return sample.with_camera_views(updated)


def nested_noise(rng: np.random.Generator, shape: tuple[int, int], count: int) -> np.ndarray:
    """Stable prefix mask: larger severities contain all lower-severity marks."""
    total = shape[0] * shape[1]
    count = min(total, max(0, count))
    order = rng.permutation(total)
    mask = np.zeros(total, dtype=bool)
    mask[order[:count]] = True
    return mask.reshape(shape)


def streak_layer(rng: np.random.Generator, height: int, width: int, count: int, length: int) -> np.ndarray:
    layer = np.zeros((height, width), dtype=np.float32)
    ys = rng.integers(0, height, size=count)
    xs = rng.integers(0, width, size=count)
    for y, x in zip(ys, xs, strict=True):
        end = min(height, y + max(1, length))
        layer[y:end, x] = np.maximum(layer[y:end, x], 1.0)
    return layer


def fog(image: np.ndarray, depth: np.ndarray, beta: float, airlight: float = 1.0) -> np.ndarray:
    transmission = np.exp(-beta * depth)[..., None]
    return clip01(image * transmission + airlight * (1.0 - transmission))
