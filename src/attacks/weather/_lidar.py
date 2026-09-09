"""Numpy LiDAR weather primitives with deterministic, injected randomness."""

from __future__ import annotations

import numpy as np

from src.core.types import LidarFrame


def fog_frame(frame: LidarFrame, alpha: float, rng: np.random.Generator) -> LidarFrame:
    points = frame.points.copy()
    xyz = points[:, :3].astype(np.float64)
    ranges = np.linalg.norm(xyz, axis=1)
    attenuation = np.exp(-2.0 * alpha * ranges)
    intensity_index = frame.fields.index("intensity") if "intensity" in frame.fields else None
    if intensity_index is not None:
        points[:, intensity_index] *= attenuation.astype(points.dtype)
        # Add a small deterministic backscatter population for distant returns.
        candidates = np.flatnonzero((ranges > 15.0) & (rng.random(len(points)) < alpha * 0.25))
        if len(candidates):
            points[candidates, intensity_index] = np.maximum(
                points[candidates, intensity_index],
                np.asarray(0.02 * points[:, intensity_index].max(), dtype=points.dtype),
            )
    return LidarFrame(points.astype(np.float32, copy=False), frame.fields, frame.sensor_model)


def snow_frame(frame: LidarFrame, rate: float, rng: np.random.Generator) -> LidarFrame:
    points = frame.points.copy()
    xyz = points[:, :3].astype(np.float64)
    ranges = np.linalg.norm(xyz, axis=1)
    intensity_index = frame.fields.index("intensity") if "intensity" in frame.fields else None
    drop_probability = np.clip(rate / 2.5 * 0.35 * (ranges / max(float(ranges.max()), 1.0)), 0, 0.35)
    keep = rng.random(len(points)) >= drop_probability
    points = points[keep]
    if intensity_index is not None and len(points):
        points[:, intensity_index] *= np.float32(max(0.05, 1.0 - rate * 0.12))
    return LidarFrame(points.astype(np.float32, copy=False), frame.fields, frame.sensor_model)
