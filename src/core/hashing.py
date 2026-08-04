"""Content-addressed keys for the variant cache (plan §5).

A variant is uniquely identified by ``(sample, attack, params, severity,
model_version)``. Same key means the forward pass can be skipped.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from src.core.types import Sample


def stable_digest(payload: Any, *, length: int = 16) -> str:
    """SHA-256 of a JSON-normalised payload, truncated for readability."""
    encoded = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]


def array_digest(array: np.ndarray, *, length: int = 16) -> str:
    """Digest of raw array bytes; use to assert two images are identical."""
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()[:length]


def sample_digest(sample: Sample, *, length: int = 32) -> str:
    """Content fingerprint covering every sensor carried by a sample.

    The legacy image-only digest was insufficient for multimodal cache keys:
    changing a LiDAR frame or a non-front camera could otherwise reuse a stale
    prediction.  Metadata (field names and camera names) is included alongside
    raw bytes so shape-compatible sensor swaps are still detected.
    """
    cameras = []
    for view in sample.camera_views:
        cameras.append({
            "name": view.name,
            "image": array_digest(view.image, length=length),
            "depth": array_digest(view.depth, length=length) if view.depth is not None else None,
            "previous": (
                array_digest(view.previous_image, length=length)
                if view.previous_image is not None else None
            ),
        })
    lidar = None
    if sample.lidar_frame is not None:
        lidar = {
            "fields": list(sample.lidar_frame.fields),
            "sensor_model": sample.lidar_frame.sensor_model,
            "points": array_digest(sample.lidar_frame.points, length=length),
        }
    elif sample.lidar is not None:
        lidar = {"fields": ["x", "y", "z", "intensity"], "points": array_digest(sample.lidar, length=length)}
    boxes3d = [asdict(box) for box in sample.boxes3d]
    return stable_digest({
        "sample_id": sample.sample_id,
        "image": array_digest(sample.image, length=length),
        "cameras": cameras,
        "lidar": lidar,
        "boxes3d": boxes3d,
    }, length=length)


def file_digest(path: str | Path, *, length: int = 32) -> str:
    """SHA-256 of a file without loading the whole checkpoint into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:length]


def variant_key(
    *,
    sample_id: str,
    attack: str,
    params: dict[str, Any],
    severity: int,
    model_version: str,
    sample_hash: str | None = None,
) -> str:
    """Cache key for one (image, attack, params, severity, model) combination."""
    return stable_digest(
        {
            "sample": sample_id,
            "attack": attack,
            "params": params,
            "severity": severity,
            "model": model_version,
            "sample_hash": sample_hash,
        }
    )


def clean_key(*, sample_id: str, model_version: str) -> str:
    """Cache key for a clean prediction (reused across every comparison)."""
    return stable_digest({"sample": sample_id, "model": model_version, "attack": "clean"})
