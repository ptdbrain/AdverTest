"""Content-addressed prediction cache (plan §5).

Two wins, both from the same store: the clean prediction of a sample is computed
once per (model, dataset) and reused by every comparison, and a repeated
``(image, attack, params, severity, model_version)`` combination never triggers a
second forward pass.

Open slot: a disk-backed implementation of :class:`PredictionCache` so the cache
survives process restarts (and can be shared by Celery workers).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Protocol

import numpy as np

from src.core.types import (
    Box,
    Box3D,
    DetectionPrediction,
    MaskPrediction,
    ModelPrediction,
    SegmentationPrediction,
)


class PredictionCache(Protocol):
    """Minimal cache interface the runner depends on."""

    hits: int
    misses: int

    def get(self, key: str) -> ModelPrediction | None: ...

    def put(self, key: str, prediction: ModelPrediction) -> None: ...


class MemoryCache:
    """Process-local cache; enough for a single-node run."""

    def __init__(self) -> None:
        self._store: dict[str, ModelPrediction] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> ModelPrediction | None:
        found = self._store.get(key)
        if found is None:
            self.misses += 1
        else:
            self.hits += 1
        return found

    def put(self, key: str, prediction: ModelPrediction) -> None:
        self._store[key] = prediction

    def __len__(self) -> int:
        return len(self._store)


class NullCache:
    """Disables caching — useful to measure the real cost of a run."""

    def __init__(self) -> None:
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> ModelPrediction | None:
        self.misses += 1
        return None

    def put(self, key: str, prediction: ModelPrediction) -> None:
        return None


class SqliteCache:
    """Process-restart-safe prediction cache keyed by complete provenance."""

    def __init__(self, path: str | Path) -> None:
        database = Path(path).expanduser().resolve()
        database.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(database, timeout=30, check_same_thread=False)
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS predictions (cache_key TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> ModelPrediction | None:
        row = self._connection.execute("SELECT payload FROM predictions WHERE cache_key=?", (key,)).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        payload = json.loads(row[0])
        prediction_type = payload.get("prediction_type", "detection")
        if prediction_type == "detection":
            return DetectionPrediction(
                sample_id=payload["sample_id"],
                boxes=tuple(
                    Box(
                        x1=box[0],
                        y1=box[1],
                        x2=box[2],
                        y2=box[3],
                        label=box[4],
                        score=box[5],
                    )
                    for box in payload.get("boxes", [])
                ),
                boxes3d=tuple(Box3D(**box) for box in payload.get("boxes3d", [])),
                latency_ms=payload.get("latency_ms", 0.0),
                metadata=payload.get("metadata", {}),
            )
        if prediction_type == "segmentation":
            return SegmentationPrediction(
                sample_id=payload["sample_id"],
                instances=tuple(
                    MaskPrediction(
                        instance_id=instance["instance_id"],
                        mask=_decode_mask(instance["mask"]),
                        label=instance.get("label"),
                        score=instance.get("score", 1.0),
                    )
                    for instance in payload.get("instances", [])
                ),
                prompt_id=payload.get("prompt_id"),
                latency_ms=payload.get("latency_ms", 0.0),
                metadata=payload.get("metadata", {}),
            )
        raise ValueError(f"unknown prediction_type: {prediction_type!r}")

    def put(self, key: str, prediction: ModelPrediction) -> None:
        if isinstance(prediction, DetectionPrediction):
            payload = {
                "prediction_type": "detection",
                "sample_id": prediction.sample_id,
                "boxes": [(*box.as_tuple(), box.label, box.score) for box in prediction.boxes],
                "boxes3d": [
                    {
                        "x": box.x,
                        "y": box.y,
                        "z": box.z,
                        "length": box.length,
                        "width": box.width,
                        "height": box.height,
                        "yaw": box.yaw,
                        "label": box.label,
                        "score": box.score,
                        "vx": box.vx,
                        "vy": box.vy,
                        "native_label": box.native_label,
                    }
                    for box in prediction.boxes3d
                ],
                "latency_ms": prediction.latency_ms,
                "metadata": prediction.metadata,
            }
        elif isinstance(prediction, SegmentationPrediction):
            payload = {
                "prediction_type": "segmentation",
                "sample_id": prediction.sample_id,
                "instances": [
                    {
                        "instance_id": instance.instance_id,
                        "mask": _encode_mask(instance.mask),
                        "label": instance.label,
                        "score": instance.score,
                    }
                    for instance in prediction.instances
                ],
                "prompt_id": prediction.prompt_id,
                "latency_ms": prediction.latency_ms,
                "metadata": prediction.metadata,
            }
        else:
            raise TypeError(f"unsupported prediction type: {type(prediction).__name__}")
        self._connection.execute(
            "INSERT OR REPLACE INTO predictions(cache_key, payload) VALUES (?, ?)",
            (key, json.dumps(payload)),
        )
        self._connection.commit()


def _encode_mask(mask: np.ndarray) -> dict[str, object]:
    """Encode a boolean mask as deterministic row-major zero-first RLE."""
    flat = mask.reshape(-1)
    runs: list[int] = []
    expected = False
    count = 0
    for value in flat:
        current = bool(value)
        if current == expected:
            count += 1
        else:
            runs.append(count)
            expected = current
            count = 1
    runs.append(count)
    return {"version": "1.0.0", "shape": list(mask.shape), "runs": runs}


def _decode_mask(payload: dict[str, object]) -> np.ndarray:
    if payload.get("version") != "1.0.0":
        raise ValueError(f"unsupported mask encoding version: {payload.get('version')!r}")
    raw_shape = payload.get("shape")
    raw_runs = payload.get("runs")
    if (
        not isinstance(raw_shape, list)
        or len(raw_shape) != 2
        or not all(isinstance(value, int) and value >= 0 for value in raw_shape)
        or not isinstance(raw_runs, list)
        or not all(isinstance(value, int) and value >= 0 for value in raw_runs)
    ):
        raise ValueError("invalid mask RLE payload")
    shape = (raw_shape[0], raw_shape[1])
    values = np.concatenate(
        [
            np.full(run, index % 2 == 1, dtype=np.bool_)
            for index, run in enumerate(raw_runs)
        ]
    )
    if values.size != shape[0] * shape[1]:
        raise ValueError("mask RLE length does not match shape")
    return values.reshape(shape)
