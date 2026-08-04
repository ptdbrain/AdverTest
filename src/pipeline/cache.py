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

from src.core.types import Box, Prediction


class PredictionCache(Protocol):
    """Minimal cache interface the runner depends on."""

    hits: int
    misses: int

    def get(self, key: str) -> Prediction | None: ...

    def put(self, key: str, prediction: Prediction) -> None: ...


class MemoryCache:
    """Process-local cache; enough for a single-node run."""

    def __init__(self) -> None:
        self._store: dict[str, Prediction] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Prediction | None:
        found = self._store.get(key)
        if found is None:
            self.misses += 1
        else:
            self.hits += 1
        return found

    def put(self, key: str, prediction: Prediction) -> None:
        self._store[key] = prediction

    def __len__(self) -> int:
        return len(self._store)


class NullCache:
    """Disables caching — useful to measure the real cost of a run."""

    def __init__(self) -> None:
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Prediction | None:
        self.misses += 1
        return None

    def put(self, key: str, prediction: Prediction) -> None:
        return None


class SqliteCache:
    """Process-restart-safe prediction cache keyed by complete provenance."""

    def __init__(self, path: str) -> None:
        database = Path(path).expanduser().resolve()
        database.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(database, timeout=30, check_same_thread=False)
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS predictions (cache_key TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Prediction | None:
        row = self._connection.execute("SELECT payload FROM predictions WHERE cache_key=?", (key,)).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        payload = json.loads(row[0])
        return Prediction(
            payload["sample_id"],
            tuple(Box(*box[:4], box[4], box[5]) for box in payload["boxes"]),
            latency_ms=payload["latency_ms"],
        )

    def put(self, key: str, prediction: Prediction) -> None:
        payload = {
            "sample_id": prediction.sample_id,
            "boxes": [(*box.as_tuple(), box.label, box.score) for box in prediction.boxes],
            "latency_ms": prediction.latency_ms,
        }
        self._connection.execute(
            "INSERT OR REPLACE INTO predictions(cache_key, payload) VALUES (?, ?)",
            (key, json.dumps(payload)),
        )
        self._connection.commit()
