"""Content-addressed prediction cache (plan §5).

Two wins, both from the same store: the clean prediction of a sample is computed
once per (model, dataset) and reused by every comparison, and a repeated
``(image, attack, params, severity, model_version)`` combination never triggers a
second forward pass.

Open slot: a disk-backed implementation of :class:`PredictionCache` so the cache
survives process restarts (and can be shared by Celery workers).
"""

from __future__ import annotations

from typing import Protocol

from src.core.types import Prediction


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
