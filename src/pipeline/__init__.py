"""Run orchestration and cost control.

Implemented: :class:`~src.pipeline.runner.TestRunner` (sequential, cached,
reproducible) and :class:`~src.pipeline.cache.MemoryCache`.

Open slots from plan §5: two-tier scan (cheap sweep -> expensive attacks on the
worst configurations only), statistical early stopping, GPU batch tuning,
Optuna-based red-team search, and a Celery/Redis queue for background runs.
"""

from __future__ import annotations

from src.pipeline.cache import MemoryCache, NullCache, PredictionCache
from src.pipeline.runner import CostEstimate, RunConfig, TestRunner

__all__ = [
    "CostEstimate",
    "MemoryCache",
    "NullCache",
    "PredictionCache",
    "RunConfig",
    "TestRunner",
]
