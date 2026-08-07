"""Paired bootstrap over sample identities, never aggregate benchmark cells."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from src.evaluation.contracts import MetricEnvelope, MetricUnit
from src.pipeline.protocol import BenchmarkProtocol


def paired_bootstrap(
    sample_ids: tuple[str, ...],
    statistic: Callable[[tuple[str, ...]], float],
    protocol: BenchmarkProtocol,
    *,
    name: str,
    unit: MetricUnit,
    version: str = "1.0.0",
    higher_is_better: bool = True,
) -> MetricEnvelope:
    if protocol.status != "LOCKED":
        raise ValueError("paired bootstrap requires a LOCKED protocol")
    if sample_ids != protocol.sample_ids:
        raise ValueError("bootstrap sample IDs/order must match the locked protocol")
    if not sample_ids:
        raise ValueError("paired bootstrap requires at least one sample ID")

    point = float(statistic(sample_ids))
    rng = np.random.default_rng(protocol.bootstrap_seed)
    ids = np.asarray(sample_ids, dtype=object)
    replicates = [
        float(statistic(tuple(rng.choice(ids, size=len(ids), replace=True).tolist())))
        for _ in range(protocol.bootstrap_iterations)
    ]
    low, high = np.quantile(np.asarray(replicates), [0.025, 0.975])
    percent_value = point * 100.0 if unit == "ratio" else point if unit == "percent" else None
    return MetricEnvelope(
        name=name,
        value=point,
        unit=unit,
        percent_value=percent_value,
        version=version,
        higher_is_better=higher_is_better,
        ci95=(float(low), float(high)),
        metadata={
            "resampling_unit": "sample_id",
            "iterations": protocol.bootstrap_iterations,
            "seed": protocol.bootstrap_seed,
            "paired": True,
        },
    )
