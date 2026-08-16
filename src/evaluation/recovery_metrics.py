"""Direction-aware recovery metrics with explicit undefined results."""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict

from src.evaluation.contracts import MetricEnvelope


class UndefinedMetric(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    reason: str
    version: str = "1.0.0"


def recovery_rate(
    baseline_clean: float,
    baseline_attacked: float,
    defended_attacked: float,
    *,
    higher_is_better: bool = True,
) -> MetricEnvelope | UndefinedMetric:
    """Fraction of attack damage recovered; deliberately not clipped to [0, 1]."""
    values = (baseline_clean, baseline_attacked, defended_attacked)
    if not all(math.isfinite(value) for value in values):
        return UndefinedMetric(name="recovery_rate", reason="inputs must be finite")
    denominator = (
        baseline_clean - baseline_attacked
        if higher_is_better
        else baseline_attacked - baseline_clean
    )
    if math.isclose(denominator, 0.0, abs_tol=1e-12):
        return UndefinedMetric(
            name="recovery_rate",
            reason="recovery denominator is zero because baseline has no measured attack damage",
        )
    numerator = (
        defended_attacked - baseline_attacked
        if higher_is_better
        else baseline_attacked - defended_attacked
    )
    value = numerator / denominator
    return MetricEnvelope(
        name="recovery_rate",
        value=value,
        unit="ratio",
        percent_value=value * 100.0,
        version="1.0.0",
        higher_is_better=True,
        metadata={
            "unbounded": True,
            "source_metric_higher_is_better": higher_is_better,
            "formula": "recovered_damage / baseline_attack_damage",
        },
    )
