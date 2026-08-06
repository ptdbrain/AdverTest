"""Immutable, unit-explicit evaluation and failure handoff contracts."""

from __future__ import annotations

import math
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.core.contracts import MaskWireV1

MetricUnit = Literal[
    "ratio",
    "percent",
    "points",
    "count",
    "milliseconds",
    "seconds",
    "score",
]
AllowedFailureUse = Literal["benchmark", "training", "review"]


class _FrozenContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class MetricEnvelope(_FrozenContract):
    name: str
    value: float
    unit: MetricUnit
    percent_value: float | None
    version: str
    higher_is_better: bool
    ci95: tuple[float, float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_units(self) -> Self:
        if not self.version.strip():
            raise ValueError("metric version must not be empty")
        if self.unit == "ratio":
            if self.percent_value is None:
                raise ValueError("ratio metrics require percent_value")
            if not math.isclose(self.percent_value, self.value * 100.0, abs_tol=1e-9):
                raise ValueError("percent_value must equal ratio value multiplied by 100")
        if self.unit == "percent" and (
            self.percent_value is None
            or not math.isclose(self.percent_value, self.value, abs_tol=1e-9)
        ):
            raise ValueError("percent metrics require percent_value equal to value")
        if self.ci95 is not None and self.ci95[0] > self.ci95[1]:
            raise ValueError("ci95 lower bound must not exceed upper bound")
        return self


class FailureCase(_FrozenContract):
    case_id: str
    sample_id: str
    model_id: str
    protocol_id: str
    clean_metrics: tuple[MetricEnvelope, ...]
    attacked_metrics: tuple[MetricEnvelope, ...]
    reason: str
    affected_object_id: str | None = None
    affected_mask: MaskWireV1 | None = None
    artifact_links: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class FailureCluster(_FrozenContract):
    cluster_id: str
    member_ids: tuple[str, ...]
    selection_allowed: bool
    allowed_uses: tuple[AllowedFailureUse, ...] = ()
    selection_reason: str
    method: str
    version: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_members(self) -> Self:
        if not self.member_ids:
            raise ValueError("failure cluster must contain at least one member")
        if tuple(sorted(set(self.member_ids))) != self.member_ids:
            raise ValueError("member_ids must be unique and deterministically sorted")
        if self.selection_allowed and not self.allowed_uses:
            raise ValueError("selection_allowed clusters require allowed_uses")
        return self
