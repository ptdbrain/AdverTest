"""Canonical, evidence-gated Defence comparison response contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _DefenseContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class EligibilityOut(_DefenseContract):
    status: Literal["ELIGIBLE", "NOT_ELIGIBLE"]
    reasons: tuple[str, ...] = ()
    next_action: str


class MetricDeltaOut(_DefenseContract):
    key: str
    label: str
    unit: Literal["ratio", "percent", "count"]
    higher_is_better: bool
    baseline_clean: float | None = None
    candidate_clean: float | None = None
    baseline_attacked: float | None = None
    candidate_attacked: float | None = None
    clean_delta: float | None = None
    attacked_delta: float | None = None


class RecoveryOut(_DefenseContract):
    metric_key: str | None = None
    value: float | None = None
    percent_value: float | None = None
    unit: Literal["ratio", "percent"] = "ratio"
    reason: str | None = None
    unbounded: bool = True


class FailureSummaryOut(_DefenseContract):
    baseline_count: int = Field(ge=0)
    candidate_count: int = Field(ge=0)
    recovered_count: int = Field(ge=0)
    regressed_count: int = Field(ge=0)
    transitions: tuple[dict[str, str], ...] = ()


class DecisionOut(_DefenseContract):
    status: Literal["NOT_ELIGIBLE", "READY_FOR_REVIEW"]
    summary: str
    next_action: str


class DefenseReportOut(_DefenseContract):
    comparison_id: str
    project_id: str
    baseline_run_id: str
    candidate_run_id: str
    eligibility: EligibilityOut
    baseline_signature: dict[str, object]
    candidate_signature: dict[str, object]
    incompatibilities: tuple[str, ...] = ()
    metric_deltas: tuple[MetricDeltaOut, ...] = ()
    recovery: RecoveryOut
    failures: FailureSummaryOut
    decision: DecisionOut
