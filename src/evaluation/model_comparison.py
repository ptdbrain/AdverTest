"""Protocol-aware model comparison and versioned promotion gate."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from src.evaluation.contracts import MetricEnvelope
from src.evaluation.recovery_metrics import UndefinedMetric, recovery_rate


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ComparisonInput(_Frozen):
    model_id: str
    protocol_id: str
    sample_ids: tuple[str, ...]
    preprocessing_version: str
    thresholds: dict[str, float]
    prompt_protocol: str | None = None
    class_mapping_version: str
    metric_versions: dict[str, str]
    clean: MetricEnvelope
    attacked: MetricEnvelope
    robust_score: MetricEnvelope
    mean_degradation: MetricEnvelope
    attack_success_rate: MetricEnvelope
    external: MetricEnvelope
    critical_scenarios: dict[str, MetricEnvelope] = Field(default_factory=dict)
    failure_count: int = Field(ge=0)
    seen_metrics: dict[str, MetricEnvelope] = Field(default_factory=dict)
    unseen_metrics: dict[str, MetricEnvelope] = Field(default_factory=dict)


class ModelComparison(_Frozen):
    baseline_model_id: str
    candidate_model_id: str
    paired: bool
    incompatibilities: tuple[str, ...] = ()
    deltas: dict[str, MetricEnvelope] = Field(default_factory=dict)
    critical_scenario_deltas: dict[str, MetricEnvelope] = Field(default_factory=dict)
    recovery: MetricEnvelope | UndefinedMetric
    failure_count_delta: MetricEnvelope
    warnings: tuple[str, ...] = ()


class CheckpointGateConfig(_Frozen):
    version: str = "1.0.0"
    max_clean_decrease_points: float = 0.02
    min_robust_score_gain_points: float = 8.0
    min_relative_degradation_improvement: float = 0.15
    max_critical_regression_points: float = 0.03
    max_asr_regression_points: float = 0.05
    max_external_regression_points: float = 0.03
    require_paired_ci: bool = True


class CheckpointGateResult(_Frozen):
    passed: bool
    version: str
    checks: dict[str, bool]
    reasons: tuple[str, ...] = ()


def compare_models(baseline: ComparisonInput, candidate: ComparisonInput) -> ModelComparison:
    incompatibilities = tuple(
        field
        for field in (
            "protocol_id",
            "sample_ids",
            "preprocessing_version",
            "thresholds",
            "prompt_protocol",
            "class_mapping_version",
            "metric_versions",
        )
        if getattr(baseline, field) != getattr(candidate, field)
    )
    failure_delta = _delta_metric(
        "failure_count_delta",
        float(baseline.failure_count),
        float(candidate.failure_count),
        unit="count",
        higher_is_better=False,
    )
    recovery = recovery_rate(
        baseline.clean.value,
        baseline.attacked.value,
        candidate.attacked.value,
        higher_is_better=baseline.attacked.higher_is_better,
    )
    if incompatibilities:
        return ModelComparison(
            baseline_model_id=baseline.model_id,
            candidate_model_id=candidate.model_id,
            paired=False,
            incompatibilities=incompatibilities,
            recovery=recovery,
            failure_count_delta=failure_delta,
            warnings=("direct paired comparison suppressed due to incompatible inputs",),
        )

    named = {
        "clean": (baseline.clean, candidate.clean),
        "attacked": (baseline.attacked, candidate.attacked),
        "robust_score": (baseline.robust_score, candidate.robust_score),
        "mean_degradation": (baseline.mean_degradation, candidate.mean_degradation),
        "attack_success_rate": (baseline.attack_success_rate, candidate.attack_success_rate),
        "external": (baseline.external, candidate.external),
    }
    deltas = {
        name: _delta_metric(
            f"{name}_delta",
            before.value,
            after.value,
            unit=before.unit,
            higher_is_better=before.higher_is_better,
        )
        for name, (before, after) in named.items()
    }
    shared_critical = sorted(set(baseline.critical_scenarios) & set(candidate.critical_scenarios))
    critical = {
        name: _delta_metric(
            f"critical_{name}_delta",
            baseline.critical_scenarios[name].value,
            candidate.critical_scenarios[name].value,
            unit=baseline.critical_scenarios[name].unit,
            higher_is_better=baseline.critical_scenarios[name].higher_is_better,
        )
        for name in shared_critical
    }
    warnings = ()
    if set(baseline.critical_scenarios) != set(candidate.critical_scenarios):
        warnings = ("critical scenario sets differ; only their intersection was compared",)
    return ModelComparison(
        baseline_model_id=baseline.model_id,
        candidate_model_id=candidate.model_id,
        paired=True,
        deltas=deltas,
        critical_scenario_deltas=critical,
        recovery=recovery,
        failure_count_delta=failure_delta,
        warnings=warnings,
    )


def checkpoint_gate(
    comparison: ModelComparison,
    config: CheckpointGateConfig | None = None,
) -> CheckpointGateResult:
    policy = config or CheckpointGateConfig()
    if not comparison.paired:
        return CheckpointGateResult(
            passed=False,
            version=policy.version,
            checks={"paired": False},
            reasons=("promotion requires a valid paired comparison",),
        )
    deltas = comparison.deltas
    clean = deltas["clean"].value
    robust = deltas["robust_score"].value
    degradation = deltas["mean_degradation"].value
    baseline_degradation = deltas["mean_degradation"].metadata["baseline"]
    relative_degradation_gain = (
        -degradation / abs(baseline_degradation) if baseline_degradation else 0.0
    )
    critical_ok = all(
        item.value >= -policy.max_critical_regression_points
        for item in comparison.critical_scenario_deltas.values()
    )
    checks = {
        "paired": True,
        "clean": clean >= -policy.max_clean_decrease_points,
        "robustness": (
            robust >= policy.min_robust_score_gain_points
            or relative_degradation_gain >= policy.min_relative_degradation_improvement
        ),
        "critical": critical_ok,
        "attack_success_rate": (
            deltas["attack_success_rate"].value <= policy.max_asr_regression_points
        ),
        "external": deltas["external"].value >= -policy.max_external_regression_points,
        "paired_ci": (
            not policy.require_paired_ci
            or all(item.ci95 is not None for item in deltas.values())
        ),
    }
    reasons = tuple(name for name, passed in checks.items() if not passed)
    return CheckpointGateResult(
        passed=all(checks.values()),
        version=policy.version,
        checks=checks,
        reasons=reasons,
    )


def _delta_metric(
    name: str,
    baseline: float,
    candidate: float,
    *,
    unit: str,
    higher_is_better: bool,
) -> MetricEnvelope:
    value = candidate - baseline
    percent_value = value * 100.0 if unit == "ratio" else value if unit == "percent" else None
    return MetricEnvelope(
        name=name,
        value=value,
        unit=unit,  # type: ignore[arg-type]
        percent_value=percent_value,
        version="1.0.0",
        higher_is_better=higher_is_better,
        ci95=(value, value),
        metadata={"baseline": baseline, "candidate": candidate, "paired": True},
    )
