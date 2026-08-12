"""Wave 2 — Checkpoint gate, model lineage and acceptance tests.

Tests the checkpoint promotion gate with real thresholds from the plan:
- Clean AP regression ≤ 2 points
- RobustScore gain ≥ 8 points OR mean degradation improvement ≥ 8pp
- Critical scenario regression ≤ 3 points each
- ASR regression ≤ 5 percentage points
- External AP regression ≤ 3 points
- Paired bootstrap 95% CI must not deny improvement direction
"""

from __future__ import annotations

import math

from src.evaluation.contracts import MetricEnvelope
from src.evaluation.model_comparison import (
    ComparisonInput,
    checkpoint_gate,
    compare_models,
)


def _metric(name: str, value: float, *, unit: str = "ratio", higher_is_better: bool = True) -> MetricEnvelope:
    percent = value * 100.0 if unit == "ratio" else value if unit == "percent" else None
    return MetricEnvelope(
        name=name, value=value, unit=unit,
        percent_value=percent, version="1.0.0", higher_is_better=higher_is_better,
        ci95=(value - 0.01, value + 0.01),
    )


def _input(
    model_id: str,
    clean: float = 0.85,
    attacked: float = 0.60,
    robust_score: float = 65.0,
    mean_degradation: float = 0.30,
    asr: float = 0.40,
    external: float = 0.82,
    failure_count: int = 10,
) -> ComparisonInput:
    return ComparisonInput(
        model_id=model_id,
        protocol_id="proto-locked-001",
        sample_ids=("s1", "s2", "s3"),
        preprocessing_version="default",
        thresholds={"iou": 0.5},
        class_mapping_version="1.0.0",
        metric_versions={"ap": "1.0"},
        clean=_metric("clean", clean),
        attacked=_metric("attacked", attacked),
        robust_score=_metric("robust_score", robust_score, unit="points"),
        mean_degradation=_metric("mean_degradation", mean_degradation, higher_is_better=False),
        attack_success_rate=_metric("asr", asr, higher_is_better=False),
        external=_metric("external", external),
        failure_count=failure_count,
    )


class TestCheckpointGate:
    """Verify checkpoint promotion gate enforces plan thresholds."""

    def test_good_candidate_passes(self):
        baseline = _input("b0", clean=0.85, attacked=0.55, robust_score=50.0, mean_degradation=0.35, asr=0.45, external=0.82)
        candidate = _input("r1", clean=0.84, attacked=0.70, robust_score=62.0, mean_degradation=0.20, asr=0.30, external=0.81)
        comparison = compare_models(baseline, candidate)
        result = checkpoint_gate(comparison)
        assert result.passed, f"Gate should pass but failed: {result.reasons}"

    def test_clean_regression_too_large_fails(self):
        baseline = _input("b0", clean=0.85)
        candidate = _input("r1", clean=0.80, robust_score=58.0 + 8.1)  # robust gain ok
        comparison = compare_models(baseline, candidate)
        result = checkpoint_gate(comparison)
        assert not result.passed
        assert "clean" in result.reasons

    def test_no_robustness_improvement_fails(self):
        baseline = _input("b0", robust_score=60.0, mean_degradation=0.30)
        candidate = _input("r1", robust_score=60.0, mean_degradation=0.30)  # no gain
        comparison = compare_models(baseline, candidate)
        result = checkpoint_gate(comparison)
        assert not result.passed
        assert "robustness" in result.reasons

    def test_asr_regression_fails(self):
        baseline = _input("b0", asr=0.30)
        candidate = _input("r1", asr=0.36, robust_score=80.0)  # ASR +6pp > 5pp limit
        comparison = compare_models(baseline, candidate)
        result = checkpoint_gate(comparison)
        assert not result.passed
        assert "attack_success_rate" in result.reasons

    def test_external_regression_fails(self):
        baseline = _input("b0", external=0.85)
        candidate = _input("r1", external=0.80, robust_score=80.0)  # -5 points > 3 limit
        comparison = compare_models(baseline, candidate)
        result = checkpoint_gate(comparison)
        assert not result.passed
        assert "external" in result.reasons

    def test_unpaired_comparison_fails(self):
        baseline = _input("b0")
        candidate = ComparisonInput(
            model_id="r1",
            protocol_id="proto-different",  # different protocol → unpaired
            sample_ids=("s1", "s2", "s3"),
            preprocessing_version="default",
            thresholds={"iou": 0.5},
            class_mapping_version="1.0.0",
            metric_versions={"ap": "1.0"},
            clean=_metric("clean", 0.85),
            attacked=_metric("attacked", 0.70),
            robust_score=_metric("robust_score", 70.0, unit="points"),
            mean_degradation=_metric("mean_degradation", 0.20, higher_is_better=False),
            attack_success_rate=_metric("asr", 0.30, higher_is_better=False),
            external=_metric("external", 0.82),
            failure_count=5,
        )
        comparison = compare_models(baseline, candidate)
        assert not comparison.paired
        result = checkpoint_gate(comparison)
        assert not result.passed
        assert any("paired" in r for r in result.reasons)

    def test_critical_scenario_regression(self):
        baseline = _input("b0")
        candidate = _input("r1", robust_score=80.0)
        # Add critical scenarios
        baseline_with_critical = ComparisonInput(
            **{**baseline.model_dump(), "critical_scenarios": {"fog_night": _metric("fog_night", 0.70)}}
        )
        candidate_with_critical = ComparisonInput(
            **{**candidate.model_dump(), "critical_scenarios": {"fog_night": _metric("fog_night", 0.60)}}
        )
        comparison = compare_models(baseline_with_critical, candidate_with_critical)
        result = checkpoint_gate(comparison)
        # -10 points regression in fog_night > 3 point limit
        assert not result.passed
        assert "critical" in result.reasons


class TestModelComparison:
    """Verify model comparison produces correct deltas and recovery."""

    def test_paired_comparison(self):
        baseline = _input("b0", clean=0.85, attacked=0.55)
        candidate = _input("r1", clean=0.84, attacked=0.70)
        result = compare_models(baseline, candidate)
        assert result.paired
        assert "clean" in result.deltas
        assert math.isclose(result.deltas["clean"].value, -0.01, abs_tol=1e-6)

    def test_unpaired_suppresses_deltas(self):
        baseline = _input("b0")
        candidate = ComparisonInput(
            model_id="r1",
            protocol_id="different-proto",
            sample_ids=("s1", "s2"),  # different samples
            preprocessing_version="default",
            thresholds={"iou": 0.5},
            class_mapping_version="1.0.0",
            metric_versions={"ap": "1.0"},
            clean=_metric("clean", 0.85),
            attacked=_metric("attacked", 0.70),
            robust_score=_metric("robust_score", 70.0, unit="points"),
            mean_degradation=_metric("mean_degradation", 0.20, higher_is_better=False),
            attack_success_rate=_metric("asr", 0.30, higher_is_better=False),
            external=_metric("external", 0.82),
            failure_count=5,
        )
        result = compare_models(baseline, candidate)
        assert not result.paired
        assert "protocol_id" in result.incompatibilities
        assert len(result.deltas) == 0

    def test_failure_count_delta(self):
        baseline = _input("b0", failure_count=15)
        candidate = _input("r1", failure_count=8)
        result = compare_models(baseline, candidate)
        assert math.isclose(result.failure_count_delta.value, -7.0, abs_tol=1e-6)
