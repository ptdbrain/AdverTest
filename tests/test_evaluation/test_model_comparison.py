from src.evaluation.contracts import MetricEnvelope
from src.evaluation.model_comparison import (
    CheckpointGateConfig,
    ComparisonInput,
    checkpoint_gate,
    compare_models,
)


def metric(name: str, value: float, *, higher: bool = True) -> MetricEnvelope:
    return MetricEnvelope(
        name=name,
        value=value,
        unit="points",
        percent_value=None,
        version="1.0.0",
        higher_is_better=higher,
        ci95=(value - 0.01, value + 0.01),
    )


def snapshot(model: str, *, protocol_id: str = "p1", attacked: float = 0.5) -> ComparisonInput:
    return ComparisonInput(
        model_id=model,
        protocol_id=protocol_id,
        sample_ids=("a", "b"),
        preprocessing_version="prep-1",
        thresholds={"score": 0.25},
        prompt_protocol="box-v1",
        class_mapping_version="1.0.0",
        metric_versions={"score": "1.0.0"},
        clean=metric("clean", 0.9),
        attacked=metric("attacked", attacked),
        robust_score=metric("robust_score", attacked * 100),
        mean_degradation=metric("mean_degradation", 0.4, higher=False),
        attack_success_rate=metric("asr", 0.3, higher=False),
        external=metric("external", 0.6),
        critical_scenarios={"fog": metric("fog", attacked)},
        failure_count=4,
    )


def test_comparison_is_paired_only_when_protocol_inputs_match() -> None:
    paired = compare_models(snapshot("base"), snapshot("defended", attacked=0.75))
    assert paired.paired is True
    assert paired.recovery.value > 0  # type: ignore[union-attr]
    assert paired.failure_count_delta.value == -0.0

    incompatible = compare_models(snapshot("base"), snapshot("defended", protocol_id="p2"))
    assert incompatible.paired is False
    assert "protocol_id" in incompatible.incompatibilities
    assert incompatible.deltas == {}


def test_default_checkpoint_gate_enforces_clean_and_robustness_thresholds() -> None:
    comparison = compare_models(snapshot("base"), snapshot("defended", attacked=0.75))
    gate = checkpoint_gate(comparison, CheckpointGateConfig())
    assert gate.passed is True
    assert gate.version == "1.0.0"
