from src.evaluation.contracts import MetricEnvelope
from src.evaluation.model_comparison import ComparisonInput, checkpoint_gate, compare_models


def _metric(name: str, value: float, higher: bool = True) -> MetricEnvelope:
    return MetricEnvelope(
        name=name,
        value=value,
        unit="points",
        percent_value=None,
        version="1.0.0",
        higher_is_better=higher,
        ci95=(value, value),
    )


def _snapshot(model: str, attacked: float) -> ComparisonInput:
    return ComparisonInput(
        model_id=model,
        protocol_id="locked-p1",
        sample_ids=("s1",),
        preprocessing_version="prep-v1",
        thresholds={"score": 0.25},
        class_mapping_version="1.0.0",
        metric_versions={"score": "1.0.0"},
        clean=_metric("clean", 0.9),
        attacked=_metric("attacked", attacked),
        robust_score=_metric("robust", attacked * 100),
        mean_degradation=_metric("degradation", 0.4 if model == "base" else 0.2, False),
        attack_success_rate=_metric("asr", 0.2, False),
        external=_metric("external", 0.7),
        critical_scenarios={"fog": _metric("fog", attacked)},
        failure_count=1,
    )


def test_recovery_comparison_uses_one_locked_protocol() -> None:
    comparison = compare_models(_snapshot("base", 0.5), _snapshot("defended", 0.75))
    assert comparison.paired
    assert checkpoint_gate(comparison).passed
