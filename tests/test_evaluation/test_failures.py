from __future__ import annotations

from src.evaluation.contracts import FailureCase, MetricEnvelope
from src.evaluation.failures import FailureGrouper


def _metric(value: float) -> MetricEnvelope:
    return MetricEnvelope(
        name="ap",
        value=value,
        unit="ratio",
        percent_value=value * 100,
        version="1.0.0",
        higher_is_better=True,
    )


def _case(case_id: str, *, severity: int, allowed: bool = True) -> FailureCase:
    return FailureCase(
        case_id=case_id,
        sample_id=f"sample-{case_id}",
        model_id="model-1",
        protocol_id="protocol-1",
        clean_metrics=(_metric(0.9),),
        attacked_metrics=(_metric(0.3),),
        reason="object_vanishing",
        affected_object_id="car-1",
        metadata={
            "task": "detection2d",
            "failure_type": "miss",
            "class_label": "Car",
            "object_size_bucket": "medium",
            "attack_family": "white_box",
            "severity": severity,
            "allowed_uses": ["training", "review"] if allowed else ["benchmark"],
        },
    )


def test_failure_grouping_is_deterministic_and_semantically_keyed() -> None:
    cases = (_case("failure-b", severity=4), _case("failure-a", severity=5))
    grouper = FailureGrouper()

    first = grouper.group(cases)
    second = grouper.group(tuple(reversed(cases)))

    assert first == second
    assert len(first) == 1
    assert first[0].member_ids == ("failure-a", "failure-b")
    assert first[0].selection_allowed is True
    assert first[0].metadata["severity_band"] == "high"


def test_failure_grouping_preserves_training_permission() -> None:
    clusters = FailureGrouper().group(
        (_case("failure-a", severity=2, allowed=False),)
    )

    assert clusters[0].selection_allowed is False
    assert clusters[0].allowed_uses == ("benchmark",)
