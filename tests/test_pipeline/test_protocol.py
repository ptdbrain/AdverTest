from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from src.pipeline.protocol import BenchmarkProtocol


def _protocol(**updates):
    payload = {
        "name": "fixture",
        "dataset_version_id": "dataset-1",
        "sample_ids": ("s1",),
        "sample_hashes": {"s1": "source-1"},
        "ground_truth_hashes": {"s1": "gt-1"},
        "recipe_hashes": ("recipe-1",),
        "seeds": (195,),
        "preprocessing_versions": {"model": "prep-1"},
        "thresholds": {"score": 0.25},
        "prompt_protocol": "box-v1",
        "metric_versions": {"ap": "1.0.0"},
        "bootstrap_iterations": 100,
        "bootstrap_seed": 7,
        "environment": {"python": "3.11"},
        "framework_versions": {"numpy": "2"},
        "class_mapping_version": "classes-v1",
    }
    payload.update(updates)
    return BenchmarkProtocol(**payload)


def test_protocol_identity_excludes_creation_time_but_includes_execution_inputs() -> None:
    first = _protocol(created_at=datetime(2026, 1, 1, tzinfo=UTC))
    second = _protocol(created_at=datetime(2026, 2, 1, tzinfo=UTC))
    assert first.protocol_id == second.protocol_id

    variants = (
        _protocol(dataset_version_id="dataset-2"),
        _protocol(sample_hashes={"s1": "changed"}),
        _protocol(ground_truth_hashes={"s1": "changed"}),
        _protocol(recipe_hashes=("recipe-2",)),
        _protocol(seeds=(196,)),
        _protocol(preprocessing_versions={"model": "prep-2"}),
        _protocol(thresholds={"score": 0.5}),
        _protocol(prompt_protocol="point-v1"),
        _protocol(metric_versions={"ap": "2.0.0"}),
        _protocol(bootstrap_seed=8),
        _protocol(environment={"python": "3.12"}),
        _protocol(framework_versions={"numpy": "3"}),
    )
    assert all(item.protocol_id != first.protocol_id for item in variants)


def test_protocol_transitions_and_locking_are_explicit() -> None:
    draft = _protocol()
    validated = draft.transition("VALIDATED")
    locked = validated.transition("LOCKED")
    retired = locked.transition("RETIRED")
    assert retired.status == "RETIRED"
    with pytest.raises(ValueError, match="transition"):
        draft.transition("LOCKED")
    with pytest.raises(ValidationError):
        locked.name = "changed"
