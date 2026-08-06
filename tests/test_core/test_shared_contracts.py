from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from src.core.contracts import (
    DetectionPredictionWire,
    MaskWireV1,
    ModelVersionMetadata,
    SegmentationPredictionWire,
)
from src.core.events import JobRequest, ProgressEvent
from src.evaluation.contracts import FailureCase, FailureCluster, MetricEnvelope
from src.training.contracts import DefenseProfile, TrainingRunConfig

FIXTURES = Path(__file__).parents[1] / "fixtures" / "contracts"


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_metric_envelope_never_confuses_ratio_and_percent() -> None:
    metric = MetricEnvelope(
        name="degradation",
        value=0.42,
        unit="ratio",
        version="1.0.0",
        percent_value=42.0,
        higher_is_better=False,
    )

    assert MetricEnvelope.model_validate_json(metric.model_dump_json()) == metric

    with pytest.raises(ValidationError, match="percent_value"):
        MetricEnvelope(
            name="degradation",
            value=0.42,
            unit="ratio",
            version="1.0.0",
            percent_value=0.42,
            higher_is_better=False,
        )


def test_mask_wire_v1_round_trips_row_major_boolean_masks() -> None:
    mask = np.array([[True, True, False], [False, True, False]], dtype=np.bool_)

    wire = MaskWireV1.from_array(mask)

    assert wire.runs == (0, 2, 2, 1, 1)
    assert np.array_equal(wire.to_array(), mask)
    assert MaskWireV1.model_validate_json(wire.model_dump_json()) == wire


def test_contracts_are_frozen_and_reject_unknown_fields() -> None:
    event = ProgressEvent.model_validate_json(
        """{
          "job_id": "job-1",
          "job_type": "generation",
          "state": "running",
          "progress_ratio": 0.5,
          "sequence": 2,
          "detail": {},
          "created_at": "2026-08-06T12:00:00Z"
        }"""
    )
    with pytest.raises(ValidationError):
        event.state = "completed"

    with pytest.raises(ValidationError, match="extra"):
        ProgressEvent.model_validate(
            {
                **event.model_dump(),
                "unexpected": True,
            }
        )


def test_training_handoff_contracts_are_versioned() -> None:
    profile = DefenseProfile(
        profile_id="defense-balanced-v1",
        recipe_ids=("weather-balanced", "adversarial-balanced"),
        clean_replay_ratio=0.4,
        generated_ratio=0.6,
    )
    config = TrainingRunConfig(
        run_id="training-1",
        trainer_name="fake-yolo",
        model_version="yolo-fixture-1",
        dataset_version_id="dataset-fixture-1",
        split_manifest_id="split-fixture-1",
        defense_profile_id=profile.profile_id,
        seed=195,
        epochs=2,
        batch_size=4,
        learning_rate=0.001,
    )

    assert profile.version == "1.0.0"
    assert config.contract_version == "1.0.0"


def test_committed_contract_fixtures_are_loadable() -> None:
    detection = DetectionPredictionWire.model_validate_json(
        _fixture("detection_prediction.json")
    )
    segmentation = SegmentationPredictionWire.model_validate_json(
        _fixture("segmentation_prediction.json")
    )
    metric = MetricEnvelope.model_validate_json(_fixture("metric_envelope.json"))
    failure_payload = __import__("json").loads(_fixture("failure_payload.json"))
    failure = FailureCase.model_validate(failure_payload["case"])
    cluster = FailureCluster.model_validate(failure_payload["cluster"])
    job_payload = __import__("json").loads(_fixture("job_request_event.json"))
    request = JobRequest.model_validate(job_payload["request"])
    event = ProgressEvent.model_validate(job_payload["event"])
    metadata = ModelVersionMetadata.model_validate_json(
        _fixture("model_metadata.json")
    )

    assert detection.to_domain().sample_id == "sample-detection-1"
    assert segmentation.to_domain().instances[0].mask.dtype == np.bool_
    assert metric.percent_value == 18.5
    assert failure.case_id in cluster.member_ids
    assert request.job_id == event.job_id
    assert metadata.model_id == "sam2"
