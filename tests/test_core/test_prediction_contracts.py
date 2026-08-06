import json
import sqlite3

import numpy as np
import pytest

from src.core.objectives import AttackObjective, TrainingObjective
from src.core.types import (
    Box,
    DetectionPrediction,
    MaskPrediction,
    SegmentationPrediction,
)
from src.pipeline.cache import SqliteCache


def test_detection_prediction_is_keyword_only() -> None:
    with pytest.raises(TypeError):
        DetectionPrediction("sample-1", ())  # type: ignore[misc]

    prediction = DetectionPrediction(sample_id="sample-1", boxes=())

    assert prediction.sample_id == "sample-1"


def test_segmentation_prediction_accepts_a_boolean_2d_mask() -> None:
    mask = np.array([[False, True], [True, False]], dtype=np.bool_)
    instance = MaskPrediction(instance_id="instance-1", mask=mask, score=0.75)

    prediction = SegmentationPrediction(
        sample_id="sample-1",
        instances=(instance,),
        prompt_id="prompt-1",
    )

    assert prediction.instances == (instance,)
    assert prediction.prompt_id == "prompt-1"


@pytest.mark.parametrize(
    ("mask", "score"),
    [
        (np.zeros((2, 2), dtype=np.uint8), 0.5),
        (np.zeros((2, 2, 1), dtype=np.bool_), 0.5),
        (np.zeros((2, 2), dtype=np.bool_), float("nan")),
        (np.zeros((2, 2), dtype=np.bool_), -0.1),
        (np.zeros((2, 2), dtype=np.bool_), 1.1),
    ],
)
def test_mask_prediction_rejects_invalid_masks_and_scores(
    mask: np.ndarray,
    score: float,
) -> None:
    with pytest.raises(ValueError):
        MaskPrediction(instance_id="instance-1", mask=mask, score=score)


def test_targeted_attack_objective_requires_a_target_label() -> None:
    with pytest.raises(ValueError, match="target_label"):
        AttackObjective(kind="targeted")

    objective = AttackObjective(kind="targeted", target_label=7)

    assert objective.objective_version == "1.0.0"


def test_training_objective_rejects_invalid_weights() -> None:
    with pytest.raises(ValueError, match="weights"):
        TrainingObjective(kind="robust_mix", weights={"clean": -0.1})

    objective = TrainingObjective(
        kind="robust_mix",
        weights={"clean": 0.4, "adversarial": 0.6},
    )

    assert objective.version == "1.0.0"


def test_sqlite_cache_round_trips_prediction_discriminators(tmp_path) -> None:
    cache_path = tmp_path / "predictions.sqlite3"
    cache = SqliteCache(cache_path)
    detection = DetectionPrediction(
        sample_id="detection-1",
        boxes=(Box(1.0, 2.0, 3.0, 4.0, "Car", 0.9),),
        latency_ms=2.5,
        metadata={"model": "detector"},
    )
    segmentation = SegmentationPrediction(
        sample_id="segmentation-1",
        instances=(
            MaskPrediction(
                instance_id="mask-1",
                mask=np.array([[True, True], [False, True]], dtype=np.bool_),
                label="Car",
                score=0.8,
            ),
        ),
        prompt_id="prompt-1",
        latency_ms=3.5,
        metadata={"model": "segmenter"},
    )

    cache.put("detection-key", detection)
    cache.put("segmentation-key", segmentation)

    assert cache.get("detection-key") == detection
    restored = cache.get("segmentation-key")
    assert isinstance(restored, SegmentationPrediction)
    assert restored.sample_id == segmentation.sample_id
    assert restored.prompt_id == segmentation.prompt_id
    assert restored.latency_ms == segmentation.latency_ms
    assert restored.metadata == segmentation.metadata
    assert len(restored.instances) == 1
    assert restored.instances[0].instance_id == "mask-1"
    assert restored.instances[0].label == "Car"
    assert restored.instances[0].score == 0.8
    assert np.array_equal(restored.instances[0].mask, segmentation.instances[0].mask)


def test_sqlite_cache_rejects_unknown_prediction_type(tmp_path) -> None:
    cache_path = tmp_path / "predictions.sqlite3"
    cache = SqliteCache(cache_path)
    payload = json.dumps(
        {
            "prediction_type": "future-model-output",
            "sample_id": "bad-sample",
        }
    )
    with sqlite3.connect(cache_path) as connection:
        connection.execute(
            "INSERT INTO predictions(cache_key, payload) VALUES (?, ?)",
            ("bad-key", payload),
        )

    with pytest.raises(ValueError, match="prediction_type"):
        cache.get("bad-key")
