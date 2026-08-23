"""Unit tests for the KITTI 3D evaluator."""

import numpy as np
import pytest

from src.core.types import Box3D, DetectionPrediction, Sample
from src.evaluation.kitti3d import Kitti3DEvaluator
from src.pipeline.protocol import BenchmarkProtocol


@pytest.fixture
def protocol():
    """Locked protocol with one sample and correct metric versions."""
    return BenchmarkProtocol(
        name="test-3d",
        dataset_version_id="kitti3d-test",
        sample_ids=("test-001",),
        sample_hashes={"test-001": "hash-1"},
        ground_truth_hashes={"test-001": "gt-1"},
        metric_versions={"kitti_3d_ap": "advertest-bev-v1", "bev_iou": "1.0.0"},
    ).transition("VALIDATED").transition("LOCKED")


@pytest.fixture
def sample():
    """Sample with one Car GT box."""
    return Sample(
        sample_id="test-001",
        image=np.zeros((4, 4, 3), dtype=np.float32),
        boxes3d=(
            Box3D(
                x=10, y=0, z=1,
                length=4, width=2, height=1.5,
                yaw=0, label="Car", score=1.0,
            ),
        ),
    )


def test_perfect_prediction_no_failure(protocol, sample):
    evaluator = Kitti3DEvaluator()
    prediction = DetectionPrediction(
        sample_id="test-001",
        boxes3d=(
            Box3D(
                x=10, y=0, z=1,
                length=4, width=2, height=1.5,
                yaw=0, label="Car", score=0.9,
            ),
        ),
    )
    result = evaluator.evaluate([prediction], [sample], protocol)

    assert result.headline.value == pytest.approx(1.0)
    assert result.headline.name == "kitti_3d_ap"
    assert len(result.failures) == 0


def test_no_predictions_all_missed(protocol, sample):
    evaluator = Kitti3DEvaluator()
    prediction = DetectionPrediction(sample_id="test-001", boxes3d=())

    result = evaluator.evaluate([prediction], [sample], protocol)

    assert result.headline.value == pytest.approx(0.0)
    assert len(result.failures) == 1
    assert result.failures[0].reason == "MISSED_GT"


def test_wrong_class_mismatch(protocol, sample):
    evaluator = Kitti3DEvaluator()
    prediction = DetectionPrediction(
        sample_id="test-001",
        boxes3d=(
            Box3D(
                x=10, y=0, z=1,
                length=4, width=2, height=1.5,
                yaw=0, label="Pedestrian", score=0.9,
            ),
        ),
    )
    result = evaluator.evaluate([prediction], [sample], protocol)

    assert result.headline.value == pytest.approx(0.0)
    failure_reasons = {f.reason for f in result.failures}
    assert "MISSED_GT" in failure_reasons
    assert "CLASS_MISMATCH" in failure_reasons


def test_far_away_prediction_localization_failure(protocol, sample):
    evaluator = Kitti3DEvaluator()
    prediction = DetectionPrediction(
        sample_id="test-001",
        boxes3d=(
            Box3D(
                x=12.0, y=0.5, z=1,
                length=4, width=2, height=1.5,
                yaw=0, label="Car", score=0.9,
            ),
        ),
    )
    result = evaluator.evaluate([prediction], [sample], protocol)

    failure_reasons = {f.reason for f in result.failures}
    assert "LOCALIZATION_FAILURE" in failure_reasons or "FALSE_POSITIVE" in failure_reasons
    assert "MISSED_GT" in failure_reasons


def test_extra_prediction_false_positive(protocol, sample):
    evaluator = Kitti3DEvaluator()
    prediction = DetectionPrediction(
        sample_id="test-001",
        boxes3d=(
            Box3D(
                x=10, y=0, z=1,
                length=4, width=2, height=1.5,
                yaw=0, label="Car", score=0.9,
            ),
            Box3D(
                x=100, y=100, z=1,
                length=4, width=2, height=1.5,
                yaw=0, label="Car", score=0.8,
            ),
        ),
    )
    result = evaluator.evaluate([prediction], [sample], protocol)

    failure_reasons = [f.reason for f in result.failures]
    assert failure_reasons == ["FALSE_POSITIVE"]


def test_per_sample_metrics_have_required_keys(protocol, sample):
    evaluator = Kitti3DEvaluator()
    prediction = DetectionPrediction(
        sample_id="test-001",
        boxes3d=(
            Box3D(
                x=10, y=0, z=1,
                length=4, width=2, height=1.5,
                yaw=0, label="Car", score=0.9,
            ),
        ),
    )
    result = evaluator.evaluate([prediction], [sample], protocol)

    metric_names = {m.name for m in result.per_sample_metrics["test-001"]}
    expected_names = {
        "gt_count",
        "prediction_count",
        "matched_count",
        "missed_count",
        "false_positive_count",
        "mean_bev_iou",
    }
    assert metric_names == expected_names


def test_multi_class_ap_is_macro_averaged(protocol):
    evaluator = Kitti3DEvaluator()
    multi_class_sample = Sample(
        sample_id="test-001",
        image=np.zeros((4, 4, 3), dtype=np.float32),
        boxes3d=(
            Box3D(
                x=10, y=0, z=1,
                length=4, width=2, height=1.5,
                yaw=0, label="Car", score=1.0,
            ),
            Box3D(
                x=20, y=0, z=1,
                length=1, width=1, height=1.5,
                yaw=0, label="Pedestrian", score=1.0,
            ),
        ),
    )
    prediction = DetectionPrediction(
        sample_id="test-001",
        boxes3d=(
            Box3D(
                x=10, y=0, z=1,
                length=4, width=2, height=1.5,
                yaw=0, label="Car", score=0.9,
            ),
        ),
    )
    result = evaluator.evaluate([prediction], [multi_class_sample], protocol)

    # Car AP=1.0, Pedestrian AP=0.0 → macro average = 0.5
    assert result.headline.value == pytest.approx(0.5, abs=0.01)
