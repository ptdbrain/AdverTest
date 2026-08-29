"""Tests for distance-bucketed 3D evaluation metrics and failure metadata."""

from __future__ import annotations

import numpy as np

from src.core.types import Box3D, DetectionPrediction, Sample
from src.evaluation.kitti3d import Kitti3DEvaluator
from src.pipeline.protocol import BenchmarkProtocol


def _make_protocol() -> BenchmarkProtocol:
    return BenchmarkProtocol(
        name="test-3d-distance",
        dataset_version_id="kitti3d-test",
        sample_ids=("sample-near", "sample-far"),
        sample_hashes={"sample-near": "hash-1", "sample-far": "hash-2"},
        ground_truth_hashes={"sample-near": "gt-1", "sample-far": "gt-2"},
        metric_versions={"kitti_3d_ap": "advertest-bev-v1", "bev_iou": "1.0.0"},
    ).transition("VALIDATED").transition("LOCKED")


def test_distance_bucketed_metrics_computed_correctly() -> None:
    # Near box: x=10, y=0 -> distance = 10m (<20m -> near)
    # Far box: x=50, y=0 -> distance = 50m (>=40m -> far)
    gt_near = Box3D(x=10.0, y=0.0, z=0.0, length=4.0, width=2.0, height=1.5, yaw=0.0, label="Car")
    gt_far = Box3D(x=50.0, y=0.0, z=0.0, length=4.0, width=2.0, height=1.5, yaw=0.0, label="Car")

    sample_near = Sample(
        sample_id="sample-near",
        image=np.zeros((4, 4, 3), dtype=np.float32),
        boxes3d=(gt_near,),
    )
    sample_far = Sample(
        sample_id="sample-far",
        image=np.zeros((4, 4, 3), dtype=np.float32),
        boxes3d=(gt_far,),
    )

    # Perfect prediction for near, no prediction for far
    pred_near = DetectionPrediction(
        sample_id="sample-near",
        boxes3d=(Box3D(x=10.0, y=0.0, z=0.0, length=4.0, width=2.0, height=1.5, yaw=0.0, label="Car", score=0.95),),
    )
    pred_far = DetectionPrediction(
        sample_id="sample-far",
        boxes3d=(),
    )

    evaluator = Kitti3DEvaluator(iou_threshold=0.5)
    result = evaluator.evaluate(
        predictions=[pred_near, pred_far],
        samples=[sample_near, sample_far],
        protocol=_make_protocol(),
    )

    metrics_dict = {m.name: m.value for m in result.supplemental_metrics}
    assert "kitti_3d_ap_near" in metrics_dict
    assert "kitti_3d_ap_medium" in metrics_dict
    assert "kitti_3d_ap_far" in metrics_dict

    assert metrics_dict["kitti_3d_ap_near"] > 0.9  # Near matched
    assert metrics_dict["kitti_3d_ap_far"] == 0.0   # Far missed

    # Check failure metadata contains distance
    assert len(result.failures) == 1
    failure = result.failures[0]
    assert failure.reason == "MISSED_GT"
    assert failure.metadata.get("distance_bucket") == "far"
    assert failure.metadata.get("distance") == 50.0
