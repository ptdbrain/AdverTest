"""Task 6: Generic benchmark runner integration test for detection3d.

Follows the exact pattern of ``test_generic_benchmark.py`` but exercises
the 3D detection path with a fake adapter and the real Kitti3DEvaluator.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

import numpy as np

from src.adapters.base import ModelAdapter
from src.core.types import (
    Box3D,
    DetectionPrediction,
    LidarFrame,
    ModelInfo,
    ModelPrediction,
    Sample,
)
from src.evaluation.kitti3d import Kitti3DEvaluator
from src.pipeline.generic_benchmark import BenchmarkRunner
from src.pipeline.protocol import BenchmarkProtocol

# ---------------------------------------------------------------------------
# Fake adapter that echoes GT as predictions
# ---------------------------------------------------------------------------


class Fake3DAdapter(ModelAdapter):
    """Returns the sample's ground-truth boxes3d as predictions."""

    def metadata(self) -> ModelInfo:
        return ModelInfo(
            name="fake-3d",
            task="detection3d",
            version="fake-3d-v1",
            modality="lidar",
            runnable=True,
        )

    def predict(self, samples: Sequence[Sample]) -> list[ModelPrediction]:
        return [
            DetectionPrediction(
                sample_id=sample.sample_id,
                boxes3d=tuple(
                    Box3D(
                        x=box.x,
                        y=box.y,
                        z=box.z,
                        length=box.length,
                        width=box.width,
                        height=box.height,
                        yaw=box.yaw,
                        label=box.label,
                        score=0.95,
                    )
                    for box in sample.boxes3d
                ),
            )
            for sample in samples
        ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_generic_runner_runs_detection3d() -> None:
    """Run a complete 3D benchmark: clean + one attack variant."""
    rng = np.random.default_rng(42)
    points = rng.standard_normal((100, 4)).astype(np.float32)

    sample = Sample(
        sample_id="3d-s1",
        image=np.zeros((4, 4, 3), dtype=np.float32),
        lidar_frame=LidarFrame(points, ("x", "y", "z", "intensity"), "test-sensor"),
        boxes3d=(
            Box3D(
                x=10,
                y=0,
                z=1,
                length=4,
                width=2,
                height=1.5,
                yaw=0,
                label="Car",
                score=1.0,
            ),
        ),
    )

    evaluator = Kitti3DEvaluator()

    protocol = (
        BenchmarkProtocol(
            name="test-3d-benchmark",
            dataset_version_id="kitti3d-test",
            sample_ids=("3d-s1",),
            sample_hashes={"3d-s1": "hash-1"},
            ground_truth_hashes={"3d-s1": "gt-1"},
            recipe_hashes=("recipe-fog-1",),
            seeds=(195,),
            metric_versions={"kitti_3d_ap": "advertest-bev-v1", "bev_iou": "1.0.0"},
            created_at=datetime.now(UTC),
        )
        .transition("VALIDATED")
        .transition("LOCKED")
    )

    runner = BenchmarkRunner(
        sample_provider=lambda _protocol: [sample],
        variant_provider=lambda _recipe_hash, samples: [
            s.with_lidar_frame(
                LidarFrame(
                    s.lidar_frame.points * 0.9,
                    s.lidar_frame.fields,
                    s.lidar_frame.sensor_model,
                )
            )
            for s in samples
        ],
    )

    report = runner.run(
        protocol,
        [Fake3DAdapter()],
        {"detection3d": evaluator},
    )

    assert len(report.models) == 1

    model_result = report.models[0]
    assert model_result.task == "detection3d"
    assert model_result.clean.headline.name == "kitti_3d_ap"
    assert model_result.clean.headline.value > 0.0
    assert model_result.paired_sample_ids == ("3d-s1",)
    assert len(model_result.cells) == 1
    assert report.complete is True


def test_generic_runner_skips_detection3d_without_evaluator() -> None:
    """Model is skipped when no evaluator is registered for detection3d."""
    sample = Sample(
        sample_id="3d-s1",
        image=np.zeros((4, 4, 3), dtype=np.float32),
        boxes3d=(
            Box3D(
                x=10,
                y=0,
                z=1,
                length=4,
                width=2,
                height=1.5,
                yaw=0,
                label="Car",
                score=1.0,
            ),
        ),
    )

    protocol = (
        BenchmarkProtocol(
            name="test-3d-no-eval",
            dataset_version_id="kitti3d-test",
            sample_ids=("3d-s1",),
            sample_hashes={"3d-s1": "hash-1"},
            ground_truth_hashes={"3d-s1": "gt-1"},
            metric_versions={},
        )
        .transition("VALIDATED")
        .transition("LOCKED")
    )

    runner = BenchmarkRunner(
        sample_provider=lambda _protocol: [sample],
        variant_provider=lambda _recipe_hash, samples: samples,
    )

    report = runner.run(
        protocol,
        [Fake3DAdapter()],
        {},  # no evaluators
    )

    assert report.models == ()
    assert len(report.skipped) == 1
    assert "detection3d" in report.skipped[0].reason
