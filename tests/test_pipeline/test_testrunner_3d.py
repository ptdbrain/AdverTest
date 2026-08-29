"""End-to-end integration test for TestRunner with 3D perception models, datasets, and LiDAR attacks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from src.adapters.base import ModelAdapter
from src.core.types import Box3D, DetectionPrediction, LidarFrame, ModelInfo, Sample
from src.datasets.base import DatasetParams, DatasetSource
from src.pipeline.runner import RunConfig, TestRunner


class Synthetic3DDataset(DatasetSource):
    """Minimal in-memory 3D dataset for end-to-end runner testing."""

    name = "synthetic_3d"
    modality = "lidar"
    task_id = "detection3d"
    loader_version = "1.0.0"
    params_model = DatasetParams
    input_schema = ("lidar_point_cloud",)
    annotation_schema = ("boxes3d",)

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.anonymized = True

    def load(self, limit: int | None = None) -> list[Sample]:
        rng = np.random.default_rng(195)
        samples = []
        for idx in range(limit or 5):
            pts = rng.uniform(-20, 20, size=(200, 4)).astype(np.float32)
            frame = LidarFrame(points=pts, fields=("x", "y", "z", "intensity"), sensor_model="KITTI")
            boxes = (
                Box3D(
                    x=float(10.0 + idx),
                    y=float(idx * 2.0),
                    z=0.5,
                    length=4.2,
                    width=1.9,
                    height=1.6,
                    yaw=0.1,
                    label="Car",
                    score=1.0,
                ),
                Box3D(
                    x=float(25.0 + idx),
                    y=float(-idx * 2.0),
                    z=0.0,
                    length=0.8,
                    width=0.8,
                    height=1.7,
                    yaw=0.0,
                    label="Pedestrian",
                    score=1.0,
                ),
            )
            samples.append(
                Sample(
                    sample_id=f"synth-3d-{idx}",
                    image=np.zeros((10, 10, 3), dtype=np.float32),
                    lidar_frame=frame,
                    boxes3d=boxes,
                )
            )
        return samples

    def require_anonymized(self) -> None:
        pass


class MockPointPillarsAdapter(ModelAdapter):
    """Simulates PointPillars inference by detecting GT with small perturbation."""

    name = "pointpillars"
    task = "detection3d"
    modality = "lidar"
    version = "mmdet3d-1.4.0"
    runnable = True

    def metadata(self) -> ModelInfo:
        return ModelInfo(
            name=self.name,
            task=self.task,
            version=self.version,
            modality=self.modality,
            supports_gradients=False,
            classes=("Car", "Pedestrian", "Cyclist"),
            checkpoint_hash="mock-checkpoint-hash",
            runnable=True,
        )

    def predict(self, samples: Sequence[Sample]) -> list[DetectionPrediction]:
        predictions: list[DetectionPrediction] = []
        for sample in samples:
            boxes: list[Box3D] = []
            if sample.boxes3d:
                for gt in sample.boxes3d:
                    # Drop far objects or degraded frames if LiDAR points dropped severely
                    point_count = len(sample.lidar_frame.points) if sample.lidar_frame else 0
                    if point_count < 50:
                        # Heavy attack lost detection
                        continue
                    boxes.append(
                        Box3D(
                            x=gt.x + 0.05,
                            y=gt.y + 0.05,
                            z=gt.z,
                            length=gt.length,
                            width=gt.width,
                            height=gt.height,
                            yaw=gt.yaw,
                            label=gt.label,
                            score=0.92,
                        )
                    )
            predictions.append(
                DetectionPrediction(
                    sample_id=sample.sample_id,
                    boxes3d=tuple(boxes),
                    latency_ms=15.0,
                    metadata={"coordinate_frame": "LIDAR"},
                )
            )
        return predictions


def test_testrunner_preflight_for_3d(monkeypatch: pytest.MonkeyPatch) -> None:
    """Preflight check correctly identifies 3D LiDAR attacks vs 2D image attacks."""
    from src.adapters import MODELS
    from src.datasets import DATASETS

    monkeypatch.setitem(MODELS._plugins, "mock_pointpillars", MockPointPillarsAdapter)
    monkeypatch.setitem(DATASETS._plugins, "synthetic_3d", Synthetic3DDataset)

    runner = TestRunner()
    config = RunConfig(
        model="mock_pointpillars",
        dataset="synthetic_3d",
        attacks=["lidar_fog", "lidar_snow", "lidar_point_dropout", "gaussian_noise"],
        severities=[1, 3],
        limit=2,
    )

    preflight = runner.preflight(config)
    assert "lidar_fog" in preflight.compatible
    assert "lidar_snow" in preflight.compatible
    assert "lidar_point_dropout" in preflight.compatible
    assert not preflight.fatal_errors


def test_testrunner_full_3d_benchmark_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute complete 3D benchmark pipeline through TestRunner."""
    from src.adapters import MODELS
    from src.datasets import DATASETS

    monkeypatch.setitem(MODELS._plugins, "mock_pointpillars", MockPointPillarsAdapter)
    monkeypatch.setitem(DATASETS._plugins, "synthetic_3d", Synthetic3DDataset)

    runner = TestRunner()
    config = RunConfig(
        model="mock_pointpillars",
        dataset="synthetic_3d",
        attacks=["lidar_fog", "lidar_point_dropout"],
        severities=[1, 5],
        limit=3,
        execution_mode="benchmark",
        bootstrap_repetitions=0,
    )

    report = runner.run(config)

    # Verify high-level metrics
    assert report.model == "pointpillars"
    assert report.dataset == "synthetic_3d"
    assert report.n_samples == 3
    assert report.ap_clean > 0.0

    # Verify clean metrics structure
    clean_metrics = report.metrics["clean"]
    assert "kitti_3d_ap" in clean_metrics
    assert "mean_bev_iou" in clean_metrics
    assert "kitti_3d_ap_near" in clean_metrics
    assert "kitti_3d_ap_medium" in clean_metrics
    assert "kitti_3d_ap_far" in clean_metrics

    # Verify cells
    assert len(report.cells) == 4  # 2 attacks * 2 severities
    for cell in report.cells:
        assert cell.attack in ("lidar_fog", "lidar_point_dropout")
        assert cell.severity in (1, 5)
        assert cell.ap >= 0.0
        assert "objects_broken" in cell.metrics
        assert "attack_success_rate" in cell.metrics

    # Verify evidence payload includes 3D coordinates
    assert len(report.sample_results) > 0
    for item in report.sample_results:
        assert item.ground_truth is not None
        assert "objects3d" in item.ground_truth
        assert len(item.ground_truth["objects3d"]) > 0
        obj3d = item.ground_truth["objects3d"][0]
        assert all(k in obj3d for k in ("x", "y", "z", "length", "width", "height", "yaw", "label"))

        assert item.clean_prediction is not None
        assert "boxes3d" in item.clean_prediction
        pred3d = item.clean_prediction["boxes3d"][0]
        assert all(k in pred3d for k in ("x", "y", "z", "length", "width", "height", "yaw", "label", "score"))
