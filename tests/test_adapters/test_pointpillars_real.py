"""Real PointPillars validation gate; intentionally skipped without its environment."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from src.adapters.pointpillars import PointPillarsAdapter
from src.core.types import LidarFrame, Sample


def _real_validation_inputs() -> tuple[Path, Path, Path] | None:
    config = os.environ.get("POINTPILLARS_CONFIG")
    weights = os.environ.get("POINTPILLARS_WEIGHTS")
    lidar = os.environ.get("POINTPILLARS_LIDAR_BIN")
    if not all((config, weights, lidar)):
        return None
    paths = tuple(Path(value).expanduser() for value in (config, weights, lidar))
    return paths if all(path.is_file() for path in paths) else None


@pytest.mark.gpu
def test_real_pointpillars_kitti_inference() -> None:
    """Runs only against explicit real GPU, MMDetection3D and KITTI artifacts."""
    inputs = _real_validation_inputs()
    has_mmdet3d = importlib.util.find_spec("mmdet3d") is not None
    try:
        import torch
    except ImportError:
        torch = None  # type: ignore[assignment]
    if inputs is None or not has_mmdet3d or torch is None or not torch.cuda.is_available():
        pytest.skip("WAITING_FOR_GPU_VALIDATION")
    config, weights, lidar_path = inputs
    points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
    adapter = PointPillarsAdapter(config=str(config), weights=str(weights), device="cuda:0")
    sample = Sample(
        sample_id="real-kitti-smoke",
        image=np.zeros((1, 1, 3), dtype=np.float32),
        lidar_frame=LidarFrame(points, fields=("x", "y", "z", "intensity"), sensor_model="KITTI"),
    )

    prediction = adapter.predict([sample])[0]

    assert prediction.sample_id == sample.sample_id
    assert prediction.metadata["coordinate_frame"] == "LIDAR"
