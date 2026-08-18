"""CPU-only contract tests for the optional PointPillars boundary.

The fake backend isolates MMDetection3D; these tests do not validate a real
PointPillars configuration, checkpoint, or GPU inference.
"""

from __future__ import annotations

import builtins
import importlib
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.adapters import pointpillars
from src.core.types import LidarFrame, Sample


class _Tensor:
    def __init__(self, values: list[list[float]] | list[float] | list[int]) -> None:
        self._values = np.asarray(values)

    def detach(self) -> _Tensor:
        return self

    def cpu(self) -> _Tensor:
        return self

    def numpy(self) -> np.ndarray:
        return self._values


def _sample(sample_id: str, *, with_lidar: bool = True) -> Sample:
    frame = (
        LidarFrame(
            np.array([[1.0, 2.0, 3.0, 0.5], [4.0, 5.0, 6.0, 0.25]], dtype=np.float32),
            fields=("x", "y", "z", "intensity"),
            sensor_model="KITTI",
        )
        if with_lidar
        else None
    )
    return Sample(
        sample_id=sample_id,
        image=np.zeros((2, 2, 3), dtype=np.float32),
        lidar_frame=frame,
    )


def _fake_result(
    rows: list[list[float]] | None = None,
    scores: list[float] | None = None,
    labels: list[int] | None = None,
) -> Any:
    instances = types.SimpleNamespace(
        bboxes_3d=types.SimpleNamespace(
            # LiDARInstance3DBoxes uses (x, y, z, dx, dy, dz, yaw).
            tensor=_Tensor(rows or [[10.0, 20.0, 30.0, 4.0, 2.0, 1.5, 0.75], [1, 2, 3, 9, 8, 7, 0]]),
        ),
        scores_3d=_Tensor(scores or [0.9, 0.1]),
        labels_3d=_Tensor(labels or [0, 1]),
    )
    return types.SimpleNamespace(pred_instances_3d=instances)


@pytest.fixture
def fake_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    config = tmp_path / "pointpillars.py"
    weights = tmp_path / "pointpillars.pth"
    config.write_text("model = dict()\n", encoding="utf-8")
    weights.write_bytes(b"unit-test-checkpoint")
    calls: dict[str, Any] = {"init": [], "inference": []}
    fake_model = types.SimpleNamespace(dataset_meta={"classes": ("Car", "Pedestrian", "Cyclist")})

    def fake_init_model(config_path: str, weights_path: str, device: str) -> Any:
        calls["init"].append((config_path, weights_path, device))
        return fake_model

    def fake_inference_detector(model: Any, point_path: str) -> Any:
        calls["inference"].append((model, point_path, Path(point_path).read_bytes()))
        return calls.get("result", _fake_result())

    # Makes the initial stub importable during RED; production implementation
    # must instead call this lazily imported boundary.
    monkeypatch.setitem(sys.modules, "mmdet3d", types.ModuleType("mmdet3d"))
    monkeypatch.setattr(
        pointpillars,
        "_load_mmdet3d_backend",
        lambda: (fake_init_model, fake_inference_detector),
        raising=False,
    )
    return {"config": config, "weights": weights, "calls": calls, "model": fake_model}


def _adapter(
    fake_backend: dict[str, Any],
    *,
    max_detections: int = 10,
) -> pointpillars.PointPillarsAdapter:
    return pointpillars.PointPillarsAdapter(
        config=str(fake_backend["config"]),
        weights=str(fake_backend["weights"]),
        device="cpu",
        score_threshold=0.25,
        max_detections=max_detections,
    )


def test_pointpillars_initializes_once_and_maps_lidar_output(fake_backend: dict[str, Any]) -> None:
    """Catches a missing adapter boundary or incorrect MMDet box conversion."""
    adapter = _adapter(fake_backend)

    predictions = adapter.predict([_sample("first")])

    assert fake_backend["calls"]["init"] == [(str(fake_backend["config"]), str(fake_backend["weights"]), "cpu")]
    assert len(predictions) == 1
    assert predictions[0].sample_id == "first"
    assert len(predictions[0].boxes3d) == 1
    box = predictions[0].boxes3d[0]
    assert (box.x, box.y, box.z, box.length, box.width, box.height, box.yaw) == pytest.approx(
        (10.0, 20.0, 30.75, 4.0, 2.0, 1.5, 0.75)
    )
    assert (box.label, box.score) == ("Car", pytest.approx(0.9))
    assert predictions[0].metadata["coordinate_frame"] == "LIDAR"
    assert predictions[0].metadata["checkpoint_hash"] == adapter.checkpoint_hash


def test_pointpillars_rejects_sample_without_lidar(fake_backend: dict[str, Any]) -> None:
    """Catches accidental image-only inference for a LiDAR adapter."""
    adapter = _adapter(fake_backend)

    with pytest.raises(ValueError, match="lidar_frame"):
        adapter.predict([_sample("no-lidar", with_lidar=False)])

    assert fake_backend["calls"]["inference"] == []


def test_pointpillars_preserves_sample_order_and_writes_float32_bin(fake_backend: dict[str, Any]) -> None:
    """Catches reordered prediction batches or a non-MMD3D-compatible point file."""
    adapter = _adapter(fake_backend)
    first, second = _sample("sample-a"), _sample("sample-b")

    predictions = adapter.predict([first, second])

    assert [prediction.sample_id for prediction in predictions] == ["sample-a", "sample-b"]
    assert [row[2] for row in fake_backend["calls"]["inference"]] == [
        first.lidar_frame.points.astype(np.float32).tobytes(),
        second.lidar_frame.points.astype(np.float32).tobytes(),
    ]
    assert all(not Path(row[1]).exists() for row in fake_backend["calls"]["inference"])


def test_pointpillars_sorts_scores_before_capping_with_stable_ties(fake_backend: dict[str, Any]) -> None:
    """Catches backend-order truncation that discards a higher-score detection."""
    fake_backend["calls"]["result"] = _fake_result(
        rows=[
            [10, 0, 0, 1, 1, 2, 0],
            [20, 0, 0, 1, 1, 2, 0],
            [30, 0, 0, 1, 1, 2, 0],
            [40, 0, 0, 1, 1, 2, 0],
        ],
        scores=[0.4, 0.8, 0.8, 0.95],
        labels=[0, 1, 2, 0],
    )
    adapter = _adapter(fake_backend, max_detections=3)

    prediction = adapter.predict([_sample("unsorted")])[0]

    assert [(box.x, box.score, box.label) for box in prediction.boxes3d] == [
        (40.0, 0.95, "Car"),
        (20.0, 0.8, "Pedestrian"),
        (30.0, 0.8, "Cyclist"),
    ]


def test_pointpillars_module_import_does_not_import_optional_mmdet3d(monkeypatch: pytest.MonkeyPatch) -> None:
    """Catches moving the optional MMDetection3D import to module scope."""
    real_import = builtins.__import__

    def forbid_mmdet3d(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "mmdet3d" or name.startswith("mmdet3d."):
            raise AssertionError("adapter module import attempted to import MMDetection3D")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbid_mmdet3d)

    importlib.reload(pointpillars)


def test_pointpillars_wraps_missing_optional_runtime_after_file_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Catches leaking optional-dependency ImportError from adapter construction."""
    config = tmp_path / "pointpillars.py"
    weights = tmp_path / "pointpillars.pth"
    config.write_text("model = dict()\n", encoding="utf-8")
    weights.write_bytes(b"unit-test-checkpoint")
    real_import = builtins.__import__

    def missing_mmdet3d(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "mmdet3d.apis":
            raise ImportError("forced missing MMDetection3D")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_mmdet3d)

    with pytest.raises(RuntimeError, match="requires MMDetection3D 1.4.0") as raised:
        pointpillars.PointPillarsAdapter(config=str(config), weights=str(weights), device="cpu")

    assert isinstance(raised.value.__cause__, ImportError)


def test_pointpillars_metadata_stays_non_runnable_until_real_validation(fake_backend: dict[str, Any]) -> None:
    """Catches publishing a mocked adapter as a runnable catalog model."""
    metadata = _adapter(fake_backend).metadata()

    assert metadata.task == "detection3d"
    assert metadata.modality == "lidar"
    assert metadata.runnable is False
