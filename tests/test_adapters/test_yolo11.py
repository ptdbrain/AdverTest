"""CPU-safe tests for the unified YOLO11 inference and surrogate adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.adapters import get_adapter
from src.adapters.yolo11 import Yolo11Adapter

WEIGHTS = "checkpoints/surrogates/yolo11s.pt"


def _adapter(**params: object) -> Yolo11Adapter:
    params.setdefault("weights", WEIGHTS)
    return get_adapter("yolo11", **params)  # type: ignore[return-value]


def test_metadata_is_lazy_and_exposes_attack_capabilities() -> None:
    info = _adapter().metadata()
    assert info.name == "yolo11"
    assert info.task == "detection2d"
    assert info.supports_gradients is True
    assert "yolo11s" in info.version
    assert "input_gradient" in Yolo11Adapter.describe()["capabilities"]


def test_version_distinguishes_checkpoints_and_settings() -> None:
    coco = _adapter(weights="yolo11s.pt").metadata().version
    tuned = _adapter(weights="runs/detect/train/weights/best.pt").metadata().version
    resized = _adapter(weights="yolo11s.pt", image_size=1280).metadata().version
    assert len({coco, tuned, resized}) == 3


def test_catalog_entry_is_complete() -> None:
    described = Yolo11Adapter.describe()
    assert described["owner"] == "group-d-e"
    assert described["docstring"]


def test_conversion_produces_bgr_uint8() -> None:
    image = np.zeros((4, 4, 3), dtype=np.float32)
    image[..., 0] = 1.0
    converted = Yolo11Adapter.to_backend_image(image)
    assert converted.dtype == np.uint8
    assert converted.shape == image.shape
    assert list(converted[0, 0]) == [0, 0, 255]
    assert converted.flags["C_CONTIGUOUS"]


def test_conversion_scales_and_clips_the_range() -> None:
    image = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
    assert list(Yolo11Adapter.to_backend_image(image)[0, 0]) == [255, 128, 0]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("car", "Car"), ("person", "Pedestrian"), ("bicycle", "Cyclist"), ("truck", None)],
)
def test_label_map_matches_normalised_classes(raw: str, expected: str | None) -> None:
    assert _adapter().map_label(raw) == expected


def test_truck_and_bus_can_be_folded_into_car() -> None:
    adapter = _adapter(map_truck_bus_to_car=True)
    assert adapter.map_label("truck") == "Car"
    assert adapter.map_label("bus") == "Car"


def test_convert_drops_unmapped_and_degenerate_boxes() -> None:
    rows = [
        (10.0, 10.0, 50.0, 60.0, "car", 0.9),
        (0.0, 0.0, 5.0, 5.0, "traffic light", 0.8),
        (20.0, 20.0, 20.0, 40.0, "person", 0.7),
    ]
    boxes = _adapter().convert(rows)
    assert [box.label for box in boxes] == ["Car"]
    assert boxes[0].score == pytest.approx(0.9)


def test_postprocess_applies_the_confidence_floor() -> None:
    adapter = _adapter(score_threshold=0.5)
    rows = [(0.0, 0.0, 10.0, 10.0, "car", 0.9), (0.0, 0.0, 10.0, 10.0, "person", 0.1)]
    assert len(adapter.postprocess(adapter.convert(rows))) == 1


def test_batching_covers_every_sample() -> None:
    adapter = _adapter(batch_size=3)
    chunks = list(adapter._chunks(list(range(7))))  # type: ignore[arg-type]
    assert [len(chunk) for chunk in chunks] == [3, 3, 1]


def test_missing_checkpoint_does_not_trigger_download(tmp_path: Path) -> None:
    adapter = _adapter(weights=str(tmp_path / "missing.pt"))
    with pytest.raises(FileNotFoundError, match="automatic download is disabled"):
        adapter._load()
