"""YOLO11 adapter: catalog metadata, colour order, label mapping.

Everything here runs without ``ultralytics`` or ``torch`` installed — that is the
point of keeping the heavy imports inside ``_load``. The one test that really
runs the model is skipped unless the extras are present.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from src.adapters import get_adapter
from src.adapters.yolo11 import Yolo11Adapter

HAS_ULTRALYTICS = importlib.util.find_spec("ultralytics") is not None


def _adapter(**params: object) -> Yolo11Adapter:
    return get_adapter("yolo11", **params)  # type: ignore[return-value]


# ------------------------------------------------------------------ catalog


def test_metadata_works_without_weights_or_gpu() -> None:
    """The catalog endpoint calls this on every request; it must never load."""
    info = _adapter().metadata()
    assert info.name == "yolo11"
    assert info.task == "detection2d"
    assert info.supports_gradients is False


def test_version_distinguishes_checkpoints_and_settings() -> None:
    """The prediction cache is keyed on version: COCO and fine-tuned must differ."""
    coco = _adapter(weights="yolo11s.pt").metadata().version
    tuned = _adapter(weights="runs/detect/train/weights/best.pt").metadata().version
    resized = _adapter(weights="yolo11s.pt", imgsz=1280).metadata().version
    assert len({coco, tuned, resized}) == 3
    assert "yolo11s" in coco and "best" in tuned


def test_catalog_entry_is_complete() -> None:
    described = Yolo11Adapter.describe()
    assert described["owner"] == "phong"
    assert described["docstring"]


# ------------------------------------------------------------- colour order


def test_conversion_produces_bgr_uint8() -> None:
    """Ultralytics reads raw arrays as BGR; a red frame must come back blue-last."""
    image = np.zeros((4, 4, 3), dtype=np.float32)
    image[..., 0] = 1.0  # pure red in RGB
    converted = Yolo11Adapter.to_backend_image(image)
    assert converted.dtype == np.uint8
    assert converted.shape == image.shape
    assert int(converted[0, 0, 2]) == 255, "red must land in the last (BGR) channel"
    assert int(converted[0, 0, 0]) == 0


def test_conversion_scales_and_clips_the_range() -> None:
    image = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
    converted = Yolo11Adapter.to_backend_image(image)
    assert list(converted[0, 0]) == [255, 128, 0]


def test_conversion_output_is_contiguous() -> None:
    """A reversed view would be rejected by the backend's tensor conversion."""
    converted = Yolo11Adapter.to_backend_image(np.zeros((3, 3, 3), dtype=np.float32))
    assert converted.flags["C_CONTIGUOUS"]


# -------------------------------------------------------------- label space


@pytest.mark.parametrize(
    ("coco", "expected"),
    [
        ("car", "Car"),
        ("person", "Pedestrian"),
        ("bicycle", "Cyclist"),
        ("motorcycle", "Cyclist"),
        ("truck", None),
        ("traffic light", None),
    ],
)
def test_label_map_matches_the_normalised_classes(coco: str, expected: str | None) -> None:
    assert _adapter().map_label(coco) == expected


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
    samples = list(range(7))
    chunks = list(adapter._chunks(samples))  # type: ignore[arg-type]
    assert [len(chunk) for chunk in chunks] == [3, 3, 1]


# ------------------------------------------------------------ real inference


@pytest.mark.skipif(not HAS_ULTRALYTICS, reason="needs the ultralytics extra")
def test_predict_returns_one_prediction_per_sample() -> None:
    from src.core.types import Sample

    samples = [
        Sample(f"s{index}", np.full((64, 64, 3), 0.4, dtype=np.float32)) for index in range(3)
    ]
    predictions = _adapter(batch_size=2).predict(samples)
    assert [prediction.sample_id for prediction in predictions] == ["s0", "s1", "s2"]
