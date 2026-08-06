"""KITTI loader tests with a strict real-anonymization gate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.datasets import get_dataset
from src.datasets.base import AnonymizationRequiredError
from src.datasets.kitti import Kitti

IMAGE_HEIGHT, IMAGE_WIDTH = 120, 200


def _write_png(path: Path, seed: int) -> None:
    from PIL import Image

    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path)


@pytest.fixture
def kitti_root(tmp_path: Path) -> Path:
    images = tmp_path / "image_2"
    labels = tmp_path / "label_2"
    images.mkdir()
    labels.mkdir()
    rows = {
        "000000": [
            "Car 0.00 0 -1.57 20.00 30.00 90.00 100.00 1.5 1.6 4.0 1 2 3 -1.5",
            "Pedestrian 0.00 0 -1.57 120.00 20.00 140.00 90.00 1.8 0.6 0.8 1 2 3 -1.5",
            "DontCare 0.00 0 -10 0.00 0.00 10.00 10.00 -1 -1 -1 -1000 -1000 -1000 -10",
        ],
        "000001": [
            "Car 0.00 0 -1.57 10.00 10.00 60.00 20.00 1.5 1.6 4.0 1 2 3 -1.5",
            "Cyclist 0.40 2 -1.57 30.00 30.00 70.00 110.00 1.7 0.6 1.8 1 2 3 -1.5",
        ],
        "000002": [
            "Car 0.00 0 -1.57 5.00 5.00 95.00 105.00 1.5 1.6 4.0 1 2 3 -1.5"
        ],
    }
    for index, (image_id, label_rows) in enumerate(rows.items()):
        (labels / f"{image_id}.txt").write_text("\n".join(label_rows) + "\n")
        _write_png(images / f"{image_id}.png", index)
    (tmp_path / "ImageSets").mkdir()
    (tmp_path / "ImageSets" / "val.txt").write_text("000000\n000001\n000002\n")
    (tmp_path / "ImageSets" / "train.txt").write_text("000000\n")
    (tmp_path / "dataset.json").write_text(json.dumps({"anonymized": False}))
    return tmp_path


def _dataset(root: Path, **params: object) -> Kitti:
    return get_dataset("kitti", root=str(root), **params)  # type: ignore[return-value]


def _mark_anonymized(root: Path) -> None:
    (root / "dataset.json").write_text(json.dumps({"anonymized": True}))
    (root / "manifest.jsonl").write_text("{\"format\": \"kitti\"}\n")


def test_catalog_does_not_claim_raw_kitti_is_anonymized() -> None:
    assert Kitti.describe()["anonymized"] is False


def test_gate_rejects_raw_kitti(kitti_root: Path) -> None:
    dataset = _dataset(kitti_root)
    assert dataset.anonymized is False
    with pytest.raises(AnonymizationRequiredError):
        dataset.require_anonymized()


def test_gate_accepts_only_descriptor_and_manifest(kitti_root: Path) -> None:
    _mark_anonymized(kitti_root)
    dataset = _dataset(kitti_root)
    dataset.require_anonymized()
    assert dataset.info().anonymized is True


def test_labels_map_and_difficulty_filter(kitti_root: Path) -> None:
    dataset = _dataset(kitti_root, difficulty="moderate")
    boxes, dropped = dataset._read_labels(
        kitti_root / "label_2" / "000000.txt", (IMAGE_HEIGHT, IMAGE_WIDTH)
    )
    assert [box.label for box in boxes] == ["Car", "Pedestrian"]
    assert dropped["DontCare"] == 1
    boxes, _ = dataset._read_labels(
        kitti_root / "label_2" / "000001.txt", (IMAGE_HEIGHT, IMAGE_WIDTH)
    )
    assert boxes == ()
    boxes, _ = _dataset(kitti_root, difficulty="hard")._read_labels(
        kitti_root / "label_2" / "000001.txt", (IMAGE_HEIGHT, IMAGE_WIDTH)
    )
    assert [box.label for box in boxes] == ["Cyclist"]


def test_merge_van_truck_and_clip_boxes(kitti_root: Path) -> None:
    (kitti_root / "label_2" / "000000.txt").write_text(
        "Van 0 0 0 -10 -10 250 140 1 1 1 1 1 1 1\n"
    )
    boxes, dropped = _dataset(kitti_root, merge_van_truck=True, difficulty="all")._read_labels(
        kitti_root / "label_2" / "000000.txt", (IMAGE_HEIGHT, IMAGE_WIDTH)
    )
    assert boxes[0].label == "Car"
    assert boxes[0].x1 == 0.0 and boxes[0].x2 == IMAGE_WIDTH
    assert "Van" not in dropped


def test_anonymized_load_preserves_ids_and_pixels(kitti_root: Path) -> None:
    _mark_anonymized(kitti_root)
    samples = _dataset(kitti_root, difficulty="all").load()
    assert len(samples) == 3
    assert samples[0].meta["image_id"] == "000000"
    assert samples[0].image.dtype == np.float32
    assert samples[0].image.shape == (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    assert samples[0].anonymized is True
    assert samples[0].meta["source_uri"] == "kitti://000000"
    assert samples[0].meta["native_labels"] == ("Car", "Pedestrian", "DontCare")
    assert samples[0].meta["loader_version"] == Kitti.loader_version
    assert samples[0].meta["split"] == "val"
    assert len(samples[0].meta["anonymization_manifest_hash"]) == 64


def test_split_and_explicit_sample_ids_are_honoured(kitti_root: Path) -> None:
    _mark_anonymized(kitti_root)
    assert [sample.meta["image_id"] for sample in _dataset(kitti_root, split="train").load()] == [
        "000000"
    ]
    assert [
        sample.meta["image_id"]
        for sample in _dataset(kitti_root, sample_ids=("000002",)).load()
    ] == ["000002"]


def test_missing_root_is_explicit(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="image_2/label_2"):
        _dataset(tmp_path / "missing").load()
