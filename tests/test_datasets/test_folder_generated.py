from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.datasets import get_dataset
from src.datasets.base import AnonymizationRequiredError


def _write_folder_dataset(root: Path, *, anonymized: bool) -> None:
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    (root / "masks").mkdir()
    image = np.full((16, 20, 3), 0.25, dtype=np.float32)
    np.save(root / "images" / "frame_001.npy", image, allow_pickle=False)
    (root / "labels" / "frame_001.json").write_text(
        json.dumps(
            {
                "boxes": [
                    {
                        "x1": 2,
                        "y1": 3,
                        "x2": 12,
                        "y2": 14,
                        "label": "Car",
                        "score": 1.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    np.save(
        root / "masks" / "frame_001.npy",
        np.ones((16, 20), dtype=np.uint8),
        allow_pickle=False,
    )
    (root / "dataset.json").write_text(
        json.dumps({"anonymized": anonymized}),
        encoding="utf-8",
    )


def test_folder_dataset_loads_canonical_arrays_and_annotations(tmp_path: Path) -> None:
    _write_folder_dataset(tmp_path, anonymized=True)
    dataset = get_dataset("folder_dataset", root=str(tmp_path))
    dataset.require_anonymized()
    sample = dataset.load()[0]
    assert sample.sample_id == "frame_001"
    assert sample.image.dtype == np.float32
    assert sample.boxes[0].label == "Car"
    assert sample.mask is not None


def test_folder_dataset_enforces_anonymization_manifest(tmp_path: Path) -> None:
    _write_folder_dataset(tmp_path, anonymized=False)
    dataset = get_dataset("folder_dataset", root=str(tmp_path))
    with pytest.raises(AnonymizationRequiredError):
        dataset.require_anonymized()


def test_folder_dataset_loads_kitti_labels(tmp_path: Path) -> None:
    (tmp_path / "image_2").mkdir()
    (tmp_path / "label_2").mkdir()
    np.save(
        tmp_path / "image_2" / "000001.npy",
        np.zeros((12, 20, 3), dtype=np.float32),
        allow_pickle=False,
    )
    (tmp_path / "label_2" / "000001.txt").write_text(
        "Car 0 0 0 1 2 10 11 1 1 1 1 1 1 1\n",
        encoding="utf-8",
    )
    (tmp_path / "dataset.json").write_text(
        json.dumps({"anonymized": True}),
        encoding="utf-8",
    )
    sample = get_dataset(
        "folder_dataset",
        root=str(tmp_path),
        input_format="kitti",
    ).load()[0]
    assert sample.boxes[0].label == "Car"
    assert sample.boxes[0].as_tuple() == (1.0, 2.0, 10.0, 11.0)


def test_folder_dataset_applies_limit_before_loading_images(tmp_path: Path) -> None:
    (tmp_path / "image_2").mkdir()
    (tmp_path / "label_2").mkdir()
    np.save(
        tmp_path / "image_2" / "000001.npy",
        np.zeros((12, 20, 3), dtype=np.float32),
        allow_pickle=False,
    )
    (tmp_path / "image_2" / "000002.png").write_bytes(b"not an image")

    dataset = get_dataset(
        "folder_dataset",
        root=str(tmp_path),
        input_format="kitti",
    )

    assert [sample.sample_id for sample in dataset.load(limit=1)] == ["000001"]
