from __future__ import annotations

import json

import numpy as np
from PIL import Image

from src.datasets import get_dataset


def _image(path, array) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def test_cityscapes_loader_creates_reviewed_instances_and_fixed_prompts(tmp_path) -> None:
    root = tmp_path / "city"
    root.mkdir()
    (root / "approved.json").write_text("{}")
    _image(root / "leftImg8bit/train/a/a_000_leftImg8bit.png", np.zeros((8, 9, 3), dtype=np.uint8))
    instance = np.zeros((8, 9), dtype=np.uint16)
    instance[2:6, 3:7] = 26001
    _image(root / "gtFine/train/a/a_000_gtFine_instanceIds.png", instance)
    sample = get_dataset("cityscapes_segmentation", root=str(root), anonymization_manifest="approved.json").load()[0]
    assert sample.meta["mask_reviewed"] is True
    assert sample.meta["instance_labels"] == {1: "Car"}
    assert sample.meta["sam_prompts"][0]["coordinates"] == [3.0, 2.0, 7.0, 6.0]


def test_bdd_loader_is_explicitly_semantic_and_nonpaired(tmp_path) -> None:
    root = tmp_path / "bdd"
    root.mkdir()
    (root / "approved.json").write_text(json.dumps({"anonymized": True}))
    _image(root / "10k/val/frame.jpg", np.zeros((6, 7, 3), dtype=np.uint8))
    semantic = np.zeros((6, 7), dtype=np.uint8)
    semantic[1:4, 2:5] = 13
    _image(root / "labels/val/frame_val_id.png", semantic)
    sample = get_dataset("bdd100k_semantic", root=str(root), split="val", anonymization_manifest="approved.json").load()[0]
    assert sample.mask is None
    assert sample.meta["comparison_scope"] == "external_semantic_nonpaired"
    assert sample.meta["semantic_mask"].sum() == 9
