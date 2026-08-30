from __future__ import annotations

import json
from pathlib import Path

from src.datasets.kitti import Kitti


def test_kitti_resolves_relative_manifest_under_dataset_root(tmp_path: Path) -> None:
    (tmp_path / "dataset.json").write_text(json.dumps({"anonymized": True}), encoding="utf-8")
    (tmp_path / "manifest.jsonl").write_text('{"sample_id":"000000"}\n', encoding="utf-8")

    dataset = Kitti(root=str(tmp_path), manifest_path="manifest.jsonl")

    assert dataset.anonymized is True
