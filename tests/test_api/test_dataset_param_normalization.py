from __future__ import annotations

from src.api.routes import normalize_dataset_params
from src.pipeline.runner import RunConfig


def test_normalize_dataset_params_removes_stale_bdd_fields() -> None:
    config = RunConfig(
        dataset="bdd100k_detection",
        dataset_params={
            "root": "/tmp/bdd",
            "split": "val",
            "merge_van_truck": True,
            "difficulty": "all",
        },
    )

    normalized = normalize_dataset_params(config)

    assert normalized.dataset_params == {"root": "/tmp/bdd", "split": "val"}


def test_normalize_dataset_params_keeps_kitti_manifest_path() -> None:
    config = RunConfig(
        dataset="kitti",
        dataset_params={"root": "/tmp/kitti", "manifest_path": "manifest.jsonl"},
    )

    assert normalize_dataset_params(config).dataset_params == config.dataset_params
