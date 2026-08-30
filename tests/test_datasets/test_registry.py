from __future__ import annotations

import json

from src.datasets.nuscenes import NuScenesDataset
from src.datasets.registry import DatasetRegistry


def test_kitti3d_manifest_without_calibration_or_lidar_is_not_ready(tmp_path) -> None:
    manifest = tmp_path / "kitti2d.json"
    manifest.write_text(
        json.dumps({"dataset_id": "kitti2d-100", "task_id": "detection3d", "modalities": ["image_2", "label_2"]}),
        encoding="utf-8",
    )

    readiness = DatasetRegistry.load(manifest).readiness("detection3d")

    assert readiness.status == "INSUFFICIENT_FOR_REQUESTED_SUBSET"
    assert set(readiness.missing) == {"calib", "velodyne"}


def test_registry_returns_declared_ids_without_scanning_unselected_source_files(tmp_path) -> None:
    manifest = tmp_path / "kitti.json"
    manifest.write_text(
        json.dumps({"dataset_id": "kitti2d-100", "task_id": "detection2d", "modalities": ["image_2", "label_2"], "sample_ids": ["000001", "000002"]}),
        encoding="utf-8",
    )

    registry = DatasetRegistry.load(manifest)

    assert registry.sample_ids == ("000001", "000002")
    assert registry.readiness("detection2d").status == "READY"


def test_registry_rejects_task_mismatch_and_preserves_declared_limitations(tmp_path) -> None:
    manifest = tmp_path / "nuscenes.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset_id": "nuscenes-mini-local-100",
                "task_id": "detection3d",
                "modalities": ["camera_360", "lidar", "radar", "calibration", "ego_pose", "annotations"],
                "sample_ids": ["sample-1"],
                "limitations": ["num_sweeps=1"],
            }
        ),
        encoding="utf-8",
    )

    registry = DatasetRegistry.load(manifest)

    assert registry.readiness("segmentation").status == "INSUFFICIENT_FOR_REQUESTED_SUBSET"
    assert registry.readiness("detection3d").limitations == ("num_sweeps=1",)


def test_nuscenes_loader_uses_only_manifest_sample_tokens(tmp_path) -> None:
    manifest = tmp_path / "nuscenes.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset_id": "nuscenes-mini-local-100",
                "task_id": "detection3d",
                "modalities": ["camera_360", "lidar", "radar", "calibration", "ego_pose", "annotations"],
                "sample_ids": ["sample-a", "sample-b"],
                "limitations": ["num_sweeps=1"],
            }
        ),
        encoding="utf-8",
    )

    assert DatasetRegistry.load(manifest).readiness("detection3d").status == "READY"
    dataset = NuScenesDataset(dataroot=str(tmp_path), curation_manifest_path=str(manifest))

    assert dataset.curated_sample_tokens == {"sample-a", "sample-b"}
