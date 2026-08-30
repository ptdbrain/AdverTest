from __future__ import annotations

import json
from pathlib import Path

from scripts.curate_local_datasets import build_plan, validate_plan


def _write_kitti_pair(root: Path, sample_id: str) -> None:
    for directory, suffix, content in (
        (root / "kitti" / "Kitti" / "raw" / "training" / "image_2", ".png", b"image"),
        (root / "kitti" / "Kitti" / "raw" / "training" / "label_2", ".txt", b"Car 0 0 0 0 0 1 1\n"),
    ):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{sample_id}{suffix}").write_bytes(content)


def _write_semantic_pair(root: Path, sample_id: str) -> None:
    for directory, suffix in (
        (root / "datasets" / "kitti_semantics" / "training" / "image_2", ".png"),
        (root / "datasets" / "kitti_semantics" / "training" / "semantic", ".png"),
    ):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{sample_id}{suffix}").write_bytes(b"mask" if directory.name == "semantic" else b"image")


def _write_nuscenes_metadata(root: Path, *, scenes: int, samples_per_scene: int) -> None:
    version_dir = root / "nuscenes" / "v1.0-mini"
    version_dir.mkdir(parents=True, exist_ok=True)
    scene_rows = []
    sample_rows = []
    sample_data_rows = []
    for scene_index in range(scenes):
        scene_token = f"scene-{scene_index}"
        scene_rows.append({"token": scene_token, "name": f"scene-{scene_index}"})
        for sample_index in range(samples_per_scene):
            token = f"sample-{scene_index}-{sample_index}"
            sample_rows.append({"token": token, "scene_token": scene_token, "data": {"LIDAR_TOP": f"lidar-{token}"}, "anns": []})
            sample_data_rows.append({"token": f"lidar-{token}", "sample_token": token, "filename": f"samples/LIDAR_TOP/{token}.bin"})
            asset = root / "nuscenes" / "samples" / "LIDAR_TOP" / f"{token}.bin"
            asset.parent.mkdir(parents=True, exist_ok=True)
            asset.write_bytes(b"lidar")
    for name, rows in (("scene", scene_rows), ("sample", sample_rows), ("sample_data", sample_data_rows)):
        (version_dir / f"{name}.json").write_text(json.dumps(rows), encoding="utf-8")


def test_plan_keeps_exactly_100_paired_kitti_and_semantic_ids(tmp_path: Path) -> None:
    for value in range(120):
        sample_id = f"{value:06d}"
        _write_kitti_pair(tmp_path, sample_id)
        _write_semantic_pair(tmp_path, sample_id)

    plan = build_plan(tmp_path, count=100)

    assert len(plan.datasets["kitti2d"].retained_ids) == 100
    assert len(plan.datasets["kitti_semantic"].retained_ids) == 100
    assert plan.datasets["kitti2d"].missing_pairs == ()
    assert plan.datasets["kitti_semantic"].missing_pairs == ()
    assert all("sha256" not in asset for asset in plan.datasets["kitti2d"].removal_assets)


def test_nuscenes_plan_keeps_ten_keyframes_per_scene_without_dangling_tokens(tmp_path: Path) -> None:
    _write_nuscenes_metadata(tmp_path, scenes=10, samples_per_scene=12)

    plan = build_plan(tmp_path, count=100)
    validation = validate_plan(plan)

    assert len(plan.datasets["nuscenes3d"].retained_ids) == 100
    assert validation.dangling_tokens == ()
    assert plan.datasets["nuscenes3d"].limitations == ("num_sweeps=1",)


def test_validation_reports_missing_pair_without_mutating_sources(tmp_path: Path) -> None:
    _write_kitti_pair(tmp_path, "000001")
    _write_semantic_pair(tmp_path, "000001")
    _write_nuscenes_metadata(tmp_path, scenes=10, samples_per_scene=10)
    (tmp_path / "kitti" / "Kitti" / "raw" / "training" / "label_2" / "000001.txt").unlink()

    plan = build_plan(tmp_path, count=100)
    validation = validate_plan(plan)

    assert "000001" in plan.datasets["kitti2d"].missing_pairs
    assert "kitti2d:000001" in validation.missing_pairs
    assert not (tmp_path / "curated").exists()


def test_semantic_plan_accepts_the_documented_flat_source_layout(tmp_path: Path) -> None:
    for directory in (tmp_path / "datasets" / "kitti_semantics" / "image_2", tmp_path / "datasets" / "kitti_semantics" / "semantic"):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "000001.png").write_bytes(b"source")
    _write_kitti_pair(tmp_path, "000001")
    _write_nuscenes_metadata(tmp_path, scenes=10, samples_per_scene=10)

    plan = build_plan(tmp_path, count=1)

    assert plan.datasets["kitti_semantic"].retained_ids == ("000001",)
