from __future__ import annotations

from pathlib import Path

from src.models.versions import scan_yolo_training_runs


def _write_run(root: Path, role: str, *, best_weights: bytes | None = b"weights") -> Path:
    run = root / role / "run-yolo11s-example" / "ultralytics_run"
    run.mkdir(parents=True)
    (run / "args.yaml").write_text(
        "model: yolo11s.pt\n"
        "data: /data/yolo_kitti/kitti.yaml\n"
        "seed: 20260807\n"
        "imgsz: 640\n",
        encoding="utf-8",
    )
    (run / "results.csv").write_text(
        "epoch,metrics/mAP50-95(B),metrics/mAP50(B)\n"
        "0,0.20,0.30\n"
        "29,0.61,0.82\n",
        encoding="utf-8",
    )
    if best_weights is not None:
        weights = run / "weights"
        weights.mkdir()
        (weights / "best.pt").write_bytes(best_weights)
    return run


def test_scan_registers_yolo_baseline_and_robust_lineage(tmp_path: Path) -> None:
    _write_run(tmp_path, "yolo_b0", best_weights=b"b0")
    _write_run(tmp_path, "yolo_r1", best_weights=b"r1")

    versions = scan_yolo_training_runs(tmp_path)

    assert [version.id for version in versions] == [
        "yolo11s-kitti-clean-b0",
        "yolo11s-kitti-robust-r1",
    ]
    assert versions[0].runnable is True
    assert versions[1].parent_id == versions[0].id
    assert versions[1].training_metadata["result"]["metrics/mAP50-95(B)"] == 0.61


def test_scan_registers_missing_checkpoint_as_visible_but_blocked(tmp_path: Path) -> None:
    _write_run(tmp_path, "yolo_r2_fog", best_weights=None)

    version = scan_yolo_training_runs(tmp_path)[0]

    assert version.id == "yolo11s-kitti-repaired-r2-fog"
    assert version.runnable is False
    assert version.checkpoint_hash is None
    assert version.blocked_reason == "CHECKPOINT_MISSING"
