"""Tests for YOLO11 ModelTrainer implementation (Person B scope)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.training.contracts import TrainingRunConfig
from src.training.registry import TrainerRegistry
from src.training.yolo_trainer import YoloTrainer


@pytest.fixture
def trainer(tmp_path: Path) -> YoloTrainer:
    return YoloTrainer(checkpoints_dir=tmp_path / "checkpoints")


@pytest.fixture
def valid_config() -> TrainingRunConfig:
    return TrainingRunConfig(
        run_id="run-yolo-test-01",
        trainer_name="yolo11",
        model_version="yolo11s-kitti-robust-v2",
        dataset_version_id="kitti-v1",
        split_manifest_id="kitti-train-split-v1",
        defense_profile_id="profile-robust-mix-01",
        seed=20260807,
        epochs=5,
        batch_size=8,
        learning_rate=0.001,
        max_gpu_hours=2.0,
        max_storage_bytes=500_000_000,
        max_wall_time_seconds=600,
        metadata={"sample_count": 200},
    )


def test_yolo_trainer_metadata(trainer: YoloTrainer) -> None:
    meta = trainer.metadata()
    assert meta.name == "yolo11"
    assert meta.task == "detection2d"
    assert meta.version == "1.0.0"


def test_yolo_trainer_registration(trainer: YoloTrainer) -> None:
    registry = TrainerRegistry()
    registry.register(trainer)
    assert "yolo11" in registry.names()
    assert registry.get("yolo11") is trainer


def test_validate_valid_config(trainer: YoloTrainer, valid_config: TrainingRunConfig) -> None:
    report = trainer.validate_config(valid_config)
    assert report.valid is True
    assert len(report.errors) == 0


def test_validate_invalid_config_catches_errors(trainer: YoloTrainer) -> None:
    invalid_config = TrainingRunConfig(
        run_id="run-invalid",
        trainer_name="unknown_trainer",
        model_version="v1",
        dataset_version_id="d1",
        split_manifest_id="s1",
        defense_profile_id="p1",
        seed=42,
        epochs=1,
        batch_size=1,
        learning_rate=0.001,
    )
    report = trainer.validate_config(invalid_config)
    assert report.valid is False
    assert any("Invalid trainer_name" in err for err in report.errors)



def test_estimate_computes_resources(trainer: YoloTrainer, valid_config: TrainingRunConfig) -> None:
    estimate = trainer.estimate(valid_config)
    assert estimate.gpu_hours > 0.0
    assert estimate.storage_bytes > 0
    assert estimate.wall_time_seconds > 0


def test_prepare_data_detects_leakage(trainer: YoloTrainer) -> None:
    leakage_config = TrainingRunConfig(
        run_id="run-leakage",
        trainer_name="yolo11",
        model_version="v1",
        dataset_version_id="d1",
        split_manifest_id="kitti-locked_test-split-v1",
        defense_profile_id="p1",
        seed=42,
        epochs=1,
        batch_size=1,
        learning_rate=0.001,
    )
    with pytest.raises(ValueError, match="Data leakage detected"):
        trainer.prepare_data(leakage_config)


def test_full_training_loop_lifecycle(trainer: YoloTrainer, valid_config: TrainingRunConfig) -> None:
    logged_epochs: list[tuple[int, dict[str, float]]] = []

    from src.training.base import TrainerCallbacks

    callbacks = TrainerCallbacks(
        on_epoch=lambda ep, metrics: logged_epochs.append((ep, metrics)),
        is_cancelled=lambda: False,
    )

    report = trainer.train(valid_config, callbacks)

    assert report.state == "COMPLETED"
    assert report.run_id == valid_config.run_id
    assert len(report.epoch_metrics) == valid_config.epochs
    assert len(logged_epochs) == valid_config.epochs

    # Checkpoints
    assert report.checkpoint is not None
    assert Path(report.checkpoint.path).is_file()
    assert report.exported_checkpoint is not None
    assert report.exported_checkpoint.load_valid is True
    assert report.registration is not None
    assert report.registration["status"] == "registered"


def test_training_cancellation_handles_state(trainer: YoloTrainer, valid_config: TrainingRunConfig) -> None:
    from src.training.base import TrainerCallbacks

    # Cancel immediately
    callbacks = TrainerCallbacks(
        on_epoch=lambda ep, metrics: None,
        is_cancelled=lambda: True,
    )

    report = trainer.train(valid_config, callbacks)
    assert report.state == "CANCELLED"


def test_acceptance_gate_evaluation() -> None:
    baseline = {"clean_map50_95": 0.685, "robust_score": 62.0}

    # Candidate meets gate: clean drop = -0.007 (<= 0.02), robust score gain = +14 (>= 8.0)
    candidate_pass = {"clean_map50_95": 0.678, "robust_score": 76.0}
    gate_res = YoloTrainer.evaluate_acceptance_gate(baseline, candidate_pass)
    assert gate_res["passed"] is True
    assert gate_res["clean_gate_passed"] is True
    assert gate_res["robust_gate_passed"] is True

    # Candidate fails gate: clean drop = -0.050 (> 0.02)
    candidate_fail_clean = {"clean_map50_95": 0.630, "robust_score": 78.0}
    gate_res_fail = YoloTrainer.evaluate_acceptance_gate(baseline, candidate_fail_clean)
    assert gate_res_fail["passed"] is False
    assert gate_res_fail["clean_gate_passed"] is False

    # Candidate fails gate: robust score gain = +2.0 (< 8.0)
    candidate_fail_robust = {"clean_map50_95": 0.684, "robust_score": 64.0}
    gate_res_fail_robust = YoloTrainer.evaluate_acceptance_gate(baseline, candidate_fail_robust)
    assert gate_res_fail_robust["passed"] is False
    assert gate_res_fail_robust["robust_gate_passed"] is False


def test_build_robust_yolo_dataset(tmp_path: Path) -> None:
    from src.training.yolo_dataset_formatter import (
        build_robust_yolo_dataset,
        create_starter_kitti_dataset,
    )

    clean_dir = tmp_path / "clean_kitti"
    create_starter_kitti_dataset(output_dir=clean_dir, num_samples=10)

    robust_dir = tmp_path / "robust_kitti"
    yaml_path = build_robust_yolo_dataset(
        clean_yolo_dir=clean_dir,
        output_dir=robust_dir,
        mode="r1",
        clean_ratio=0.5,
        seed=42,
    )

    assert yaml_path.is_file()
    train_images = list((robust_dir / "images" / "train").glob("*.jpg"))
    assert len(train_images) == 9  # 10 - 1 val = 9 train
    # Check that augmented images exist
    aug_images = [img for img in train_images if "aug_" in img.name]
    assert len(aug_images) > 0
