from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.attacks import get_attack
from src.pipeline.generator import SurrogateConfig
from src.training import PatchTrainer, PatchTrainingConfig


def test_patch_trainer_writes_reloadable_artifact(tmp_path: Path) -> None:
    artifact = PatchTrainer().train(
        PatchTrainingConfig(
            dataset_name="synthetic_shapes",
            dataset_params={"n_samples": 2, "seed": 7},
            surrogate=SurrogateConfig(name="blob_detector"),
            target_label=None,
            patch_size=12,
            iterations=2,
            train_limit=2,
            output_dir=str(tmp_path),
        )
    )
    patch = np.load(artifact.patch_path, allow_pickle=False)
    assert patch.shape == (12, 12, 3)
    assert patch.dtype == np.float32
    assert artifact.preview_path.is_file()
    assert artifact.manifest_path.is_file()
    assert artifact.artifact_hash
    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    assert manifest["training_source_fingerprint"]
    assert manifest["train_sample_ids"]

    loaded = get_attack(
        "dpatch",
        patch_path=str(artifact.patch_path),
        artifact_hash=artifact.artifact_hash,
    )
    assert loaded.artifact_hash == artifact.artifact_hash


def test_patch_artifact_hash_is_checked_before_use(tmp_path: Path) -> None:
    patch_path = tmp_path / "patch.npy"
    np.save(patch_path, np.ones((8, 8, 3), dtype=np.float32), allow_pickle=False)
    (tmp_path / "patch-manifest.json").write_text(
        json.dumps({"artifact_hash": "tampered"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        get_attack("dpatch", patch_path=str(patch_path))


def test_targeted_dpatch_separates_source_and_target_labels(
    tmp_path: Path,
) -> None:
    config = PatchTrainingConfig(
        dataset_name="synthetic_shapes",
        dataset_params={"n_samples": 2, "seed": 7},
        surrogate=SurrogateConfig(
            name="blob_detector",
            objective="targeted",
        ),
        source_label="Car",
        target_label="Cyclist",
        patch_size=8,
        iterations=1,
        train_limit=2,
        output_dir=str(tmp_path),
    )
    artifact = PatchTrainer().train(config)
    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))

    assert manifest["placement_label"] == "Car"
    assert manifest["objective_target_label"] == "Cyclist"
    assert manifest["trainer_version"] == PatchTrainer.version

    with pytest.raises(ValueError, match="target label mismatch"):
        get_attack(
            "dpatch",
            patch_path=str(artifact.patch_path),
            objective="targeted",
            source_label="Car",
            target_label="Pedestrian",
        )


def test_patch_artifact_algorithm_and_objective_are_checked(
    tmp_path: Path,
) -> None:
    artifact = PatchTrainer().train(
        PatchTrainingConfig(
            dataset_name="synthetic_shapes",
            dataset_params={"n_samples": 1, "seed": 9},
            surrogate=SurrogateConfig(name="blob_detector"),
            patch_size=8,
            iterations=1,
            output_dir=str(tmp_path),
        )
    )

    with pytest.raises(ValueError, match="algorithm mismatch"):
        get_attack("thys_patch", patch_path=str(artifact.patch_path))
    with pytest.raises(ValueError, match="objective mismatch"):
        get_attack(
            "dpatch",
            patch_path=str(artifact.patch_path),
            objective="targeted",
            target_label="Cyclist",
        )
