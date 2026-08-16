from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.core.hashing import array_digest, stable_digest
from src.datasets.contracts import (
    DatasetVersion,
    GeneratedStepRecord,
    GeneratedVariantRecord,
    SampleRecord,
)
from src.datasets.splits import SplitBuilder, SplitPolicy
from src.training.contracts import DefenseProfile
from src.training.dataset_builder import (
    GeneratedTrainingSource,
    TrainingDatasetBuilder,
    TrainingDatasetConfig,
)


def _base_version() -> DatasetVersion:
    records = tuple(
        SampleRecord(
            sample_id=f"sample-{index}",
            source_uri=f"fixture://sample-{index}",
            source_hash=f"source-{index}",
            ground_truth_hash=f"gt-{index}",
            annotation_type="boxes",
            class_labels=("Car",),
            anonymized=True,
            provenance={"logical_source_id": "fixture", "loader_version": "v1"},
        )
        for index in range(10)
    )
    return DatasetVersion(
        name="fixture",
        logical_source_id="fixture",
        version_id="dataset-1",
        manifest_hash="manifest-1",
        schema_version="1.0.0",
        loader_version="v1",
        records=records,
    )


def _generated(root: Path, source_id: str = "sample-0") -> GeneratedTrainingSource:
    root.mkdir(parents=True)
    source_index = source_id.removeprefix("sample-")
    image = np.full((4, 5, 3), 0.5, dtype=np.float32)
    image_path = root / "image.npy"
    label_path = root / "label.json"
    np.save(image_path, image, allow_pickle=False)
    label = {"boxes": [{"x1": 0, "y1": 0, "x2": 2, "y2": 3, "label": "Car", "score": 1.0}], "boxes3d": []}
    label_path.write_text(json.dumps(label), encoding="utf-8")
    step = GeneratedStepRecord(
        position=0,
        attack_name="gaussian_noise",
        implementation_version="1.0.0",
        severity=2,
        requested_seed=1,
        derived_seed=2,
        resolved_parameters={"sigma": 0.1},
        input_hash="input",
        output_hash=array_digest(image),
        intermediate_path="image.npy",
        cost=1.0,
        status="completed",
    )
    record = GeneratedVariantRecord(
        variant_id="variant-1",
        source_sample_id=source_id,
        source_dataset_version_id="dataset-1",
        source_hash=f"source-{source_index}",
        source_sample_hash="sample-hash",
        ground_truth_hash=f"gt-{source_index}",
        recipe_hash="recipe-1",
        catalog_version="1.0.0",
        ordered_steps=(step,),
        intermediate_paths=("image.npy",),
        intermediate_hashes=(array_digest(image),),
        output_hash=array_digest(image, length=32),
        image_path="image.npy",
        label_path="label.json",
        label_hash=stable_digest(label, length=32),
        intended_use="training",
        validation_status="passed",
        status="complete",
        anonymized=True,
    )
    return GeneratedTrainingSource(root=str(root), records=(record,))


def _config(tmp_path: Path) -> TrainingDatasetConfig:
    version = _base_version()
    splits = SplitBuilder().build(version, SplitPolicy(strategy="seeded", seed=1))
    train_source = splits.train_ids[0]
    generated = _generated(tmp_path / "generated", train_source)
    return TrainingDatasetConfig(
        base_version=version,
        split_manifest=splits,
        defense_profile=DefenseProfile(
            profile_id="balanced",
            recipe_ids=("recipe-1",),
            clean_replay_ratio=0.5,
            generated_ratio=0.5,
        ),
        generated_sources=(generated,),
        output_dir=str(tmp_path / "output"),
        storage_budget_bytes=1_000_000,
    )


def test_estimate_is_dry_run_and_build_is_deterministic(tmp_path: Path) -> None:
    config = _config(tmp_path)
    builder = TrainingDatasetBuilder()

    estimate = builder.estimate(config)
    assert estimate.clean_count > 0
    assert estimate.generated_count == 1
    assert estimate.estimated_bytes > 0
    assert not Path(config.output_dir).exists()

    first = builder.build(config)
    second = builder.build(config)
    assert first.manifest_hash == second.manifest_hash
    assert first.entries == second.entries
    assert Path(config.output_dir, "training-manifest.json").is_file()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("locked", "locked"),
        ("benchmark", "benchmark"),
        ("missing_transform", "transform"),
        ("duplicate", "variant"),
        ("missing_file", "missing"),
        ("invalid_gt", "ground-truth"),
        ("storage", "storage"),
    ],
)
def test_builder_rejects_mandatory_unsafe_inputs(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    config = _config(tmp_path)
    source = config.generated_sources[0]
    record = source.records[0]
    if mutation == "locked":
        record = record.model_copy(
            update={"source_sample_id": config.split_manifest.test_ids[0]}
        )
    elif mutation == "benchmark":
        record = record.model_copy(update={"intended_use": "benchmark"})
    elif mutation == "missing_transform":
        record = record.model_copy(update={"ordered_steps": ()})
    elif mutation == "duplicate":
        source = source.model_copy(update={"records": (record, record)})
    elif mutation == "missing_file":
        record = record.model_copy(update={"image_path": "missing.npy"})
    elif mutation == "invalid_gt":
        record = record.model_copy(update={"ground_truth_hash": "wrong"})
    if mutation not in {"duplicate"}:
        source = source.model_copy(update={"records": (record,)})
    updates = {"generated_sources": (source,)}
    if mutation == "storage":
        updates["storage_budget_bytes"] = 1
    config = config.model_copy(update=updates)

    with pytest.raises(ValueError, match=message):
        TrainingDatasetBuilder().estimate(config)
