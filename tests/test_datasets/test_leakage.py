from __future__ import annotations

from src.datasets.contracts import DatasetVersion, SampleRecord
from src.datasets.leakage import (
    GeneratedLeakageInput,
    GeneratedLineageRecord,
    LeakageValidator,
    TrainingLeakageInput,
)
from src.datasets.splits import SplitBuilder, SplitPolicy


def _record(
    sample_id: str,
    *,
    source_hash: str | None = None,
    ground_truth_hash: str | None = None,
    label: str = "Car",
    provenance: bool = True,
) -> SampleRecord:
    return SampleRecord(
        sample_id=sample_id,
        source_uri=f"fixture://{sample_id}",
        source_hash=source_hash or f"source-{sample_id}",
        ground_truth_hash=ground_truth_hash or f"gt-{sample_id}",
        annotation_type="boxes",
        class_labels=(label,),
        anonymized=True,
        provenance=(
            {
                "logical_source_id": "fixture",
                "loader_version": "fixture-v1",
            }
            if provenance
            else {}
        ),
    )


def _version(version_id: str, records: tuple[SampleRecord, ...]) -> DatasetVersion:
    return DatasetVersion(
        name="fixture",
        logical_source_id="fixture",
        version_id=version_id,
        manifest_hash=f"manifest-{version_id}",
        schema_version="1.0.0",
        loader_version="fixture-v1",
        records=records,
    )


def _codes(report) -> set[str]:
    return {finding.code for finding in report.errors}


def test_valid_source_splits_pass_with_stable_report_hash() -> None:
    version = _version(
        "dataset-1",
        tuple(_record(f"sample-{index}") for index in range(10)),
    )
    manifest = SplitBuilder().build(version, SplitPolicy(strategy="seeded", seed=1))
    validator = LeakageValidator()

    first = validator.validate_splits(version, manifest)
    second = validator.validate_splits(version, manifest)

    assert first.passed
    assert first.report_hash == second.report_hash


def test_split_validation_blocks_overlap_and_duplicate_source_hash() -> None:
    records = (
        _record("train-a", source_hash="duplicate"),
        _record("test-a", source_hash="duplicate"),
        _record("val-a"),
    )
    version = _version("dataset-1", records)
    manifest = SplitBuilder().build(
        version,
        SplitPolicy(strategy="seeded", seed=1),
    )
    tampered = manifest.model_copy(
        update={
            "train_ids": ("train-a", "test-a"),
            "val_ids": ("val-a",),
            "test_ids": ("test-a",),
            "locked_test_ids": ("test-a",),
        }
    )

    report = LeakageValidator().validate_splits(version, tampered)

    assert not report.passed
    assert {"sample_overlap", "duplicate_source_cross_split"} <= _codes(report)
    assert report.duplicate_groups[0].digest == "duplicate"


def test_training_validation_blocks_all_mandatory_source_failures() -> None:
    records = (
        _record("train-ok"),
        _record("locked-test"),
        _record("invalid-class", label="Alien", provenance=False),
    )
    version = _version("dataset-1", records)
    manifest = SplitBuilder().build(
        version,
        SplitPolicy(strategy="official_or_seeded", seed=2),
    )
    locked = manifest.test_ids[0]
    config = TrainingLeakageInput(
        versions=(version,),
        splits=(manifest,),
        selected_sample_ids=(locked, "missing-source", "benchmark-1", "invalid-class"),
        benchmark_artifact_ids=("benchmark-1",),
        class_mapping={"Alien": "Unknown"},
    )

    report = LeakageValidator().validate_training(config)

    assert {
        "locked_test_reuse",
        "missing_source",
        "benchmark_artifact_reuse",
        "invalid_class_mapping",
        "missing_provenance",
    } <= _codes(report)


def test_training_validation_blocks_duplicate_image_gt_pairs_across_versions() -> None:
    first = _version(
        "dataset-1",
        (_record("sample-a", source_hash="same-image", ground_truth_hash="same-gt"),),
    )
    second = _version(
        "dataset-2",
        (_record("sample-b", source_hash="same-image", ground_truth_hash="same-gt"),),
    )

    report = LeakageValidator().validate_training(
        TrainingLeakageInput(versions=(first, second))
    )

    assert "duplicate_image_gt_across_versions" in _codes(report)


def test_generated_validation_requires_lineage_and_training_permission() -> None:
    generated = GeneratedLeakageInput(
        version_id="generated-1",
        records=(
            GeneratedLineageRecord(
                generated_id="generated-a",
                source_sample_id="sample-a",
                source_version_id="",
                recipe_id="",
                transform_log_hash="",
                seed=1,
                artifact_kind="benchmark",
                allowed_uses=("benchmark",),
            ),
        ),
    )

    report = LeakageValidator().validate_generated(generated)

    assert {"missing_provenance", "benchmark_artifact_reuse"} <= _codes(report)
