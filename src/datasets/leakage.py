"""Machine-readable leakage, duplicate, and lineage validation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.core.hashing import stable_digest
from src.core.types import CLASSES
from src.datasets.contracts import DatasetVersion, SplitManifest
from src.datasets.splits import SplitBuilder

Severity = Literal["error", "warning"]
ArtifactKind = Literal["source", "generated", "benchmark"]
AllowedUse = Literal["training", "benchmark", "review"]


class _FrozenContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class LeakageFinding(_FrozenContract):
    code: str
    severity: Severity
    sample_ids: tuple[str, ...] = ()
    detail: str
    context: dict[str, Any] = Field(default_factory=dict)


class DuplicateGroup(_FrozenContract):
    hash_kind: Literal["source", "image_ground_truth", "generated_id"]
    digest: str
    sample_ids: tuple[str, ...]
    split_names: tuple[str, ...] = ()


class LeakageReport(_FrozenContract):
    validator_version: str
    passed: bool
    errors: tuple[LeakageFinding, ...] = ()
    warnings: tuple[LeakageFinding, ...] = ()
    duplicate_groups: tuple[DuplicateGroup, ...] = ()
    report_hash: str


class TrainingLeakageInput(_FrozenContract):
    versions: tuple[DatasetVersion, ...] = ()
    splits: tuple[SplitManifest, ...] = ()
    selected_sample_ids: tuple[str, ...] = ()
    benchmark_artifact_ids: tuple[str, ...] = ()
    class_mapping: dict[str, str] = Field(default_factory=dict)
    duplicate_allowlist: dict[str, str] = Field(default_factory=dict)


class GeneratedLineageRecord(_FrozenContract):
    generated_id: str
    source_sample_id: str
    source_version_id: str
    recipe_id: str
    transform_log_hash: str
    seed: int = Field(ge=0)
    artifact_kind: ArtifactKind = "generated"
    allowed_uses: tuple[AllowedUse, ...] = ("training",)


class GeneratedLeakageInput(_FrozenContract):
    version_id: str
    records: tuple[GeneratedLineageRecord, ...]
    duplicate_allowlist: dict[str, str] = Field(default_factory=dict)


class LeakageValidator:
    version = "1.0.0"

    def validate_splits(
        self,
        version: DatasetVersion,
        splits: SplitManifest,
    ) -> LeakageReport:
        errors: list[LeakageFinding] = []
        warnings: list[LeakageFinding] = []
        duplicate_groups: list[DuplicateGroup] = []
        split_report = SplitBuilder().validate(version, splits)
        for code in split_report.errors:
            errors.append(
                _finding(
                    code,
                    detail=f"split manifest failed validation: {code}",
                )
            )
        membership: dict[str, set[str]] = defaultdict(set)
        for split_name, sample_ids in (
            ("train", splits.train_ids),
            ("val", splits.val_ids),
            ("test", splits.test_ids),
        ):
            for sample_id in sample_ids:
                membership[sample_id].add(split_name)
        records_by_hash: dict[str, list[str]] = defaultdict(list)
        for record in version.records:
            records_by_hash[record.source_hash].append(record.sample_id)
        for digest, sample_ids in sorted(records_by_hash.items()):
            unique_ids = tuple(sorted(set(sample_ids)))
            if len(unique_ids) < 2:
                continue
            split_names = tuple(
                sorted(
                    {
                        split_name
                        for sample_id in unique_ids
                        for split_name in membership.get(sample_id, set())
                    }
                )
            )
            duplicate_groups.append(
                DuplicateGroup(
                    hash_kind="source",
                    digest=digest,
                    sample_ids=unique_ids,
                    split_names=split_names,
                )
            )
            if len(split_names) > 1:
                errors.append(
                    _finding(
                        "duplicate_source_cross_split",
                        sample_ids=unique_ids,
                        detail="identical source content appears in multiple splits",
                        context={"source_hash": digest, "splits": split_names},
                    )
                )
        return self._report(errors, warnings, duplicate_groups)

    def validate_training(self, config: TrainingLeakageInput) -> LeakageReport:
        errors: list[LeakageFinding] = []
        warnings: list[LeakageFinding] = []
        duplicate_groups: list[DuplicateGroup] = []
        versions_by_id = {version.version_id: version for version in config.versions}
        for manifest in config.splits:
            version = versions_by_id.get(manifest.dataset_version_id)
            if version is None:
                errors.append(
                    _finding(
                        "missing_source",
                        detail="split manifest references an unavailable dataset version",
                        context={"dataset_version_id": manifest.dataset_version_id},
                    )
                )
                continue
            split_report = self.validate_splits(version, manifest)
            errors.extend(split_report.errors)
            warnings.extend(split_report.warnings)
            duplicate_groups.extend(split_report.duplicate_groups)

        record_index = {
            record.sample_id: record
            for version in config.versions
            for record in version.records
        }
        locked_ids = {
            sample_id
            for manifest in config.splits
            for sample_id in manifest.locked_test_ids
        }
        benchmark_ids = set(config.benchmark_artifact_ids)
        for sample_id in sorted(set(config.selected_sample_ids)):
            if sample_id in benchmark_ids:
                errors.append(
                    _finding(
                        "benchmark_artifact_reuse",
                        sample_ids=(sample_id,),
                        detail="benchmark-only artifact cannot be used for training",
                    )
                )
            record = record_index.get(sample_id)
            if record is None:
                errors.append(
                    _finding(
                        "missing_source",
                        sample_ids=(sample_id,),
                        detail="selected training sample has no source record",
                    )
                )
                continue
            if sample_id in locked_ids:
                errors.append(
                    _finding(
                        "locked_test_reuse",
                        sample_ids=(sample_id,),
                        detail="locked test membership cannot enter training",
                    )
                )
            if not {
                "logical_source_id",
                "loader_version",
            }.issubset(record.provenance):
                errors.append(
                    _finding(
                        "missing_provenance",
                        sample_ids=(sample_id,),
                        detail="source record lacks required logical source or loader provenance",
                    )
                )
            invalid_labels = tuple(
                sorted(
                    label
                    for label in record.class_labels
                    if label not in CLASSES
                    and config.class_mapping.get(label) not in CLASSES
                )
            )
            if invalid_labels:
                errors.append(
                    _finding(
                        "invalid_class_mapping",
                        sample_ids=(sample_id,),
                        detail="source labels do not map to the canonical class space",
                        context={"labels": invalid_labels},
                    )
                )

        pairs: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
        for version in config.versions:
            for record in version.records:
                pairs[(record.source_hash, record.ground_truth_hash)].append(
                    (version.version_id, record.sample_id)
                )
        for (source_hash, ground_truth_hash), members in sorted(pairs.items()):
            version_ids = {version_id for version_id, _ in members}
            if len(version_ids) < 2:
                continue
            digest = stable_digest(
                {
                    "source_hash": source_hash,
                    "ground_truth_hash": ground_truth_hash,
                },
                length=64,
            )
            sample_ids = tuple(sorted(sample_id for _, sample_id in members))
            duplicate_groups.append(
                DuplicateGroup(
                    hash_kind="image_ground_truth",
                    digest=digest,
                    sample_ids=sample_ids,
                )
            )
            explanation = config.duplicate_allowlist.get(digest, "").strip()
            finding = _finding(
                "duplicate_image_gt_across_versions",
                severity="warning" if explanation else "error",
                sample_ids=sample_ids,
                detail=(
                    explanation
                    if explanation
                    else "identical image and ground truth appear across dataset versions"
                ),
                context={"dataset_version_ids": tuple(sorted(version_ids))},
            )
            (warnings if explanation else errors).append(finding)
        return self._report(errors, warnings, duplicate_groups)

    def validate_generated(
        self,
        generated: GeneratedLeakageInput,
    ) -> LeakageReport:
        errors: list[LeakageFinding] = []
        warnings: list[LeakageFinding] = []
        duplicate_groups: list[DuplicateGroup] = []
        by_id: dict[str, list[GeneratedLineageRecord]] = defaultdict(list)
        for record in generated.records:
            by_id[record.generated_id].append(record)
            missing = tuple(
                field
                for field, value in (
                    ("source_sample_id", record.source_sample_id),
                    ("source_version_id", record.source_version_id),
                    ("recipe_id", record.recipe_id),
                    ("transform_log_hash", record.transform_log_hash),
                )
                if not value.strip()
            )
            if missing:
                errors.append(
                    _finding(
                        "missing_provenance",
                        sample_ids=(record.generated_id,),
                        detail="generated artifact lacks required lineage",
                        context={"fields": missing},
                    )
                )
            if record.artifact_kind == "benchmark" or "training" not in record.allowed_uses:
                errors.append(
                    _finding(
                        "benchmark_artifact_reuse",
                        sample_ids=(record.generated_id,),
                        detail="generated artifact is not permitted for training",
                    )
                )
        for generated_id, records in sorted(by_id.items()):
            if len(records) < 2:
                continue
            digest = stable_digest(
                [record.model_dump(mode="json") for record in records],
                length=64,
            )
            duplicate_groups.append(
                DuplicateGroup(
                    hash_kind="generated_id",
                    digest=digest,
                    sample_ids=(generated_id,),
                )
            )
            explanation = generated.duplicate_allowlist.get(generated_id, "").strip()
            finding = _finding(
                "duplicate_generated_id",
                severity="warning" if explanation else "error",
                sample_ids=(generated_id,),
                detail=explanation or "generated ID is not unique",
            )
            (warnings if explanation else errors).append(finding)
        return self._report(errors, warnings, duplicate_groups)

    def _report(
        self,
        errors: list[LeakageFinding],
        warnings: list[LeakageFinding],
        duplicate_groups: list[DuplicateGroup],
    ) -> LeakageReport:
        ordered_errors = tuple(sorted(errors, key=_finding_key))
        ordered_warnings = tuple(sorted(warnings, key=_finding_key))
        ordered_duplicates = tuple(
            sorted(
                set(duplicate_groups),
                key=lambda group: (group.hash_kind, group.digest, group.sample_ids),
            )
        )
        payload = {
            "validator_version": self.version,
            "errors": [finding.model_dump(mode="json") for finding in ordered_errors],
            "warnings": [finding.model_dump(mode="json") for finding in ordered_warnings],
            "duplicate_groups": [
                group.model_dump(mode="json") for group in ordered_duplicates
            ],
        }
        return LeakageReport(
            validator_version=self.version,
            passed=not ordered_errors,
            errors=ordered_errors,
            warnings=ordered_warnings,
            duplicate_groups=ordered_duplicates,
            report_hash=stable_digest(payload, length=64),
        )


def _finding(
    code: str,
    *,
    detail: str,
    severity: Severity = "error",
    sample_ids: tuple[str, ...] = (),
    context: dict[str, Any] | None = None,
) -> LeakageFinding:
    return LeakageFinding(
        code=code,
        severity=severity,
        sample_ids=tuple(sorted(sample_ids)),
        detail=detail,
        context=context or {},
    )


def _finding_key(finding: LeakageFinding) -> tuple[Any, ...]:
    return (
        finding.code,
        finding.sample_ids,
        stable_digest(finding.context, length=64),
        finding.detail,
    )
