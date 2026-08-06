"""Deterministic official, seeded, and class-stratified split manifests."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.core.hashing import stable_digest
from src.datasets.contracts import (
    DatasetVersion,
    SampleRecord,
    SplitManifest,
    SplitValidationReport,
)

SplitStrategy = Literal["official", "seeded", "class_stratified", "official_or_seeded"]


class SplitPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    strategy: SplitStrategy = "official_or_seeded"
    seed: int = Field(default=195, ge=0)
    train_ratio: float = Field(default=0.70, ge=0.0, le=1.0)
    val_ratio: float = Field(default=0.15, ge=0.0, le=1.0)
    test_ratio: float = Field(default=0.15, ge=0.0, le=1.0)
    lock_test: bool = True

    @model_validator(mode="after")
    def validate_ratios(self) -> Self:
        if not math.isclose(
            self.train_ratio + self.val_ratio + self.test_ratio,
            1.0,
            abs_tol=1e-9,
        ):
            raise ValueError("split ratios must sum to 1")
        return self


class SplitBuilder:
    def build(self, version: DatasetVersion, policy: SplitPolicy) -> SplitManifest:
        has_official = bool(version.records) and all(
            record.official_split is not None for record in version.records
        )
        if policy.strategy == "official" and not has_official:
            raise ValueError("official split requested but membership is incomplete")
        if policy.strategy == "official" or (
            policy.strategy == "official_or_seeded" and has_official
        ):
            assignments = {
                name: tuple(
                    record.sample_id
                    for record in version.records
                    if record.official_split == name
                )
                for name in ("train", "val", "test")
            }
        elif policy.strategy == "class_stratified":
            assignments = _class_stratified(version.records, policy)
        else:
            ordered = sorted(
                (record.sample_id for record in version.records),
                key=lambda sample_id: _seeded_key(sample_id, policy.seed),
            )
            assignments = _partition(tuple(ordered), policy)
        locked = assignments["test"] if policy.lock_test else ()
        payload: dict[str, Any] = {
            "dataset_version_id": version.version_id,
            "train_ids": assignments["train"],
            "val_ids": assignments["val"],
            "test_ids": assignments["test"],
            "locked_test_ids": locked,
            "policy": policy.model_dump(mode="json"),
        }
        return SplitManifest(
            **payload,
            manifest_hash=stable_digest(payload, length=64),
        )

    def validate(
        self,
        version: DatasetVersion,
        manifest: SplitManifest,
    ) -> SplitValidationReport:
        errors: list[str] = []
        warnings: list[str] = []
        expected = {record.sample_id for record in version.records}
        train = set(manifest.train_ids)
        val = set(manifest.val_ids)
        test = set(manifest.test_ids)
        if train & val or train & test or val & test:
            errors.append("sample_overlap")
        assigned = train | val | test
        if assigned != expected:
            errors.append("sample_membership_mismatch")
        if set(manifest.locked_test_ids) != test and manifest.policy.get("lock_test", True):
            errors.append("locked_test_membership_changed")
        if manifest.dataset_version_id != version.version_id:
            errors.append("dataset_version_mismatch")
        payload = {
            "dataset_version_id": manifest.dataset_version_id,
            "train_ids": manifest.train_ids,
            "val_ids": manifest.val_ids,
            "test_ids": manifest.test_ids,
            "locked_test_ids": manifest.locked_test_ids,
            "policy": manifest.policy,
        }
        if stable_digest(payload, length=64) != manifest.manifest_hash:
            errors.append("manifest_hash_mismatch")
        ordered_errors = tuple(sorted(set(errors)))
        ordered_warnings = tuple(sorted(set(warnings)))
        report_payload = {
            "validator_version": "1.0.0",
            "errors": ordered_errors,
            "warnings": ordered_warnings,
        }
        return SplitValidationReport(
            passed=not ordered_errors,
            errors=ordered_errors,
            warnings=ordered_warnings,
            report_hash=stable_digest(report_payload, length=64),
        )


def _seeded_key(sample_id: str, seed: int) -> str:
    return stable_digest({"sample_id": sample_id, "seed": seed}, length=64)


def _counts(total: int, policy: SplitPolicy) -> tuple[int, int, int]:
    train = int(total * policy.train_ratio + 0.5)
    val = int(total * policy.val_ratio + 0.5)
    if train + val > total:
        val = max(0, total - train)
    test = total - train - val
    return train, val, test


def _partition(
    ordered: tuple[str, ...],
    policy: SplitPolicy,
) -> dict[str, tuple[str, ...]]:
    train_count, val_count, _ = _counts(len(ordered), policy)
    return {
        "train": tuple(sorted(ordered[:train_count])),
        "val": tuple(sorted(ordered[train_count : train_count + val_count])),
        "test": tuple(sorted(ordered[train_count + val_count :])),
    }


def _class_stratified(
    records: tuple[SampleRecord, ...],
    policy: SplitPolicy,
) -> dict[str, tuple[str, ...]]:
    groups: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for record in records:
        groups[record.class_labels].append(record.sample_id)
    merged: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for signature in sorted(groups):
        ordered = tuple(
            sorted(
                groups[signature],
                key=lambda sample_id: _seeded_key(sample_id, policy.seed),
            )
        )
        partitioned = _partition(ordered, policy)
        for split_name in merged:
            merged[split_name].extend(partitioned[split_name])
    return {name: tuple(sorted(sample_ids)) for name, sample_ids in merged.items()}
