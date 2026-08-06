"""Immutable dataset-version and split-manifest contracts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

AnnotationType = Literal["none", "boxes", "mask", "boxes_and_mask", "boxes3d"]
SplitName = Literal["train", "val", "test"]


class _FrozenContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class SampleRecord(_FrozenContract):
    sample_id: str
    source_uri: str
    source_hash: str
    ground_truth_hash: str
    annotation_type: AnnotationType
    class_labels: tuple[str, ...] = ()
    anonymized: bool
    official_split: SplitName | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)


class DatasetVersion(_FrozenContract):
    name: str
    logical_source_id: str
    version_id: str
    manifest_hash: str
    schema_version: str
    loader_version: str
    records: tuple[SampleRecord, ...]
    metadata: dict[str, Any] = Field(default_factory=dict)


class SplitManifest(_FrozenContract):
    dataset_version_id: str
    train_ids: tuple[str, ...]
    val_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    locked_test_ids: tuple[str, ...]
    policy: dict[str, Any]
    manifest_hash: str


class SplitValidationReport(_FrozenContract):
    passed: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    report_hash: str
