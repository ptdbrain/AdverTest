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


class GeneratedStepRecord(_FrozenContract):
    position: int = Field(ge=0)
    attack_name: str
    implementation_version: str
    severity: int = Field(ge=0, le=5)
    requested_seed: int = Field(ge=0)
    derived_seed: int = Field(ge=0)
    resolved_parameters: dict[str, Any] = Field(default_factory=dict)
    input_hash: str
    output_hash: str
    intermediate_path: str
    cost: float = Field(ge=0.0)
    transform_log: dict[str, Any] | None = None
    status: Literal["completed", "not_selected_by_probability"]


class GeneratedVariantRecord(_FrozenContract):
    schema_version: Literal["3.0.0"] = "3.0.0"
    variant_id: str
    source_sample_id: str
    source_dataset_version_id: str
    source_hash: str
    source_sample_hash: str
    ground_truth_hash: str
    recipe_hash: str
    catalog_version: str
    ordered_steps: tuple[GeneratedStepRecord, ...]
    intermediate_paths: tuple[str, ...]
    intermediate_hashes: tuple[str, ...]
    output_hash: str
    image_path: str
    label_path: str
    label_hash: str
    annotation_format: str = "advertest-annotations-v2"
    mask_path: str | None = None
    mask_hash: str | None = None
    intended_use: Literal["training", "benchmark", "review"]
    validation_status: Literal["passed", "failed"]
    status: Literal["incomplete", "failed", "complete"]
    anonymized: bool
    transform_logs: tuple[dict[str, Any], ...] = ()
    camera_paths: dict[str, str] = Field(default_factory=dict)
    camera_payloads: tuple[dict[str, Any], ...] = ()
    lidar_path: str | None = None
    lidar_hash: str | None = None
    lidar_fields: tuple[str, ...] | None = None
    lidar_sensor_model: str | None = None
    preview_path: str | None = None


class GeneratedDatasetVersion(_FrozenContract):
    format: Literal["advertest-generated-v3"] = "advertest-generated-v3"
    schema_version: Literal["3.0.0"] = "3.0.0"
    generation_id: str
    version_id: str
    source_dataset_version_id: str
    source_manifest_hash: str
    recipe_hash: str
    catalog_version: str
    intended_use: Literal["training", "benchmark", "review"]
    status: Literal["in_progress", "incomplete", "failed", "complete"]
    anonymized: bool
    n_source_samples: int = Field(ge=0)
    n_variants: int = Field(ge=0)
    manifest_hash: str | None = None
    validation_status: Literal["pending", "passed", "failed"] = "pending"
    lineage_report_hash: str | None = None
