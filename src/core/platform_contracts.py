"""Versioned contracts for the multi-user platform foundation.

These types define the ownership and lifecycle boundaries shared by the
storage, checkpoint, job, training, and user-product domains.  They contain no
database, object-storage, queue, or HTTP implementation details.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _PlatformContractV1(BaseModel):
    """Immutable platform wire contract with strict compatibility rules."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class TaskId(StrEnum):
    DETECTION2D = "detection2d"
    SEGMENTATION = "segmentation"
    DETECTION3D = "detection3d"


class DatasetVersionStatus(StrEnum):
    UPLOADING = "UPLOADING"
    VALIDATING = "VALIDATING"
    READY = "READY"
    FAILED = "FAILED"


class ArtifactKind(StrEnum):
    DATASET_BUNDLE = "dataset_bundle"
    CHECKPOINT = "checkpoint"
    EVIDENCE = "evidence"
    PREDICTION = "prediction"
    REPORT = "report"
    EXPORT = "export"
    TRAINING_OUTPUT = "training_output"


class ArtifactState(StrEnum):
    UPLOADING = "UPLOADING"
    QUARANTINED = "QUARANTINED"
    READY = "READY"
    FAILED = "FAILED"
    DELETED = "DELETED"


class CheckpointSource(StrEnum):
    UPLOAD = "upload"
    ULTRALYTICS_IMPORT = "ultralytics_import"
    TRAINING_OUTPUT = "training_output"


class CheckpointStatus(StrEnum):
    UPLOADING = "UPLOADING"
    UPLOADED = "UPLOADED"
    QUARANTINED = "QUARANTINED"
    INTEGRITY_VALIDATING = "INTEGRITY_VALIDATING"
    SANDBOX_LOADING = "SANDBOX_LOADING"
    METADATA_EXTRACTING = "METADATA_EXTRACTING"
    SMOKE_TESTING = "SMOKE_TESTING"
    READY = "READY"
    REJECTED = "REJECTED"
    FAILED_VALIDATION = "FAILED_VALIDATION"
    UNSUPPORTED = "UNSUPPORTED"


class JobStatus(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ClassMappingStatus(StrEnum):
    MAPPED = "MAPPED"
    UNMAPPED = "UNMAPPED"
    IGNORED = "IGNORED"


class BenchmarkRunStatus(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class TrainingRunStatus(StrEnum):
    DRAFT = "DRAFT"
    VALIDATING = "VALIDATING"
    ESTIMATING = "ESTIMATING"
    QUEUED = "QUEUED"
    PREPARING_DATA = "PREPARING_DATA"
    TRAINING = "TRAINING"
    VALIDATING_CHECKPOINT = "VALIDATING_CHECKPOINT"
    EXPORTING = "EXPORTING"
    REGISTERING_MODEL = "REGISTERING_MODEL"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"


class DefenceRunStatus(StrEnum):
    DRAFT = "DRAFT"
    GENERATING_DATA = "GENERATING_DATA"
    TRAINING = "TRAINING"
    EVALUATING = "EVALUATING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ModelComparisonStatus(StrEnum):
    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    INCOMPATIBLE = "INCOMPATIBLE"
    FAILED = "FAILED"


class TaskDefinitionV1(_PlatformContractV1):
    """Global catalog entry; task identity is not inferred from model names."""

    id: TaskId
    display_name: str = Field(min_length=1, max_length=100)
    required_modalities: tuple[Literal["image", "lidar", "multi"], ...]
    prediction_contract: str = Field(min_length=1, max_length=100)
    annotation_schema: tuple[str, ...] = ()
    version: Literal["1.0.0"] = "1.0.0"


class ModelFamilyV1(_PlatformContractV1):
    """Global model-family catalog entry, separate from a checkpoint artifact."""

    id: str = Field(min_length=1, max_length=100)
    display_name: str = Field(min_length=1, max_length=100)
    supported_task_ids: tuple[TaskId, ...] = Field(min_length=1)
    checkpoint_extensions: tuple[str, ...] = Field(min_length=1)
    adapter_key: str = Field(min_length=1, max_length=100)
    runnable: bool = False
    blocked_reason: str | None = None
    version: Literal["1.0.0"] = "1.0.0"

    @model_validator(mode="after")
    def validate_extensions(self) -> ModelFamilyV1:
        if any(not extension.startswith(".") for extension in self.checkpoint_extensions):
            raise ValueError("checkpoint extensions must start with a dot")
        return self


class UserV1(_PlatformContractV1):
    id: UUID
    email: str
    role: Literal["USER", "ADMIN"]
    created_at: datetime
    contract_version: Literal["1.0.0"] = "1.0.0"


class ProjectV1(_PlatformContractV1):
    id: UUID
    owner_user_id: UUID
    name: str = Field(min_length=1, max_length=200)
    created_at: datetime
    contract_version: Literal["1.0.0"] = "1.0.0"


class ProjectMembershipV1(_PlatformContractV1):
    """Explicit access record used by every project-scoped authorization query."""

    project_id: UUID
    user_id: UUID
    role: Literal["OWNER", "MEMBER", "VIEWER"]
    created_at: datetime
    contract_version: Literal["1.0.0"] = "1.0.0"


class DatasetV1(_PlatformContractV1):
    """Logical dataset identity; its versions are immutable snapshots."""

    id: UUID
    project_id: UUID
    created_by_user_id: UUID
    name: str = Field(min_length=1, max_length=200)
    task_id: TaskId
    created_at: datetime
    contract_version: Literal["1.0.0"] = "1.0.0"


class ArtifactV1(_PlatformContractV1):
    """Object-storage metadata, never a signed URL or inline file bytes."""

    id: UUID
    project_id: UUID
    created_by_user_id: UUID
    kind: ArtifactKind
    state: ArtifactState
    storage_key: str = Field(min_length=1, max_length=1024)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    size_bytes: int | None = Field(default=None, ge=0)
    mime_type: str | None = Field(default=None, min_length=1, max_length=255)
    original_filename: str = Field(min_length=1, max_length=1024)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    finalized_at: datetime | None = None
    deleted_at: datetime | None = None
    contract_version: Literal["1.0.0"] = "1.0.0"

    @model_validator(mode="after")
    def require_integrity_metadata_after_upload(self) -> ArtifactV1:
        if self.state in {ArtifactState.QUARANTINED, ArtifactState.READY}:
            if self.sha256 is None or self.size_bytes is None or self.finalized_at is None:
                raise ValueError("finalized artifacts require hash, size, and finalized_at")
        if self.state is ArtifactState.DELETED and self.deleted_at is None:
            raise ValueError("deleted artifacts require deleted_at")
        return self


class DatasetVersionV1(_PlatformContractV1):
    """Immutable, project-scoped snapshot selected by training and benchmarks."""

    id: UUID
    dataset_id: UUID
    project_id: UUID
    manifest_artifact_id: UUID
    split_manifest_artifact_id: UUID | None = None
    version: int = Field(ge=1)
    status: DatasetVersionStatus
    schema_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_dataset_version_id: UUID | None = None
    sample_count: int | None = Field(default=None, ge=0)
    created_at: datetime
    contract_version: Literal["1.0.0"] = "1.0.0"


class CheckpointArtifactV1(_PlatformContractV1):
    """Checkpoint semantics referencing one artifact; it does not expose bytes."""

    checkpoint_id: UUID
    project_id: UUID
    artifact_id: UUID
    task_id: TaskId
    model_family_id: str = Field(min_length=1, max_length=100)
    source: CheckpointSource
    status: CheckpointStatus
    native_class_names: tuple[str, ...] = ()
    num_classes: int | None = Field(default=None, ge=1)
    class_schema_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    created_at: datetime
    contract_version: Literal["1.0.0"] = "1.0.0"

    @model_validator(mode="after")
    def require_class_metadata_when_ready(self) -> CheckpointArtifactV1:
        if self.status is CheckpointStatus.READY:
            if self.num_classes is None or len(self.native_class_names) != self.num_classes:
                raise ValueError("ready checkpoints require complete native class metadata")
            if self.class_schema_hash is None:
                raise ValueError("ready checkpoints require class_schema_hash")
        return self


class ClassMappingV1(_PlatformContractV1):
    """Links a checkpoint's native classes to one dataset-version label schema."""

    id: UUID
    project_id: UUID
    checkpoint_id: UUID
    dataset_version_id: UUID
    native_class_id: int = Field(ge=0)
    canonical_class_id: str | None = Field(default=None, min_length=1, max_length=255)
    status: ClassMappingStatus
    created_at: datetime
    contract_version: Literal["1.0.0"] = "1.0.0"

    @model_validator(mode="after")
    def validate_mapping_status(self) -> ClassMappingV1:
        if self.status is ClassMappingStatus.MAPPED and self.canonical_class_id is None:
            raise ValueError("mapped class mappings require canonical_class_id")
        if self.status is not ClassMappingStatus.MAPPED and self.canonical_class_id is not None:
            raise ValueError("only mapped class mappings may set canonical_class_id")
        return self


class JobV1(_PlatformContractV1):
    id: UUID
    project_id: UUID
    owner_user_id: UUID
    type: str = Field(min_length=1, max_length=100)
    status: JobStatus
    stage: str = Field(min_length=1, max_length=100)
    completed_units: int = Field(ge=0)
    total_units: int = Field(ge=0)
    idempotency_key: str | None = Field(default=None, max_length=255)
    error_code: str | None = Field(default=None, max_length=100)
    error_message: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    contract_version: Literal["1.0.0"] = "1.0.0"

    @model_validator(mode="after")
    def validate_progress_and_terminal_state(self) -> JobV1:
        if self.total_units and self.completed_units > self.total_units:
            raise ValueError("completed_units cannot exceed total_units")
        if self.status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED} and self.completed_at is None:
            raise ValueError("terminal jobs require completed_at")
        return self


class JobProgressEventV1(_PlatformContractV1):
    job_id: UUID
    sequence: int = Field(ge=0)
    stage: str = Field(min_length=1, max_length=100)
    completed: int = Field(ge=0)
    total: int = Field(ge=0)
    percent: float = Field(ge=0.0, le=100.0)
    message: str = Field(min_length=1, max_length=1000)
    timestamp: datetime
    contract_version: Literal["1.0.0"] = "1.0.0"

    @model_validator(mode="after")
    def validate_event_progress(self) -> JobProgressEventV1:
        if self.total and self.completed > self.total:
            raise ValueError("completed cannot exceed total")
        return self


class ArtifactReferenceV1(_PlatformContractV1):
    """Append-only provenance edge from a platform entity to an artifact."""

    artifact_id: UUID
    project_id: UUID
    subject_type: Literal[
        "dataset_version",
        "checkpoint",
        "attack_recipe",
        "benchmark_run",
        "training_run",
        "defence_run",
        "model_comparison",
    ]
    subject_id: UUID
    relation: Literal["INPUT", "OUTPUT", "EVIDENCE", "REPORT", "EXPORT"]
    created_at: datetime
    contract_version: Literal["1.0.0"] = "1.0.0"


class AttackRecipeV1(_PlatformContractV1):
    """Project-scoped immutable record of a domain AttackRecipe definition."""

    id: UUID
    project_id: UUID
    created_by_user_id: UUID
    task_id: TaskId
    recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    catalog_version: str = Field(min_length=1, max_length=100)
    implementation_version: str = Field(min_length=1, max_length=100)
    definition: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    contract_version: Literal["1.0.0"] = "1.0.0"


class BenchmarkRunV1(_PlatformContractV1):
    """One execution of a locked benchmark protocol against one checkpoint."""

    id: UUID
    project_id: UUID
    created_by_user_id: UUID
    checkpoint_id: UUID
    dataset_version_id: UUID
    attack_recipe_id: UUID | None = None
    class_mapping_id: UUID | None = None
    job_id: UUID
    protocol_id: str = Field(min_length=1, max_length=255)
    status: BenchmarkRunStatus
    scientific: bool = True
    metrics_artifact_id: UUID | None = None
    report_artifact_id: UUID | None = None
    created_at: datetime
    completed_at: datetime | None = None
    contract_version: Literal["1.0.0"] = "1.0.0"

    @model_validator(mode="after")
    def validate_scientific_run(self) -> BenchmarkRunV1:
        if self.scientific and self.class_mapping_id is None:
            raise ValueError("scientific benchmark runs require class_mapping_id")
        if self.status is BenchmarkRunStatus.COMPLETED and self.completed_at is None:
            raise ValueError("completed benchmark runs require completed_at")
        return self


class TrainingRunV1(_PlatformContractV1):
    """Persistent training execution and lineage, separate from worker reports."""

    id: UUID
    project_id: UUID
    created_by_user_id: UUID
    parent_checkpoint_id: UUID
    dataset_version_id: UUID
    defence_profile_id: str = Field(min_length=1, max_length=255)
    job_id: UUID | None = None
    status: TrainingRunStatus
    seed: int = Field(ge=0)
    output_checkpoint_id: UUID | None = None
    report_artifact_id: UUID | None = None
    created_at: datetime
    completed_at: datetime | None = None
    contract_version: Literal["1.0.0"] = "1.0.0"

    @model_validator(mode="after")
    def validate_completed_training(self) -> TrainingRunV1:
        if self.status is TrainingRunStatus.COMPLETED:
            if self.output_checkpoint_id is None or self.completed_at is None:
                raise ValueError("completed training runs require output_checkpoint_id and completed_at")
        return self


class DefenceRunV1(_PlatformContractV1):
    """Closed-loop record tying defence data, training, and recovery together."""

    id: UUID
    project_id: UUID
    created_by_user_id: UUID
    baseline_checkpoint_id: UUID
    baseline_benchmark_run_id: UUID
    defence_profile_id: str = Field(min_length=1, max_length=255)
    generated_dataset_version_id: UUID | None = None
    training_run_id: UUID | None = None
    defended_checkpoint_id: UUID | None = None
    defended_benchmark_run_id: UUID | None = None
    status: DefenceRunStatus
    created_at: datetime
    completed_at: datetime | None = None
    contract_version: Literal["1.0.0"] = "1.0.0"

    @model_validator(mode="after")
    def validate_completed_defence(self) -> DefenceRunV1:
        if self.status is DefenceRunStatus.COMPLETED:
            required = (self.training_run_id, self.defended_checkpoint_id, self.defended_benchmark_run_id)
            if any(value is None for value in required) or self.completed_at is None:
                raise ValueError("completed defence runs require training, checkpoint, benchmark, and completed_at")
        return self


class ModelComparisonV1(_PlatformContractV1):
    """Persisted paired-comparison record; metric math remains in evaluation."""

    id: UUID
    project_id: UUID
    created_by_user_id: UUID
    baseline_benchmark_run_id: UUID
    candidate_benchmark_run_id: UUID
    status: ModelComparisonStatus
    paired: bool
    report_artifact_id: UUID | None = None
    incompatibilities: tuple[str, ...] = ()
    created_at: datetime
    completed_at: datetime | None = None
    contract_version: Literal["1.0.0"] = "1.0.0"

    @model_validator(mode="after")
    def validate_comparison_result(self) -> ModelComparisonV1:
        if self.status is ModelComparisonStatus.COMPLETED and self.completed_at is None:
            raise ValueError("completed comparisons require completed_at")
        if self.status is ModelComparisonStatus.INCOMPATIBLE and not self.incompatibilities:
            raise ValueError("incompatible comparisons require incompatibilities")
        return self
