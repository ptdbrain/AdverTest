from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.api.schemas import ValidationSummary
from src.core.types import Modality, Task
from src.pipeline.generator import SurrogateConfig

DatasetKind = Literal["clean", "attacked_standalone", "attacked_paired"]

class AttackedDatasetManifest(BaseModel):
    """Provenance required before a clean-versus-attacked comparison is allowed."""
    model_config = ConfigDict(extra="forbid")

    clean_dataset_version_id: str | None = None
    pairs: list[dict[str, str]] = Field(default_factory=list)
    attack_name: str | None = None
    attack_version: str | None = None
    severity: int | None = Field(default=None, ge=0, le=5)
    seed: int | None = Field(default=None, ge=0)
    source_hash: str | None = None
    ground_truth_hash: str | None = None

class UploadBatchCreateIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    display_name: str = Field(min_length=1, max_length=200)
    task_id: Task
    class_map: dict[str, str] = Field(default_factory=dict)
    anonymized: bool = False
    dataset_kind: DatasetKind = "clean"
    attacked_manifest: AttackedDatasetManifest | None = None

class UploadBatchOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_id: str
    display_name: str
    task_id: Task
    dataset_kind: DatasetKind
    class_map: dict[str, str] = Field(default_factory=dict)
    anonymized: bool
    samples: dict[str, dict[str, Any]] = Field(default_factory=dict)
    validation: ValidationSummary

class GeneratedDatasetCreateIn(BaseModel):
    """An immutable request referencing durable source and recipe records."""
    model_config = ConfigDict(extra="forbid")

    dataset_version_id: str = Field(min_length=1, max_length=128)
    recipe_id: str = Field(min_length=1, max_length=128)
    task: Task = Field(default="detection2d", description="Perception task this dataset serves")
    modality: Modality = Field(default="image", description="Sensor modality of generated samples")
    seed: int = Field(default=20260730, ge=0)
    surrogate: SurrogateConfig | None = None
    intended_use: str = Field(default="training", pattern="^(training|benchmark|review)$")
    preview: bool = True
    limit: int | None = Field(default=None, ge=1)

class GeneratedDatasetJobOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    status: str
    job_type: str
    error: str | None = None
    cancel_requested: bool = False
    descriptor: dict[str, Any] | None = None
    artifact_root: str | None = None

class GeneratedDatasetEventOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sequence: int
    state: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: str

class GeneratedDatasetEventsOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    events: list[GeneratedDatasetEventOut] = Field(default_factory=list)

class GeneratedDatasetManifestOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    manifest: list[dict[str, Any]] = Field(default_factory=list)

class GeneratedDatasetVariantsOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    variants: list[dict[str, Any]] = Field(default_factory=list)

class GeneratedDatasetValidationOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    validation: dict[str, Any] = Field(default_factory=dict)

class DatasetImportIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    root: str = Field(min_length=1)
    name: str = Field(min_length=1, max_length=200)
    logical_source_id: str = Field(min_length=1, max_length=200)
    input_format: str = Field(default="advertest", pattern="^(advertest|kitti)$")
    anonymization_manifest: str | None = None
    max_samples: int | None = Field(default=None, ge=1)
    task_id: Task = "detection2d"
