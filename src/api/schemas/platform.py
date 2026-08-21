"""Public request and response schemas for the platform foundation APIs."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from src.core.platform_contracts import ArtifactKind, ArtifactState, JobStatus


class _PlatformApiSchemaV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CreateArtifactUploadSessionIn(_PlatformApiSchemaV1):
    """The project ID and actor are derived from the path and authentication."""

    kind: ArtifactKind
    original_filename: str = Field(min_length=1, max_length=1024)
    mime_type: str = Field(min_length=1, max_length=255)
    expected_size_bytes: int = Field(gt=0)
    expected_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class CreateArtifactUploadSessionOut(_PlatformApiSchemaV1):
    artifact_id: UUID
    upload_session_id: UUID
    upload_url: str
    expires_at: datetime
    contract_version: str = "1.0.0"


class FinalizeArtifactUploadIn(_PlatformApiSchemaV1):
    """Storage is authoritative; the worker verifies these claimed values."""

    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class ImportUltralyticsIn(_PlatformApiSchemaV1):
    """The handler accepts only a server-maintained model-ID whitelist."""

    model_id: str = Field(pattern=r"^yolo11[nsmlx]$")


class RegisterCheckpointIn(_PlatformApiSchemaV1):
    artifact_id: UUID
    task_id: str = Field(min_length=1, max_length=100)
    model_family_id: str = Field(min_length=1, max_length=100)


class ExportAttackedDatasetIn(_PlatformApiSchemaV1):
    source_dataset_version_id: UUID
    task: str = Field(min_length=1, max_length=100)
    attack_method: str = Field(min_length=1, max_length=255)
    severity: int = Field(ge=0, le=5)
    seed: int = Field(ge=0)
    implementation_version: str = Field(min_length=1, max_length=100)
    media_artifact_ids: tuple[UUID, ...] = ()
    label_artifact_ids: tuple[UUID, ...] = ()
    manifest: dict[str, object] = Field(default_factory=dict)
    recipe: dict[str, object] = Field(default_factory=dict)


class JobOutV1(_PlatformApiSchemaV1):
    id: UUID
    status: JobStatus
    stage: str
    completed_units: int = Field(ge=0)
    total_units: int = Field(ge=0)
    error_code: str | None = None
    error_message: str | None = None
    contract_version: str = "1.0.0"


class ArtifactOutV1(_PlatformApiSchemaV1):
    """Public representation intentionally excludes the private storage key."""

    id: UUID
    kind: ArtifactKind
    state: ArtifactState
    sha256: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    mime_type: str | None = None
    original_filename: str
    created_at: datetime
    contract_version: str = "1.0.0"
