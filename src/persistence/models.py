"""Platform-owned relational records.

Project and user tables are deliberately not duplicated here: person D owns
their migrations. Platform records retain project/user UUIDs and gain foreign
keys once those shared tables land in the migration registry.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class UserRecord(Base):
    """Auth table representation used by Alembic schema comparison."""

    __tablename__ = "users"
    __table_args__ = (Index("ix_users_email", "email"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[str] = mapped_column(String(200), nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False, server_default="USER", index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, server_default="ACTIVE", index=True)
    storage_quota_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False, server_default="10737418240")
    compute_quota_hours: Mapped[float] = mapped_column(Float, nullable=False, server_default="100.0")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class AuditLogRecord(Base):
    """Audit table representation used by Alembic schema comparison."""

    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    actor_user_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    action: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    resource_type: Mapped[str] = mapped_column(String(64), nullable=False)
    resource_id: Mapped[str] = mapped_column(String(64), nullable=False)
    detail_json: Mapped[str] = mapped_column(Text, nullable=False, server_default="{}")
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), index=True
    )


class ArtifactRecord(Base):
    __tablename__ = "artifacts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    project_id: Mapped[str] = mapped_column(String(36), index=True)
    created_by_user_id: Mapped[str] = mapped_column(String(36), index=True)
    kind: Mapped[str] = mapped_column(String(64), index=True)
    state: Mapped[str] = mapped_column(String(32), index=True)
    storage_key: Mapped[str] = mapped_column(String(1024), unique=True)
    sha256: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    mime_type: Mapped[str | None] = mapped_column(String(255), nullable=True)
    original_filename: Mapped[str] = mapped_column(String(1024))
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finalized_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ArtifactUploadSessionRecord(Base):
    __tablename__ = "artifact_upload_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    artifact_id: Mapped[str] = mapped_column(ForeignKey("artifacts.id", ondelete="CASCADE"), unique=True)
    project_id: Mapped[str] = mapped_column(String(36), index=True)
    expected_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    expected_size_bytes: Mapped[int] = mapped_column(Integer)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class CheckpointRecord(Base):
    __tablename__ = "checkpoints"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    project_id: Mapped[str] = mapped_column(String(36), index=True)
    created_by_user_id: Mapped[str] = mapped_column(String(36), index=True)
    artifact_id: Mapped[str] = mapped_column(ForeignKey("artifacts.id", ondelete="RESTRICT"), unique=True)
    task_id: Mapped[str] = mapped_column(String(100))
    model_family_id: Mapped[str] = mapped_column(String(100))
    source: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(64), index=True)
    native_class_names_json: Mapped[str] = mapped_column(Text, default="[]")
    num_classes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    class_schema_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    validation_error_code: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class CheckpointValidationRecord(Base):
    __tablename__ = "checkpoint_validations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    checkpoint_id: Mapped[str] = mapped_column(ForeignKey("checkpoints.id", ondelete="CASCADE"), index=True)
    job_id: Mapped[str] = mapped_column(String(36), index=True)
    stage: Mapped[str] = mapped_column(String(64))
    passed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(100), nullable=True)
    details_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class DatasetVersionRecord(Base):
    """Immutable dataset bundle registration backed by a finalized artifact."""

    __tablename__ = "dataset_versions"
    __table_args__ = (UniqueConstraint("project_id", "artifact_id", name="uq_dataset_versions_project_artifact"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    dataset_id: Mapped[str] = mapped_column(String(36), index=True)
    project_id: Mapped[str] = mapped_column(String(36), index=True)
    created_by_user_id: Mapped[str] = mapped_column(String(36), index=True)
    artifact_id: Mapped[str] = mapped_column(ForeignKey("artifacts.id", ondelete="RESTRICT"), nullable=False)
    display_name: Mapped[str] = mapped_column(String(200))
    task_id: Mapped[str] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(32), index=True)
    schema_hash: Mapped[str] = mapped_column(String(64))
    sample_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    manifest_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class PlatformJobRecord(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        UniqueConstraint("project_id", "idempotency_key", name="uq_jobs_project_idempotency"),
        Index("ix_jobs_project_status", "project_id", "status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    project_id: Mapped[str] = mapped_column(String(36), index=True)
    owner_user_id: Mapped[str] = mapped_column(String(36), index=True)
    type: Mapped[str] = mapped_column(String(100), index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    stage: Mapped[str] = mapped_column(String(100))
    completed_units: Mapped[int] = mapped_column(Integer, default=0)
    total_units: Mapped[int] = mapped_column(Integer, default=0)
    idempotency_key: Mapped[str | None] = mapped_column(String(255), nullable=True)
    request_json: Mapped[str] = mapped_column(Text, default="{}")
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(100), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    attempt: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class JobEventRecord(Base):
    __tablename__ = "job_events"
    __table_args__ = (UniqueConstraint("job_id", "sequence", name="uq_job_events_sequence"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    sequence: Mapped[int] = mapped_column(Integer)
    stage: Mapped[str] = mapped_column(String(100))
    completed: Mapped[int] = mapped_column(Integer)
    total: Mapped[int] = mapped_column(Integer)
    percent: Mapped[float] = mapped_column()
    message: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ProjectRecord(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    owner_user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ProjectMembershipRecord(Base):
    __tablename__ = "project_memberships"
    __table_args__ = (
        UniqueConstraint("project_id", "user_id", name="uq_project_user_membership"),
        Index("ix_project_memberships_user", "user_id", "status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    project_id: Mapped[str] = mapped_column(String(36), index=True)
    user_id: Mapped[str] = mapped_column(String(36), index=True)
    role: Mapped[str] = mapped_column(String(32), default="MEMBER")
    status: Mapped[str] = mapped_column(String(32), default="ACTIVE")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ExperimentSessionRecord(Base):
    __tablename__ = "experiment_sessions"
    __table_args__ = (Index("ix_experiment_sessions_project_status", "project_id", "status"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str] = mapped_column(String(36), index=True, default="default-project")
    owner_user_id: Mapped[str] = mapped_column(String(36), default="system")
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(Text, default="")
    task_id: Mapped[str] = mapped_column(String(100), default="detection2d")
    task_name: Mapped[str] = mapped_column(String(200), default="Object Detection")
    model_id: Mapped[str] = mapped_column(String(100), default="")
    model_name: Mapped[str] = mapped_column(String(200), default="")
    dataset_id: Mapped[str] = mapped_column(String(100), default="")
    dataset_name: Mapped[str] = mapped_column(String(200), default="")
    status: Mapped[str] = mapped_column(String(32), default="active", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    total_duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)


class SessionRunRecord(Base):
    __tablename__ = "session_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("experiment_sessions.id", ondelete="CASCADE"), index=True)
    project_id: Mapped[str] = mapped_column(String(36), index=True, default="default-project")
    name: Mapped[str] = mapped_column(String(255))
    timestamp: Mapped[str] = mapped_column(String(64))
    attack_type: Mapped[str] = mapped_column(String(100))
    attack_name: Mapped[str] = mapped_column(String(200))
    severity: Mapped[int] = mapped_column(Integer, default=1)
    # A visual inference run has no scientific metric until its evidence bundle
    # is complete.  NULL preserves that distinction; zero would be a false result.
    clean_map: Mapped[float | None] = mapped_column(Float, nullable=True)
    attacked_map: Mapped[float | None] = mapped_column(Float, nullable=True)
    map_drop_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    clean_conf: Mapped[float | None] = mapped_column(Float, nullable=True)
    attacked_conf: Mapped[float | None] = mapped_column(Float, nullable=True)
    psnr: Mapped[str] = mapped_column(String(32), default="N/A")
    ssim: Mapped[str] = mapped_column(String(32), default="N/A")
    inference_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    robustness_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    clean_bbox_count: Mapped[int] = mapped_column(Integer, default=0)
    attacked_bbox_count: Mapped[int] = mapped_column(Integer, default=0)
    sample_id: Mapped[str] = mapped_column(String(64), default="000000")
    clean_miou: Mapped[float | None] = mapped_column(Float, nullable=True)
    attacked_miou: Mapped[float | None] = mapped_column(Float, nullable=True)
    miou_drop_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_combined: Mapped[bool] = mapped_column(Boolean, default=False)
    attack_components_json: Mapped[str] = mapped_column(Text, default="[]")
    note: Mapped[str] = mapped_column(Text, default="")
    seed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    run_config_hash: Mapped[str] = mapped_column(String(128), default="")
    backend_run_id: Mapped[str] = mapped_column(String(128), default="")
    metrics_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
