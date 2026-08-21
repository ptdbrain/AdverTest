"""platform foundation tables

Revision ID: 20260819_0001
Revises:
Create Date: 2026-08-19 00:00:00
"""

from alembic import op
import sqlalchemy as sa

revision = "20260819_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "artifacts",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("project_id", sa.String(length=36), nullable=False),
        sa.Column("created_by_user_id", sa.String(length=36), nullable=False),
        sa.Column("kind", sa.String(length=64), nullable=False),
        sa.Column("state", sa.String(length=32), nullable=False),
        sa.Column("storage_key", sa.String(length=1024), nullable=False, unique=True),
        sa.Column("sha256", sa.String(length=64)),
        sa.Column("size_bytes", sa.Integer()),
        sa.Column("mime_type", sa.String(length=255)),
        sa.Column("original_filename", sa.String(length=1024), nullable=False),
        sa.Column("metadata_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("finalized_at", sa.DateTime(timezone=True)),
        sa.Column("deleted_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_artifacts_project_id", "artifacts", ["project_id"])
    op.create_index("ix_artifacts_created_by_user_id", "artifacts", ["created_by_user_id"])
    op.create_index("ix_artifacts_kind", "artifacts", ["kind"])
    op.create_index("ix_artifacts_state", "artifacts", ["state"])
    op.create_index("ix_artifacts_sha256", "artifacts", ["sha256"])

    op.create_table(
        "artifact_upload_sessions",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("artifact_id", sa.String(length=36), sa.ForeignKey("artifacts.id", ondelete="CASCADE"), nullable=False, unique=True),
        sa.Column("project_id", sa.String(length=36), nullable=False),
        sa.Column("expected_sha256", sa.String(length=64)),
        sa.Column("expected_size_bytes", sa.Integer(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_artifact_upload_sessions_project_id", "artifact_upload_sessions", ["project_id"])
    op.create_index("ix_artifact_upload_sessions_expires_at", "artifact_upload_sessions", ["expires_at"])

    op.create_table(
        "checkpoints",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("project_id", sa.String(length=36), nullable=False),
        sa.Column("created_by_user_id", sa.String(length=36), nullable=False),
        sa.Column("artifact_id", sa.String(length=36), sa.ForeignKey("artifacts.id", ondelete="RESTRICT"), nullable=False, unique=True),
        sa.Column("task_id", sa.String(length=100), nullable=False),
        sa.Column("model_family_id", sa.String(length=100), nullable=False),
        sa.Column("source", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=64), nullable=False),
        sa.Column("native_class_names_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("num_classes", sa.Integer()),
        sa.Column("class_schema_hash", sa.String(length=64)),
        sa.Column("validation_error_code", sa.String(length=100)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_checkpoints_project_id", "checkpoints", ["project_id"])
    op.create_index("ix_checkpoints_created_by_user_id", "checkpoints", ["created_by_user_id"])
    op.create_index("ix_checkpoints_status", "checkpoints", ["status"])

    op.create_table(
        "jobs",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("project_id", sa.String(length=36), nullable=False),
        sa.Column("owner_user_id", sa.String(length=36), nullable=False),
        sa.Column("type", sa.String(length=100), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("stage", sa.String(length=100), nullable=False),
        sa.Column("completed_units", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_units", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("idempotency_key", sa.String(length=255)),
        sa.Column("request_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("result_json", sa.Text()),
        sa.Column("error_code", sa.String(length=100)),
        sa.Column("error_message", sa.Text()),
        sa.Column("cancel_requested", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("attempt", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("max_attempts", sa.Integer(), nullable=False, server_default="3"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint("project_id", "idempotency_key", name="uq_jobs_project_idempotency"),
    )
    op.create_index("ix_jobs_project_id", "jobs", ["project_id"])
    op.create_index("ix_jobs_owner_user_id", "jobs", ["owner_user_id"])
    op.create_index("ix_jobs_type", "jobs", ["type"])
    op.create_index("ix_jobs_status", "jobs", ["status"])
    op.create_index("ix_jobs_project_status", "jobs", ["project_id", "status"])

    op.create_table(
        "job_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("job_id", sa.String(length=36), sa.ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("stage", sa.String(length=100), nullable=False),
        sa.Column("completed", sa.Integer(), nullable=False),
        sa.Column("total", sa.Integer(), nullable=False),
        sa.Column("percent", sa.Float(), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("job_id", "sequence", name="uq_job_events_sequence"),
    )
    op.create_index("ix_job_events_job_id", "job_events", ["job_id"])

    op.create_table(
        "checkpoint_validations",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("checkpoint_id", sa.String(length=36), sa.ForeignKey("checkpoints.id", ondelete="CASCADE"), nullable=False),
        sa.Column("job_id", sa.String(length=36), nullable=False),
        sa.Column("stage", sa.String(length=64), nullable=False),
        sa.Column("passed", sa.Boolean()),
        sa.Column("error_code", sa.String(length=100)),
        sa.Column("details_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_checkpoint_validations_checkpoint_id", "checkpoint_validations", ["checkpoint_id"])
    op.create_index("ix_checkpoint_validations_job_id", "checkpoint_validations", ["job_id"])


def downgrade() -> None:
    op.drop_table("checkpoint_validations")
    op.drop_table("job_events")
    op.drop_table("jobs")
    op.drop_table("checkpoints")
    op.drop_table("artifact_upload_sessions")
    op.drop_table("artifacts")
