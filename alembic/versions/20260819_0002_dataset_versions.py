"""dataset bundle versions

Revision ID: 20260819_0002
Revises: 20260819_0001
"""

import sqlalchemy as sa

from alembic import op

revision = "20260819_0002"
down_revision = "20260819_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "dataset_versions",
        sa.Column("id", sa.String(36), primary_key=True), sa.Column("dataset_id", sa.String(36), nullable=False),
        sa.Column("project_id", sa.String(36), nullable=False), sa.Column("created_by_user_id", sa.String(36), nullable=False),
        sa.Column("artifact_id", sa.String(36), sa.ForeignKey("artifacts.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("display_name", sa.String(200), nullable=False), sa.Column("task_id", sa.String(100), nullable=False),
        sa.Column("status", sa.String(32), nullable=False), sa.Column("schema_hash", sa.String(64), nullable=False),
        sa.Column("sample_count", sa.Integer()), sa.Column("manifest_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("project_id", "artifact_id", name="uq_dataset_versions_project_artifact"),
    )
    op.create_index("ix_dataset_versions_project_id", "dataset_versions", ["project_id"])
    op.create_index("ix_dataset_versions_dataset_id", "dataset_versions", ["dataset_id"])
    op.create_index("ix_dataset_versions_created_by_user_id", "dataset_versions", ["created_by_user_id"])
    op.create_index("ix_dataset_versions_status", "dataset_versions", ["status"])


def downgrade() -> None:
    op.drop_table("dataset_versions")
