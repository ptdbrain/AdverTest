"""experiment sessions and project memberships

Revision ID: 20260830_0004
Revises: 20260823_0003
"""

from alembic import op
import sqlalchemy as sa

revision = "20260830_0004"
down_revision = "20260823_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "project_memberships",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("project_id", sa.String(36), nullable=False),
        sa.Column("user_id", sa.String(36), nullable=False),
        sa.Column("role", sa.String(32), nullable=False, server_default="MEMBER"),
        sa.Column("status", sa.String(32), nullable=False, server_default="ACTIVE"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("project_id", "user_id", name="uq_project_user_membership"),
    )
    op.create_index("ix_project_memberships_project_id", "project_memberships", ["project_id"])
    op.create_index("ix_project_memberships_user_id", "project_memberships", ["user_id"])
    op.create_index("ix_project_memberships_user", "project_memberships", ["user_id", "status"])

    op.create_table(
        "experiment_sessions",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("project_id", sa.String(36), nullable=False, server_default="default-project"),
        sa.Column("owner_user_id", sa.String(36), nullable=False, server_default="system"),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("task_id", sa.String(100), nullable=False, server_default="detection2d"),
        sa.Column("task_name", sa.String(200), nullable=False, server_default="Object Detection"),
        sa.Column("model_id", sa.String(100), nullable=False, server_default=""),
        sa.Column("model_name", sa.String(200), nullable=False, server_default=""),
        sa.Column("dataset_id", sa.String(100), nullable=False, server_default=""),
        sa.Column("dataset_name", sa.String(200), nullable=False, server_default=""),
        sa.Column("status", sa.String(32), nullable=False, server_default="active"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("total_duration_seconds", sa.Float(), nullable=True),
    )
    op.create_index("ix_experiment_sessions_project_id", "experiment_sessions", ["project_id"])
    op.create_index("ix_experiment_sessions_project_status", "experiment_sessions", ["project_id", "status"])
    op.create_index("ix_experiment_sessions_status", "experiment_sessions", ["status"])

    op.create_table(
        "session_runs",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("session_id", sa.String(64), sa.ForeignKey("experiment_sessions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("project_id", sa.String(36), nullable=False, server_default="default-project"),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("timestamp", sa.String(64), nullable=False),
        sa.Column("attack_type", sa.String(100), nullable=False),
        sa.Column("attack_name", sa.String(200), nullable=False),
        sa.Column("severity", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("clean_map", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("attacked_map", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("map_drop_pct", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("clean_conf", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("attacked_conf", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("psnr", sa.String(32), nullable=False, server_default="N/A"),
        sa.Column("ssim", sa.String(32), nullable=False, server_default="N/A"),
        sa.Column("inference_ms", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("robustness_score", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("clean_bbox_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("attacked_bbox_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("sample_id", sa.String(64), nullable=False, server_default="000000"),
        sa.Column("clean_miou", sa.Float(), nullable=True),
        sa.Column("attacked_miou", sa.Float(), nullable=True),
        sa.Column("miou_drop_pct", sa.Float(), nullable=True),
        sa.Column("is_combined", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("attack_components_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("note", sa.Text(), nullable=False, server_default=""),
        sa.Column("seed", sa.Integer(), nullable=True),
        sa.Column("run_config_hash", sa.String(128), nullable=False, server_default=""),
        sa.Column("backend_run_id", sa.String(128), nullable=False, server_default=""),
        sa.Column("metrics_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_session_runs_session_id", "session_runs", ["session_id"])
    op.create_index("ix_session_runs_project_id", "session_runs", ["project_id"])


def downgrade() -> None:
    op.drop_table("session_runs")
    op.drop_table("experiment_sessions")
    op.drop_table("project_memberships")
