"""project task type and session class mapping

Revision ID: 20260904_0007
Revises: 20260831_0006
Create Date: 2026-09-04 00:00:00
"""

import sqlalchemy as sa

from alembic import op


revision = "20260904_0007"
down_revision = "20260831_0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Workflow §3: one project pursues exactly one task type. Existing rows
    # keep working under the detection2d default.
    op.add_column(
        "projects",
        sa.Column(
            "task_type",
            sa.String(100),
            nullable=False,
            server_default="detection2d",
        ),
    )
    # Workflow §4: the class mapping matrix chosen in the UI must survive
    # refreshes, so it is persisted on the session instead of localStorage.
    op.add_column(
        "experiment_sessions",
        sa.Column(
            "class_mapping_json",
            sa.Text(),
            nullable=False,
            server_default="{}",
        ),
    )


def downgrade() -> None:
    op.drop_column("experiment_sessions", "class_mapping_json")
    op.drop_column("projects", "task_type")