"""add missing projects updated timestamp

Revision ID: 20260831_0006
Revises: 20260830_0005
Create Date: 2026-08-31 00:00:00
"""

import sqlalchemy as sa

from alembic import op


revision = "20260831_0006"
down_revision = "20260830_0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # `ProjectRecord` gained this field after the original projects migration.
    # A server default populates existing production rows during ALTER TABLE.
    op.add_column(
        "projects",
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )


def downgrade() -> None:
    op.drop_column("projects", "updated_at")
