"""preserve absent visual metrics as null

Revision ID: 20260830_0005
Revises: 20260830_0004
"""

from alembic import op
import sqlalchemy as sa

revision = "20260830_0005"
down_revision = "20260830_0004"
branch_labels = None
depends_on = None

_NULLABLE_METRICS = (
    "clean_map",
    "attacked_map",
    "map_drop_pct",
    "clean_conf",
    "attacked_conf",
    "inference_ms",
    "robustness_score",
)


def upgrade() -> None:
    with op.batch_alter_table("session_runs") as batch:
        for column in _NULLABLE_METRICS:
            batch.alter_column(column, existing_type=sa.Float(), nullable=True, server_default=None)


def downgrade() -> None:
    with op.batch_alter_table("session_runs") as batch:
        for column in _NULLABLE_METRICS:
            batch.alter_column(column, existing_type=sa.Float(), nullable=False, server_default="0.0")
