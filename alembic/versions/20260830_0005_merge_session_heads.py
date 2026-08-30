"""merge durable-session and NewUI session migration heads

Revision ID: 20260830_0005
Revises: 20260829_0004, 20260830_0004
"""

from alembic import op


revision = "20260830_0005"
down_revision = ("20260829_0004", "20260830_0004")
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Join independent session-schema branches without DDL."""


def downgrade() -> None:
    """The two parent revisions remain independently reversible."""

