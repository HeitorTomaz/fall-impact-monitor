"""Recorded-span index.

Revision ID: 0001
Revises:
Create Date: 2026-11-09
"""

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "recording_spans",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("camera_id", sa.Text(), nullable=False),
        sa.Column("start_utc", sa.DateTime(timezone=True), nullable=False),
        sa.Column("end_utc", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("camera_id", "start_utc"),
    )


def downgrade() -> None:
    op.drop_table("recording_spans")
