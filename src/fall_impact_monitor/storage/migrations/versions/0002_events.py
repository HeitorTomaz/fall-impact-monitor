"""Events, notifications, feedback and configuration snapshots.

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-09
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "events",
        sa.Column("event_id", sa.Uuid(as_uuid=False), primary_key=True),
        sa.Column("camera_id", sa.Text(), nullable=False),
        sa.Column("category", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("near_miss", sa.Boolean(), nullable=False),
        sa.Column("started_utc", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_utc", sa.DateTime(timezone=True), nullable=False),
        sa.Column("detected_utc", sa.DateTime(timezone=True), nullable=False),
        sa.Column("track_id", sa.Integer(), nullable=True),
        sa.Column("identity", sa.Text(), nullable=False),
        sa.Column("evidence", JSONB(), nullable=False),
        sa.Column("analysed_seqs", JSONB(), nullable=False),
        sa.Column("config_hash", sa.Text(), nullable=False),
        sa.Column("state", sa.Text(), nullable=False, server_default="detected"),
        sa.Column("feedback", sa.Text(), nullable=True),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_index("ix_events_camera_started", "events", ["camera_id", "started_utc"])
    op.create_table(
        "notifications",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("event_id", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("sink", sa.Text(), nullable=False),
        sa.Column("message_ref", sa.Text(), nullable=True),
        sa.Column("sent_utc", sa.DateTime(timezone=True), nullable=False),
        sa.Column("late_s", sa.Float(), nullable=False),
        sa.UniqueConstraint("event_id", "sink"),
    )
    op.create_table(
        "feedback",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("event_id", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("action", sa.Text(), nullable=False),
        sa.Column("created_utc", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_feedback_event", "feedback", ["event_id"])
    op.create_table(
        "config_snapshots",
        sa.Column("config_hash", sa.Text(), primary_key=True),
        sa.Column("snapshot", JSONB(), nullable=False),
        sa.Column(
            "created_utc", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )


def downgrade() -> None:
    op.drop_table("config_snapshots")
    op.drop_table("feedback")
    op.drop_table("notifications")
    op.drop_table("events")
