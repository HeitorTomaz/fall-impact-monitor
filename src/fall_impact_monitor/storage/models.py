"""SQLAlchemy models. PostgreSQL never stores video or images; times are UTC timestamptz."""

from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    Integer,
    Text,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class RecordingSpan(Base):
    """A contiguous recorded time range of one camera, as listed by MediaMTX. The space
    between two spans of a camera is a gap in its recording."""

    __tablename__ = "recording_spans"
    __table_args__ = (UniqueConstraint("camera_id", "start_utc"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    camera_id: Mapped[str] = mapped_column(Text)
    start_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    end_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class EventRow(Base):
    """One event (near-misses included). State: detected -> notified -> seen -> feedback."""

    __tablename__ = "events"

    event_id: Mapped[str] = mapped_column(Uuid(as_uuid=False), primary_key=True)
    camera_id: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float)
    near_miss: Mapped[bool] = mapped_column(Boolean)
    started_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    ended_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    detected_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    track_id: Mapped[int | None] = mapped_column(Integer)
    identity: Mapped[str] = mapped_column(Text)
    evidence: Mapped[dict[str, Any]] = mapped_column(JSONB)
    analysed_seqs: Mapped[list[int]] = mapped_column(JSONB)
    config_hash: Mapped[str] = mapped_column(Text)
    state: Mapped[str] = mapped_column(Text, server_default="detected")
    feedback: Mapped[str | None] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class NotificationRow(Base):
    """A delivery of an event to one notifying sink (e.g. one Telegram chat)."""

    __tablename__ = "notifications"
    __table_args__ = (UniqueConstraint("event_id", "sink"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    event_id: Mapped[str] = mapped_column(Uuid(as_uuid=False))
    sink: Mapped[str] = mapped_column(Text)
    message_ref: Mapped[str | None] = mapped_column(Text)
    sent_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    late_s: Mapped[float] = mapped_column(Float)


class FeedbackRow(Base):
    """A human answer to an alert: seen, false_alarm or confirmed (by a person)."""

    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    event_id: Mapped[str] = mapped_column(Uuid(as_uuid=False))
    action: Mapped[str] = mapped_column(Text)
    created_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class ConfigSnapshotRow(Base):
    """The configuration (secrets masked) that produced events with this hash."""

    __tablename__ = "config_snapshots"

    config_hash: Mapped[str] = mapped_column(Text, primary_key=True)
    snapshot: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
