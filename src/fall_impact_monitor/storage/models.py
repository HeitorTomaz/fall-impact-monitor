"""SQLAlchemy models. PostgreSQL never stores video or images; times are UTC timestamptz."""

from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Text, UniqueConstraint, func
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
