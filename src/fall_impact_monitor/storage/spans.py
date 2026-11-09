"""The recorded-span index: what MediaMTX has on disk per camera, and therefore its gaps."""

from datetime import UTC

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncEngine

from fall_impact_monitor.recording.mediamtx import Span
from fall_impact_monitor.storage.models import RecordingSpan


async def sync_spans(engine: AsyncEngine, camera_id: str, spans: list[Span]) -> None:
    """Store the current listing of a camera: upsert each span by its start (a growing
    last span updates its end) and drop spans older than the oldest listed one (deleted
    by retention). An empty listing changes nothing."""
    if not spans:
        return
    rows = [{"camera_id": camera_id, "start_utc": s.start, "end_utc": s.end} for s in spans]
    stmt = insert(RecordingSpan).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=["camera_id", "start_utc"],
        set_={"end_utc": stmt.excluded.end_utc, "updated_at": func.now()},
    )
    oldest = min(s.start for s in spans)
    async with engine.begin() as conn:
        await conn.execute(stmt)
        await conn.execute(
            delete(RecordingSpan).where(
                RecordingSpan.camera_id == camera_id, RecordingSpan.start_utc < oldest
            )
        )


async def stored_spans(engine: AsyncEngine, camera_id: str) -> list[Span]:
    query = (
        select(RecordingSpan.start_utc, RecordingSpan.end_utc)
        .where(RecordingSpan.camera_id == camera_id)
        .order_by(RecordingSpan.start_utc)
    )
    async with engine.connect() as conn:
        rows = (await conn.execute(query)).all()
    return [Span(start.astimezone(UTC), end.astimezone(UTC)) for start, end in rows]
