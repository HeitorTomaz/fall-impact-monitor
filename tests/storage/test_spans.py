import asyncio
import json
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text

from fall_impact_monitor.cli.main import main
from fall_impact_monitor.recording import Span
from fall_impact_monitor.storage import make_engine, stored_spans, sync_spans, upgrade
from tests.conftest import FakePlayback
from tests.storage.conftest import URL, needs_database

pytestmark = needs_database
T0 = datetime(2026, 11, 9, 3, 0, 0, tzinfo=UTC)


def span(start_s: float, end_s: float) -> Span:
    return Span(T0 + timedelta(seconds=start_s), T0 + timedelta(seconds=end_s))


def sync(camera_id: str, spans: list[Span]) -> list[Span]:
    async def run() -> list[Span]:
        engine = make_engine(URL)
        try:
            await sync_spans(engine, camera_id, spans)
            return await stored_spans(engine, camera_id)
        finally:
            await engine.dispose()

    return asyncio.run(run())


def test_upgrade_creates_the_span_table_and_is_idempotent(database: str) -> None:
    upgrade(database)

    async def tables() -> set[str]:
        engine = make_engine(database)
        async with engine.connect() as conn:
            rows = await conn.execute(
                text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
            )
            names = {r[0] for r in rows}
        await engine.dispose()
        return names

    assert {"recording_spans", "alembic_version"} <= asyncio.run(tables())


def test_sync_inserts_then_extends_the_last_span_without_duplicates(database: str) -> None:
    assert sync("cam1", [span(0, 60), span(70, 100)]) == [span(0, 60), span(70, 100)]
    assert sync("cam1", [span(0, 60), span(70, 160)]) == [span(0, 60), span(70, 160)]


def test_spans_older_than_the_listing_are_removed(database: str) -> None:
    sync("cam1", [span(0, 60), span(70, 100)])
    assert sync("cam1", [span(70, 130)]) == [span(70, 130)]


def test_cameras_are_independent_and_times_round_trip_as_utc(database: str) -> None:
    sync("cam1", [span(0, 60)])
    stored = sync("cam2", [span(5.25, 10.5)])
    assert stored == [span(5.25, 10.5)]
    assert stored[0].start.utcoffset() == timedelta(0)
    assert sync("cam1", [span(0, 60)]) == [span(0, 60)]


def test_an_empty_listing_keeps_the_stored_spans(database: str) -> None:
    sync("cam1", [span(0, 60)])
    assert sync("cam1", []) == [span(0, 60)]


def test_index_command_stores_spans_and_prints_gaps(
    database: str,
    playback: FakePlayback,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    now = datetime.now(UTC).replace(microsecond=0)
    items = [
        {"start": (now - timedelta(seconds=300)).isoformat(), "duration": 120.0},
        {"start": (now - timedelta(seconds=150)).isoformat(), "duration": 150.0},
    ]
    playback.routes["/list"] = lambda q: (200, json.dumps(items).encode())
    monkeypatch.setenv("FIM_CAMERAS", "cam1")
    monkeypatch.setenv("FIM_CAMERA_CAM1_MAIN_URL", "rtsp://user:secret@host/1")
    monkeypatch.setenv("FIM_MEDIAMTX_PLAYBACK_URL", playback.url)
    monkeypatch.setattr("fall_impact_monitor.cli.recordings._database_url", lambda: database)
    assert main(["recordings", "index"]) == 0
    out = capsys.readouterr().out
    assert "cam1: 2 spans" in out
    assert "gap" in out and "30" in out
    assert "secret" not in out
