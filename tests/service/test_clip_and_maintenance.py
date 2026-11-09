import asyncio
import threading
from datetime import datetime
from pathlib import Path

import pytest

from fall_impact_monitor.config import Settings
from fall_impact_monitor.events import EventCategory
from fall_impact_monitor.service.clips import ClipWorker
from fall_impact_monitor.service.maintenance import maintenance_loop, maintenance_step
from tests.sinks.helpers import make_record


def test_clips_are_exported_for_notified_detections_only() -> None:
    done = threading.Event()
    calls: list[tuple[str, str]] = []

    def export(camera_id: str, event_id: str, start: datetime, end: datetime) -> None:
        calls.append((camera_id, event_id))
        done.set()

    worker = ClipWorker(export)
    assert not worker.request(make_record(0, near_miss=True))
    assert not worker.request(make_record(1, category=EventCategory.MONITORING_DEGRADED))
    record = make_record(2)
    worker.start()
    try:
        assert worker.request(record)
        assert done.wait(5)
    finally:
        worker.stop()
    assert calls == [("cam1", record.event.event_id)]


def test_an_export_failure_does_not_stop_the_worker() -> None:
    done = threading.Event()
    attempts: list[str] = []

    def export(camera_id: str, event_id: str, start: datetime, end: datetime) -> None:
        attempts.append(event_id)
        if len(attempts) == 1:
            raise RuntimeError("nothing recorded")
        done.set()

    worker = ClipWorker(export)
    worker.start()
    try:
        worker.request(make_record(0))
        worker.request(make_record(1))
        assert done.wait(5)
    finally:
        worker.stop()
    assert len(attempts) == 2


def test_a_full_queue_refuses_instead_of_blocking() -> None:
    worker = ClipWorker(lambda *a: None, max_queue=1)  # not started: nothing is consumed
    assert worker.request(make_record(0))
    assert not worker.request(make_record(1))


def test_the_maintenance_step_applies_retention_without_a_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recordings = tmp_path / "rec" / "cam1"
    recordings.mkdir(parents=True)
    for minute in range(3):
        (recordings / f"2026-11-01_00-0{minute}-00-000000_+0000.mp4").write_bytes(b"\0" * 1000)
    monkeypatch.setenv("FIM_RECORDINGS_DIR", str(tmp_path / "rec"))
    monkeypatch.setenv("FIM_EVENT_CLIP_DIR", str(tmp_path / "clips"))
    monkeypatch.setenv("FIM_RECORDINGS_MAX_GB", "0.0000015")  # 1500 bytes
    monkeypatch.setenv("FIM_RECORDINGS_DISK_MIN_FREE_GB", "0")
    asyncio.run(maintenance_step(Settings(_env_file=None), cameras=[], engine=None))
    assert len(list(recordings.iterdir())) == 1


def test_the_maintenance_loop_survives_a_failing_step() -> None:
    runs: list[int] = []

    async def step() -> None:
        runs.append(1)
        if len(runs) == 1:
            raise OSError("disk unavailable")

    async def main() -> None:
        async def fast(seconds: float) -> None:
            await asyncio.sleep(0)

        task = asyncio.create_task(maintenance_loop(step, interval_s=600, sleep=fast))
        for _ in range(100):
            if len(runs) >= 3:
                break
            await asyncio.sleep(0)
        task.cancel()

    asyncio.run(main())
    assert len(runs) >= 3


def test_a_database_error_in_the_span_index_does_not_fail_the_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from fall_impact_monitor.config import CameraSettings
    from fall_impact_monitor.storage import make_engine

    monkeypatch.setenv("FIM_RECORDINGS_DIR", str(tmp_path / "rec"))
    monkeypatch.setenv("FIM_EVENT_CLIP_DIR", str(tmp_path / "clips"))
    monkeypatch.setattr("fall_impact_monitor.service.maintenance.list_spans", lambda url, path: [])

    async def main() -> None:
        engine = make_engine("postgresql+asyncpg://fim:fim@127.0.0.1:1/missing")
        try:
            await maintenance_step(
                Settings(_env_file=None),
                cameras=[CameraSettings("cam1", "rtsp://c", None)],
                engine=engine,
            )
        finally:
            await engine.dispose()

    monkeypatch.setattr("fall_impact_monitor.service.maintenance.sync_spans", _failing_sync)
    asyncio.run(main())
    assert "span index not updated" in caplog.text


async def _failing_sync(*args: object) -> None:
    raise ConnectionError("database starting")
