import asyncio
import json
import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from fall_impact_monitor.config import CameraSettings, Settings
from fall_impact_monitor.config.logs import JsonFormatter
from fall_impact_monitor.service.app import ServiceParts, run_service
from fall_impact_monitor.service.camera import DecodedFrame
from tests.service.test_camera_worker import FakeAnalyzer, FakePipeline, fall, frame_record
from tests.sinks.test_telegram import FakeApi

CAMERA = CameraSettings("cam1", "rtsp://user:secret@cam/1", None)


def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **env: str) -> Settings:
    defaults = {
        "FIM_OUTBOX_DIR": str(tmp_path / "outbox"),
        "FIM_EVENT_LOG_FILE": str(tmp_path / "events.jsonl"),
        "FIM_EVENT_CLIP_DIR": str(tmp_path / "clips"),
        "FIM_RECORDINGS_DIR": str(tmp_path / "rec"),
        "FIM_MEDIAMTX_PLAYBACK_URL": "http://127.0.0.1:1",
    }
    for key, value in {**defaults, **env}.items():
        monkeypatch.setenv(key, value)
    return Settings(_env_file=None)


def parts(api: FakeApi | None = None) -> ServiceParts:
    def source(camera: CameraSettings) -> Iterator[DecodedFrame]:
        yield DecodedFrame(frame_record(100.0), np.zeros((36, 64, 3), np.uint8))

    return ServiceParts(
        make_pipeline=FakePipeline,
        make_analyzer=lambda camera: FakeAnalyzer([[fall(98.0, 99.5)]]),
        make_source=source,
        telegram_api=api,
    )


def run(settings_: Settings, tmp_path: Path, api: FakeApi | None = None) -> list[dict[str, object]]:
    log = tmp_path / "events.jsonl"

    async def main() -> None:
        stop = asyncio.Event()
        task = asyncio.create_task(
            run_service(
                settings_, [CAMERA], parts=parts(api), heartbeat=tmp_path / "alive", stop=stop
            )
        )
        for _ in range(500):
            if log.exists() and "possible_fall" in log.read_text(encoding="utf-8"):
                break
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
        stop.set()
        await asyncio.wait_for(task, 10)

    asyncio.run(main())
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]


def test_an_event_flows_from_the_camera_to_every_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = FakeApi()
    s = settings(
        tmp_path,
        monkeypatch,
        FIM_TELEGRAM_ENABLED="true",
        FIM_TELEGRAM_BOT_TOKEN="1:t",
        FIM_TELEGRAM_CHAT_IDS="-1",
    )
    lines = run(s, tmp_path, api)
    assert [line["category"] for line in lines] == ["possible_fall"]
    assert lines[0]["started_utc"] == "2026-11-09T03:00:08+00:00"
    assert api.calls and api.calls[0][0] == "send"
    assert (tmp_path / "alive").exists()


def test_an_unreachable_database_does_not_stop_alerts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = settings(
        tmp_path,
        monkeypatch,
        FIM_DB_PASSWORD="x",
        FIM_DB_HOST="127.0.0.1",
        FIM_DB_PORT="1",
    )
    lines = run(s, tmp_path)
    assert [line["category"] for line in lines] == ["possible_fall"]
    assert any((tmp_path / "outbox" / "database").glob("*.json"))  # kept for later


def test_logs_are_json_lines_with_extra_fields() -> None:
    record = logging.LogRecord("fim", logging.WARNING, "x.py", 1, "delivery failed", None, None)
    record.sink = "webhook"
    line = json.loads(JsonFormatter().format(record))
    assert line["level"] == "WARNING" and line["message"] == "delivery failed"
    assert line["sink"] == "webhook" and line["logger"] == "fim" and "time" in line


def test_fim_run_needs_cameras(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from fall_impact_monitor.cli.main import main

    monkeypatch.delenv("FIM_CAMERAS", raising=False)
    assert main(["run"]) == 2
    assert "FIM_CAMERAS" in capsys.readouterr().err
