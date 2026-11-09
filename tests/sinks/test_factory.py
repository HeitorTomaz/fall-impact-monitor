import asyncio
import json
from pathlib import Path

import pytest

from fall_impact_monitor.cli.main import main
from fall_impact_monitor.config import Settings
from fall_impact_monitor.recording.clips import find_clip
from fall_impact_monitor.sinks.factory import build_dispatcher
from tests.database import URL, needs_database
from tests.sinks.helpers import make_record
from tests.sinks.test_telegram import FakeApi


def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **env: str) -> Settings:
    monkeypatch.setenv("FIM_OUTBOX_DIR", str(tmp_path / "outbox"))
    monkeypatch.setenv("FIM_EVENT_LOG_FILE", str(tmp_path / "events.jsonl"))
    monkeypatch.setenv("FIM_EVENT_CLIP_DIR", str(tmp_path / "clips"))
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return Settings(_env_file=None)


def test_only_the_json_log_runs_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dispatcher, outputs = build_dispatcher(settings(tmp_path, monkeypatch), engine=None)
    assert [s.name for s in dispatcher.sinks] == ["event_log"]
    assert outputs.telegram == {}


def test_enabled_outputs_are_built(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    s = settings(
        tmp_path,
        monkeypatch,
        FIM_WEBHOOK_URL="http://hooks.invalid/x",
        FIM_TELEGRAM_ENABLED="true",
        FIM_TELEGRAM_BOT_TOKEN="1:t",
        FIM_TELEGRAM_CHAT_IDS="-1,-2",
    )
    dispatcher, outputs = build_dispatcher(s, engine=None, telegram_api=FakeApi())
    names = [x.name for x in dispatcher.sinks]
    assert names[:2] == ["event_log", "webhook"]
    assert len(names) == 4 and all(n.startswith("telegram:") for n in names[2:])
    assert sorted(outputs.telegram) == [-2, -1]


def test_alert_test_command_reports_each_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    settings(tmp_path, monkeypatch)
    assert main(["alert", "test", "--timeout-s", "5"]) == 0
    out = capsys.readouterr().out
    assert "event_log: delivered" in out
    lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["kind"] == "test"


@needs_database
def test_a_synthetic_event_reaches_every_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, database: str
) -> None:
    from sqlalchemy import text

    from fall_impact_monitor.storage import make_engine

    s = settings(
        tmp_path,
        monkeypatch,
        FIM_TELEGRAM_ENABLED="true",
        FIM_TELEGRAM_BOT_TOKEN="1:t",
        FIM_TELEGRAM_CHAT_IDS="-1",
    )
    api = FakeApi()
    record = make_record(0)

    async def scenario() -> list[tuple[str, str]]:
        engine = make_engine(URL)
        dispatcher, _ = build_dispatcher(s, engine=engine, telegram_api=api)
        task = asyncio.create_task(dispatcher.run())
        await asyncio.sleep(0)
        dispatcher.submit(record)
        for _ in range(500):
            if not any(dispatcher.pending_counts().values()) and api.calls:
                break
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
        for _ in range(500):  # the Telegram notification record follows the event
            if not any(dispatcher.pending_counts().values()):
                break
            await asyncio.sleep(0.01)
        task.cancel()
        async with engine.connect() as conn:
            rows = (await conn.execute(text("SELECT state, config_hash FROM events"))).all()
        await engine.dispose()
        return [tuple(r) for r in rows]

    assert asyncio.run(scenario()) == [("notified", "cfg")]
    assert api.calls[0][0] == "send"
    assert (tmp_path / "events.jsonl").exists()
    clip = (
        tmp_path / "clips" / "cam1" / "2026" / "11" / "09" / f"030000_{record.event.event_id}.mp4"
    )
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"mp4")
    clip.with_suffix(".json").write_text("{}")
    assert find_clip(tmp_path / "clips", record.event.event_id) == clip


def test_alert_test_leaves_the_service_outbox_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fall_impact_monitor.sinks.outbox import Outbox

    settings(tmp_path, monkeypatch)
    live = Outbox(tmp_path / "outbox")
    live.put("event_log", {"kind": "event", "record": make_record(0).to_json()})
    assert main(["alert", "test", "--timeout-s", "5"]) == 0
    assert live.count("event_log") == 1  # the running service delivers its own queue
