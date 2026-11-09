import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from fall_impact_monitor.sinks.base import RetryLater
from fall_impact_monitor.sinks.dispatcher import Dispatcher
from fall_impact_monitor.sinks.event_log import JsonLogSink
from fall_impact_monitor.sinks.outbox import Outbox
from tests.sinks.helpers import make_record


class RecordingSink:
    def __init__(self, name: str, failures: int = 0, retry_after: float | None = None) -> None:
        self.name = name
        self.failures = failures
        self.retry_after = retry_after
        self.delivered: list[dict[str, Any]] = []

    def accepts(self, item: dict[str, Any]) -> bool:
        return item["kind"] in ("event", "test")

    async def deliver(self, item: dict[str, Any]) -> None:
        if self.failures:
            self.failures -= 1
            if self.retry_after is not None:
                raise RetryLater(self.retry_after)
            raise ConnectionError("unreachable")
        self.delivered.append(item)


def run_until(
    dispatcher: Dispatcher, done: Callable[[], bool], before: Callable[[], None] | None = None
) -> None:
    async def main() -> None:
        task = asyncio.create_task(dispatcher.run())
        await asyncio.sleep(0)
        if before is not None:
            before()
        for _ in range(500):
            if done():
                break
            await asyncio.sleep(0.01)
        task.cancel()

    asyncio.run(main())


def fast_sleep(delays: list[float]) -> Callable[[float], Awaitable[None]]:
    async def sleep(seconds: float) -> None:
        delays.append(seconds)
        await asyncio.sleep(0)

    return sleep


def test_items_are_delivered_in_order_and_removed(tmp_path: Path) -> None:
    sink = RecordingSink("a")
    dispatcher = Dispatcher(Outbox(tmp_path), [sink])
    records = [make_record(i) for i in range(3)]

    def submit() -> None:
        for r in records:
            dispatcher.submit(r)

    run_until(dispatcher, lambda: len(sink.delivered) == 3, before=submit)
    ids = [item["record"]["event_id"] for item in sink.delivered]
    assert ids == [r.event.event_id for r in records]
    assert dispatcher.pending_counts() == {"a": 0}


def test_a_failing_sink_retries_with_backoff_and_does_not_delay_others(tmp_path: Path) -> None:
    good, bad = RecordingSink("good"), RecordingSink("bad", failures=3)
    delays: list[float] = []
    dispatcher = Dispatcher(Outbox(tmp_path), [good, bad], sleep=fast_sleep(delays))
    run_until(
        dispatcher,
        lambda: len(bad.delivered) == 1,
        before=lambda: dispatcher.submit(make_record(0)),
    )
    assert len(good.delivered) == 1
    assert delays == [1.0, 2.0, 4.0]


def test_flood_control_waits_as_told(tmp_path: Path) -> None:
    sink = RecordingSink("telegram", failures=1, retry_after=7.0)
    delays: list[float] = []
    dispatcher = Dispatcher(Outbox(tmp_path), [sink], sleep=fast_sleep(delays))
    run_until(
        dispatcher,
        lambda: len(sink.delivered) == 1,
        before=lambda: dispatcher.submit(make_record(0)),
    )
    assert delays == [7.0]


def test_pending_items_survive_a_restart_and_are_delivered_once(tmp_path: Path) -> None:
    first = Dispatcher(Outbox(tmp_path), [RecordingSink("a")])
    for i in range(2):
        first.submit(make_record(i))  # never run: the process "stops" here
    assert first.pending_counts() == {"a": 2}
    sink = RecordingSink("a")
    second = Dispatcher(Outbox(tmp_path), [sink])
    run_until(second, lambda: len(sink.delivered) == 2)
    third_sink = RecordingSink("a")
    third = Dispatcher(Outbox(tmp_path), [third_sink])
    run_until(third, lambda: False)
    assert third_sink.delivered == []


def test_sinks_with_unsafe_names_get_their_own_folder(tmp_path: Path) -> None:
    outbox = Outbox(tmp_path)
    outbox.put("telegram:-100123", {"kind": "test"})
    assert [p.name for p in tmp_path.iterdir()] == ["telegram_-100123"]
    assert len(outbox.pending("telegram:-100123")) == 1


def test_the_json_log_keeps_every_event_including_near_misses(tmp_path: Path) -> None:
    log = tmp_path / "events" / "events.jsonl"
    sink = JsonLogSink(log)
    dispatcher = Dispatcher(Outbox(tmp_path / "outbox"), [sink])

    def submit() -> None:
        dispatcher.submit(make_record(0))
        dispatcher.submit(make_record(1, near_miss=True))

    run_until(
        dispatcher,
        lambda: (
            dispatcher.pending_counts()["event_log"] == 0
            and log.exists()
            and len(log.read_text().splitlines()) == 2
        ),
        before=submit,
    )
    lines = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [line["near_miss"] for line in lines] == [False, True]
    assert all(line["kind"] == "event" and "logged_utc" in line for line in lines)
