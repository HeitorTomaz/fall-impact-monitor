import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from aiohttp import web

from fall_impact_monitor.sinks.webhook import WebhookSink
from tests.sinks.helpers import make_record


def with_server(
    statuses: list[int], body: Callable[[str, list[Any]], Awaitable[None]]
) -> list[Any]:
    """Run ``body(url, received)`` against a local server answering ``statuses`` in turn."""
    received: list[Any] = []

    async def handler(request: web.Request) -> web.Response:
        received.append(await request.json())
        return web.Response(status=statuses.pop(0) if statuses else 200)

    async def main() -> None:
        app = web.Application()
        app.router.add_post("/hook", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
        try:
            await body(f"http://127.0.0.1:{port}/hook", received)
        finally:
            await runner.cleanup()

    asyncio.run(main())
    return received


def test_a_notified_event_is_posted_as_json() -> None:
    record = make_record(0)

    async def body(url: str, received: list[Any]) -> None:
        sink = WebhookSink(url, timeout_s=5, late_after_s=60, clock=clock_at(2.0))
        await sink.deliver({"kind": "event", "record": record.to_json()})

    received = with_server([], body)
    assert received == [{"kind": "event", **record.to_json(), "late": False, "late_s": 2.0}]


def test_near_misses_are_not_posted_but_tests_are() -> None:
    sink = WebhookSink("http://unused", timeout_s=5, late_after_s=60)
    assert not sink.accepts({"kind": "event", "record": make_record(0, near_miss=True).to_json()})
    assert sink.accepts({"kind": "test"})
    assert not sink.accepts({"kind": "feedback"})


def test_a_server_error_raises_so_the_item_is_retried() -> None:
    async def body(url: str, received: list[Any]) -> None:
        sink = WebhookSink(url, timeout_s=5, late_after_s=60)
        with pytest.raises(ConnectionError, match="HTTP 500"):
            await sink.deliver({"kind": "test"})
        await sink.deliver({"kind": "test"})

    assert len(with_server([500], body)) == 2


def test_the_url_never_appears_in_errors() -> None:
    async def body() -> None:
        sink = WebhookSink("http://127.0.0.1:1/secret-path", timeout_s=1, late_after_s=60)
        with pytest.raises(ConnectionError) as info:
            await sink.deliver({"kind": "test"})
        assert "secret-path" not in str(info.value)

    asyncio.run(body())


def clock_at(*offsets_s: float) -> Any:
    """A clock that returns the record's end plus each offset in turn."""
    from datetime import timedelta

    ended = make_record(0).ended_utc
    times = [ended + timedelta(seconds=o) for o in offsets_s]
    return lambda: times.pop(0) if len(times) > 1 else times[0]


def test_an_alert_delivered_after_the_budget_is_marked_late() -> None:
    record = make_record(0).to_json()

    async def body(url: str, received: list[Any]) -> None:
        sink = WebhookSink(url, timeout_s=5, late_after_s=60, clock=clock_at(90.0))
        await sink.deliver({"kind": "event", "record": record})

    [payload] = with_server([200], body)
    assert payload["late"] is True and payload["late_s"] == 90.0


def test_an_alert_within_the_budget_is_not_late() -> None:
    record = make_record(0).to_json()

    async def body(url: str, received: list[Any]) -> None:
        sink = WebhookSink(url, timeout_s=5, late_after_s=60, clock=clock_at(10.0))
        await sink.deliver({"kind": "event", "record": record})

    [payload] = with_server([200], body)
    assert payload["late"] is False and payload["late_s"] == 10.0


def test_a_retry_after_the_budget_is_marked_late() -> None:
    record = make_record(0).to_json()

    async def body(url: str, received: list[Any]) -> None:
        sink = WebhookSink(url, timeout_s=5, late_after_s=60, clock=clock_at(30.0, 70.0))
        with pytest.raises(ConnectionError):
            await sink.deliver({"kind": "event", "record": record})
        await sink.deliver({"kind": "event", "record": record})  # the dispatcher retries

    first, retry = with_server([500, 200], body)
    assert first["late"] is False and retry["late"] is True


def test_delivery_tests_carry_no_late_marker() -> None:
    async def body(url: str, received: list[Any]) -> None:
        sink = WebhookSink(url, timeout_s=5, late_after_s=60, clock=clock_at(999.0))
        await sink.deliver({"kind": "test", "record": {"text": "test"}})

    [payload] = with_server([200], body)
    assert "late" not in payload and "late_s" not in payload
