import json
from datetime import UTC, datetime, timedelta

import pytest

from fall_impact_monitor.recording import Span, gaps, list_spans, wait_for_coverage
from tests.conftest import FakePlayback

T0 = datetime(2026, 11, 9, 3, 0, 0, tzinfo=UTC)


def listing(*items: tuple[str, float]) -> bytes:
    return json.dumps([{"start": s, "duration": d, "url": "x"} for s, d in items]).encode()


def test_list_spans_parses_offsets_into_utc_spans(playback: FakePlayback) -> None:
    playback.routes["/list"] = lambda q: (
        200,
        listing(("2026-11-09T03:00:00Z", 60.0), ("2026-11-09T00:02:00-03:00", 30.5)),
    )
    spans = list_spans(playback.url, "cam1", T0, T0 + timedelta(minutes=5))
    assert spans == [
        Span(T0, T0 + timedelta(seconds=60)),
        Span(T0 + timedelta(minutes=2), T0 + timedelta(minutes=2, seconds=30.5)),
    ]
    assert all(s.start.tzinfo is UTC for s in spans)
    path, query = playback.requests[-1]
    assert path == "/list"
    assert query["path"] == "cam1"
    assert query["start"] == "2026-11-09T03:00:00Z"


def test_list_spans_accepts_nanosecond_timestamps(playback: FakePlayback) -> None:
    playback.routes["/list"] = lambda q: (200, listing(("2026-11-09T03:00:00.123456789Z", 1.0)))
    spans = list_spans(playback.url, "cam1")
    assert spans[0].start == T0 + timedelta(microseconds=123456)


def test_nothing_recorded_is_an_empty_list(playback: FakePlayback) -> None:
    assert list_spans(playback.url, "cam1") == []  # MediaMTX answers 404


def test_naive_bounds_are_rejected(playback: FakePlayback) -> None:
    with pytest.raises(ValueError, match="timezone"):
        list_spans(playback.url, "cam1", datetime(2026, 11, 9))


def test_gaps_lists_uncovered_parts_of_the_window() -> None:
    end = T0 + timedelta(seconds=100)
    full = [Span(T0 - timedelta(seconds=10), end + timedelta(seconds=10))]
    assert gaps(full, T0, end) == []
    holed = [
        Span(T0, T0 + timedelta(seconds=40)),
        Span(T0 + timedelta(seconds=45), T0 + timedelta(seconds=70)),
        Span(T0 + timedelta(seconds=71), end),
    ]
    assert gaps(holed, T0, end) == [Span(T0 + timedelta(seconds=40), T0 + timedelta(seconds=45))]
    assert gaps([], T0, end) == [Span(T0, end)]
    late = [Span(T0 + timedelta(seconds=30), T0 + timedelta(seconds=60))]
    assert gaps(late, T0, end) == [
        Span(T0, T0 + timedelta(seconds=30)),
        Span(T0 + timedelta(seconds=60), end),
    ]


class FakeClock:
    def __init__(self) -> None:
        self.now = T0

    def __call__(self) -> datetime:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


def test_wait_for_coverage_returns_once_the_recording_reaches_the_target(
    playback: FakePlayback,
) -> None:
    clock = FakeClock()
    # The recording grows with the (fake) clock: always 2 s behind it.
    playback.routes["/list"] = lambda q: (
        200,
        listing(("2026-11-09T02:59:00Z", (clock.now - T0).total_seconds() + 58.0)),
    )
    until = T0 + timedelta(seconds=10)
    assert wait_for_coverage(
        playback.url, "cam1", until, timeout_s=60, poll_s=2, clock=clock, sleep=clock.sleep
    )
    assert clock.now >= T0 + timedelta(seconds=12)
    assert clock.now <= T0 + timedelta(seconds=14)


def test_wait_for_coverage_gives_up_at_the_timeout(playback: FakePlayback) -> None:
    clock = FakeClock()
    assert not wait_for_coverage(
        playback.url, "cam1", T0, timeout_s=30, poll_s=5, clock=clock, sleep=clock.sleep
    )
    assert clock.now == T0 + timedelta(seconds=30)
