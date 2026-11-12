from datetime import UTC, datetime, timedelta
from typing import Any

from fall_impact_monitor.events import Event, EventCategory, make_event
from fall_impact_monitor.events.record import EventRecord
from fall_impact_monitor.service.confirmation import ConfirmationWorker

T_END = datetime(2026, 11, 12, 3, 0, 0, tzinfo=UTC)


def fall(start_s: float, end_s: float, *, near_miss: bool, evidence: Any = None) -> Event:
    return make_event(
        camera_id="cam1",
        category=EventCategory.POSSIBLE_FALL,
        confidence=0.3 if near_miss else 0.8,
        start_s=start_s,
        end_s=end_s,
        track_id=1,
        identity="stable",
        evidence=evidence if evidence is not None else {},
        analysed_seqs=[],
        threshold=0.5,
    )


def candidate(evidence: Any = None) -> EventRecord:
    event = fall(10.0, 11.0, near_miss=True, evidence=evidence or {"near_fall": True})
    return EventRecord(event, T_END - timedelta(seconds=1), T_END, T_END, "cfg")


class FakeBuffer:
    def __init__(self, newest_s: float) -> None:
        self.newest_s = newest_s
        self.windows: list[tuple[int, int]] = []

    def newest_arrival_ns(self) -> int | None:
        return int(self.newest_s * 1e9)

    def decode_window(self, start_ns: int, end_ns: int) -> list[Any]:
        self.windows.append((start_ns, end_ns))
        return ["frame"]


class FakeConfirmer:
    def __init__(self, events: list[Event]) -> None:
        self.events = events
        self.frames: list[Any] = []

    def confirm(self, frames: Any) -> list[Event]:
        self.frames = list(frames)
        return self.events


def worker(
    buffer: FakeBuffer, confirmer: FakeConfirmer, sent: list[EventRecord], **kw: Any
) -> ConfirmationWorker:
    return ConfirmationWorker(
        lambda camera: confirmer,
        {"cam1": buffer},
        sent.append,
        before_s=2.0,
        after_s=3.0,
        clock=lambda: T_END + timedelta(seconds=9),
        **kw,
    )


def test_a_confirmed_fall_is_sent_with_utc_times_from_the_candidate() -> None:
    buffer, sent = FakeBuffer(newest_s=20.0), []
    confirmed = fall(10.2, 10.8, near_miss=False, evidence={"reanalysed_at_full_rate": True})
    w = worker(buffer, FakeConfirmer([confirmed]), sent)
    assert w.request(candidate())
    w.process_next(timeout_s=0)
    assert buffer.windows == [(int(8.0e9), int(14.0e9))]  # 2 s before to 3 s after
    [record] = sent
    assert record.event is confirmed
    assert record.started_utc == T_END - timedelta(seconds=0.8)  # end_s 11.0 is T_END
    assert record.ended_utc == T_END - timedelta(seconds=0.2)
    assert record.detected_utc == T_END + timedelta(seconds=9)
    assert record.config_hash == "cfg"


def test_nothing_is_sent_when_the_fall_is_not_confirmed() -> None:
    sent: list[EventRecord] = []
    w = worker(FakeBuffer(newest_s=20.0), FakeConfirmer([]), sent)
    w.request(candidate())
    w.process_next(timeout_s=0)
    assert sent == []


def test_it_waits_for_the_frames_after_the_candidate() -> None:
    buffer, sent, slept = FakeBuffer(newest_s=12.0), [], []

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        buffer.newest_s += 1.0  # the camera keeps streaming

    w = worker(buffer, FakeConfirmer([]), sent, sleep=sleep)
    w.request(candidate())
    w.process_next(timeout_s=0)
    assert buffer.newest_s >= 14.0 and slept
    assert buffer.windows == [(int(8.0e9), int(14.0e9))]


def test_it_does_not_wait_forever_for_a_camera_that_stopped() -> None:
    buffer, sent, slept = FakeBuffer(newest_s=11.5), [], []
    w = worker(buffer, FakeConfirmer([]), sent, sleep=slept.append)
    w.request(candidate())
    w.process_next(timeout_s=0)
    assert sum(slept) <= 3.0 + 5.0 + 0.5  # after_s plus a grace, then what exists
    assert buffer.windows


def test_requests_never_block_and_a_full_queue_refuses() -> None:
    w = worker(FakeBuffer(newest_s=20.0), FakeConfirmer([]), [], max_queue=1)
    assert w.request(candidate())
    assert not w.request(candidate())  # the near-miss itself was already recorded


def test_only_fall_near_misses_outside_safe_zones_are_confirmed() -> None:
    w = worker(FakeBuffer(newest_s=20.0), FakeConfirmer([]), [])
    alert = EventRecord(fall(10.0, 11.0, near_miss=False), T_END, T_END, T_END, "cfg")
    assert not w.request(alert)
    assert not w.request(candidate({"near_fall": True, "safe_zone": "bed"}))


def test_a_camera_without_a_buffer_is_skipped() -> None:
    w = ConfirmationWorker(
        lambda camera: FakeConfirmer([]),
        {},
        lambda r: None,
        before_s=2.0,
        after_s=3.0,
    )
    assert not w.request(candidate())
