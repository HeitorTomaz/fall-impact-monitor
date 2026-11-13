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

    def snapshot(self, start_ns: int, end_ns: int) -> Any:
        self.windows.append((start_ns, end_ns))
        return FakeWindow()


class FakeWindow:
    def decode(self) -> list[Any]:
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


def test_the_started_worker_keeps_running_and_confirms() -> None:
    import threading

    buffer, sent, done = FakeBuffer(newest_s=20.0), [], threading.Event()
    confirmed = fall(10.2, 10.8, near_miss=False, evidence={"reanalysed_at_full_rate": True})

    def on_event(record: EventRecord) -> None:
        sent.append(record)
        done.set()

    w = ConfirmationWorker(
        lambda camera: FakeConfirmer([confirmed]),
        {"cam1": buffer},
        on_event,
        before_s=2.0,
        after_s=3.0,
    )
    w.start()
    try:
        import time

        time.sleep(0.2)  # the threads must still be waiting for work
        assert w.request(candidate())
        assert done.wait(5)
    finally:
        w.stop()
    assert len(sent) == 1


def test_packets_are_copied_before_a_long_analysis_of_another_candidate() -> None:
    # The buffer keeps 20 s; a candidate queued behind a 15 s analysis must not lose
    # the start of its window.
    import threading
    import time

    buffer, release = FakeBuffer(newest_s=60.0), threading.Event()

    class SlowConfirmer(FakeConfirmer):
        def confirm(self, frames: Any) -> list[Event]:
            release.wait(5)
            return []

    w = ConfirmationWorker(
        lambda camera: SlowConfirmer([]),
        {"cam1": buffer},
        lambda r: None,
        before_s=2.0,
        after_s=3.0,
    )
    w.start()
    try:
        assert w.request(candidate())
        assert w.request(candidate())
        deadline = time.monotonic() + 5
        while len(buffer.windows) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(buffer.windows) == 2  # both copied while the first is still analysed
    finally:
        release.set()
        w.stop()


def test_a_fall_already_alerted_live_is_not_sent_again() -> None:
    sent: list[EventRecord] = []
    confirmed = fall(10.2, 10.8, near_miss=False, evidence={"reanalysed_at_full_rate": True})
    w = worker(FakeBuffer(newest_s=20.0), FakeConfirmer([confirmed]), sent)
    live = EventRecord(fall(11.5, 12.0, near_miss=False), T_END, T_END, T_END, "cfg")
    w.note_alert(live)  # the live analysis raised the same fall a moment later
    w.request(candidate())
    w.process_next(timeout_s=0)
    assert sent == []


def test_a_separate_fall_is_still_sent() -> None:
    sent: list[EventRecord] = []
    confirmed = fall(10.2, 10.8, near_miss=False, evidence={"reanalysed_at_full_rate": True})
    w = worker(FakeBuffer(newest_s=20.0), FakeConfirmer([confirmed]), sent)
    w.note_alert(EventRecord(fall(100.0, 101.0, near_miss=False), T_END, T_END, T_END, "cfg"))
    w.request(candidate())
    w.process_next(timeout_s=0)
    assert len(sent) == 1


def test_the_service_creates_the_buffers_it_confirms_from() -> None:
    from fall_impact_monitor.config import CameraSettings, Settings
    from fall_impact_monitor.service.app import confirmation_worker

    class Sharing:
        def process(self, image: Any, record: Any) -> Any:
            raise AssertionError("not used here")

        def sharing_models(self, tracker: Any) -> "Sharing":
            return self

    buffers: dict[str, Any] = {}  # make_source has not run yet when the service wires this
    camera = CameraSettings("cam1", "rtsp://h/1", None)
    w = confirmation_worker(
        Settings(_env_file=None), [camera], {"cam1": Sharing()}, buffers, lambda r: None
    )
    assert w is not None and "cam1" in buffers
