from typing import Any

import numpy as np
import pytest

from fall_impact_monitor.capture.reader import StateEvent
from fall_impact_monitor.config import Settings
from fall_impact_monitor.events import EventCategory
from fall_impact_monitor.events.record import EventRecord
from fall_impact_monitor.service.camera import CameraWorker, DecodedFrame
from fall_impact_monitor.service.health import HealthLoop
from tests.service.test_camera_worker import FakeAnalyzer, FakePipeline, frame_record


class Clock:
    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class FakeDispatcher:
    def __init__(self) -> None:
        self.records: list[EventRecord] = []

    def submit(self, record: EventRecord) -> None:
        self.records.append(record)

    def oldest_pending_age_s(self) -> dict[str, float | None]:
        return {}


def setup(
    monkeypatch: pytest.MonkeyPatch, **env: str
) -> tuple[CameraWorker, HealthLoop, FakeDispatcher, Clock, list[int]]:
    for key, value in {
        "FIM_HEALTH_CAMERA_LOST_S": "30",
        "FIM_HEALTH_RESTORE_AFTER_S": "10",
        **env,
    }.items():
        monkeypatch.setenv(key, value)
    clock = Clock(100.0)
    worker = CameraWorker(
        "cam1",
        pipeline=FakePipeline(),
        analyzer=FakeAnalyzer([]),
        on_event=lambda r: None,
        config_hash="cfg",
        max_width_px=1280,
        monotonic=clock,
    )
    dispatcher = FakeDispatcher()
    exits: list[int] = []
    loop = HealthLoop(
        [worker],
        dispatcher,
        settings=Settings(_env_file=None),
        config_hash="cfg",
        monotonic=clock,
        disk_free_bytes=lambda: 10**12,
        exit=exits.append,
    )
    return worker, loop, dispatcher, clock, exits


def frame(arrival_s: float) -> DecodedFrame:
    return DecodedFrame(frame_record(arrival_s), np.zeros((36, 64, 3), np.uint8))


def test_a_camera_without_video_is_reported_and_then_restored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker, loop, dispatcher, clock, _ = setup(monkeypatch)
    worker.process(frame(100.0))
    clock.now = 125.0
    loop.check()
    assert dispatcher.records == []
    clock.now = 131.0
    worker.process(StateEvent("disconnected", "x"))  # the reader keeps trying
    loop.check()
    assert [r.event.category for r in dispatcher.records] == [EventCategory.MONITORING_DEGRADED]
    assert dispatcher.records[0].event.evidence == {"reasons": ["camera_lost"]}
    assert "cam1: DEGRADED" in loop.status_text()
    worker.process(frame(140.0))
    clock.now = 145.0
    loop.check()
    clock.now = 151.0
    loop.check()
    assert [r.event.category for r in dispatcher.records][-1] is EventCategory.MONITORING_RESTORED


def test_a_stuck_worker_makes_the_process_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    worker, loop, dispatcher, clock, exits = setup(monkeypatch, FIM_HEALTH_STALL_EXIT_S="180")
    worker.process(frame(100.0))
    clock.now = 281.0  # no frame and no stream state for 181 s: stuck in processing
    loop.check()
    assert exits == [70]
    assert any(r.event.evidence.get("reasons") == ["worker_stalled"] for r in dispatcher.records)


def test_an_offline_camera_does_not_trigger_the_watchdog(monkeypatch: pytest.MonkeyPatch) -> None:
    worker, loop, _, clock, exits = setup(monkeypatch, FIM_HEALTH_STALL_EXIT_S="180")
    for second in range(100, 400, 20):
        clock.now = float(second)
        worker.process(StateEvent("disconnected", "x"))  # reader backing off, still alive
        loop.check()
    assert exits == []


def test_low_disk_degrades_the_system(monkeypatch: pytest.MonkeyPatch) -> None:
    worker, loop, dispatcher, _, _ = setup(monkeypatch, FIM_RECORDINGS_DISK_MIN_FREE_GB="10")
    loop._disk_free_bytes = lambda: 1  # type: ignore[method-assign]
    worker.process(frame(100.0))
    loop.check()
    assert [r.event.camera_id for r in dispatcher.records] == ["system"]


def test_status_lists_every_component(monkeypatch: Any) -> None:
    _, loop, _, _, _ = setup(monkeypatch)
    assert "cam1: OK" in loop.status_text() and "system: OK" in loop.status_text()


def test_the_watchdog_exits_even_if_reporting_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    worker, loop, dispatcher, clock, exits = setup(monkeypatch, FIM_HEALTH_STALL_EXIT_S="180")

    def broken(record: EventRecord) -> None:
        raise RuntimeError("outbox unavailable")

    dispatcher.submit = broken  # type: ignore[method-assign]
    worker.process(frame(100.0))
    clock.now = 281.0
    loop.check()
    assert exits == [70]


def test_a_blocked_event_loop_makes_the_process_exit() -> None:
    from fall_impact_monitor.service.health import LoopWatchdog

    clock = Clock(100.0)
    exits: list[int] = []
    watchdog = LoopWatchdog(stall_s=60.0, monotonic=clock, exit=exits.append)
    watchdog.tick()
    clock.now = 150.0
    watchdog.check()
    assert exits == []
    clock.now = 161.0
    watchdog.check()
    assert exits == [70]


def test_worker_counts_reconnections_and_processing_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker, _, _, clock, _ = setup(monkeypatch)
    for _ in range(3):
        worker.process(StateEvent("connected", "x"))
    assert worker.probe(clock.now).reconnects == 2  # the first connection is not a reconnection
    worker._analyzer = None  # type: ignore[assignment]  # every frame now fails
    for second in range(5):
        worker.run([frame(100.0 + second)])
    assert worker.probe(clock.now).failures == 5


def restartable(
    monkeypatch: pytest.MonkeyPatch, state_file: Any, disk_free: list[int]
) -> tuple[HealthLoop, FakeDispatcher, Clock]:
    """A loop whose disk is low while ``disk_free[0]`` is 1; its state lives in a file."""
    monkeypatch.setenv("FIM_HEALTH_RESTORE_AFTER_S", "10")
    monkeypatch.setenv("FIM_RECORDINGS_DISK_MIN_FREE_GB", "10")
    monkeypatch.setenv("FIM_HEALTH_STATE_FILE", str(state_file))
    clock = Clock(100.0)
    dispatcher = FakeDispatcher()
    loop = HealthLoop(
        [],
        dispatcher,
        settings=Settings(_env_file=None),
        config_hash="cfg",
        monotonic=clock,
        disk_free_bytes=lambda: disk_free[0],
        exit=lambda code: None,
    )
    return loop, dispatcher, clock


def test_a_restart_does_not_announce_the_same_degraded_state_again(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    state, disk = tmp_path / "health_state.json", [1]
    first, sent, _ = restartable(monkeypatch, state, disk)
    first.check()
    assert [r.event.category for r in sent.records] == [EventCategory.MONITORING_DEGRADED]
    again, resent, clock = restartable(monkeypatch, state, disk)  # the service restarted
    again.check()
    assert resent.records == []
    disk[0] = 10**12  # the condition clears: "restored" still comes
    clock.now = 111.0
    again.check()
    assert [r.event.category for r in resent.records] == [EventCategory.MONITORING_RESTORED]


def test_a_corrupted_health_state_file_is_ignored(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    state = tmp_path / "health_state.json"
    state.write_text("{not json", encoding="utf-8")
    loop, sent, _ = restartable(monkeypatch, state, [1])
    loop.check()
    assert [r.event.category for r in sent.records] == [EventCategory.MONITORING_DEGRADED]


def test_an_unwritable_health_state_file_does_not_stop_monitoring(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    folder = tmp_path / "missing" / "deeper"
    loop, sent, _ = restartable(monkeypatch, folder / "state.json", [1])
    monkeypatch.setattr("os.replace", lambda *a: (_ for _ in ()).throw(OSError("read-only")))
    loop.check()
    assert [r.event.category for r in sent.records] == [EventCategory.MONITORING_DEGRADED]


def test_a_stall_exit_is_restored_after_the_restart(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    # The watchdog restarts the service itself: "restored" must still follow.
    monkeypatch.setenv("FIM_HEALTH_STATE_FILE", str(tmp_path / "health_state.json"))
    worker, loop, _, clock, exits = setup(monkeypatch, FIM_HEALTH_STALL_EXIT_S="180")
    worker.process(frame(100.0))
    clock.now = 281.0
    loop.check()
    assert exits == [70]
    worker, restarted, sent, clock, _ = setup(monkeypatch)  # Docker started it again
    worker.process(frame(clock.now))
    restarted.check()
    clock.now += 11.0
    worker.process(frame(clock.now))
    restarted.check()
    assert [r.event.category for r in sent.records] == [EventCategory.MONITORING_RESTORED]
