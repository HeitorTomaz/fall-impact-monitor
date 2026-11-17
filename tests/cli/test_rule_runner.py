from datetime import UTC, datetime
from fractions import Fraction

from fall_impact_monitor.api.perception import FrameResult
from fall_impact_monitor.capture import FrameRecord, TimestampQuality
from fall_impact_monitor.cli.analyze import RuleRunner
from fall_impact_monitor.config import Settings
from fall_impact_monitor.events import Event, EventCategory


def fall(event_id: str, start_s: float, end_s: float, *, near_miss: bool) -> Event:
    return Event(
        event_id, "cam1", EventCategory.POSSIBLE_FALL, 0.3 if near_miss else 0.8,
        start_s, end_s, 0, "stable", near_miss,
    )  # fmt: skip


class ScriptedDetectors:
    """Returns the scripted events at the given frame times."""

    def __init__(self, script: dict[float, list[Event]]) -> None:
        self._script = script

    def update(self, result: FrameResult, size: tuple[int, int]) -> list[Event]:
        return self._script.get(round(result.record.arrival_ns / 1e9, 3), [])

    def flush(self) -> list[Event]:
        return []


def frame(t_s: float) -> FrameResult:
    record = FrameRecord(
        "cam1", int(t_s * 10), int(t_s * 90000), Fraction(1, 90000), int(t_s * 1e9),
        datetime(2026, 1, 1, tzinfo=UTC), False, 100, TimestampQuality.OK,
    )  # fmt: skip
    return FrameResult(record, [], analysed=True)


def run(script: dict[float, list[Event]], seen: list[Event]) -> list[str]:
    written: list[str] = []
    runner = RuleRunner(
        Settings(),
        ScriptedDetectors(script),  # type: ignore[arg-type]
        lambda event, emitted_s: written.append(event.event_id),
        lambda start_ns, end_ns: seen,
    )
    for i in range(100):
        runner.frame(frame(i / 10), (640, 480))
    runner.close()
    return written


def test_a_reanalysis_alert_for_a_fall_already_alerted_live_is_not_written() -> None:
    """The service drops it (`_already_alerted`); analysis and replay must too, or the
    evaluation counts an alert nobody would receive (#103)."""
    script = {
        1.0: [fall("near", 1.0, 1.5, near_miss=True)],
        2.0: [fall("live", 1.8, 2.0, near_miss=False)],
    }
    written = run(script, [fall("again", 1.2, 1.6, near_miss=False)])
    assert written == ["near", "live"]


def test_a_reanalysis_alert_for_a_fall_not_alerted_live_is_written() -> None:
    script = {1.0: [fall("near", 1.0, 1.5, near_miss=True)]}
    written = run(script, [fall("seen", 1.2, 1.6, near_miss=False)])
    assert written == ["near", "seen"]
