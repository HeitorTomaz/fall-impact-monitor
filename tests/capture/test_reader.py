from collections.abc import Iterator
from pathlib import Path
from typing import Any

import av

from fall_impact_monitor.capture import TimestampQuality
from fall_impact_monitor.capture.reader import (
    FrameEvent,
    ReconnectingReader,
    SourceOptions,
    StateEvent,
    demux_options,
    iter_frame_records,
)

RTSP = "rtsp://user:secret@cam.local:554/stream2"


class FakeClock:
    def __init__(self) -> None:
        self.ns = 0

    def __call__(self) -> int:
        self.ns += 40_000_000  # 40 ms per call
        return self.ns


def test_file_yields_one_record_per_video_packet(synthetic_video: Path) -> None:
    items = list(iter_frame_records(str(synthetic_video), "cam1", SourceOptions()))
    records = [record for record, _ in items]
    assert len(records) == 75
    assert [r.seq for r in records] == list(range(75))
    assert {r.quality for r in records} == {TimestampQuality.OK}
    assert sum(r.keyframe for r in records) == 3
    assert all(r.camera_id == "cam1" and r.size_bytes > 0 for r in records)
    assert all(bytes(packet) for _, packet in items)


def test_arrival_times_come_from_the_injected_clock(synthetic_video: Path) -> None:
    clock = FakeClock()
    records = [
        r
        for r, _ in iter_frame_records(
            str(synthetic_video), "cam1", SourceOptions(), clock_ns=clock
        )
    ]
    assert records[1].arrival_ns - records[0].arrival_ns == 40_000_000


def test_rtsp_demux_options() -> None:
    assert demux_options(RTSP, SourceOptions()) == {
        "rtsp_transport": "tcp",
        "allowed_media_types": "video",
    }
    assert demux_options(RTSP, SourceOptions(video_only=False, wallclock=True)) == {
        "rtsp_transport": "tcp",
        "use_wallclock_as_timestamps": "1",
    }
    assert demux_options("/media/file.mkv", SourceOptions()) == {}


class FlakyOpener:
    """Fails ``failures`` times, then opens ``path``; optionally stops after ``stop_after``."""

    def __init__(
        self,
        path: Path,
        failures: int = 0,
        stop_after: int | None = None,
        clock: FakeClock | None = None,
    ) -> None:
        self.path, self.failures, self.stop_after, self.clock = path, failures, stop_after, clock
        self.calls = 0

    def __call__(self, url: str, **kwargs: Any) -> Any:
        self.calls += 1
        if self.calls <= self.failures:
            raise OSError(f"Connection refused: {url}")
        container = av.open(str(self.path))
        if self.stop_after is None:
            return container
        return _StoppingContainer(container, self.stop_after, self.clock)


class _StoppingContainer:
    def __init__(self, inner: Any, stop_after: int, clock: FakeClock | None) -> None:
        self._inner, self._stop_after, self._clock = inner, stop_after, clock
        self.streams = inner.streams

    def demux(self, stream: Any) -> Iterator[Any]:
        for i, packet in enumerate(self._inner.demux(stream)):
            if i == self._stop_after:
                if self._clock is not None:
                    self._clock.ns += 20_000_000_000  # 20 s of silence
                raise TimeoutError("read timed out")
            yield packet

    def close(self) -> None:
        self._inner.close()


def collect(reader: ReconnectingReader, frames: int) -> list[FrameEvent | StateEvent]:
    events: list[FrameEvent | StateEvent] = []
    seen = 0
    for event in reader:
        events.append(event)
        seen += isinstance(event, FrameEvent)
        if seen >= frames:
            break
    reader.close()
    return events


def test_reconnects_with_exponential_backoff_and_masks_errors(synthetic_video: Path) -> None:
    sleeps: list[float] = []
    opener = FlakyOpener(synthetic_video, failures=2)
    reader = ReconnectingReader(RTSP, "cam1", SourceOptions(), opener=opener, sleep=sleeps.append)
    events = collect(reader, frames=5)
    states = [e for e in events if isinstance(e, StateEvent)]
    assert [s.state for s in states] == ["disconnected", "disconnected", "connected"]
    assert sleeps == [1.0, 2.0]
    assert all("secret" not in s.detail for s in states)
    assert "***@cam.local" in states[0].detail


def test_backoff_is_capped(synthetic_video: Path) -> None:
    sleeps: list[float] = []
    opener = FlakyOpener(synthetic_video, failures=7)
    reader = ReconnectingReader(
        RTSP, "cam1", SourceOptions(), opener=opener, sleep=sleeps.append, backoff_max_s=8.0
    )
    collect(reader, frames=1)
    assert sleeps == [1.0, 2.0, 4.0, 8.0, 8.0, 8.0, 8.0]


def test_silent_stream_is_reported_stale_then_reconnected(synthetic_video: Path) -> None:
    clock = FakeClock()
    opener = FlakyOpener(synthetic_video, stop_after=3, clock=clock)
    reader = ReconnectingReader(
        RTSP,
        "cam1",
        SourceOptions(),
        opener=opener,
        sleep=lambda s: None,
        clock_ns=clock,
        stale_timeout_s=10.0,
    )
    events = collect(reader, frames=5)
    states = [e.state for e in events if isinstance(e, StateEvent)]
    assert states == ["connected", "stale", "connected"]
    assert opener.calls == 2


def test_file_source_ends_without_reconnecting(synthetic_video: Path) -> None:
    opener = FlakyOpener(synthetic_video)
    reader = ReconnectingReader(
        str(synthetic_video), "cam1", SourceOptions(), opener=opener, sleep=lambda s: None
    )
    events = list(reader)
    assert isinstance(events[-1], StateEvent) and events[-1].state == "ended"
    assert sum(isinstance(e, FrameEvent) for e in events) == 75
    assert opener.calls == 1
