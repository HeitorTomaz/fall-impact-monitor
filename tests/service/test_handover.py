import time
from collections.abc import Iterator
from datetime import UTC, datetime
from fractions import Fraction

import numpy as np
import pytest

from fall_impact_monitor.capture import FrameRecord, TimestampQuality
from fall_impact_monitor.capture.reader import StateEvent
from fall_impact_monitor.service.camera import DecodedFrame
from fall_impact_monitor.service.handover import LatestFrames

FPS = 100.0


def frame(seq: int, arrival_ns: int = 0) -> DecodedFrame:
    record = FrameRecord(
        "cam1",
        seq,
        seq,
        Fraction(1, 100),
        arrival_ns,
        datetime(2026, 11, 10, tzinfo=UTC),
        False,
        100,
        TimestampQuality.OK,
    )
    return DecodedFrame(record, np.zeros((4, 4, 3), np.uint8))


def live_stream(frames: int) -> Iterator[DecodedFrame]:
    """A camera: frame i exists only from start + i / FPS, and is stamped when read."""
    start = time.monotonic()
    for i in range(frames):
        delay = start + i / FPS - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        yield frame(i, time.monotonic_ns())


def test_a_slow_analysis_stays_in_real_time_and_skips_frames() -> None:
    analysis_s = 0.05  # five frame intervals per analysis
    handover = LatestFrames(live_stream(100), "cam1")
    lags_s, seqs = [], []
    for item in handover:
        assert isinstance(item, DecodedFrame)
        capture_s = item.record.seq / FPS
        lags_s.append(time.monotonic() - handover.started_s - capture_s)
        seqs.append(item.record.seq)
        time.sleep(analysis_s)
    assert seqs == sorted(seqs) and seqs[-1] == 99
    assert len(seqs) < 50 and handover.skipped == 100 - len(seqs)
    assert max(lags_s) < 3 * analysis_s  # no backlog builds up


def test_without_the_handover_a_slow_analysis_falls_behind() -> None:
    # The behaviour this replaces: reading only when analysis asks builds a backlog.
    start = time.monotonic()
    lags_s = []
    for item in live_stream(30):
        lags_s.append(time.monotonic() - start - item.record.seq / FPS)
        time.sleep(0.05)
    assert lags_s[-1] > 1.0


def test_stream_states_are_never_dropped_and_keep_their_order() -> None:
    def source() -> Iterator[DecodedFrame | StateEvent]:
        yield frame(0)
        yield StateEvent("connected", "")
        yield frame(1)
        yield frame(2)
        yield StateEvent("stale", "")
        yield frame(3)

    handover = LatestFrames(source(), "cam1")
    time.sleep(0.2)  # the reader runs ahead of a consumer that has not started yet
    items = list(handover)
    order = [i.state if isinstance(i, StateEvent) else i.record.seq for i in items]
    assert order == [0, "connected", 2, "stale", 3]
    assert handover.skipped == 1


def test_an_error_while_reading_reaches_the_analysis_after_the_frames_before_it() -> None:
    def source() -> Iterator[DecodedFrame]:
        yield frame(0)
        raise ConnectionError("stream lost")

    handover = LatestFrames(source(), "cam1")
    got = []
    with pytest.raises(ConnectionError):
        for item in handover:
            got.append(item)
    assert len(got) == 1


def test_closing_stops_the_reader() -> None:
    handover = LatestFrames(live_stream(10_000), "cam1")
    next(iter(handover))
    handover.close()
    handover.join(timeout_s=1.0)
    assert not handover.reading


def test_the_service_reads_each_camera_through_the_handover() -> None:
    from fall_impact_monitor.config import CameraSettings, Settings
    from fall_impact_monitor.service.app import default_parts

    parts = default_parts(Settings(_env_file=None))
    source = parts.make_source(CameraSettings("cam1", "publisher", None))
    try:
        assert isinstance(source, LatestFrames)
    finally:
        source.close()  # type: ignore[attr-defined]
