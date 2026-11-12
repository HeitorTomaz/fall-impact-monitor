from datetime import UTC, datetime, timedelta
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from fall_impact_monitor.api.perception import FrameResult
from fall_impact_monitor.capture import FrameRecord, TimestampQuality
from fall_impact_monitor.capture.reader import SourceOptions, StateEvent, iter_frame_records
from fall_impact_monitor.events import Event, EventCategory, make_event
from fall_impact_monitor.events.record import EventRecord
from fall_impact_monitor.service.camera import CameraWorker, DecodedFrame, analysis_url, decode
from fall_impact_monitor.service.timebase import to_utc

WALL = datetime(2026, 11, 9, 3, 0, 10, tzinfo=UTC)


def frame_record(arrival_s: float, seq: int = 0) -> FrameRecord:
    return FrameRecord(
        "cam1",
        seq,
        seq,
        Fraction(1, 90000),
        int(arrival_s * 1e9),
        WALL + timedelta(seconds=arrival_s - 100.0),
        False,
        100,
        TimestampQuality.OK,
    )


def test_arrival_seconds_map_to_utc_through_the_frame_anchor() -> None:
    anchor = frame_record(100.0)
    assert to_utc(95.5, anchor) == WALL - timedelta(seconds=4.5)
    assert to_utc(100.0, anchor) == WALL


class FakePipeline:
    def __init__(self) -> None:
        self.shapes: list[tuple[int, ...]] = []

    def process(self, image: Any, record: FrameRecord) -> FrameResult:
        self.shapes.append(image.shape)
        return FrameResult(record, [], analysed=True)


class FakeAnalyzer:
    def __init__(self, events: list[list[Event]]) -> None:
        self.events = events
        self.sizes: list[tuple[int, int]] = []

    def update(self, result: FrameResult, frame_size: tuple[int, int]) -> list[Event]:
        self.sizes.append(frame_size)
        return self.events.pop(0) if self.events else []


def fall(start_s: float, end_s: float) -> Event:
    return make_event(
        camera_id="cam1",
        category=EventCategory.POSSIBLE_FALL,
        confidence=0.8,
        start_s=start_s,
        end_s=end_s,
        track_id=1,
        identity="stable",
        evidence={},
        analysed_seqs=[],
        threshold=0.5,
    )


def test_an_event_becomes_a_record_with_utc_times_and_the_config_hash() -> None:
    records: list[EventRecord] = []
    event = fall(98.0, 99.5)
    worker = CameraWorker(
        "cam1",
        pipeline=FakePipeline(),
        analyzer=FakeAnalyzer([[event]]),
        on_event=records.append,
        config_hash="cfg",
        max_width_px=1280,
    )
    worker.process(DecodedFrame(frame_record(100.0), np.zeros((360, 640, 3), np.uint8)))
    assert len(records) == 1
    record = records[0]
    assert record.event is event and record.config_hash == "cfg"
    assert record.started_utc == WALL - timedelta(seconds=2)
    assert record.ended_utc == WALL - timedelta(seconds=0.5)
    assert record.detected_utc == WALL


def test_wide_frames_are_downscaled_before_analysis() -> None:
    pipeline, analyzer = FakePipeline(), FakeAnalyzer([])
    worker = CameraWorker(
        "cam1",
        pipeline=pipeline,
        analyzer=analyzer,
        on_event=lambda r: None,
        config_hash="cfg",
        max_width_px=1280,
    )
    worker.process(DecodedFrame(frame_record(100.0), np.zeros((1440, 2560, 3), np.uint8)))
    assert pipeline.shapes == [(720, 1280, 3)]
    assert analyzer.sizes == [(1280, 720)]


def test_status_follows_the_stream_state() -> None:
    worker = CameraWorker(
        "cam1",
        pipeline=FakePipeline(),
        analyzer=FakeAnalyzer([]),
        on_event=lambda r: None,
        config_hash="cfg",
        max_width_px=1280,
        monotonic=lambda: 100.5,
    )
    assert "cam1: starting" in worker.status_line()
    worker.process(StateEvent("connected", "rtsp://mediamtx:8554/cam1sub"))
    worker.process(DecodedFrame(frame_record(100.0), np.zeros((36, 64, 3), np.uint8)))
    line = worker.status_line()
    assert line.startswith("cam1: connected") and "last frame 0 s ago" in line
    worker.process(StateEvent("disconnected", "boom"))
    assert worker.status_line().startswith("cam1: disconnected")


def test_frames_are_decoded_from_reader_packets(synthetic_video: Path) -> None:
    from fall_impact_monitor.capture.reader import FrameEvent

    events = (
        FrameEvent(r, p)
        for r, p in iter_frame_records(str(synthetic_video), "cam1", SourceOptions())
    )
    frames = [f for f in decode(events) if isinstance(f, DecodedFrame)]
    assert len(frames) >= 70
    assert frames[0].image.shape == (360, 640, 3)


def test_analysis_uses_the_substream_when_there_is_one() -> None:
    from fall_impact_monitor.config import CameraSettings

    assert (
        analysis_url("rtsp://mediamtx:8554", CameraSettings("cam1", "rtsp://c/1", "rtsp://c/2"))
        == "rtsp://mediamtx:8554/cam1sub"
    )
    assert (
        analysis_url("rtsp://mediamtx:8554/", CameraSettings("cam2", "rtsp://c/1", None))
        == "rtsp://mediamtx:8554/cam2"
    )


class _BadPacket:
    def decode(self) -> list[Any]:
        raise ValueError("corrupt packet")


def test_a_corrupt_packet_is_skipped_and_decoding_goes_on(synthetic_video: Path) -> None:
    from fall_impact_monitor.capture.reader import FrameEvent

    good = [
        FrameEvent(r, p)
        for r, p in iter_frame_records(str(synthetic_video), "cam1", SourceOptions())
    ]
    events = [FrameEvent(good[0].record, _BadPacket()), *good]  # type: ignore[arg-type]
    frames = [f for f in decode(events) if isinstance(f, DecodedFrame)]
    assert len(frames) >= 70


def test_the_worker_reopens_its_source_after_a_failure() -> None:
    opened: list[int] = []

    def source() -> Any:
        opened.append(1)
        if len(opened) == 1:
            raise OSError("stream lost")
        worker.stop()
        yield StateEvent("connected", "x")

    worker = CameraWorker(
        "cam1",
        pipeline=FakePipeline(),
        analyzer=FakeAnalyzer([]),
        on_event=lambda r: None,
        config_hash="cfg",
        max_width_px=1280,
    )
    worker.run_forever(source, retry_s=0.0)
    assert len(opened) == 2


def test_the_worker_asks_for_full_rate_analysis_when_the_detectors_need_it() -> None:
    class AttentivePipeline(FakePipeline):
        def __init__(self) -> None:
            super().__init__()
            self.attention: list[float] = []

        def request_attention(self, until_s: float) -> None:
            self.attention.append(until_s)

    class NearHazard(FakeAnalyzer):
        def attention_until(self) -> float | None:
            return 102.0

    pipeline = AttentivePipeline()
    worker = CameraWorker(
        "cam1",
        pipeline=pipeline,
        analyzer=NearHazard([]),
        on_event=lambda r: None,
        config_hash="cfg",
        max_width_px=1280,
    )
    worker.process(DecodedFrame(frame_record(100.0), np.zeros((36, 64, 3), np.uint8)))
    assert pipeline.attention == [102.0]


def test_the_worker_closes_its_source_when_it_stops() -> None:
    class Source:
        closed = False

        def __iter__(self) -> Any:
            worker.stop()
            yield StateEvent("connected", "x")
            yield StateEvent("stale", "x")

        def close(self) -> None:
            self.closed = True

    worker = CameraWorker(
        "cam1",
        pipeline=FakePipeline(),
        analyzer=FakeAnalyzer([]),
        on_event=lambda r: None,
        config_hash="cfg",
        max_width_px=1280,
    )
    source = Source()
    worker.run(source)
    assert source.closed
