import dataclasses
from datetime import UTC, datetime, timedelta
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest

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


def test_a_reconnection_restarts_the_analysis_rate_window() -> None:
    class ResumablePipeline(FakePipeline):
        def __init__(self) -> None:
            super().__init__()
            self.resumed: list[float] = []

        def resume(self, now_s: float) -> None:
            self.resumed.append(now_s)

    pipeline = ResumablePipeline()
    worker = CameraWorker(
        "cam1",
        pipeline=pipeline,
        analyzer=FakeAnalyzer([]),
        on_event=lambda r: None,
        config_hash="cfg",
        max_width_px=1280,
        monotonic=lambda: 42.0,
    )
    worker.process(StateEvent("connected", "x"))
    worker.process(StateEvent("stale", "x"))
    assert pipeline.resumed == [42.0]


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


def test_frames_are_rotated_for_a_camera_mounted_upside_down(synthetic_video: Path) -> None:
    from fall_impact_monitor.capture.reader import FrameEvent

    def frames(rotate_deg: int) -> list[Any]:
        events = (
            FrameEvent(r, p)
            for r, p in iter_frame_records(str(synthetic_video), "cam1", SourceOptions())
        )
        decoded = decode(events, rotate_deg=rotate_deg)
        return [f.image for f in decoded if isinstance(f, DecodedFrame)][:3]

    upright, rotated = frames(0), frames(180)
    for a, b in zip(upright, rotated, strict=True):
        assert np.array_equal(b, a[::-1, ::-1])


def test_the_status_line_hints_at_an_upside_down_image() -> None:
    from fall_impact_monitor.perception.tracking import Track

    keypoints = np.zeros((17, 2), np.float32)
    keypoints[5] = keypoints[6] = (100.0, 200.0)  # shoulders below the hips
    keypoints[11] = keypoints[12] = (100.0, 100.0)
    upside_down = Track(
        1, (50.0, 0.0, 150.0, 300.0), 0.9, keypoints, np.full(17, 0.9, np.float32), "stable", 9, 0
    )

    class InvertedPipeline(FakePipeline):
        def process(self, image: Any, record: FrameRecord) -> FrameResult:
            return FrameResult(record, [upside_down], analysed=True)

    worker = CameraWorker(
        "cam1",
        pipeline=InvertedPipeline(),
        analyzer=FakeAnalyzer([]),
        on_event=lambda r: None,
        config_hash="cfg",
        max_width_px=1280,
        rotate_deg=0,
    )
    image = np.zeros((36, 64, 3), np.uint8)
    for i in range(60):
        worker.process(DecodedFrame(frame_record(100.0 + i * 0.1, i), image))
    assert "upside down" in worker.status_line()
    assert "FIM_CAMERA_CAM1_ROTATE_DEG (now 0)" in worker.status_line()


def test_frames_are_converted_only_when_analysis_takes_them() -> None:
    class CountingFrame:
        conversions = 0

        def to_ndarray(self, format: str) -> Any:
            CountingFrame.conversions += 1
            return np.zeros((36, 64, 3), np.uint8)

    class FakePacket:
        def decode(self) -> list[CountingFrame]:
            return [CountingFrame()]

    from fall_impact_monitor.capture.reader import FrameEvent

    events = [FrameEvent(frame_record(100.0 + i * 0.05, i), FakePacket()) for i in range(10)]
    decoded = [f for f in decode(events) if isinstance(f, DecodedFrame)]  # type: ignore[arg-type]
    assert len(decoded) == 10 and CountingFrame.conversions == 0
    image = decoded[-1].image
    assert image.shape == (36, 64, 3) and decoded[-1].image is image
    assert CountingFrame.conversions == 1


def test_the_worker_reports_how_far_the_reader_is_behind() -> None:
    worker = CameraWorker(
        "cam1",
        pipeline=FakePipeline(),
        analyzer=FakeAnalyzer([]),
        on_event=lambda r: None,
        config_hash="cfg",
        max_width_px=1280,
    )
    image = np.zeros((36, 64, 3), np.uint8)
    for i in range(21):  # camera time 0.1 s per frame, read 0.15 s apart
        record = dataclasses.replace(
            frame_record(100.0 + i * 0.15, i), pts=i * 9000, time_base=Fraction(1, 90000)
        )
        worker.process(DecodedFrame(record, image))
    assert worker.probe(105.0).reader_lag_s == pytest.approx(1.0, abs=0.01)
    worker.process(StateEvent("connected", "x"))  # a new stream: a new reference
    worker.process(DecodedFrame(dataclasses.replace(frame_record(200.0, 99), pts=0), image))
    assert worker.probe(200.0).reader_lag_s == 0.0


def test_decoding_feeds_the_packet_buffer_and_a_reconnection_resets_it(
    synthetic_video: Path,
) -> None:
    from fall_impact_monitor.capture.packet_buffer import LivePacketBuffer
    from fall_impact_monitor.capture.reader import FrameEvent

    live = LivePacketBuffer(window_s=10.0)
    events: list[Any] = [
        FrameEvent(r, p)
        for r, p in iter_frame_records(str(synthetic_video), "cam1", SourceOptions())
    ]
    list(decode(events, buffer=live))
    assert live.newest_arrival_ns() == events[-1].record.arrival_ns
    list(decode([StateEvent("connected", "x")], buffer=live))
    assert live.newest_arrival_ns() is None
