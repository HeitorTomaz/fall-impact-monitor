from pathlib import Path

import pytest

from fall_impact_monitor.capture import FrameRecord
from fall_impact_monitor.capture.packet_buffer import PacketBuffer
from fall_impact_monitor.capture.reader import SourceOptions, iter_frame_records

FRAME_NS = 40_000_000


class StepClock:
    def __init__(self) -> None:
        self.ns = 0

    def __call__(self) -> int:
        self.ns += FRAME_NS
        return self.ns


def fill(path: Path, window_s: float) -> tuple[PacketBuffer, list[FrameRecord]]:
    buffer: PacketBuffer | None = None
    records: list[FrameRecord] = []
    for record, packet in iter_frame_records(
        str(path), "cam1", SourceOptions(), clock_ns=StepClock()
    ):
        if buffer is None:
            ctx = packet.stream.codec_context
            buffer = PacketBuffer(window_s, ctx.name, ctx.extradata)
        buffer.append(record, bytes(packet))
        records.append(record)
    assert buffer is not None
    return buffer, records


def test_eviction_keeps_only_the_configured_window(synthetic_video: Path) -> None:
    buffer, records = fill(synthetic_video, window_s=1.0)
    newest = records[-1].arrival_ns
    kept = buffer.window(0, newest)
    assert buffer.span_ns() <= 1_000_000_000
    assert len(buffer) < len(records)
    assert all(newest - record.arrival_ns <= 1_000_000_000 for record, _ in kept)


def test_window_starting_mid_gop_includes_previous_keyframe(synthetic_video: Path) -> None:
    buffer, records = fill(synthetic_video, window_s=10.0)
    items = buffer.window(records[30].arrival_ns, records[40].arrival_ns)
    assert items[0][0].seq == 25 and items[0][0].keyframe
    assert items[-1][0].seq == 40


def test_decode_window_yields_only_frames_inside_the_window(synthetic_video: Path) -> None:
    buffer, records = fill(synthetic_video, window_s=10.0)
    decoded = list(buffer.decode_window(records[30].arrival_ns, records[40].arrival_ns))
    assert [record.seq for record, _ in decoded] == list(range(30, 41))
    assert all(frame.width == 640 and frame.height == 360 for _, frame in decoded)


def test_window_without_keyframe_in_buffer_starts_at_first_keyframe(synthetic_video: Path) -> None:
    buffer, records = fill(synthetic_video, window_s=1.0)  # oldest kept packets are mid-GOP
    oldest = buffer.window(0, records[-1].arrival_ns)
    decoded = list(buffer.decode_window(0, records[-1].arrival_ns))
    assert decoded, "frames after the first kept keyframe must decode"
    assert decoded[0][0].seq >= next(r.seq for r, _ in oldest if r.keyframe)


def test_rejects_non_positive_window() -> None:
    with pytest.raises(ValueError):
        PacketBuffer(0, "h264", None)


def test_window_after_newest_packet_is_empty(synthetic_video: Path) -> None:
    buffer, records = fill(synthetic_video, window_s=10.0)
    later = records[-1].arrival_ns + 1_000_000_000
    assert buffer.window(later, later + 1_000_000_000) == []
    assert list(buffer.decode_window(later, later + 1_000_000_000)) == []


def test_appending_and_reading_from_two_threads_is_safe(synthetic_video: Path) -> None:
    import threading

    buffer, records = fill(synthetic_video, window_s=0.5)
    payload = b"x" * 10
    errors: list[BaseException] = []

    def writer() -> None:
        try:
            for i in range(3000):
                buffer.append(records[i % len(records)], payload)
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=writer)
    thread.start()
    try:
        for _ in range(3000):
            buffer.window(0, 10**18)
            buffer.newest_arrival_ns()
    finally:
        thread.join()
    assert errors == []


def test_the_live_buffer_is_fed_from_reader_packets_and_reset(synthetic_video: Path) -> None:
    from fall_impact_monitor.capture.packet_buffer import LivePacketBuffer

    live = LivePacketBuffer(window_s=10.0)
    assert live.newest_arrival_ns() is None and list(live.decode_window(0, 10**18)) == []
    records = []
    for record, packet in iter_frame_records(
        str(synthetic_video), "cam1", SourceOptions(), clock_ns=StepClock()
    ):
        live.add(record, packet)
        records.append(record)
    assert live.newest_arrival_ns() == records[-1].arrival_ns
    decoded = list(live.decode_window(records[30].arrival_ns, records[40].arrival_ns))
    assert [record.seq for record, _ in decoded] == list(range(30, 41))
    live.reset()  # a reconnection: the next stream may differ
    assert live.newest_arrival_ns() is None


def test_a_window_works_with_packets_out_of_order(synthetic_video: Path) -> None:
    # Offline, packets are timed by presentation time but arrive in decode order
    # (B-frames): the window must not assume sorted times.
    import dataclasses

    buffer, records = fill(synthetic_video, window_s=10.0)
    shuffled = PacketBuffer(10.0, "h264", None)
    items = buffer.window(0, 10**18)
    for index, (record, payload) in enumerate(items):
        if index % 4 == 2 and index + 1 < len(items):  # swap a pair of neighbours' times
            nxt = items[index + 1][0]
            record = dataclasses.replace(record, arrival_ns=nxt.arrival_ns + 1)
        shuffled.append(record, payload)
    start, end = records[30].arrival_ns, records[40].arrival_ns
    picked = [r.seq for r, _ in shuffled.window(start, end)]
    assert set(range(30, 40)) <= set(picked)
