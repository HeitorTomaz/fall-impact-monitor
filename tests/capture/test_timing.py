from datetime import UTC, datetime
from fractions import Fraction

from fall_impact_monitor.capture import FrameRecord, TimestampClassifier, TimestampQuality

TB = Fraction(1, 90_000)  # RTP video clock
STEP = 3_000  # 33.3 ms at 90 kHz


def classify_all(pts_values: list[int | None]) -> list[TimestampQuality]:
    classifier = TimestampClassifier()
    return [classifier.classify(pts, TB) for pts in pts_values]


def steady(n: int, start: int = 0) -> list[int | None]:
    return [start + i * STEP for i in range(n)]


def test_steady_stream_is_ok() -> None:
    assert set(classify_all(steady(30))) == {TimestampQuality.OK}


def test_missing_pts() -> None:
    assert classify_all([0, None, STEP])[1] is TimestampQuality.MISSING


def test_equal_pts_is_zero_delta() -> None:
    assert classify_all([0, STEP, STEP])[2] is TimestampQuality.ZERO_DELTA


def test_decreasing_pts_is_backward() -> None:
    assert classify_all([0, 2 * STEP, STEP])[2] is TimestampQuality.BACKWARD


def test_large_gap_after_warmup_is_outlier() -> None:
    values = steady(15)
    values.append(values[-1] + 36_000)  # +400 ms
    assert classify_all(values)[-1] is TimestampQuality.OUTLIER_GAP


def test_large_gap_during_warmup_is_ok() -> None:
    assert classify_all([0, STEP, STEP + 36_000])[2] is TimestampQuality.OK


def test_observed_burst_then_jump_pattern() -> None:
    values = steady(15)
    burst = values[-1] + STEP
    values += [burst] * 12  # first of the run is a normal step, then 11 repeats
    values += [burst + STEP, burst + 2 * STEP]
    values.append(burst + 2 * STEP + 36_000)
    qualities = classify_all(values)[15:]
    assert qualities.count(TimestampQuality.ZERO_DELTA) == 11
    assert qualities.count(TimestampQuality.OUTLIER_GAP) == 1
    assert qualities[-1] is TimestampQuality.OUTLIER_GAP


def test_frame_record_converts_pts_to_seconds() -> None:
    record = FrameRecord(
        camera_id="cam1",
        seq=0,
        pts=90_000,
        time_base=TB,
        arrival_ns=5,
        wall_utc=datetime(2026, 1, 1, tzinfo=UTC),
        keyframe=True,
        size_bytes=100,
        quality=TimestampQuality.OK,
    )
    assert record.pts_s == 1.0
    assert record.quality == "ok"


def test_frame_record_without_pts_has_no_seconds() -> None:
    record = FrameRecord(
        "cam1", 0, None, TB, 0, datetime(2026, 1, 1, tzinfo=UTC), False, 1, TimestampQuality.MISSING
    )
    assert record.pts_s is None


def test_bursty_tiny_deltas_do_not_turn_normal_intervals_into_outliers() -> None:
    # Bursty delivery: 3 deltas of 2 ms then 2 of 80 ms (mean ~33 ms, median 2 ms).
    deltas_ms = [2, 2, 2, 80, 80] * 8
    values = [0]
    for d in deltas_ms:
        values.append(values[-1] + d * 90)
    qualities = classify_all(values)
    assert TimestampQuality.OUTLIER_GAP not in qualities
    values.append(values[-1] + 400 * 90)
    assert classify_all(values)[-1] is TimestampQuality.OUTLIER_GAP


def test_one_long_freeze_does_not_hide_later_gaps() -> None:
    values = steady(30)
    values.append(values[-1] + 5 * 90_000)  # 5 s freeze
    for _ in range(30):
        values.append(values[-1] + STEP)
    values.append(values[-1] + 36_000)  # 400 ms gap within the next 50 frames
    qualities = classify_all(values)
    assert qualities[30] is TimestampQuality.OUTLIER_GAP
    assert qualities[-1] is TimestampQuality.OUTLIER_GAP
