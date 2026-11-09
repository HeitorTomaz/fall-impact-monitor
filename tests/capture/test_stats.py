from datetime import UTC, datetime
from fractions import Fraction

import pytest

from fall_impact_monitor.capture import FrameRecord, TimestampClassifier
from fall_impact_monitor.capture.stats import summarize, to_markdown

TB = Fraction(1, 1000)
WALL = datetime(2026, 1, 1, tzinfo=UTC)


def make_records(
    pts_ms: list[int | None], arrival_ms: list[int], keyframe_every: int = 25
) -> list[FrameRecord]:
    classifier = TimestampClassifier()
    return [
        FrameRecord(
            "cam1",
            i,
            pts,
            TB,
            arr * 1_000_000,
            WALL,
            i % keyframe_every == 0,
            1000,
            classifier.classify(pts, TB),
        )
        for i, (pts, arr) in enumerate(zip(pts_ms, arrival_ms, strict=True))
    ]


def test_steady_25_fps_stream() -> None:
    times = [i * 40 for i in range(101)]  # 4 s at 25 fps
    s = summarize(make_records(list(times), times))
    assert s.frames == 101
    assert s.duration_pts_s == pytest.approx(4.0)
    assert s.fps_pts == pytest.approx(25.0)
    assert s.fps_arrival == pytest.approx(25.0)
    assert s.quality_counts["ok"] == 101
    assert s.delta_ms_histogram == {
        "<=0": 0,
        "<30": 0,
        "30-37": 0,
        "37-45": 100,
        "45-70": 0,
        ">=70": 0,
    }
    assert s.keyframes == 5
    assert s.mean_gop_s == pytest.approx(1.0)
    assert (s.frames_per_second_min, s.frames_per_second_max) == (25, 25)


def test_zero_deltas_and_jumps_are_counted() -> None:
    pts: list[int | None] = [i * 33 for i in range(20)] + [660] * 5 + [1100]
    arrival = [i * 33 for i in range(26)]
    s = summarize(make_records(pts, arrival))
    assert s.quality_counts["zero_delta"] == 4
    assert s.quality_counts["outlier_gap"] == 1
    assert s.delta_ms_histogram["<=0"] == 4
    assert s.delta_ms_histogram[">=70"] == 1


def test_missing_pts_are_excluded_from_pts_statistics() -> None:
    pts: list[int | None] = [0, None, 80, 120]
    s = summarize(make_records(pts, [0, 40, 80, 120]))
    assert s.quality_counts["missing"] == 1
    assert s.fps_pts == pytest.approx(2 / 0.12)
    assert sum(s.delta_ms_histogram.values()) == 2


def test_stream_without_any_pts_has_no_pts_rates() -> None:
    s = summarize(make_records([None, None, None], [0, 40, 80]))
    assert s.duration_pts_s is None
    assert s.fps_pts is None


def test_empty_input_is_rejected() -> None:
    with pytest.raises(ValueError):
        summarize([])


def test_markdown_lists_both_time_domains() -> None:
    times = [i * 40 for i in range(26)]
    text = to_markdown(summarize(make_records(list(times), times)))
    assert "FPS (source PTS)" in text
    assert "FPS (arrival)" in text


def test_reconnect_does_not_mix_pts_timelines() -> None:
    times = [i * 40 for i in range(51)]  # 2 s per connection, PTS restart at 0
    first = make_records(list(times), times)
    second = [
        FrameRecord(
            r.camera_id,
            r.seq + 51,
            r.pts,
            r.time_base,
            r.arrival_ns + 3_000_000_000,
            r.wall_utc,
            r.keyframe,
            r.size_bytes,
            r.quality,
            connection=1,
        )
        for r in make_records(list(times), times)
    ]
    s = summarize(first + second)
    assert s.duration_pts_s == pytest.approx(4.0)
    assert s.fps_pts == pytest.approx(25.0)
    assert sum(s.delta_ms_histogram.values()) == 100
    assert s.mean_gop_s == pytest.approx(1.0)
    assert s.connections == 2
