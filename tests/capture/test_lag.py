import pytest

from fall_impact_monitor.capture.lag import ReaderLag


def test_a_reader_that_keeps_up_has_no_lag() -> None:
    lag = ReaderLag()
    values = [lag.update(t * 0.05, 100.0 + t * 0.05) for t in range(200)]
    assert max(values) == pytest.approx(0.0, abs=1e-9)


def test_a_reader_falling_behind_shows_a_growing_lag() -> None:
    # Each second of camera time takes 1.5 s to read: 0.5 s more lag per second.
    lag = ReaderLag()
    values = [lag.update(t * 0.1, 100.0 + t * 0.15) for t in range(101)]
    assert values[-1] == pytest.approx(5.0, abs=0.05)  # minus the allowed clock drift


def test_camera_timestamp_jitter_is_not_lag() -> None:
    # Bursts of frames with equal timestamps, then a 400 ms jump (seen on real cameras).
    lag, pts, values = ReaderLag(), 0.0, []
    for i in range(300):
        pts += 0.4 if i % 10 == 0 else 0.0
        values.append(lag.update(pts, 100.0 + i * 0.04))
    assert max(values) < 0.6


def test_timestamps_going_backwards_start_a_new_reference() -> None:
    lag = ReaderLag()
    for t in range(50):
        lag.update(1000.0 + t * 0.1, 100.0 + t * 0.1)
    assert lag.update(0.0, 105.1) == 0.0  # the camera restarted its clock
    assert lag.update(0.1, 105.2) == pytest.approx(0.0, abs=1e-9)


def test_frames_without_timestamps_report_no_lag() -> None:
    lag = ReaderLag()
    assert lag.update(None, 100.0) == 0.0


def test_reset_forgets_the_reference() -> None:
    lag = ReaderLag()
    lag.update(0.0, 100.0)
    lag.update(1.0, 110.0)  # 9 s behind
    lag.reset()  # reconnected: a new stream
    assert lag.update(50.0, 200.0) == 0.0


def test_a_camera_clock_that_stands_still_is_not_lag() -> None:
    # Broken timestamps: the camera repeats one presentation time. No information.
    lag = ReaderLag()
    values = [lag.update(5.0, 100.0 + i * 0.5) for i in range(100)]
    assert max(values) <= 1.0


def test_a_slow_reader_is_seen_with_one_second_timestamp_runs() -> None:
    # The reference camera's pattern: equal timestamps, then a jump, about once a
    # second. A reader at 0.8x real time must still show a growing lag.
    lag, values = ReaderLag(), []
    for second in range(60):
        for i in range(20):  # 20 frames share one timestamp
            arrival = 100.0 + (second * 20 + i) * 0.05 / 0.8
            values.append(lag.update(float(second), arrival))
    assert values[-1] > 10.0


def test_a_constant_lag_is_not_forgotten_after_ten_minutes() -> None:
    # The reader fell 20 s behind and stays there (the relay drops what does not fit).
    lag = ReaderLag()
    for i in range(100):
        lag.update(i * 0.1, 100.0 + i * 0.1)
    value = 0.0
    for i in range(100, 100 + 20 * 60 * 10):  # 20 minutes at 10 fps
        value = lag.update(i * 0.1, 120.0 + i * 0.1)
    assert value > 15.0
