import pytest

from fall_impact_monitor.perception.cadence import CadencePolicy, RateMonitor


def test_cadence_intervals_per_scene_state() -> None:
    policy = CadencePolicy(empty_fps=1.0, person_fps=5.0)
    assert policy.interval_s("empty") == pytest.approx(1.0)
    assert policy.interval_s("person") == pytest.approx(0.2)
    assert policy.interval_s("attention") == 0.0  # every frame


def test_rate_monitor_is_not_degraded_during_warmup() -> None:
    monitor = RateMonitor(min_fps=1.0, window_s=10.0, started_s=0.0)
    assert not monitor.degraded(5.0)


def test_rate_monitor_degrades_below_minimum_and_recovers() -> None:
    monitor = RateMonitor(min_fps=1.0, window_s=10.0, started_s=0.0)
    for t in range(0, 20):
        monitor.record(float(t))  # 1 fps
    assert not monitor.degraded(19.5)
    assert monitor.degraded(35.0)  # nothing analysed for 15 s
    for t in range(35, 46):
        monitor.record(float(t))
    assert not monitor.degraded(45.0)


def test_rate_monitor_rejects_invalid_settings() -> None:
    with pytest.raises(ValueError):
        RateMonitor(min_fps=0.0, window_s=10.0, started_s=0.0)


def test_rate_monitor_gives_a_reconnected_camera_a_full_window() -> None:
    # Video stopped for 30 s; the stream reconnects and analysis resumes at 5 fps: the
    # window still covers the outage, but analysis is not slow.
    monitor = RateMonitor(min_fps=0.5, window_s=10.0, started_s=0.0)
    for i in range(100):
        monitor.record(i * 0.2)  # 5 fps for 20 s
    monitor.resume(50.0)  # the stream connected again
    for i in range(3):
        monitor.record(50.0 + i * 0.2)
    assert not monitor.degraded(50.5)
    assert not monitor.degraded(59.9)


def test_a_very_slow_analysis_stays_degraded_while_video_flows() -> None:
    # One analysed frame every 11 s (inference slower than the window), no reconnection:
    # every probe must report it.
    monitor = RateMonitor(min_fps=0.5, window_s=10.0, started_s=0.0)
    probes = []
    for t in range(120):  # health probes every second, in time order
        if t % 11 == 0:
            monitor.record(float(t))
        if t >= 12:
            probes.append(monitor.degraded(float(t)))
    assert all(probes)


def test_rate_monitor_still_reports_a_slow_analysis_after_a_return() -> None:
    monitor = RateMonitor(min_fps=0.5, window_s=10.0, started_s=0.0)
    for i in range(100):
        monitor.record(i * 0.2)
    monitor.resume(50.0)
    monitor.record(50.0)
    monitor.record(58.0)  # one frame in 8 s after the return
    assert monitor.degraded(61.0)
