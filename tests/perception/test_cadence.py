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
