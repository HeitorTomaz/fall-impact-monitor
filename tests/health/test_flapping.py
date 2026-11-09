"""Simulated outages through the real observe() + HealthMonitor, checked every 5 s."""

from fall_impact_monitor.health.monitor import Condition, HealthMonitor, Transition
from fall_impact_monitor.health.observe import CameraProbe, HealthLimits, observe

LIMITS = HealthLimits(camera_lost_s=30, camera_gap_s=10, max_reconnects=3, reconnect_window_s=600)


def simulate(
    down_s: int, up_s: int, *, count_reconnects: bool, minutes: int = 20
) -> list[Transition]:
    monitor = HealthMonitor(["cam1"], restore_after_s=30)
    transitions: list[Transition] = []
    last_frame = 0.0
    connects: list[float] = []
    period = down_s + up_s
    for t in range(0, minutes * 60, 5):
        phase = t % period
        up = phase >= down_s
        if up:
            if phase - 5 < down_s:  # the camera came back: a reconnection
                connects.append(float(t))
            last_frame = float(t)
        recent = [c for c in connects if t - c <= LIMITS.reconnect_window_s]
        probe = CameraProbe(
            "cam1",
            last_frame_s=last_frame,
            started_s=0.0,
            rate_degraded=False,
            reconnects=len(recent) if count_reconnects else 0,
        )
        degraded = {name for name, (state, _) in monitor.states().items() if state == "DEGRADED"}
        found = observe(
            float(t),
            cameras=[probe],
            limits=LIMITS,
            degraded=degraded,
            disk_free_bytes=1,
            backlog_ages_s={},
        )
        transitions += monitor.update(float(t), found)
    return transitions


def test_a_camera_dropping_for_a_minute_at_a_time_does_not_flood_the_chat() -> None:
    transitions = simulate(60, 5, count_reconnects=False)
    assert [t.degraded for t in transitions] == [True]


def test_short_repeated_drops_are_reported_as_an_unstable_camera() -> None:
    transitions = simulate(20, 5, count_reconnects=True)
    assert transitions and transitions[0].degraded
    assert "camera_unstable" in transitions[0].reasons
    assert len(transitions) <= 2


def test_a_new_reason_on_a_degraded_component_is_reported() -> None:
    monitor = HealthMonitor(["system"], restore_after_s=30)
    monitor.update(0.0, {Condition("system", "database_backlog")})
    added = monitor.update(
        5.0, {Condition("system", "database_backlog"), Condition("system", "disk_low")}
    )
    assert added == [Transition("system", degraded=True, reasons=("database_backlog", "disk_low"))]
    assert monitor.update(10.0, {Condition("system", "disk_low")}) == []  # a removal is silent
