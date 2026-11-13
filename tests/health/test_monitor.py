from fall_impact_monitor.health.monitor import Condition, HealthMonitor, Transition

LOST = Condition("cam1", "camera lost")
SLOW = Condition("cam1", "analysis below the minimum rate")
DISK = Condition("system", "disk below the free-space floor")


def test_a_component_degrades_once_with_all_its_reasons() -> None:
    monitor = HealthMonitor(["cam1", "system"], restore_after_s=10)
    assert monitor.update(0.0, {LOST, SLOW}) == [
        Transition(
            "cam1", degraded=True, reasons=("analysis below the minimum rate", "camera lost")
        )
    ]
    assert monitor.update(5.0, {LOST}) == []
    assert monitor.states() == {"cam1": ("DEGRADED", ["camera lost"]), "system": ("OK", [])}


def test_restored_only_after_the_hold_time_without_any_condition() -> None:
    monitor = HealthMonitor(["system"], restore_after_s=10)
    monitor.update(0.0, {DISK})
    assert monitor.update(5.0, {DISK}) == []  # last active at 5 s
    assert monitor.update(14.9, set()) == []
    assert monitor.update(15.0, set()) == [Transition("system", degraded=False, reasons=())]
    assert monitor.states()["system"] == ("OK", [])


def test_a_flapping_camera_gives_one_degraded_and_one_restored_message() -> None:
    monitor = HealthMonitor(["cam1"], restore_after_s=10)
    transitions = []
    for second in range(0, 40):
        active = {LOST} if second < 20 and second % 4 < 2 else set()
        transitions += monitor.update(float(second), active)
    assert [t.degraded for t in transitions] == [True, False]


def test_unknown_components_are_tracked_when_they_degrade() -> None:
    monitor = HealthMonitor([], restore_after_s=10)
    assert monitor.update(0.0, {LOST})[0].component == "cam1"


def test_a_restored_state_is_not_announced_again() -> None:
    # After a restart: the camera was already announced as lost before it.
    monitor = HealthMonitor(["cam1", "system"], restore_after_s=10)
    monitor.restore({"cam1": [LOST.reason]}, now_s=0.0)
    assert monitor.states()["cam1"] == ("DEGRADED", [LOST.reason])
    assert monitor.update(1.0, {LOST}) == []


def test_a_restored_state_is_restored_when_the_condition_clears() -> None:
    monitor = HealthMonitor(["cam1"], restore_after_s=10)
    monitor.restore({"cam1": [LOST.reason]}, now_s=0.0)
    assert monitor.update(9.9, set()) == []
    assert monitor.update(10.0, set()) == [Transition("cam1", degraded=False, reasons=())]


def test_a_new_reason_after_a_restart_is_announced() -> None:
    monitor = HealthMonitor(["cam1"], restore_after_s=10)
    monitor.restore({"cam1": [LOST.reason]}, now_s=0.0)
    assert monitor.update(1.0, {LOST, SLOW}) == [
        Transition(
            "cam1", degraded=True, reasons=("analysis below the minimum rate", "camera lost")
        )
    ]


def test_the_snapshot_lists_announced_reasons_of_degraded_components() -> None:
    monitor = HealthMonitor(["cam1", "system"], restore_after_s=10)
    monitor.update(0.0, {LOST})
    assert monitor.snapshot() == {"cam1": [LOST.reason]}
    monitor.update(10.0, set())
    assert monitor.snapshot() == {}


def test_components_no_longer_configured_are_not_restored() -> None:
    monitor = HealthMonitor(["cam1"], restore_after_s=10)
    monitor.restore({"cam9": [LOST.reason]}, now_s=0.0)
    assert "cam9" not in monitor.states()
