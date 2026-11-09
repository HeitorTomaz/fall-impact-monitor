from fall_impact_monitor.analysis.fall_state import FallConfig, FallStateMachine, StepResult
from fall_impact_monitor.analysis.observation import Observation

FPS = 5.0
HEIGHT = 200.0  # upright box height in px
HIP_UP = 150.0


def obs(t: float, hip: float, angle: float, aspect: float, height: float = HEIGHT) -> Observation:
    return Observation(True, t, aspect, height, 0.9, hip, hip - 60.0, angle)


def run(frames: list[Observation], config: FallConfig | None = None) -> list[StepResult]:
    machine = FallStateMachine(config or FallConfig())
    return [machine.update(o) for o in frames]


def timeline(seconds: float, at: callable) -> list[Observation]:  # type: ignore[valid-type]
    return [at(i / FPS) for i in range(int(seconds * FPS))]


def standing(t: float) -> Observation:
    return obs(t, HIP_UP, 5.0, 0.4)


def fall_then(t0: float, after: str):  # type: ignore[no-untyped-def]
    """Stand until t0, fall in 0.4 s (hip down 0.6 heights), then lie or get up at once."""

    def at(t: float) -> Observation:
        if t < t0:
            return standing(t)
        k = min(1.0, (t - t0) / 0.4)
        if after == "recover" and t >= t0 + 0.8:
            return standing(t)
        return obs(t, HIP_UP + 120.0 * k, 5.0 + 80.0 * k, 0.4 + 2.0 * k, height=HEIGHT - 120 * k)

    return at


def events(results: list[StepResult]) -> list[StepResult]:
    return [r for r in results if r.event]


def test_fall_and_staying_down_raises_one_event() -> None:
    results = run(timeline(6.0, fall_then(2.0, "lie")))
    assert len(events(results)) == 1
    assert results[-1].state in {"fallen", "recovering"}


def test_fall_with_immediate_recovery_still_raises_an_event() -> None:
    results = run(timeline(6.0, fall_then(2.0, "recover")))
    fired = events(results)
    assert len(fired) == 1
    assert fired[0].features["max_drop_heights"] >= 0.5
    assert results[-1].state == "normal"


def test_sitting_down_quickly_is_not_a_fall() -> None:
    def at(t: float) -> Observation:
        k = min(1.0, max(0.0, (t - 2.0) / 0.5))
        return obs(t, HIP_UP + 50.0 * k, 5.0 + 15.0 * k, 0.4 + 0.3 * k)

    assert events(run(timeline(5.0, at))) == []


def test_bending_to_pick_something_up_is_not_a_fall() -> None:
    def at(t: float) -> Observation:
        k = min(1.0, max(0.0, (t - 2.0) / 1.0))
        return obs(t, HIP_UP + 10.0 * k, 5.0 + 70.0 * k, 0.4 + 0.5 * k)

    assert events(run(timeline(5.0, at))) == []


def test_slowly_lying_down_is_left_to_the_floor_rules() -> None:
    def at(t: float) -> Observation:
        k = min(1.0, max(0.0, (t - 1.0) / 6.0))
        return obs(t, HIP_UP + 120.0 * k, 5.0 + 80.0 * k, 0.4 + 2.0 * k)

    assert events(run(timeline(9.0, at))) == []


def test_suspicion_that_does_not_confirm_is_a_near_miss() -> None:
    def at(t: float) -> Observation:
        if 2.0 <= t < 2.4:  # stumble: fast dip, torso tilts, then upright again
            return obs(t, HIP_UP + 30.0, 60.0, 0.6)
        return standing(t)

    results = run(timeline(6.0, at))
    assert events(results) == []
    assert any(r.near_miss for r in results)


def test_cooldown_prevents_duplicate_events_for_one_fall() -> None:
    def at(t: float) -> Observation:
        if t < 2.0:
            return standing(t)
        bounce = 2.6 <= t < 2.8  # hip bounces up for one sample right after landing
        k = min(1.0, (t - 2.0) / 0.4)
        hip = HIP_UP + 120.0 * k - (40.0 if bounce else 0.0)
        return obs(t, hip, 85.0 * k, 0.4 + 2.0 * k)

    assert len(events(run(timeline(6.0, at)))) == 1


def test_invalid_observations_never_start_a_fall() -> None:
    machine = FallStateMachine(FallConfig())
    for i in range(10):
        machine.update(standing(i / FPS))
    result = machine.update(Observation(False, 2.0, 3.0, 50.0, 0.9))
    assert not result.event and result.state == "normal"
