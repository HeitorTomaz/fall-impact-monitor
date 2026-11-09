"""Health states per component (each camera, and "system"): OK or DEGRADED.

A component is DEGRADED as soon as any of its conditions is active, and RESTORED only
after none has been active for ``restore_after_s``, so a flapping camera produces one
degraded and one restored message instead of a storm. Pure logic: the service feeds it
observations and turns transitions into monitoring events.
"""

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Condition:
    component: str
    reason: str


@dataclass(frozen=True)
class Transition:
    component: str
    degraded: bool
    reasons: tuple[str, ...]


@dataclass
class _State:
    degraded: bool = False
    reasons: tuple[str, ...] = ()
    last_bad_s: float = 0.0


class HealthMonitor:
    def __init__(self, components: Iterable[str], *, restore_after_s: float) -> None:
        self._restore_after_s = restore_after_s
        self._states: dict[str, _State] = {name: _State() for name in components}

    def update(self, now_s: float, active: set[Condition]) -> list[Transition]:
        by_component: dict[str, list[str]] = {}
        for condition in active:
            by_component.setdefault(condition.component, []).append(condition.reason)
        for name in by_component:
            self._states.setdefault(name, _State())
        transitions = []
        for name, state in self._states.items():
            reasons = tuple(sorted(by_component.get(name, [])))
            if reasons:
                state.last_bad_s = now_s
                state.reasons = reasons
                if not state.degraded:
                    state.degraded = True
                    transitions.append(Transition(name, degraded=True, reasons=reasons))
            elif state.degraded and now_s - state.last_bad_s >= self._restore_after_s:
                state.degraded = False
                state.reasons = ()
                transitions.append(Transition(name, degraded=False, reasons=()))
        return transitions

    def states(self) -> dict[str, tuple[str, list[str]]]:
        return {
            name: ("DEGRADED" if s.degraded else "OK", list(s.reasons))
            for name, s in self._states.items()
        }
