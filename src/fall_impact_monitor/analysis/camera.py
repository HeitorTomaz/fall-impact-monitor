"""All rule-based detectors of one camera: analysed frames in, events out."""

from fall_impact_monitor.analysis.audio import Thump
from fall_impact_monitor.analysis.fall import FallAnalyzer
from fall_impact_monitor.analysis.floor import FloorAnalyzer
from fall_impact_monitor.analysis.impact import HeadImpactAnalyzer
from fall_impact_monitor.analysis.zones import Zone
from fall_impact_monitor.api.perception import FrameResult
from fall_impact_monitor.events import Event, EventCategory


class CameraAnalyzer:
    """Runs the fall, person-on-the-floor and (experimental) head-impact detectors with the
    same zones; a possible fall (alert or near-miss) shortens the floor limit for that
    person and is reaction evidence for the head-impact rules."""

    def __init__(
        self,
        camera_id: str,
        zones: list[Zone],
        *,
        threshold: float = 0.5,
        fall_enabled: bool = True,
        floor_enabled: bool = True,
        person_down_s: float = 30.0,
        no_fall_s: float = 60.0,
        head_impact_enabled: bool = True,
        head_impact_min_evidence: int = 2,
        head_impact_window_s: float = 3.0,
    ) -> None:
        self._fall = (
            FallAnalyzer(camera_id, zones=zones, threshold=threshold) if fall_enabled else None
        )
        self._floor = (
            FloorAnalyzer(
                camera_id,
                zones,
                person_down_s=person_down_s,
                no_fall_s=no_fall_s,
                threshold=threshold,
            )
            if floor_enabled
            else None
        )

        self._impact = (
            HeadImpactAnalyzer(
                camera_id,
                zones,
                threshold=threshold,
                min_evidence=head_impact_min_evidence,
                window_s=head_impact_window_s,
            )
            if head_impact_enabled
            else None
        )

    def add_thumps(self, thumps: list[Thump]) -> None:
        if self._impact is not None:
            self._impact.add_thumps(thumps)

    def wants_audio(self) -> bool:
        """Thumps are only used by the head-impact rules."""
        return self._impact is not None

    def attention_until(self) -> float | None:
        """Analyse every frame until then (a head moving fast outside safe zones)."""
        return self._impact.attention_until() if self._impact is not None else None

    def update(self, result: FrameResult, frame_size: tuple[int, int]) -> list[Event]:
        events: list[Event] = []
        if self._fall is not None:
            events += self._fall.update(result, frame_size)
        falls = list(events)
        if self._floor is not None:
            for event in events:
                if event.category is EventCategory.POSSIBLE_FALL and event.track_id is not None:
                    self._floor.note_fall(event.track_id, event.end_s)
            events += self._floor.update(result, frame_size)
        if self._impact is not None:
            events += self._impact.update(result, frame_size, falls)
        return events
