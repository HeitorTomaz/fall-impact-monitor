"""All rule-based detectors of one camera: analysed frames in, events out."""

from fall_impact_monitor.analysis.fall import FallAnalyzer
from fall_impact_monitor.analysis.floor import FloorAnalyzer
from fall_impact_monitor.analysis.zones import Zone
from fall_impact_monitor.api.perception import FrameResult
from fall_impact_monitor.events import Event, EventCategory


class CameraAnalyzer:
    """Runs the fall and person-on-the-floor detectors with the same allowed zones; a
    possible fall (alert or near-miss) shortens the floor limit for that person."""

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

    def update(self, result: FrameResult, frame_size: tuple[int, int]) -> list[Event]:
        events: list[Event] = []
        if self._fall is not None:
            events += self._fall.update(result, frame_size)
        if self._floor is not None:
            for event in events:
                if event.category is EventCategory.POSSIBLE_FALL and event.track_id is not None:
                    self._floor.note_fall(event.track_id, event.end_s)
            events += self._floor.update(result, frame_size)
        return events
