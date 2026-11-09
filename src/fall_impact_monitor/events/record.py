"""An event ready for the outputs: the detector event plus wall-clock (UTC) times and the
hash of the configuration that produced it."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from fall_impact_monitor.events.model import Event, EventCategory


@dataclass(frozen=True)
class EventRecord:
    event: Event
    started_utc: datetime
    ended_utc: datetime
    detected_utc: datetime  # when the detector raised it
    config_hash: str

    def __post_init__(self) -> None:
        for value in (self.started_utc, self.ended_utc, self.detected_utc):
            if value.tzinfo is None:
                raise ValueError("event record times must include a timezone")

    def to_json(self) -> dict[str, Any]:
        e = self.event
        return {
            "event_id": e.event_id,
            "camera_id": e.camera_id,
            "category": e.category.value,
            "confidence": round(e.confidence, 4),
            "near_miss": e.near_miss,
            "start_s": e.start_s,
            "end_s": e.end_s,
            "track_id": e.track_id,
            "identity": e.identity,
            "evidence": e.evidence,
            "analysed_seqs": e.analysed_seqs,
            "started_utc": self.started_utc.isoformat(),
            "ended_utc": self.ended_utc.isoformat(),
            "detected_utc": self.detected_utc.isoformat(),
            "config_hash": self.config_hash,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "EventRecord":
        event = Event(
            event_id=data["event_id"],
            camera_id=data["camera_id"],
            category=EventCategory(data["category"]),
            confidence=data["confidence"],
            start_s=data["start_s"],
            end_s=data["end_s"],
            track_id=data["track_id"],
            identity=data["identity"],
            near_miss=data["near_miss"],
            evidence=data["evidence"],
            analysed_seqs=data["analysed_seqs"],
        )
        return cls(
            event=event,
            started_utc=datetime.fromisoformat(data["started_utc"]),
            ended_utc=datetime.fromisoformat(data["ended_utc"]),
            detected_utc=datetime.fromisoformat(data["detected_utc"]),
            config_hash=data["config_hash"],
        )
