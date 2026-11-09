"""Event model shared by detectors, outputs and storage."""

import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal


class EventCategory(StrEnum):
    POSSIBLE_FALL = "possible_fall"
    PERSON_DOWN = "person_down"
    POSSIBLE_HEAD_IMPACT = "possible_head_impact"
    MANUAL_REPORT = "manual_report"
    MONITORING_DEGRADED = "monitoring_degraded"
    MONITORING_RESTORED = "monitoring_restored"


@dataclass(frozen=True)
class Event:
    """A detection result. ``near_miss`` events are persisted but never notified.

    ``confidence`` is a rule-based score in [0, 1]; it is not a calibrated
    probability until calibrated on site data.
    """

    event_id: str
    camera_id: str
    category: EventCategory
    confidence: float
    start_s: float
    end_s: float
    track_id: int | None
    identity: Literal["stable", "uncertain"]
    near_miss: bool
    evidence: dict[str, Any] = field(default_factory=dict)
    analysed_seqs: list[int] = field(default_factory=list)


def is_near_miss(confidence: float, *, threshold: float) -> bool:
    """Events below the alert threshold are near-misses."""
    return confidence < threshold


def make_event(
    *,
    camera_id: str,
    category: EventCategory,
    confidence: float,
    start_s: float,
    end_s: float,
    track_id: int | None,
    identity: Literal["stable", "uncertain"],
    evidence: dict[str, Any],
    analysed_seqs: list[int],
    threshold: float,
) -> Event:
    if end_s < start_s:
        raise ValueError("event end_s must not be before start_s")
    clamped = min(1.0, max(0.0, confidence))
    return Event(
        event_id=str(uuid.uuid4()),
        camera_id=camera_id,
        category=category,
        confidence=clamped,
        start_s=start_s,
        end_s=end_s,
        track_id=track_id,
        identity=identity,
        near_miss=is_near_miss(clamped, threshold=threshold),
        evidence=evidence,
        analysed_seqs=analysed_seqs,
    )
