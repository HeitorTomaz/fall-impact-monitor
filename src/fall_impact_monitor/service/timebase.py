"""Arrival time (monotonic seconds, used by the detectors) to UTC, for messages and storage.

Each frame record carries its own pair (monotonic arrival, UTC wall clock) taken at the
same instant; a detector time is mapped through the pair of the frame that produced the
event. No time is invented: the offset is the one observed with that frame.
"""

from datetime import datetime, timedelta

from fall_impact_monitor.capture import FrameRecord


def to_utc(t_s: float, anchor: FrameRecord) -> datetime:
    return anchor.wall_utc + timedelta(seconds=t_s - anchor.arrival_ns / 1e9)
