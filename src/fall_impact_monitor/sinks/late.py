"""Late marker for outputs (safety rule 7): an alert that exceeds the latency budget is
still delivered, and says so. Computed at each delivery attempt, so a retry after the
budget is marked late even when the first attempt was not."""

from collections.abc import Mapping
from datetime import datetime
from typing import Any


def late_fields(record: Mapping[str, Any], now: datetime, late_after_s: float) -> dict[str, Any]:
    """``{"late": bool, "late_s": seconds since the event ended}``; ``{}`` for items
    without an end time (delivery tests)."""
    ended = record.get("ended_utc")
    if not isinstance(ended, str):
        return {}
    late_s = max(0.0, (now - datetime.fromisoformat(ended)).total_seconds())
    return {"late": late_s > late_after_s, "late_s": round(late_s, 3)}
