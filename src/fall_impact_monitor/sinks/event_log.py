"""JSON event log: always on, one line per event (near-misses included) and per delivery
test. Local file; never leaves the host."""

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from fall_impact_monitor.sinks.base import Item
from fall_impact_monitor.sinks.late import late_fields


class JsonLogSink:
    name = "event_log"

    def __init__(
        self,
        path: Path,
        *,
        late_after_s: float,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._path = path
        self._late_after_s = late_after_s
        self._clock = clock

    def accepts(self, item: Item) -> bool:
        return bool(item["kind"] in ("event", "test"))

    async def deliver(self, item: Item) -> None:
        now = self._clock()
        record = item.get("record", {})
        line = {"kind": item["kind"], "logged_utc": now.isoformat()}
        line.update(record)
        line.update(late_fields(record, now, self._late_after_s))
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(line) + "\n")
