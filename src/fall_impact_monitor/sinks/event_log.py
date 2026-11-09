"""JSON event log: always on, one line per event (near-misses included) and per delivery
test. Local file; never leaves the host."""

import json
from datetime import UTC, datetime
from pathlib import Path

from fall_impact_monitor.sinks.base import Item


class JsonLogSink:
    name = "event_log"

    def __init__(self, path: Path) -> None:
        self._path = path

    def accepts(self, item: Item) -> bool:
        return bool(item["kind"] in ("event", "test"))

    async def deliver(self, item: Item) -> None:
        line = {"kind": item["kind"], "logged_utc": datetime.now(UTC).isoformat()}
        line.update(item.get("record", {}))
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(line) + "\n")
