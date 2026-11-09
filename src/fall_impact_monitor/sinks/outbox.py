"""Durable outbox: one folder per sink, one JSON file per pending item.

Items are written atomically and deleted only after delivery, so pending alerts survive
restarts and outages of the database or the internet. File names start with a
zero-padded sequence number, so ``pending`` returns items oldest first.
"""

import json
import logging
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any

_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]")
log = logging.getLogger(__name__)


class Outbox:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._lock = threading.Lock()
        self._last_seq = 0

    def _folder(self, sink: str) -> Path:
        return self._root / _UNSAFE.sub("_", sink)

    def put(self, sink: str, item: dict[str, Any]) -> str:
        with self._lock:
            self._last_seq = max(self._last_seq + 1, time.time_ns())
            item_id = f"{self._last_seq:020d}-{uuid.uuid4().hex[:8]}"
        folder = self._folder(sink)
        folder.mkdir(parents=True, exist_ok=True)
        partial = folder / f"{item_id}.part"
        with partial.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(item))
            handle.flush()
            os.fsync(handle.fileno())  # survives a power cut once put() returns
        partial.replace(folder / f"{item_id}.json")
        return item_id

    def pending(self, sink: str) -> list[tuple[str, dict[str, Any]]]:
        folder = self._folder(sink)
        if not folder.is_dir():
            return []
        items = []
        for path in sorted(folder.glob("*.json")):
            try:
                items.append((path.stem, json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, ValueError):
                # Truncated by a crash: set aside for inspection, never block the queue.
                log.error("unreadable outbox item set aside", extra={"file": path.name})
                path.replace(path.with_suffix(".bad"))
        return items

    def done(self, sink: str, item_id: str) -> None:
        (self._folder(sink) / f"{item_id}.json").unlink(missing_ok=True)

    def oldest_age_s(self, sink: str) -> float | None:
        """Seconds since the oldest pending item was queued; None when nothing waits."""
        folder = self._folder(sink)
        times = [p.stat().st_mtime for p in folder.glob("*.json")] if folder.is_dir() else []
        return time.time() - min(times) if times else None

    def count(self, sink: str) -> int:
        folder = self._folder(sink)
        return len(list(folder.glob("*.json"))) if folder.is_dir() else 0
