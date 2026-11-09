"""The sink interface. Items are JSON-serializable dicts with a ``kind``:

- ``event``: ``{"kind": "event", "record": EventRecord.to_json()}``;
- ``notification``: a delivery result to persist (sink, message reference, lateness);
- ``feedback``: a human answer (seen, false alarm, confirmed) to persist;
- ``test``: a delivery test (`fim alert test`).
"""

from typing import Any, Protocol

Item = dict[str, Any]


class RetryLater(Exception):
    """The destination asked to wait (e.g. Telegram flood control)."""

    def __init__(self, seconds: float) -> None:
        super().__init__(f"retry after {seconds} s")
        self.seconds = seconds


class Sink(Protocol):
    name: str

    def accepts(self, item: Item) -> bool: ...

    async def deliver(self, item: Item) -> None:
        """Deliver or raise; a raised item is retried later."""
        ...


def is_notifiable(item: Item) -> bool:
    """Events at or above the alert threshold, and delivery tests. Near-misses are
    persisted but never notified."""
    if item["kind"] == "test":
        return True
    return item["kind"] == "event" and not item["record"]["near_miss"]
