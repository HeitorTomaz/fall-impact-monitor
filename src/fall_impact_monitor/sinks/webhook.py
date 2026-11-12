"""Generic webhook: POST each notified event (and delivery tests) as JSON.

The URL may carry a token, so it is never logged nor put in error messages. Any status
outside 2xx, or a network error, raises so the dispatcher retries the item.
"""

from collections.abc import Callable
from datetime import UTC, datetime

import aiohttp

from fall_impact_monitor.sinks.base import Item, is_notifiable
from fall_impact_monitor.sinks.late import late_fields


class WebhookSink:
    name = "webhook"

    def __init__(
        self,
        url: str,
        *,
        timeout_s: float,
        late_after_s: float,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._url = url
        self._timeout = aiohttp.ClientTimeout(total=timeout_s)
        self._late_after_s = late_after_s
        self._clock = clock

    def accepts(self, item: Item) -> bool:
        return is_notifiable(item)

    async def deliver(self, item: Item) -> None:
        record = item.get("record", {})
        payload = {"kind": item["kind"], **record}
        payload.update(late_fields(record, self._clock(), self._late_after_s))
        try:
            async with (
                aiohttp.ClientSession(timeout=self._timeout) as session,
                session.post(self._url, json=payload) as response,
            ):
                status = response.status
        except (aiohttp.ClientError, TimeoutError) as error:
            raise ConnectionError(f"webhook unreachable ({type(error).__name__})") from None
        if not 200 <= status < 300:
            raise ConnectionError(f"webhook answered HTTP {status}")
