"""Generic webhook: POST each notified event (and delivery tests) as JSON.

The URL may carry a token, so it is never logged nor put in error messages. Any status
outside 2xx, or a network error, raises so the dispatcher retries the item.
"""

import aiohttp

from fall_impact_monitor.sinks.base import Item, is_notifiable


class WebhookSink:
    name = "webhook"

    def __init__(self, url: str, *, timeout_s: float) -> None:
        self._url = url
        self._timeout = aiohttp.ClientTimeout(total=timeout_s)

    def accepts(self, item: Item) -> bool:
        return is_notifiable(item)

    async def deliver(self, item: Item) -> None:
        payload = {"kind": item["kind"], **item.get("record", {})}
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
