"""Answers to Telegram buttons and commands, as plain async methods (the aiogram glue in
``bot.py`` only forwards updates here). Chats that are not configured are ignored."""

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path

from fall_impact_monitor.events.wording import Language, message, video_caption
from fall_impact_monitor.recording.clips import find_clip
from fall_impact_monitor.sinks.base import Item
from fall_impact_monitor.sinks.telegram.api import TelegramApi, parse_callback
from fall_impact_monitor.sinks.telegram.sink import TelegramChatSink

log = logging.getLogger(__name__)
_HELP = {
    "en": "/status - state of the cameras and of the system",
    "pt-BR": "/status - estado das câmeras e do sistema",
}


class TelegramController:
    def __init__(
        self,
        api: TelegramApi,
        sinks: dict[int, TelegramChatSink],
        submit_item: Callable[[Item], None],
        *,
        clip_dir: Path,
        language: Language,
        timezone: str,
        status_text: Callable[[], str],
        clip_wait_s: float,
        clip_poll_s: float = 5.0,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._api = api
        self._sinks = sinks
        self._submit = submit_item
        self._clip_dir = clip_dir
        self._language: Language = language
        self._timezone = timezone
        self._status_text = status_text
        self._clip_wait_s = clip_wait_s  # a press before the export: wait this long for it
        self._clip_poll_s = clip_poll_s
        self._clock = clock
        self._sleep = sleep
        self._waiting: set[tuple[int, str]] = set()  # (chat, event) presses awaiting a clip

    async def on_button(
        self, chat_id: int, message_id: int, callback_id: str, data: str | None
    ) -> None:
        if chat_id not in self._sinks:
            return
        parsed = parse_callback(data)
        if parsed is None:
            await self._api.answer(callback_id, message(self._language, "unknown_event"))
            return
        action, event_id = parsed
        if action == "video":
            await self._send_video(chat_id, message_id, callback_id, event_id)
            return
        self._submit(
            {
                "kind": "feedback",
                "event_id": event_id,
                "action": action,
                "created_utc": self._clock().isoformat(),
            }
        )
        self._sinks[chat_id].mark_seen(message_id)
        await self._api.answer(callback_id, message(self._language, "feedback_saved"))

    async def _send_video(
        self, chat_id: int, message_id: int, callback_id: str, event_id: str
    ) -> None:
        clip = find_clip(self._clip_dir, event_id)
        if clip is None:
            await self._api.answer(callback_id, message(self._language, "video_when_ready"))
            if (chat_id, event_id) in self._waiting:
                return  # an earlier press already waits for this clip
            self._waiting.add((chat_id, event_id))
            try:
                clip = await self._wait_for_clip(event_id)
            finally:
                self._waiting.discard((chat_id, event_id))
            if clip is None:
                await self._api.send_message(
                    chat_id,
                    message(self._language, "video_missing"),
                    buttons=[],
                    reply_to=message_id,
                    silent=True,
                )
                return
        else:
            await self._api.answer(callback_id, message(self._language, "video_sending"))
        sidecar_path = clip.with_suffix(".json")
        caption = ""
        if sidecar_path.exists():
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            caption = video_caption(sidecar, language=self._language, timezone=self._timezone)
        try:
            await self._api.send_video(chat_id, clip, caption=caption, reply_to=message_id)
        except Exception as error:
            log.error("video not sent", extra={"event": event_id, "error": type(error).__name__})
            await self._api.send_message(
                chat_id,
                message(self._language, "video_failed"),
                buttons=[],
                reply_to=message_id,
                silent=True,
            )

    async def _wait_for_clip(self, event_id: str) -> Path | None:
        """The clip exists only after the post-event window and the export."""
        waited_s = 0.0
        while waited_s < self._clip_wait_s:
            await self._sleep(self._clip_poll_s)
            waited_s += self._clip_poll_s
            clip = find_clip(self._clip_dir, event_id)
            if clip is not None:
                return clip
        return None

    async def on_command(self, chat_id: int, command: str) -> str | None:
        if chat_id not in self._sinks:
            return None
        if command == "status":
            return self._status_text()
        if command in ("help", "start"):
            return _HELP[self._language]
        return None
