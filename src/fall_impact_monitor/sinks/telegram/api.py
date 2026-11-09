"""The Telegram operations the system needs, behind a small protocol, and their aiogram
implementation. Callback data is ``fim:<action>:<event id>`` (at most 64 bytes)."""

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest, TelegramNetworkError, TelegramRetryAfter
from aiogram.types import FSInputFile, InlineKeyboardButton, InlineKeyboardMarkup, ReplyParameters

from fall_impact_monitor.sinks.base import RetryLater

ACTIONS = ("seen", "false_alarm", "confirmed", "video")


class MessageGone(Exception):
    """The message to edit no longer exists (deleted in the chat)."""


@dataclass(frozen=True)
class Button:
    action: str
    event_id: str
    label: str


def callback_data(action: str, event_id: str) -> str:
    return f"fim:{action}:{event_id}"


def parse_callback(data: str | None) -> tuple[str, str] | None:
    parts = (data or "").split(":", 2)
    if len(parts) != 3 or parts[0] != "fim" or parts[1] not in ACTIONS:
        return None
    try:
        uuid.UUID(parts[2])
    except ValueError:
        return None
    return parts[1], parts[2]


class TelegramApi(Protocol):
    async def send_message(
        self, chat_id: int, text: str, *, buttons: list[Button], reply_to: int | None, silent: bool
    ) -> int: ...

    async def edit_message(
        self, chat_id: int, message_id: int, text: str, *, buttons: list[Button]
    ) -> None: ...

    async def send_video(
        self, chat_id: int, path: Path, *, caption: str, reply_to: int | None
    ) -> int: ...

    async def answer(self, callback_id: str, text: str) -> None: ...


def _keyboard(buttons: list[Button]) -> InlineKeyboardMarkup | None:
    if not buttons:
        return None
    feedback = [b for b in buttons if b.action != "video"]
    rows = [feedback] + [[b] for b in buttons if b.action == "video"]
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=b.label, callback_data=callback_data(b.action, b.event_id)
                )
                for b in row
            ]
            for row in rows
            if row
        ]
    )


def _reply(message_id: int | None) -> ReplyParameters | None:
    if message_id is None:
        return None
    return ReplyParameters(message_id=message_id, allow_sending_without_reply=True)


class AiogramApi:
    """Maps flood control to ``RetryLater`` and network errors to ``ConnectionError``, so
    the dispatcher retries. Tokens and chat ids never appear in the raised messages."""

    def __init__(self, bot: Bot) -> None:
        self._bot = bot

    async def send_message(
        self, chat_id: int, text: str, *, buttons: list[Button], reply_to: int | None, silent: bool
    ) -> int:
        try:
            message = await self._bot.send_message(
                chat_id,
                text,
                reply_markup=_keyboard(buttons),
                reply_parameters=_reply(reply_to),
                disable_notification=silent,
            )
        except TelegramRetryAfter as error:
            raise RetryLater(error.retry_after) from None
        except TelegramNetworkError:
            raise ConnectionError("Telegram unreachable") from None
        return message.message_id

    async def edit_message(
        self, chat_id: int, message_id: int, text: str, *, buttons: list[Button]
    ) -> None:
        try:
            await self._bot.edit_message_text(
                text=text, chat_id=chat_id, message_id=message_id, reply_markup=_keyboard(buttons)
            )
        except TelegramRetryAfter as error:
            raise RetryLater(error.retry_after) from None
        except TelegramNetworkError:
            raise ConnectionError("Telegram unreachable") from None
        except TelegramBadRequest as error:
            if "not modified" not in str(error):
                raise MessageGone(type(error).__name__) from None

    async def send_video(
        self, chat_id: int, path: Path, *, caption: str, reply_to: int | None
    ) -> int:
        message = await self._bot.send_video(
            chat_id,
            FSInputFile(path),
            caption=caption,
            reply_parameters=_reply(reply_to),
            supports_streaming=True,
        )
        return message.message_id

    async def answer(self, callback_id: str, text: str) -> None:
        await self._bot.answer_callback_query(callback_id, text=text or None)
