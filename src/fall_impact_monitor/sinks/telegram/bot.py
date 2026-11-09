"""aiogram glue: outbound long polling (no inbound port) that forwards button presses and
commands to ``TelegramController``. Used by the monitoring service."""

import asyncio
import logging
from collections.abc import Awaitable, Callable

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command
from aiogram.types import CallbackQuery, Message

from fall_impact_monitor.sinks.telegram.controller import TelegramController

log = logging.getLogger(__name__)


async def keep_polling(
    poll: Callable[[], Awaitable[None]],
    *,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> None:
    """Restart polling after any failure (no internet at start-up, Telegram outage), so
    buttons and commands come back by themselves."""
    delay_s = 5.0
    while True:
        try:
            await poll()
            return
        except asyncio.CancelledError:
            raise
        except Exception as error:
            log.warning(
                "Telegram polling stopped, restarting", extra={"error": type(error).__name__}
            )
            await sleep(delay_s)
            delay_s = min(delay_s * 2, 300.0)


def build_router(controller: TelegramController) -> Router:
    router = Router()

    @router.callback_query(F.data.startswith("fim:"))
    async def on_callback(query: CallbackQuery) -> None:
        message = query.message
        if message is None:
            return
        await controller.on_button(message.chat.id, message.message_id, query.id, query.data)

    @router.message(Command("status", "help", "start"))
    async def on_command(message: Message) -> None:
        command = (message.text or "").split()[0].lstrip("/").split("@")[0]
        reply = await controller.on_command(message.chat.id, command)
        if reply:
            await message.answer(reply)

    return router


async def run_polling(bot: Bot, controller: TelegramController) -> None:
    dispatcher = Dispatcher()
    dispatcher.include_router(build_router(controller))
    await dispatcher.start_polling(bot, handle_signals=False, close_bot_session=False)
