"""Builds the dispatcher and the enabled outputs from the settings.

The JSON event log is always on; the database sink exists when an engine is given; the
webhook when ``FIM_WEBHOOK_URL`` is set; one Telegram sink per configured chat when
``FIM_TELEGRAM_ENABLED`` is true.
"""

from dataclasses import dataclass, field
from pathlib import Path

from aiogram import Bot
from sqlalchemy.ext.asyncio import AsyncEngine

from fall_impact_monitor.config import Settings
from fall_impact_monitor.sinks.base import Item, Sink
from fall_impact_monitor.sinks.database import DatabaseSink
from fall_impact_monitor.sinks.dispatcher import Dispatcher
from fall_impact_monitor.sinks.event_log import JsonLogSink
from fall_impact_monitor.sinks.outbox import Outbox
from fall_impact_monitor.sinks.telegram.api import AiogramApi, TelegramApi
from fall_impact_monitor.sinks.telegram.sink import TelegramChatSink
from fall_impact_monitor.sinks.webhook import WebhookSink


@dataclass
class Outputs:
    telegram: dict[int, TelegramChatSink] = field(default_factory=dict)
    telegram_api: TelegramApi | None = None
    bot: Bot | None = None  # set when a real bot was created here (close its session)


def build_dispatcher(
    settings: Settings,
    *,
    engine: AsyncEngine | None,
    telegram_api: TelegramApi | None = None,
    outbox_dir: Path | None = None,
) -> tuple[Dispatcher, Outputs]:
    outputs = Outputs()
    sinks: list[Sink] = [JsonLogSink(Path(settings.event_log_file))]
    if engine is not None:
        sinks.append(DatabaseSink(engine))
    if settings.webhook_url is not None:
        sinks.append(
            WebhookSink(
                settings.webhook_url.get_secret_value(), timeout_s=settings.webhook_timeout_s
            )
        )
    created: list[Dispatcher] = []

    def report(item: Item) -> None:
        """Delivery results go back through the same dispatcher (to the database)."""
        created[0].submit_item(item)

    if settings.telegram_enabled:
        if telegram_api is None:
            outputs.bot = Bot(settings.telegram_bot_token.get_secret_value())
            telegram_api = AiogramApi(outputs.bot)
        outputs.telegram_api = telegram_api
        for chat_id in settings.telegram_chat_id_list():
            sink = TelegramChatSink(
                telegram_api,
                chat_id,
                language=settings.alert_language,
                timezone=settings.alert_timezone,
                aggregation_window_s=settings.alert_aggregation_window_s,
                late_after_s=settings.alert_late_after_s,
                report=report,
            )
            outputs.telegram[chat_id] = sink
            sinks.append(sink)
    created.append(Dispatcher(Outbox(outbox_dir or Path(settings.outbox_dir)), sinks))
    return created[0], outputs
