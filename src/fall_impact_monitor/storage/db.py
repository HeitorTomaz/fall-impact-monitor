"""Database engine and migrations (Alembic, configured in code: no alembic.ini)."""

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

_MIGRATIONS = Path(__file__).parent / "migrations"


def make_engine(url: str) -> AsyncEngine:
    return create_async_engine(url, pool_pre_ping=True)


def upgrade(url: str) -> None:
    """Apply every pending migration. Must be called outside a running event loop."""
    config = Config()
    config.set_main_option("script_location", str(_MIGRATIONS))
    config.attributes["url"] = url  # not in the ini options: '%' in passwords is safe
    command.upgrade(config, "head")
