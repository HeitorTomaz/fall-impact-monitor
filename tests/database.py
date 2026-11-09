"""Storage tests run against a throwaway PostgreSQL (FIM_TEST_DATABASE_URL): the
`postgres-test` service of the dev profile, or the CI service container."""

import asyncio
import os
from collections.abc import Iterator

import pytest
from sqlalchemy import text

from fall_impact_monitor.storage import make_engine, upgrade

URL = os.environ.get("FIM_TEST_DATABASE_URL", "")
needs_database = pytest.mark.skipif(not URL, reason="FIM_TEST_DATABASE_URL is not set")


async def _reset() -> None:
    engine = make_engine(URL)
    async with engine.begin() as conn:
        await conn.execute(text("DROP SCHEMA public CASCADE"))
        await conn.execute(text("CREATE SCHEMA public"))
    await engine.dispose()


@pytest.fixture
def database() -> Iterator[str]:
    """An empty, migrated test database."""
    asyncio.run(_reset())
    upgrade(URL)
    yield URL
