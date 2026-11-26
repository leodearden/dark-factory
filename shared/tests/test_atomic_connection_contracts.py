"""Two contracts of ``shared.async_sqlite_base`` that every AtomicConnection store relies on.

* A write unit's rollback never displaces the exception that ended the unit,
  even when a second cancellation lands while the rollback is awaited.
* ``AtomicConnection.checkpoint_or_unavailable`` is the one place a store that is not open
  answers :meth:`CheckpointResult.unavailable` instead of raising.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import pytest_asyncio

from shared.async_sqlite_base import (
    AtomicConnection,
    CheckpointResult,
    apply_wal_pragmas,
    connect_daemon,
)


@pytest_asyncio.fixture
async def one_row_access(tmp_path: Path):
    conn = await connect_daemon(str(tmp_path / 'contracts.db'))
    try:
        await apply_wal_pragmas(conn, busy_timeout_ms=0)
        await conn.execute('CREATE TABLE t (id TEXT PRIMARY KEY, v TEXT)')
        await conn.execute("INSERT INTO t VALUES ('x', '0')")
        await conn.commit()
        yield conn, AtomicConnection(conn)
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_a_cancel_during_the_rollback_does_not_displace_the_units_error(one_row_access):
    """The unit's own error escapes, and its rollback still runs.

    The cancel is delivered while the task is parked on aiosqlite's rollback
    future: ``rolling_back`` wakes this test before the worker thread can post
    the rollback's result, so the cancel always lands there.
    """
    conn, access = one_row_access
    rolling_back = asyncio.Event()
    real_rollback = conn.rollback

    async def rollback_announced() -> None:
        rolling_back.set()
        await real_rollback()

    conn.rollback = rollback_announced

    async def failing_unit() -> None:
        async with access.write() as db:
            await db.execute("UPDATE t SET v = 'doomed' WHERE id = 'x'")
            raise ValueError('the unit failed')

    unit = asyncio.create_task(failing_unit())
    await rolling_back.wait()
    unit.cancel()
    (outcome,) = await asyncio.gather(unit, return_exceptions=True)

    assert isinstance(outcome, ValueError), (
        f'the second cancellation displaced the unit error: {outcome!r}'
    )
    row = await access.read_one("SELECT v FROM t WHERE id = 'x'")
    assert row is not None
    assert row[0] == '0', 'the rollback did not run after its await was cancelled'


@pytest.mark.asyncio
async def test_a_store_that_is_not_open_answers_unavailable():
    assert await AtomicConnection.checkpoint_or_unavailable(None) == CheckpointResult.unavailable()


@pytest.mark.asyncio
async def test_an_open_access_runs_the_checkpoint(one_row_access):
    _, access = one_row_access

    result = await AtomicConnection.checkpoint_or_unavailable(access)

    assert result != CheckpointResult.unavailable()
    assert result.busy == 0
