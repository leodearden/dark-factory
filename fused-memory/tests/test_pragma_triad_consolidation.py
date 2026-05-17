"""Tests for pragma-triad consolidation across seven fused-memory SQLite stores.

Each store replaces five inline PRAGMA statements with a single call to the
canonical helper ``apply_full_durability_pragmas``.  These are *delegation*
tests — not re-verifications of the helper's pragma contract (which lives in
``shared/tests/test_async_sqlite_base.py``).  What we test here is whether
the store calls the helper at all: the failure mode this consolidation effort
is designed to prevent.

Mirrors the pattern from ``orchestrator/tests/test_sqlite_pragmas.py`` and
``dashboard/tests/test_durability.py``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import ANY, AsyncMock, patch

import aiosqlite
import pytest

from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.middleware.ticket_store import TicketStore
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.reconciliation.journal import ReconciliationJournal
from fused_memory.services.durable_queue import DurableWriteQueue
from fused_memory.services.planned_episode_registry import PlannedEpisodeRegistry
from fused_memory.services.write_journal import WriteJournal

# ---------------------------------------------------------------------------
# Lifecycle helpers
# ---------------------------------------------------------------------------


async def _std_lifecycle(store, tmp_path: Path) -> None:  # noqa: ARG001
    """Standard initialize() lifecycle used by all stores except SqliteTaskBackend."""
    await store.initialize()


async def _backend_lifecycle(store: SqliteTaskBackend, tmp_path: Path) -> None:
    """SqliteTaskBackend lifecycle: start() then trigger _get_connection via get_tasks()."""
    await store.start()
    await store.get_tasks(project_root=str(tmp_path))


# ---------------------------------------------------------------------------
# Parametrized delegation test — one case per store
# ---------------------------------------------------------------------------

_DELEGATION_CASES = [
    pytest.param(
        'fused_memory.middleware.ticket_store.apply_full_durability_pragmas',
        lambda tmp: TicketStore(tmp / 'tickets.db'),
        _std_lifecycle,
        id='TicketStore',
    ),
    pytest.param(
        'fused_memory.services.durable_queue.apply_full_durability_pragmas',
        lambda tmp: DurableWriteQueue(data_dir=tmp / 'queue', execute_write=AsyncMock()),
        _std_lifecycle,
        id='DurableWriteQueue',
    ),
    pytest.param(
        'fused_memory.services.planned_episode_registry.apply_full_durability_pragmas',
        lambda tmp: PlannedEpisodeRegistry(data_dir=tmp / 'registry'),
        _std_lifecycle,
        id='PlannedEpisodeRegistry',
    ),
    pytest.param(
        'fused_memory.services.write_journal.apply_full_durability_pragmas',
        lambda tmp: WriteJournal(data_dir=tmp / 'journal'),
        _std_lifecycle,
        id='WriteJournal',
    ),
    pytest.param(
        'fused_memory.backends.sqlite_task_backend.apply_full_durability_pragmas',
        lambda tmp: SqliteTaskBackend(),  # noqa: ARG005
        _backend_lifecycle,
        id='SqliteTaskBackend',
    ),
    pytest.param(
        'fused_memory.reconciliation.event_buffer.apply_full_durability_pragmas',
        lambda tmp: EventBuffer(db_path=tmp / 'eb.db'),
        _std_lifecycle,
        id='EventBuffer',
    ),
    pytest.param(
        'fused_memory.reconciliation.journal.apply_full_durability_pragmas',
        lambda tmp: ReconciliationJournal(data_dir=tmp / 'journal'),
        _std_lifecycle,
        id='ReconciliationJournal',
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize('patch_path,make_store,lifecycle', _DELEGATION_CASES)
async def test_pragma_delegation(
    patch_path: str,
    make_store,
    lifecycle,
    tmp_path: Path,
) -> None:
    """Each store delegates the pragma triad to apply_full_durability_pragmas exactly once."""
    with patch(patch_path, new_callable=AsyncMock, create=True) as mock_helper:
        store = make_store(tmp_path)
        await lifecycle(store, tmp_path)
        mock_helper.assert_awaited_once_with(ANY, busy_timeout_ms=5000)
        assert isinstance(mock_helper.call_args.args[0], aiosqlite.Connection)
        await store.close()


# ---------------------------------------------------------------------------
# SqliteTaskBackend: PRAGMA foreign_keys=OFF regression
# ---------------------------------------------------------------------------


class TestSqliteTaskBackendForeignKeysOff:
    """_get_connection() must explicitly set PRAGMA foreign_keys=OFF after the helper.

    The assertion is meaningful because we pre-enable foreign_keys=ON via a
    patched ``aiosqlite.connect`` wrapper.  Reading 0 afterwards proves the
    explicit ``PRAGMA foreign_keys=OFF`` statement ran — not just SQLite's
    default value of 0, which would make the assertion vacuously true.
    """

    @pytest.mark.asyncio
    async def test_foreign_keys_off_after_helper(self, tmp_path: Path) -> None:
        _real_connect = aiosqlite.connect

        async def _connect_with_fk_on(path: str) -> aiosqlite.Connection:
            """Open a real connection and enable foreign_keys before returning."""
            conn = await _real_connect(path)
            await conn.execute('PRAGMA foreign_keys=ON')
            return conn

        with patch(
            'fused_memory.backends.sqlite_task_backend.apply_full_durability_pragmas',
            new_callable=AsyncMock,
            create=True,
        ), patch('aiosqlite.connect', side_effect=_connect_with_fk_on):
            backend = SqliteTaskBackend()
            await backend.start()
            project_root = str(tmp_path)
            await backend.get_tasks(project_root=project_root)
            conn = backend._connections.get(project_root)
            assert conn is not None
            cursor = await conn.execute('PRAGMA foreign_keys')
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] == 0, f'Expected foreign_keys=0 (OFF); got {row[0]}'
            await backend.close()


# ---------------------------------------------------------------------------
# EventBuffer: in-memory fallback regression
# ---------------------------------------------------------------------------


class TestEventBufferMemoryFallback:
    """initialize() skips the pragma helper when db_path=None (in-memory SQLite).

    Regression-pins the contract for server/main.py:580 and
    server/tools.py:1598, which both pass ``db_path=None`` as a
    Taskmaster-disabled fallback.  SQLite refuses WAL on ':memory:' DBs so
    the helper must not be called.  The guard uses ``_in_memory`` (derived
    from ``db_path is None`` at construction) rather than a string comparison
    so the contract is tied to caller intent, not a specific string literal.
    """

    @pytest.mark.asyncio
    async def test_initialize_with_memory_db_skips_helper(self) -> None:
        with patch(
            'fused_memory.reconciliation.event_buffer.apply_full_durability_pragmas',
            new_callable=AsyncMock,
            create=True,
        ) as mock_helper:
            buf = EventBuffer(db_path=None)
            await buf.initialize()  # must not raise
            mock_helper.assert_not_awaited()
            await buf.close()
