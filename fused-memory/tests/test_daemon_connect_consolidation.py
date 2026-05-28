"""Tests for daemon-thread worker protection across seven fused-memory SQLite stores.

Each store must open its aiosqlite connection through ``connect_daemon`` so the
background worker thread is marked daemon before it starts.  These are
*behavioral* tests — they assert the daemon flag directly on the live connection
rather than checking delegation to the helper.  A store that bypasses the helper
and calls aiosqlite.connect directly would still fail here.

Modeled on ``test_pragma_triad_consolidation.py``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

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
# Parametrized daemon test — one case per store
# ---------------------------------------------------------------------------

_DAEMON_CASES = [
    pytest.param(
        lambda tmp: TicketStore(tmp / 'tickets.db'),
        _std_lifecycle,
        lambda store, _root: store._db,
        id='TicketStore',
    ),
    pytest.param(
        lambda tmp: DurableWriteQueue(data_dir=tmp / 'queue', execute_write=AsyncMock()),
        _std_lifecycle,
        lambda store, _root: store._db,
        id='DurableWriteQueue',
    ),
    pytest.param(
        lambda tmp: PlannedEpisodeRegistry(data_dir=tmp / 'registry'),
        _std_lifecycle,
        lambda store, _root: store._db,
        id='PlannedEpisodeRegistry',
    ),
    pytest.param(
        lambda tmp: WriteJournal(data_dir=tmp / 'journal'),
        _std_lifecycle,
        lambda store, _root: store._db,
        id='WriteJournal',
    ),
    pytest.param(
        lambda tmp: EventBuffer(db_path=tmp / 'eb.db'),  # noqa: ARG005
        _std_lifecycle,
        lambda store, _root: store._db,
        id='EventBuffer',
    ),
    pytest.param(
        lambda tmp: ReconciliationJournal(data_dir=tmp / 'journal'),  # noqa: ARG005
        _std_lifecycle,
        lambda store, _root: store._db,
        id='ReconciliationJournal',
    ),
    pytest.param(
        lambda tmp: SqliteTaskBackend(),  # noqa: ARG005
        _backend_lifecycle,
        lambda store, root: store._connections[root],
        id='SqliteTaskBackend',
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize('make_store,lifecycle,get_conn', _DAEMON_CASES)
async def test_store_connection_is_daemon(
    make_store,
    lifecycle,
    get_conn,
    tmp_path: Path,
) -> None:
    """Each store's aiosqlite worker thread must be daemon after initialization.

    RED until each store routes its connect call through connect_daemon:
    aiosqlite's default worker-thread daemon flag is False under pytest's main
    thread, so this assertion genuinely fails when a store calls aiosqlite.connect
    directly.
    """
    project_root = str(tmp_path)
    store = make_store(tmp_path)
    await lifecycle(store, tmp_path)
    try:
        conn = get_conn(store, project_root)
        assert conn is not None, 'Expected an open connection'
        thread = conn._thread
        assert thread.is_alive(), 'Worker thread should be alive while connection is open'
        assert thread.daemon is True, (
            f'{type(store).__name__}: aiosqlite worker thread must be daemon so a '
            'leaked connection cannot block interpreter shutdown'
        )
    finally:
        await store.close()
