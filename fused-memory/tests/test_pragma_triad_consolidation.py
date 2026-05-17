"""Tests for pragma-triad consolidation across seven fused-memory SQLite stores.

Each test class patches ``apply_full_durability_pragmas`` at the module path
the target store imports from, drives the init lifecycle path, and asserts the
helper was awaited exactly once with ``busy_timeout_ms=5000``.

This is a delegation test — not a re-verification of the helper's pragma
contract (which lives in shared/tests/test_async_sqlite_base.py).  What we
test here is *whether the store calls the helper at all*: the failure mode
this consolidation effort is designed to prevent.

Mirrors the pattern from orchestrator/tests/test_sqlite_pragmas.py and
dashboard/tests/test_durability.py.
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
# TestTicketStorePragmaDelegation
# ---------------------------------------------------------------------------


class TestTicketStorePragmaDelegation:
    """TicketStore.initialize() delegates the pragma triad to apply_full_durability_pragmas."""

    @pytest.mark.asyncio
    async def test_initialize_calls_apply_full_durability_pragmas(
        self, tmp_path: Path
    ) -> None:
        with patch(
            'fused_memory.middleware.ticket_store.apply_full_durability_pragmas',
            new_callable=AsyncMock,
            create=True,
        ) as mock_helper:
            store = TicketStore(tmp_path / 'tickets.db')
            await store.initialize()
            mock_helper.assert_awaited_once_with(ANY, busy_timeout_ms=5000)
            assert isinstance(mock_helper.call_args.args[0], aiosqlite.Connection)
            await store.close()


# ---------------------------------------------------------------------------
# TestDurableWriteQueuePragmaDelegation
# ---------------------------------------------------------------------------


class TestDurableWriteQueuePragmaDelegation:
    """DurableWriteQueue.initialize() delegates the pragma triad to apply_full_durability_pragmas."""

    @pytest.mark.asyncio
    async def test_initialize_calls_apply_full_durability_pragmas(
        self, tmp_path: Path
    ) -> None:
        with patch(
            'fused_memory.services.durable_queue.apply_full_durability_pragmas',
            new_callable=AsyncMock,
            create=True,
        ) as mock_helper:
            queue = DurableWriteQueue(
                data_dir=tmp_path / 'queue',
                execute_write=AsyncMock(),
            )
            await queue.initialize()
            mock_helper.assert_awaited_once_with(ANY, busy_timeout_ms=5000)
            assert isinstance(mock_helper.call_args.args[0], aiosqlite.Connection)
            await queue.close()


# ---------------------------------------------------------------------------
# TestPlannedEpisodeRegistryPragmaDelegation
# ---------------------------------------------------------------------------


class TestPlannedEpisodeRegistryPragmaDelegation:
    """PlannedEpisodeRegistry.initialize() delegates the pragma triad to apply_full_durability_pragmas."""

    @pytest.mark.asyncio
    async def test_initialize_calls_apply_full_durability_pragmas(
        self, tmp_path: Path
    ) -> None:
        with patch(
            'fused_memory.services.planned_episode_registry.apply_full_durability_pragmas',
            new_callable=AsyncMock,
            create=True,
        ) as mock_helper:
            registry = PlannedEpisodeRegistry(data_dir=tmp_path / 'registry')
            await registry.initialize()
            mock_helper.assert_awaited_once_with(ANY, busy_timeout_ms=5000)
            assert isinstance(mock_helper.call_args.args[0], aiosqlite.Connection)
            await registry.close()


# ---------------------------------------------------------------------------
# TestWriteJournalPragmaDelegation
# ---------------------------------------------------------------------------


class TestWriteJournalPragmaDelegation:
    """WriteJournal.initialize() delegates the pragma triad to apply_full_durability_pragmas."""

    @pytest.mark.asyncio
    async def test_initialize_calls_apply_full_durability_pragmas(
        self, tmp_path: Path
    ) -> None:
        with patch(
            'fused_memory.services.write_journal.apply_full_durability_pragmas',
            new_callable=AsyncMock,
            create=True,
        ) as mock_helper:
            journal = WriteJournal(data_dir=tmp_path / 'journal')
            await journal.initialize()
            mock_helper.assert_awaited_once_with(ANY, busy_timeout_ms=5000)
            assert isinstance(mock_helper.call_args.args[0], aiosqlite.Connection)
            await journal.close()


# ---------------------------------------------------------------------------
# TestSqliteTaskBackendPragmaDelegation
# ---------------------------------------------------------------------------


class TestSqliteTaskBackendPragmaDelegation:
    """SqliteTaskBackend._get_connection() delegates the pragma triad to apply_full_durability_pragmas.

    Also verifies that ``PRAGMA foreign_keys=OFF`` is still applied after the
    helper call — it is not part of the durability triad and must be preserved.
    """

    @pytest.mark.asyncio
    async def test_get_connection_calls_apply_full_durability_pragmas(
        self, tmp_path: Path
    ) -> None:
        with patch(
            'fused_memory.backends.sqlite_task_backend.apply_full_durability_pragmas',
            new_callable=AsyncMock,
            create=True,
        ) as mock_helper:
            backend = SqliteTaskBackend()
            await backend.start()
            project_root = str(tmp_path)
            # Trigger _get_connection via get_tasks
            await backend.get_tasks(project_root=project_root)
            mock_helper.assert_awaited_once_with(ANY, busy_timeout_ms=5000)
            assert isinstance(mock_helper.call_args.args[0], aiosqlite.Connection)
            # PRAGMA foreign_keys=OFF must still be applied after the helper
            conn = backend._connections.get(project_root)
            assert conn is not None
            cursor = await conn.execute('PRAGMA foreign_keys')
            row = await cursor.fetchone()
            assert row[0] == 0, f'Expected foreign_keys=0 (OFF); got {row[0]}'
            await backend.close()


# ---------------------------------------------------------------------------
# TestEventBufferPragmaDelegation
# ---------------------------------------------------------------------------


class TestEventBufferPragmaDelegation:
    """EventBuffer.initialize() delegates the pragma triad for file-backed DBs only.

    The fallback callers (server/main.py:580, server/tools.py:1598) pass
    db_path=None which becomes ':memory:'.  SQLite refuses WAL on ':memory:'
    so the helper must be skipped — this class regression-pins both cases.
    """

    @pytest.mark.asyncio
    async def test_initialize_with_file_path_calls_helper(
        self, tmp_path: Path
    ) -> None:
        """File-backed EventBuffer delegates to the helper exactly once."""
        with patch(
            'fused_memory.reconciliation.event_buffer.apply_full_durability_pragmas',
            new_callable=AsyncMock,
            create=True,
        ) as mock_helper:
            buf = EventBuffer(db_path=tmp_path / 'eb.db')
            await buf.initialize()
            mock_helper.assert_awaited_once_with(ANY, busy_timeout_ms=5000)
            assert isinstance(mock_helper.call_args.args[0], aiosqlite.Connection)
            await buf.close()

    @pytest.mark.asyncio
    async def test_initialize_with_memory_db_skips_helper(self) -> None:
        """When db_path=None (':memory:'), helper must NOT be called and initialize() must succeed.

        Regression-pins the contract for server/main.py:580 and
        server/tools.py:1598 which both pass db_path=None as a
        Taskmaster-disabled fallback.
        """
        with patch(
            'fused_memory.reconciliation.event_buffer.apply_full_durability_pragmas',
            new_callable=AsyncMock,
            create=True,
        ) as mock_helper:
            buf = EventBuffer(db_path=None)
            await buf.initialize()  # must not raise
            mock_helper.assert_not_awaited()
            await buf.close()


# ---------------------------------------------------------------------------
# TestReconciliationJournalPragmaDelegation
# ---------------------------------------------------------------------------


class TestReconciliationJournalPragmaDelegation:
    """ReconciliationJournal.initialize() delegates the pragma triad to apply_full_durability_pragmas."""

    @pytest.mark.asyncio
    async def test_initialize_calls_apply_full_durability_pragmas(
        self, tmp_path: Path
    ) -> None:
        with patch(
            'fused_memory.reconciliation.journal.apply_full_durability_pragmas',
            new_callable=AsyncMock,
            create=True,
        ) as mock_helper:
            journal = ReconciliationJournal(data_dir=tmp_path / 'journal')
            await journal.initialize()
            mock_helper.assert_awaited_once_with(ANY, busy_timeout_ms=5000)
            assert isinstance(mock_helper.call_args.args[0], aiosqlite.Connection)
            await journal.close()
