"""Verify that all four SQLite stores inherit from AsyncSqliteBase.

This module is the canonical record of the migration contract established in
Task 322.  Each store must:
  - Subclass AsyncSqliteBase
  - Expose a ``_schema`` property returning the DDL string
  - Manage lifecycle via ``open()`` / ``close()`` (NOT ``initialize()``)
  - Use ``_require_conn()`` (NOT ``_require_db()``)
  - Support the async context-manager protocol
  - NOT expose an ``initialize()`` method
"""

import pytest
import pytest_asyncio

from shared.async_sqlite_base import AsyncSqliteBase


# ---------------------------------------------------------------------------
# Step-1 / Step-2: EventBuffer migration tests
# ---------------------------------------------------------------------------


class TestEventBufferMigration:
    """EventBuffer must be a well-formed AsyncSqliteBase subclass."""

    def test_event_buffer_is_async_sqlite_base_subclass(self):
        from fused_memory.reconciliation.event_buffer import EventBuffer

        assert issubclass(EventBuffer, AsyncSqliteBase), (
            'EventBuffer must inherit from AsyncSqliteBase'
        )

    def test_event_buffer_has_schema_property(self):
        """_schema property must exist and return a non-empty DDL string."""
        from fused_memory.reconciliation.event_buffer import EventBuffer

        buf = EventBuffer(db_path=None)
        schema = buf._schema
        assert isinstance(schema, str)
        assert len(schema) > 0
        assert 'event_buffer' in schema

    def test_event_buffer_has_no_initialize_method(self):
        from fused_memory.reconciliation.event_buffer import EventBuffer

        assert not hasattr(EventBuffer, 'initialize'), (
            'EventBuffer must not expose initialize(); use open() instead'
        )

    def test_event_buffer_has_no_require_db_method(self):
        from fused_memory.reconciliation.event_buffer import EventBuffer

        assert not hasattr(EventBuffer, '_require_db'), (
            'EventBuffer must not have _require_db(); use _require_conn() from base'
        )

    def test_event_buffer_has_require_conn_method(self):
        from fused_memory.reconciliation.event_buffer import EventBuffer

        assert hasattr(EventBuffer, '_require_conn'), (
            'EventBuffer must have _require_conn() from AsyncSqliteBase'
        )

    @pytest.mark.asyncio
    async def test_event_buffer_open_close(self, tmp_path):
        from fused_memory.reconciliation.event_buffer import EventBuffer

        buf = EventBuffer(db_path=tmp_path / 'buf.db')
        await buf.open()
        assert buf._conn is not None
        await buf.close()
        assert buf._conn is None

    @pytest.mark.asyncio
    async def test_event_buffer_context_manager(self, tmp_path):
        from fused_memory.reconciliation.event_buffer import EventBuffer

        async with EventBuffer(db_path=tmp_path / 'ctx.db') as buf:
            assert buf._conn is not None
        assert buf._conn is None

    @pytest.mark.asyncio
    async def test_event_buffer_in_memory_mode(self):
        """EventBuffer(db_path=None) opens ':memory:' database correctly."""
        from fused_memory.reconciliation.event_buffer import EventBuffer

        buf = EventBuffer(db_path=None)
        await buf.open()
        try:
            conn = buf._require_conn()
            assert conn is not None
        finally:
            await buf.close()

    @pytest.mark.asyncio
    async def test_event_buffer_require_conn_raises_when_closed(self):
        from fused_memory.reconciliation.event_buffer import EventBuffer

        buf = EventBuffer(db_path=None)
        with pytest.raises(RuntimeError, match='not opened'):
            buf._require_conn()


# ---------------------------------------------------------------------------
# Step-4 / Step-5: ReconciliationJournal migration tests
# ---------------------------------------------------------------------------


class TestReconciliationJournalMigration:
    """ReconciliationJournal must be a well-formed AsyncSqliteBase subclass."""

    def test_reconciliation_journal_is_async_sqlite_base_subclass(self):
        from fused_memory.reconciliation.journal import ReconciliationJournal

        assert issubclass(ReconciliationJournal, AsyncSqliteBase), (
            'ReconciliationJournal must inherit from AsyncSqliteBase'
        )

    def test_reconciliation_journal_has_schema_property(self):
        """_schema property must exist and return a non-empty DDL string."""
        from fused_memory.reconciliation.journal import ReconciliationJournal
        from pathlib import Path

        journal = ReconciliationJournal(data_dir=Path('/tmp'))
        schema = journal._schema
        assert isinstance(schema, str)
        assert len(schema) > 0
        assert 'runs' in schema
        assert 'watermarks' in schema

    def test_reconciliation_journal_has_no_initialize_method(self):
        from fused_memory.reconciliation.journal import ReconciliationJournal

        assert not hasattr(ReconciliationJournal, 'initialize'), (
            'ReconciliationJournal must not expose initialize(); use open() instead'
        )

    def test_reconciliation_journal_has_no_require_db_method(self):
        from fused_memory.reconciliation.journal import ReconciliationJournal

        assert not hasattr(ReconciliationJournal, '_require_db'), (
            'ReconciliationJournal must not have _require_db(); use _require_conn()'
        )

    @pytest.mark.asyncio
    async def test_reconciliation_journal_open_close(self, tmp_path):
        from fused_memory.reconciliation.journal import ReconciliationJournal

        journal = ReconciliationJournal(data_dir=tmp_path)
        await journal.open()
        assert journal._conn is not None
        await journal.close()
        assert journal._conn is None

    @pytest.mark.asyncio
    async def test_reconciliation_journal_context_manager(self, tmp_path):
        from fused_memory.reconciliation.journal import ReconciliationJournal

        async with ReconciliationJournal(data_dir=tmp_path) as journal:
            assert journal._conn is not None
        assert journal._conn is None

    @pytest.mark.asyncio
    async def test_triggered_by_column_migration(self, tmp_path):
        """open() must add triggered_by column to pre-existing DBs (idempotent)."""
        import aiosqlite
        from fused_memory.reconciliation.journal import ReconciliationJournal

        db_path = tmp_path / 'reconciliation.db'

        # Create a minimal DB without triggered_by column (simulating old schema)
        async with aiosqlite.connect(str(db_path)) as db:
            await db.execute("""
                CREATE TABLE runs (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    run_type TEXT NOT NULL,
                    trigger_reason TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    events_processed INTEGER DEFAULT 0,
                    stage_reports TEXT DEFAULT '{}',
                    status TEXT DEFAULT 'running'
                )
            """)
            await db.commit()

        # open() must silently add the triggered_by column
        journal = ReconciliationJournal(data_dir=tmp_path)
        await journal.open()
        try:
            conn = journal._require_conn()
            async with conn.execute('PRAGMA table_info(runs)') as cursor:
                cols = {row[1] for row in await cursor.fetchall()}
            assert 'triggered_by' in cols
        finally:
            await journal.close()

    @pytest.mark.asyncio
    async def test_triggered_by_migration_idempotent(self, tmp_path):
        """Opening the same journal twice must not raise (column already exists)."""
        from fused_memory.reconciliation.journal import ReconciliationJournal

        journal = ReconciliationJournal(data_dir=tmp_path)
        await journal.open()
        await journal.close()

        # Second open must not raise on the duplicate ALTER TABLE
        journal2 = ReconciliationJournal(data_dir=tmp_path)
        await journal2.open()
        await journal2.close()


# ---------------------------------------------------------------------------
# Step-12: Final verification tests — all 4 stores share the same contract
# ---------------------------------------------------------------------------


import pytest


@pytest.mark.parametrize('store_class,kwargs', [
    ('fused_memory.services.durable_queue.DurableWriteQueue', {'data_dir': None}),
    ('fused_memory.services.write_journal.WriteJournal', {'data_dir': None}),
    ('fused_memory.reconciliation.event_buffer.EventBuffer', {'db_path': None}),
    ('fused_memory.reconciliation.journal.ReconciliationJournal', {'data_dir': None}),
])
def test_all_stores_are_async_sqlite_base_subclasses(store_class, kwargs, tmp_path):
    """All four SQLite stores must inherit from AsyncSqliteBase."""
    import importlib
    module_path, cls_name = store_class.rsplit('.', 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, cls_name)
    assert issubclass(cls, AsyncSqliteBase), (
        f'{cls_name} must inherit from AsyncSqliteBase'
    )


@pytest.mark.parametrize('store_class', [
    'fused_memory.services.durable_queue.DurableWriteQueue',
    'fused_memory.services.write_journal.WriteJournal',
    'fused_memory.reconciliation.event_buffer.EventBuffer',
    'fused_memory.reconciliation.journal.ReconciliationJournal',
])
def test_no_store_has_initialize_method(store_class):
    """No store must expose initialize() — the lifecycle method is open()."""
    import importlib
    module_path, cls_name = store_class.rsplit('.', 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, cls_name)
    assert not hasattr(cls, 'initialize'), (
        f'{cls_name} must not have initialize(); use open() instead'
    )


@pytest.mark.parametrize('store_class', [
    'fused_memory.services.durable_queue.DurableWriteQueue',
    'fused_memory.services.write_journal.WriteJournal',
    'fused_memory.reconciliation.event_buffer.EventBuffer',
    'fused_memory.reconciliation.journal.ReconciliationJournal',
])
def test_no_store_has_require_db_method(store_class):
    """No store must expose _require_db() — the guard method is _require_conn()."""
    import importlib
    module_path, cls_name = store_class.rsplit('.', 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, cls_name)
    assert not hasattr(cls, '_require_db'), (
        f'{cls_name} must not have _require_db(); use _require_conn() from base'
    )


@pytest.mark.parametrize('store_class,init_kwargs', [
    ('fused_memory.services.write_journal.WriteJournal', {}),
    ('fused_memory.reconciliation.event_buffer.EventBuffer', {'db_path': None}),
])
@pytest.mark.asyncio
async def test_stores_support_context_manager_protocol(store_class, init_kwargs, tmp_path):
    """Stores must support 'async with' for safe lifecycle management."""
    import importlib
    module_path, cls_name = store_class.rsplit('.', 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, cls_name)

    # Use a real path for stores that need directories
    if 'data_dir' not in init_kwargs and 'db_path' not in init_kwargs:
        init_kwargs['data_dir'] = tmp_path
    elif init_kwargs.get('db_path') is None:
        pass  # Use None (in-memory)
    else:
        init_kwargs['data_dir'] = tmp_path

    async with cls(**init_kwargs) as store:
        assert store._conn is not None
    assert store._conn is None
