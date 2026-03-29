"""Tests for CostStore — async aiosqlite-backed cost/invocation persistence."""

from __future__ import annotations

from pathlib import Path

import aiosqlite
import pytest

from shared.cost_store import CostStore


@pytest.mark.asyncio
class TestSchemaCreation:
    """Verify that CostStore creates the expected tables and indexes."""

    async def test_schema_creation(self, tmp_path: Path):
        """invocations and account_events tables exist after open()."""
        db_path = tmp_path / 'runs.db'
        await CostStore.open(db_path)

        async with aiosqlite.connect(str(db_path)) as conn, conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ) as cursor:
            tables = {row[0] for row in await cursor.fetchall()}

        assert 'invocations' in tables
        assert 'account_events' in tables

    async def test_schema_idempotent(self, tmp_path: Path):
        """Two opens on the same db_path do not raise."""
        db_path = tmp_path / 'runs.db'
        await CostStore.open(db_path)
        await CostStore.open(db_path)  # should not raise

    async def test_creates_parent_dirs(self, tmp_path: Path):
        """db_path with nested non-existent parent dirs is created."""
        db_path = tmp_path / 'a' / 'b' / 'c' / 'runs.db'
        assert not db_path.parent.exists()
        await CostStore.open(db_path)
        assert db_path.exists()

    async def test_indexes_created(self, tmp_path: Path):
        """All 4 expected indexes exist after open()."""
        db_path = tmp_path / 'runs.db'
        await CostStore.open(db_path)

        async with aiosqlite.connect(str(db_path)) as conn, conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' ORDER BY name"
        ) as cursor:
            indexes = {row[0] for row in await cursor.fetchall()}

        assert 'idx_inv_project' in indexes
        assert 'idx_inv_account' in indexes
        assert 'idx_inv_run' in indexes
        assert 'idx_acct_evt_account' in indexes


@pytest.mark.asyncio
class TestWalMode:
    """Verify WAL journal mode is set."""

    async def test_wal_mode(self, tmp_path: Path):
        """PRAGMA journal_mode returns 'wal' after open()."""
        db_path = tmp_path / 'runs.db'
        await CostStore.open(db_path)

        async with aiosqlite.connect(str(db_path)) as conn, conn.execute('PRAGMA journal_mode') as cursor:
            row = await cursor.fetchone()
        assert row is not None
        assert row[0] == 'wal'


@pytest.mark.asyncio
class TestSaveInvocation:
    """Tests for CostStore.save_invocation()."""

    async def test_save_invocation_roundtrip(self, tmp_path: Path):
        """save_invocation stores all fields correctly."""
        db_path = tmp_path / 'runs.db'
        store = await CostStore.open(db_path)

        await store.save_invocation(
            run_id='run-abc123',
            task_id='task-42',
            project_id='dark_factory',
            account_name='max-a',
            model='claude-opus-4-5',
            role='implementer',
            cost_usd=0.025,
            input_tokens=1000,
            output_tokens=500,
            cache_read_tokens=200,
            cache_create_tokens=100,
            duration_ms=3500,
            capped=False,
            started_at='2026-01-01T00:00:00',
            completed_at='2026-01-01T00:00:03',
        )

        async with aiosqlite.connect(str(db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            async with conn.execute('SELECT * FROM invocations') as cursor:
                rows = await cursor.fetchall()

        assert len(rows) == 1
        row = rows[0]
        assert row['run_id'] == 'run-abc123'
        assert row['task_id'] == 'task-42'
        assert row['project_id'] == 'dark_factory'
        assert row['account_name'] == 'max-a'
        assert row['model'] == 'claude-opus-4-5'
        assert row['role'] == 'implementer'
        assert abs(row['cost_usd'] - 0.025) < 1e-9
        assert row['input_tokens'] == 1000
        assert row['output_tokens'] == 500
        assert row['cache_read_tokens'] == 200
        assert row['cache_create_tokens'] == 100
        assert row['duration_ms'] == 3500
        assert row['capped'] == 0  # stored as integer 0/1
        assert row['started_at'] == '2026-01-01T00:00:00'
        assert row['completed_at'] == '2026-01-01T00:00:03'
        assert row['id'] is not None  # autoincrement

    async def test_save_invocation_autoincrement(self, tmp_path: Path):
        """id is autoincremented across multiple inserts."""
        db_path = tmp_path / 'runs.db'
        store = await CostStore.open(db_path)

        kwargs = dict(
            run_id='run-1',
            task_id=None,
            project_id='p',
            account_name='acc',
            model='m',
            role='r',
            cost_usd=0.0,
            input_tokens=None,
            output_tokens=None,
            cache_read_tokens=None,
            cache_create_tokens=None,
            duration_ms=0,
            capped=False,
            started_at='2026-01-01T00:00:00',
            completed_at='2026-01-01T00:00:01',
        )
        await store.save_invocation(**kwargs)
        await store.save_invocation(**{**kwargs, 'run_id': 'run-2'})

        async with aiosqlite.connect(str(db_path)) as conn, conn.execute('SELECT id FROM invocations ORDER BY id') as cursor:
            ids = [row[0] for row in await cursor.fetchall()]

        assert ids == [1, 2]

    async def test_save_invocation_nullable_fields(self, tmp_path: Path):
        """save_invocation stores NULLs for nullable columns."""
        db_path = tmp_path / 'runs.db'
        store = await CostStore.open(db_path)

        await store.save_invocation(
            run_id='run-x',
            task_id=None,
            project_id='p',
            account_name='acc',
            model='m',
            role='r',
            cost_usd=0.0,
            input_tokens=None,
            output_tokens=None,
            cache_read_tokens=None,
            cache_create_tokens=None,
            duration_ms=0,
            capped=False,
            started_at='2026-01-01T00:00:00',
            completed_at='2026-01-01T00:00:01',
        )

        async with aiosqlite.connect(str(db_path)) as conn, conn.execute('SELECT * FROM invocations') as cursor:
            rows = await cursor.fetchall()
            cols = [d[0] for d in cursor.description]

        assert len(rows) == 1
        row_dict = dict(zip(cols, rows[0], strict=False))
        assert row_dict['task_id'] is None
        assert row_dict['input_tokens'] is None
        assert row_dict['output_tokens'] is None
        assert row_dict['cache_read_tokens'] is None
        assert row_dict['cache_create_tokens'] is None


@pytest.mark.asyncio
class TestSaveAccountEvent:
    """Tests for CostStore.save_account_event()."""

    async def test_save_account_event_roundtrip(self, tmp_path: Path):
        """save_account_event stores all fields correctly."""
        db_path = tmp_path / 'runs.db'
        store = await CostStore.open(db_path)

        await store.save_account_event(
            account_name='max-a',
            event_type='cap_hit',
            project_id='dark_factory',
            run_id='run-abc',
            details='{"reason": "usage limit"}',
            created_at='2026-01-01T12:00:00',
        )

        async with aiosqlite.connect(str(db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            async with conn.execute('SELECT * FROM account_events') as cursor:
                rows = await cursor.fetchall()

        assert len(rows) == 1
        row = rows[0]
        assert row['account_name'] == 'max-a'
        assert row['event_type'] == 'cap_hit'
        assert row['project_id'] == 'dark_factory'
        assert row['run_id'] == 'run-abc'
        assert row['details'] == '{"reason": "usage limit"}'
        assert row['created_at'] == '2026-01-01T12:00:00'
        assert row['id'] is not None

    async def test_save_account_event_nullable_fields(self, tmp_path: Path):
        """save_account_event stores NULLs for nullable columns."""
        db_path = tmp_path / 'runs.db'
        store = await CostStore.open(db_path)

        await store.save_account_event(
            account_name='max-b',
            event_type='resume',
            project_id=None,
            run_id=None,
            details=None,
            created_at='2026-01-01T12:00:00',
        )

        async with aiosqlite.connect(str(db_path)) as conn, conn.execute('SELECT * FROM account_events') as cursor:
            rows = await cursor.fetchall()
            cols = [d[0] for d in cursor.description]

        assert len(rows) == 1
        row_dict = dict(zip(cols, rows[0], strict=False))
        assert row_dict['project_id'] is None
        assert row_dict['run_id'] is None
        assert row_dict['details'] is None
