"""Tests for CostStore persistent connection + async context manager."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import aiosqlite
import pytest

from shared.cost_store import CostStore


@pytest.mark.asyncio
class TestCostStoreInit:
    """step-1: CostStore.__init__ stores db_path and sets _conn to None."""

    async def test_init_stores_db_path(self, tmp_path: Path):
        db_path = tmp_path / 'costs.db'
        store = CostStore(db_path)
        assert store.db_path == db_path

    async def test_init_conn_is_none(self, tmp_path: Path):
        db_path = tmp_path / 'costs.db'
        store = CostStore(db_path)
        assert store._conn is None


@pytest.mark.asyncio
class TestSaveInvocation:
    """step-9: save_invocation() persists a row with all 15 fields."""

    async def test_save_invocation_full_row(self, tmp_path: Path):
        async with CostStore(tmp_path / 'costs.db') as store:
            await store.save_invocation(
                run_id='run-abc',
                task_id='task-42',
                project_id='dark_factory',
                account_name='max-d',
                model='claude-3-5-sonnet',
                role='agent',
                cost_usd=0.123,
                input_tokens=500,
                output_tokens=200,
                cache_read_tokens=100,
                cache_create_tokens=50,
                duration_ms=1234,
                capped=False,
                started_at='2024-01-01T00:00:00',
                completed_at='2024-01-01T00:00:01',
            )
            # Verify via raw SQL
            async with store._conn.execute(
                'SELECT run_id, task_id, project_id, account_name, model, role, '
                'cost_usd, input_tokens, output_tokens, cache_read_tokens, '
                'cache_create_tokens, duration_ms, capped, started_at, completed_at '
                'FROM invocations'
            ) as cur:
                row = await cur.fetchone()
        assert row[0] == 'run-abc'
        assert row[1] == 'task-42'
        assert row[2] == 'dark_factory'
        assert row[3] == 'max-d'
        assert row[4] == 'claude-3-5-sonnet'
        assert row[5] == 'agent'
        assert abs(row[6] - 0.123) < 1e-9
        assert row[7] == 500
        assert row[8] == 200
        assert row[9] == 100
        assert row[10] == 50
        assert row[11] == 1234
        assert row[12] == 0  # capped=False stored as 0
        assert row[13] == '2024-01-01T00:00:00'
        assert row[14] == '2024-01-01T00:00:01'

    async def test_save_invocation_nullable_fields(self, tmp_path: Path):
        """task_id and token counts can be None."""
        async with CostStore(tmp_path / 'costs.db') as store:
            await store.save_invocation(
                run_id='run-xyz',
                task_id=None,
                project_id='p',
                account_name='a',
                model='m',
                role='r',
                cost_usd=0.0,
                input_tokens=None,
                output_tokens=None,
                cache_read_tokens=None,
                cache_create_tokens=None,
                duration_ms=0,
                capped=True,
                started_at='2024-01-01T00:00:00',
                completed_at='2024-01-01T00:00:00',
            )
            async with store._conn.execute(
                'SELECT task_id, input_tokens, output_tokens, cache_read_tokens, '
                'cache_create_tokens, capped FROM invocations'
            ) as cur:
                row = await cur.fetchone()
        assert row[0] is None
        assert row[1] is None
        assert row[2] is None
        assert row[3] is None
        assert row[4] is None
        assert row[5] == 1  # capped=True stored as 1

    async def test_save_invocation_multiple_rows(self, tmp_path: Path):
        """Multiple invocations saved in same session."""
        async with CostStore(tmp_path / 'costs.db') as store:
            for i in range(3):
                await store.save_invocation(
                    run_id=f'run-{i}',
                    task_id=None,
                    project_id='p',
                    account_name='a',
                    model='m',
                    role='r',
                    cost_usd=float(i),
                    input_tokens=None,
                    output_tokens=None,
                    cache_read_tokens=None,
                    cache_create_tokens=None,
                    duration_ms=i * 100,
                    capped=False,
                    started_at='2024-01-01T00:00:00',
                    completed_at='2024-01-01T00:00:00',
                )
            async with store._conn.execute('SELECT COUNT(*) FROM invocations') as cur:
                (count,) = await cur.fetchone()
        assert count == 3


@pytest.mark.asyncio
class TestSaveAccountEvent:
    """step-11: save_account_event() persists a row with all 6 fields."""

    async def test_save_account_event_full_row(self, tmp_path: Path):
        async with CostStore(tmp_path / 'costs.db') as store:
            await store.save_account_event(
                account_name='max-d',
                event_type='cap_hit',
                project_id='dark_factory',
                run_id='run-abc',
                details='{"reset_at": "2024-01-01T05:00:00"}',
                created_at='2024-01-01T00:00:00',
            )
            async with store._conn.execute(
                'SELECT account_name, event_type, project_id, run_id, details, created_at '
                'FROM account_events'
            ) as cur:
                row = await cur.fetchone()
        assert row[0] == 'max-d'
        assert row[1] == 'cap_hit'
        assert row[2] == 'dark_factory'
        assert row[3] == 'run-abc'
        assert row[4] == '{"reset_at": "2024-01-01T05:00:00"}'
        assert row[5] == '2024-01-01T00:00:00'

    async def test_save_account_event_nullable_fields(self, tmp_path: Path):
        """project_id, run_id, and details can be None."""
        async with CostStore(tmp_path / 'costs.db') as store:
            await store.save_account_event(
                account_name='max-c',
                event_type='switch',
                project_id=None,
                run_id=None,
                details=None,
                created_at='2024-06-15T12:00:00',
            )
            async with store._conn.execute(
                'SELECT project_id, run_id, details FROM account_events'
            ) as cur:
                row = await cur.fetchone()
        assert row[0] is None
        assert row[1] is None
        assert row[2] is None

    async def test_save_account_event_multiple_rows(self, tmp_path: Path):
        """Multiple events saved in same session."""
        async with CostStore(tmp_path / 'costs.db') as store:
            for evt in ('cap_hit', 'switch', 'resume'):
                await store.save_account_event(
                    account_name='max-d',
                    event_type=evt,
                    project_id=None,
                    run_id=None,
                    details=None,
                    created_at='2024-01-01T00:00:00',
                )
            async with store._conn.execute(
                'SELECT COUNT(*) FROM account_events'
            ) as cur:
                (count,) = await cur.fetchone()
        assert count == 3


@pytest.mark.asyncio
class TestConnectionReuse:
    """step-13: aiosqlite.connect is called exactly once during open()."""

    async def test_connect_called_once_for_multiple_saves(self, tmp_path: Path):
        """Multiple save calls reuse the persistent connection; connect called once."""
        db_path = tmp_path / 'costs.db'

        real_connect = aiosqlite.connect
        connect_call_count = 0

        async def counting_connect(path):
            nonlocal connect_call_count
            connect_call_count += 1
            return await real_connect(path)

        with patch('shared.cost_store.aiosqlite.connect', side_effect=counting_connect):
            async with CostStore(db_path) as store:
                for i in range(3):
                    await store.save_invocation(
                        run_id=f'r{i}',
                        task_id=None,
                        project_id='p',
                        account_name='a',
                        model='m',
                        role='r',
                        cost_usd=0.0,
                        input_tokens=None,
                        output_tokens=None,
                        cache_read_tokens=None,
                        cache_create_tokens=None,
                        duration_ms=0,
                        capped=False,
                        started_at='2024-01-01T00:00:00',
                        completed_at='2024-01-01T00:00:00',
                    )
                for _i in range(2):
                    await store.save_account_event(
                        account_name='a',
                        event_type='cap_hit',
                        project_id=None,
                        run_id=None,
                        details=None,
                        created_at='2024-01-01T00:00:00',
                    )

        assert connect_call_count == 1


@pytest.mark.asyncio
class TestCostStoreNotOpenedGuard:
    """step-7: save methods raise RuntimeError when called before open()."""

    async def test_save_invocation_raises_if_not_opened(self, tmp_path: Path):
        store = CostStore(tmp_path / 'costs.db')
        with pytest.raises(RuntimeError, match='CostStore not opened'):
            await store.save_invocation(
                run_id='r1',
                task_id=None,
                project_id='proj',
                account_name='acct',
                model='claude-3',
                role='agent',
                cost_usd=0.01,
                input_tokens=None,
                output_tokens=None,
                cache_read_tokens=None,
                cache_create_tokens=None,
                duration_ms=100,
                capped=False,
                started_at='2024-01-01T00:00:00',
                completed_at='2024-01-01T00:00:01',
            )

    async def test_save_account_event_raises_if_not_opened(self, tmp_path: Path):
        store = CostStore(tmp_path / 'costs.db')
        with pytest.raises(RuntimeError, match='CostStore not opened'):
            await store.save_account_event(
                account_name='acct',
                event_type='cap_hit',
                project_id=None,
                run_id=None,
                details=None,
                created_at='2024-01-01T00:00:00',
            )


@pytest.mark.asyncio
class TestCostStoreContextManager:
    """step-5: async context manager opens on enter, closes on exit."""

    async def test_context_manager_opens_conn(self, tmp_path: Path):
        async with CostStore(tmp_path / 'costs.db') as store:
            assert store._conn is not None

    async def test_context_manager_closes_conn_on_exit(self, tmp_path: Path):
        async with CostStore(tmp_path / 'costs.db') as store:
            pass  # just enter and exit
        assert store._conn is None

    async def test_context_manager_returns_self(self, tmp_path: Path):
        cs = CostStore(tmp_path / 'costs.db')
        async with cs as store:
            assert store is cs

    async def test_context_manager_closes_on_exception(self, tmp_path: Path):
        """Connection is closed even when body raises."""
        store = None
        try:
            async with CostStore(tmp_path / 'costs.db') as s:
                store = s
                raise ValueError('boom')
        except ValueError:
            pass
        assert store is not None
        assert store._conn is None


@pytest.mark.asyncio
class TestCostStoreOpenClose:
    """step-3: open() creates persistent WAL connection; close() is idempotent."""

    async def test_open_sets_conn(self, tmp_path: Path):
        store = CostStore(tmp_path / 'costs.db')
        await store.open()
        try:
            assert store._conn is not None
        finally:
            await store.close()

    async def test_open_sets_wal_mode(self, tmp_path: Path):
        store = CostStore(tmp_path / 'costs.db')
        await store.open()
        try:
            async with store._conn.execute('PRAGMA journal_mode') as cur:
                row = await cur.fetchone()
            assert row[0] == 'wal'
        finally:
            await store.close()

    async def test_open_sets_busy_timeout(self, tmp_path: Path):
        store = CostStore(tmp_path / 'costs.db')
        await store.open()
        try:
            async with store._conn.execute('PRAGMA busy_timeout') as cur:
                row = await cur.fetchone()
            assert row[0] == 30000
        finally:
            await store.close()

    async def test_close_sets_conn_to_none(self, tmp_path: Path):
        store = CostStore(tmp_path / 'costs.db')
        await store.open()
        await store.close()
        assert store._conn is None

    async def test_close_is_idempotent(self, tmp_path: Path):
        """Double-close should not raise."""
        store = CostStore(tmp_path / 'costs.db')
        await store.open()
        await store.close()
        await store.close()  # should not raise
        assert store._conn is None

    async def test_open_creates_parent_dirs(self, tmp_path: Path):
        nested = tmp_path / 'a' / 'b' / 'c' / 'costs.db'
        store = CostStore(nested)
        await store.open()
        try:
            assert nested.exists()
        finally:
            await store.close()

    async def test_open_no_leak_on_setup_error(self, tmp_path: Path):
        """If executescript raises after connect, conn is closed and _conn stays None."""
        store = CostStore(tmp_path / 'costs.db')
        real_connect = aiosqlite.connect
        close_called = False

        async def fake_connect(path):
            nonlocal close_called
            conn = await real_connect(path)
            original_close = conn.close

            async def tracking_close():
                nonlocal close_called
                close_called = True
                await original_close()

            async def failing_executescript(_sql: str) -> None:
                raise RuntimeError('schema failure')

            conn.close = tracking_close
            conn.executescript = failing_executescript
            return conn

        with patch('shared.cost_store.aiosqlite.connect', side_effect=fake_connect), pytest.raises(RuntimeError, match='schema failure'):
            await store.open()

        assert store._conn is None, '_conn should remain None on setup failure'
        assert close_called, 'Connection must be closed to prevent resource leak'
