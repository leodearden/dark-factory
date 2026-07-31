"""Tests for CostStore persistent connection + async context manager."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import aiosqlite
import pytest

from shared.cost_store import CostStore


def _conn(store: CostStore) -> aiosqlite.Connection:
    """Return ``store._conn`` narrowed to the non-Optional connection.

    CostStore.__init__ sets ``_conn`` to None; after ``open()`` it holds a
    connection.  Tests that use the connection directly (e.g. raw SQL
    verification) need the non-Optional form — assert here rather than
    repeating the assertion at every call site.
    """
    conn = store._conn
    assert conn is not None, 'store._conn is None — did you forget to open()?'
    return conn


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
            async with _conn(store).execute(
                'SELECT run_id, task_id, project_id, account_name, model, role, '
                'cost_usd, input_tokens, output_tokens, cache_read_tokens, '
                'cache_create_tokens, duration_ms, capped, started_at, completed_at '
                'FROM invocations'
            ) as cur:
                row = await cur.fetchone()
        assert row is not None
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
            async with _conn(store).execute(
                'SELECT task_id, input_tokens, output_tokens, cache_read_tokens, '
                'cache_create_tokens, capped FROM invocations'
            ) as cur:
                row = await cur.fetchone()
        assert row is not None
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
            async with _conn(store).execute('SELECT COUNT(*) FROM invocations') as cur:
                row = await cur.fetchone()
        assert row is not None
        (count,) = row
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
            async with _conn(store).execute(
                'SELECT account_name, event_type, project_id, run_id, details, created_at '
                'FROM account_events'
            ) as cur:
                row = await cur.fetchone()
        assert row is not None
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
            async with _conn(store).execute(
                'SELECT project_id, run_id, details FROM account_events'
            ) as cur:
                row = await cur.fetchone()
        assert row is not None
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
            async with _conn(store).execute(
                'SELECT COUNT(*) FROM account_events'
            ) as cur:
                row = await cur.fetchone()
        assert row is not None
        (count,) = row
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

        with patch('shared.async_sqlite_base.aiosqlite.connect', side_effect=counting_connect):
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
    """step-3: open() creates persistent WAL connection; close() is idempotent; double-open raises."""

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
            async with _conn(store).execute('PRAGMA journal_mode') as cur:
                row = await cur.fetchone()
            assert row is not None
            assert row[0] == 'wal'
        finally:
            await store.close()

    async def test_open_sets_busy_timeout(self, tmp_path: Path):
        store = CostStore(tmp_path / 'costs.db')
        await store.open()
        try:
            async with _conn(store).execute('PRAGMA busy_timeout') as cur:
                row = await cur.fetchone()
            assert row is not None
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

    async def test_double_open_raises(self, tmp_path: Path):
        """Calling open() twice raises RuntimeError('CostStore already opened').

        After the second open() raises, the store must still be functional:
        _conn must not be None and save_invocation must succeed.
        """
        store = CostStore(tmp_path / 'costs.db')
        await store.open()
        try:
            with pytest.raises(RuntimeError, match='CostStore already opened'):
                await store.open()
            # _conn must not be corrupted by the failed second open
            assert store._conn is not None, '_conn was corrupted by rejected double-open'
            # store must still be usable
            await store.save_invocation(
                run_id='r-after-double-open',
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
            # Test-only monkeypatch: swap the method to simulate schema failure.
            # Pyright flags cross-signature assignment; suppress narrowly.
            conn.executescript = failing_executescript  # type: ignore[method-assign,assignment]
            return conn

        with patch('shared.async_sqlite_base.aiosqlite.connect', side_effect=fake_connect), pytest.raises(RuntimeError, match='schema failure'):
            await store.open()

        assert store._conn is None, '_conn should remain None on setup failure'
        assert close_called, 'Connection must be closed to prevent resource leak'

    async def test_concurrent_open_does_not_leak(self, tmp_path: Path):
        """Two concurrent open() calls must not both succeed; connect() called exactly once.

        Without asyncio.Lock, both coroutines pass the `_conn is not None` guard before
        either assigns _conn, causing two connections to be opened (orphaning one).
        With the lock, exactly one open() succeeds and one raises RuntimeError.
        """
        db_path = tmp_path / 'costs.db'
        store = CostStore(db_path)

        real_connect = aiosqlite.connect
        connect_call_count = 0

        async def counting_connect(path):
            nonlocal connect_call_count
            connect_call_count += 1
            return await real_connect(path)

        with patch('shared.async_sqlite_base.aiosqlite.connect', side_effect=counting_connect):
            results = await asyncio.gather(
                store.open(),
                store.open(),
                return_exceptions=True,
            )

        try:
            successes = [r for r in results if r is None]
            errors = [r for r in results if isinstance(r, RuntimeError)]
            assert len(successes) == 1, f'Expected exactly one success, got {successes!r}'
            assert len(errors) == 1, f'Expected exactly one RuntimeError, got {errors!r}'
            assert 'already opened' in str(errors[0])
            assert connect_call_count == 1, (
                f'Expected connect() called once, got {connect_call_count}'
            )
        finally:
            await store.close()


@pytest.mark.asyncio
class TestPostCloseGuard:
    """step-5: after close(), save methods must raise RuntimeError (regression guard)."""

    async def test_save_invocation_raises_after_close(self, tmp_path: Path):
        """open() -> close() -> save_invocation() must raise RuntimeError('CostStore not opened')."""
        store = CostStore(tmp_path / 'costs.db')
        await store.open()
        await store.close()
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

    async def test_save_account_event_raises_after_close(self, tmp_path: Path):
        """open() -> close() -> save_account_event() must raise RuntimeError('CostStore not opened')."""
        store = CostStore(tmp_path / 'costs.db')
        await store.open()
        await store.close()
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
class TestPersistAcrossReopen:
    """step-6: data saved before close() survives a close/reopen cycle."""

    async def test_invocation_persists_across_reopen(self, tmp_path: Path):
        """save_invocation() -> close() -> reopen() -> SELECT verifies row exists."""
        db_path = tmp_path / 'costs.db'

        # First session: write a row
        store = CostStore(db_path)
        await store.open()
        await store.save_invocation(
            run_id='run-persist',
            task_id='task-1',
            project_id='dark_factory',
            account_name='max-d',
            model='claude-3-5-sonnet',
            role='agent',
            cost_usd=0.05,
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=None,
            cache_create_tokens=None,
            duration_ms=500,
            capped=False,
            started_at='2024-06-01T10:00:00',
            completed_at='2024-06-01T10:00:01',
        )
        await store.close()

        # Second session: reopen and verify row is still there
        store2 = CostStore(db_path)
        await store2.open()
        try:
            async with _conn(store2).execute(
                'SELECT run_id, task_id, cost_usd FROM invocations'
            ) as cur:
                row = await cur.fetchone()
        finally:
            await store2.close()

        assert row is not None, 'Row must survive close/reopen cycle'
        assert row[0] == 'run-persist'
        assert row[1] == 'task-1'
        assert abs(row[2] - 0.05) < 1e-9

    async def test_account_event_persists_across_reopen(self, tmp_path: Path):
        """save_account_event() -> close() -> reopen() -> SELECT verifies row exists."""
        db_path = tmp_path / 'costs.db'

        # First session: write a row
        store = CostStore(db_path)
        await store.open()
        await store.save_account_event(
            account_name='max-d',
            event_type='cap_hit',
            project_id='dark_factory',
            run_id='run-persist',
            details='{"reset_at": "2024-01-01T05:00:00"}',
            created_at='2024-06-01T10:00:00',
        )
        await store.close()

        # Second session: reopen and verify row is still there
        store2 = CostStore(db_path)
        await store2.open()
        try:
            async with _conn(store2).execute(
                'SELECT account_name, event_type, project_id, run_id, details, created_at'
                ' FROM account_events'
            ) as cur:
                row = await cur.fetchone()
        finally:
            await store2.close()

        assert row is not None, 'Row must survive close/reopen cycle'
        assert row[0] == 'max-d'
        assert row[1] == 'cap_hit'
        assert row[2] == 'dark_factory'
        assert row[3] == 'run-persist'
        assert row[4] == '{"reset_at": "2024-01-01T05:00:00"}'
        assert row[5] == '2024-06-01T10:00:00'


# ---------------------------------------------------------------------------
# step-13: CostStore subclasses AsyncSqliteBase
# ---------------------------------------------------------------------------


class TestCostStoreInheritsAsyncSqliteBase:
    """step-13: CostStore is a subclass of AsyncSqliteBase with correct _schema and busy_timeout."""

    def test_coststore_is_subclass_of_async_sqlite_base(self) -> None:
        """CostStore must be a subclass of AsyncSqliteBase."""
        from shared.async_sqlite_base import AsyncSqliteBase

        assert issubclass(CostStore, AsyncSqliteBase)

    def test_coststore_schema_property_returns_ddl(self, tmp_path: Path) -> None:
        """CostStore._schema returns the _SCHEMA DDL string."""
        from shared.cost_store import _SCHEMA

        store = CostStore(tmp_path / 'costs.db')
        assert store._schema == _SCHEMA

    def test_coststore_busy_timeout_is_30000(self, tmp_path: Path) -> None:
        """CostStore sets busy_timeout_ms=30000 on the base class."""
        store = CostStore(tmp_path / 'costs.db')
        assert store.busy_timeout_ms == 30000


# ---------------------------------------------------------------------------
# TestCostTotalsInWindow — CostStore.cost_totals_in_window(start_iso, end_iso)
# ---------------------------------------------------------------------------


async def _seed_row(
    store: CostStore,
    *,
    role: str,
    cost_usd: float,
    completed_at: str,
    model: str = 'm',
) -> None:
    """Insert a minimal invocation row for cost_totals_in_window testing."""
    await store.save_invocation(
        run_id='r1',
        task_id=None,
        project_id='p',
        account_name='a',
        model=model,
        role=role,
        cost_usd=cost_usd,
        input_tokens=None,
        output_tokens=None,
        cache_read_tokens=None,
        cache_create_tokens=None,
        duration_ms=0,
        capped=False,
        started_at=completed_at,
        completed_at=completed_at,
    )


@pytest.mark.asyncio
class TestCostTotalsInWindow:
    """Tests for CostStore.cost_totals_in_window(start_iso, end_iso) -> (total_usd, watcher_usd)."""

    async def test_empty_db_returns_zeros(self, tmp_path: Path) -> None:
        """Empty table → (0.0, 0.0)."""
        async with CostStore(tmp_path / 'costs.db') as store:
            result = await store.cost_totals_in_window(
                '2026-01-01T00:00:00',
                '2026-12-31T23:59:59',
            )
        assert result == (0.0, 0.0)

    async def test_single_watcher_row_returns_cost_twice(self, tmp_path: Path) -> None:
        """One watcher row inside window → (cost, cost)."""
        async with CostStore(tmp_path / 'costs.db') as store:
            await _seed_row(
                store, role='watcher', cost_usd=3.5,
                completed_at='2026-06-01T12:00:00',
            )
            result = await store.cost_totals_in_window(
                '2026-06-01T00:00:00',
                '2026-06-01T23:59:59',
            )
        assert result == (pytest.approx(3.5), pytest.approx(3.5))

    async def test_single_non_watcher_row_returns_zero_watcher(
        self, tmp_path: Path
    ) -> None:
        """One non-watcher row inside window → (cost, 0.0)."""
        async with CostStore(tmp_path / 'costs.db') as store:
            await _seed_row(
                store, role='orchestrator', cost_usd=2.0,
                completed_at='2026-06-01T12:00:00',
            )
            result = await store.cost_totals_in_window(
                '2026-06-01T00:00:00',
                '2026-06-01T23:59:59',
            )
        assert result == (pytest.approx(2.0), pytest.approx(0.0))

    async def test_mixed_roles_sums_correctly(self, tmp_path: Path) -> None:
        """Mixed watcher + non-watcher → (sum_all, sum_watcher)."""
        async with CostStore(tmp_path / 'costs.db') as store:
            await _seed_row(
                store, role='watcher', cost_usd=4.0,
                completed_at='2026-06-01T10:00:00',
            )
            await _seed_row(
                store, role='orchestrator', cost_usd=1.5,
                completed_at='2026-06-01T11:00:00',
            )
            result = await store.cost_totals_in_window(
                '2026-06-01T00:00:00',
                '2026-06-01T23:59:59',
            )
        total, watcher = result
        assert total == pytest.approx(5.5)
        assert watcher == pytest.approx(4.0)

    async def test_out_of_window_rows_excluded(self, tmp_path: Path) -> None:
        """Rows outside [start, end] are not counted."""
        async with CostStore(tmp_path / 'costs.db') as store:
            # Before window
            await _seed_row(
                store, role='watcher', cost_usd=99.0,
                completed_at='2026-05-31T23:59:59',
            )
            # After window
            await _seed_row(
                store, role='watcher', cost_usd=99.0,
                completed_at='2026-06-02T00:00:00',
            )
            # Inside window
            await _seed_row(
                store, role='watcher', cost_usd=1.0,
                completed_at='2026-06-01T12:00:00',
            )
            result = await store.cost_totals_in_window(
                '2026-06-01T00:00:00',
                '2026-06-01T23:59:59',
            )
        assert result == (pytest.approx(1.0), pytest.approx(1.0))

    async def test_watcher_substring_roles_all_match(self, tmp_path: Path) -> None:
        """Roles with 'watcher' as substring (e.g. 'escalation-watcher-auto',
        'watcher-supervisor') all match the LIKE '%watcher%' pattern."""
        async with CostStore(tmp_path / 'costs.db') as store:
            for role in ('escalation-watcher-auto', 'watcher-supervisor', 'my-watcher'):
                await _seed_row(
                    store, role=role, cost_usd=1.0,
                    completed_at='2026-06-01T12:00:00',
                )
            result = await store.cost_totals_in_window(
                '2026-06-01T00:00:00',
                '2026-06-01T23:59:59',
            )
        total, watcher = result
        assert total == pytest.approx(3.0)
        assert watcher == pytest.approx(3.0)

    async def test_not_opened_raises_runtime_error(self, tmp_path: Path) -> None:
        """Calling cost_totals_in_window on a never-opened store raises RuntimeError."""
        store = CostStore(tmp_path / 'costs.db')
        with pytest.raises(RuntimeError, match='CostStore not opened'):
            await store.cost_totals_in_window(
                '2026-01-01T00:00:00',
                '2026-12-31T23:59:59',
            )


# ---------------------------------------------------------------------------
# TestModelCostInWindow — CostStore.model_cost_in_window(model, start_iso, end_iso)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestModelCostInWindow:
    """Tests for CostStore.model_cost_in_window(model, start_iso, end_iso) -> float."""

    async def test_sums_only_in_window_rows_for_model(self, tmp_path: Path) -> None:
        """Two in-window 'opus' rows + an in-window 'sonnet' row + an out-of-window
        'opus' row -> only the two in-window 'opus' rows are summed."""
        async with CostStore(tmp_path / 'costs.db') as store:
            await _seed_row(
                store, role='implementer', cost_usd=2.0,
                completed_at='2026-06-01T10:00:00', model='opus',
            )
            await _seed_row(
                store, role='debugger', cost_usd=3.0,
                completed_at='2026-06-01T11:00:00', model='opus',
            )
            await _seed_row(
                store, role='implementer', cost_usd=5.0,
                completed_at='2026-06-01T12:00:00', model='sonnet',
            )
            await _seed_row(
                store, role='implementer', cost_usd=99.0,
                completed_at='2026-05-31T23:59:59', model='opus',
            )
            result = await store.model_cost_in_window(
                'opus',
                '2026-06-01T00:00:00',
                '2026-06-01T23:59:59',
            )
        assert result == pytest.approx(5.0)

    async def test_model_with_no_rows_returns_zero(self, tmp_path: Path) -> None:
        """A model with no matching rows in the window -> 0.0."""
        async with CostStore(tmp_path / 'costs.db') as store:
            await _seed_row(
                store, role='implementer', cost_usd=2.0,
                completed_at='2026-06-01T10:00:00', model='sonnet',
            )
            result = await store.model_cost_in_window(
                'opus',
                '2026-06-01T00:00:00',
                '2026-06-01T23:59:59',
            )
        assert result == 0.0

    async def test_empty_db_returns_zero(self, tmp_path: Path) -> None:
        """Empty table -> 0.0."""
        async with CostStore(tmp_path / 'costs.db') as store:
            result = await store.model_cost_in_window(
                'opus',
                '2026-01-01T00:00:00',
                '2026-12-31T23:59:59',
            )
        assert result == 0.0

    async def test_not_opened_raises_runtime_error(self, tmp_path: Path) -> None:
        """Calling model_cost_in_window on a never-opened store raises RuntimeError."""
        store = CostStore(tmp_path / 'costs.db')
        with pytest.raises(RuntimeError, match='CostStore not opened'):
            await store.model_cost_in_window(
                'opus',
                '2026-01-01T00:00:00',
                '2026-12-31T23:59:59',
            )


# ---------------------------------------------------------------------------
# TestSaveApiErrorEvent — CostStore.save_api_error_event(...)
# (task 3325 / plans/server-side-api-error-handling-prd.md C4)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSaveApiErrorEvent:
    """step-1: save_api_error_event() writes one account_events row typed 'api_error'."""

    async def test_writes_single_row_with_api_error_type(self, tmp_path: Path) -> None:
        """All six columns round-trip verbatim, and event_type is the literal 'api_error'.

        The literal matters beyond this test: it is the discriminator ApiHealthGate
        forensics, the dashboard provider-health strip and operators all query on.
        """
        async with CostStore(tmp_path / 'costs.db') as store:
            await store.save_api_error_event(
                account_name='max-d',
                project_id='dark_factory',
                run_id='run-1',
                details='{"status": 529}',
                created_at='2026-07-31T00:00:00+00:00',
            )
            async with _conn(store).execute(
                'SELECT account_name, event_type, project_id, run_id, details, created_at '
                'FROM account_events'
            ) as cur:
                rows = await cur.fetchall()
        assert len(rows) == 1
        assert rows[0] == (
            'max-d',
            'api_error',
            'dark_factory',
            'run-1',
            '{"status": 529}',
            '2026-07-31T00:00:00+00:00',
        )

    async def test_nullable_fields_accept_none(self, tmp_path: Path) -> None:
        """project_id / run_id / details are nullable — a watcher invocation has no project."""
        async with CostStore(tmp_path / 'costs.db') as store:
            await store.save_api_error_event(
                account_name='max-c',
                project_id=None,
                run_id=None,
                details=None,
                created_at='2026-07-31T01:00:00+00:00',
            )
            async with _conn(store).execute(
                'SELECT event_type, project_id, run_id, details FROM account_events'
            ) as cur:
                row = await cur.fetchone()
        assert row == ('api_error', None, None, None)

    async def test_raises_if_not_opened(self, tmp_path: Path) -> None:
        """A never-opened store raises rather than silently dropping forensics."""
        store = CostStore(tmp_path / 'costs.db')
        with pytest.raises(RuntimeError, match='CostStore not opened'):
            await store.save_api_error_event(
                account_name='max-d',
                project_id=None,
                run_id=None,
                details=None,
                created_at='2026-07-31T00:00:00+00:00',
            )


# ---------------------------------------------------------------------------
# TestAccountEventsInWindow — CostStore.account_events_in_window(...)
# (task 3325 / plans/server-side-api-error-handling-prd.md C4)
# ---------------------------------------------------------------------------


async def _seed_event(
    store: CostStore,
    *,
    event_type: str,
    created_at: str,
    account_name: str = 'max-d',
    details: str | None = None,
) -> None:
    """Insert a minimal account_events row for account_events_in_window testing."""
    await store.save_account_event(
        account_name=account_name,
        event_type=event_type,
        project_id='dark_factory',
        run_id='run-1',
        details=details,
        created_at=created_at,
    )


@pytest.mark.asyncio
class TestAccountEventsInWindow:
    """step-3: account_events_in_window() is the read path for api_error forensics."""

    async def test_returns_dicts_keyed_by_column_name(self, tmp_path: Path) -> None:
        """Rows come back as dicts with all six columns — callers never index by position."""
        async with CostStore(tmp_path / 'costs.db') as store:
            await _seed_event(
                store,
                event_type='api_error',
                created_at='2026-07-31T12:00:00+00:00',
                details='{"status": 529}',
            )
            rows = await store.account_events_in_window(
                '2026-07-31T00:00:00+00:00',
                '2026-07-31T23:59:59+00:00',
            )
        assert len(rows) == 1
        assert rows[0] == {
            'account_name': 'max-d',
            'event_type': 'api_error',
            'project_id': 'dark_factory',
            'run_id': 'run-1',
            'details': '{"status": 529}',
            'created_at': '2026-07-31T12:00:00+00:00',
        }

    async def test_window_is_inclusive_at_both_bounds(self, tmp_path: Path) -> None:
        """Matches cost_totals_in_window's inclusive BETWEEN semantics."""
        async with CostStore(tmp_path / 'costs.db') as store:
            await _seed_event(store, event_type='api_error', created_at='2026-07-31T00:00:00+00:00')
            await _seed_event(store, event_type='api_error', created_at='2026-07-31T23:59:59+00:00')
            rows = await store.account_events_in_window(
                '2026-07-31T00:00:00+00:00',
                '2026-07-31T23:59:59+00:00',
            )
        assert len(rows) == 2

    async def test_excludes_rows_outside_window(self, tmp_path: Path) -> None:
        async with CostStore(tmp_path / 'costs.db') as store:
            await _seed_event(store, event_type='api_error', created_at='2026-07-30T23:59:59+00:00')
            await _seed_event(store, event_type='api_error', created_at='2026-07-31T12:00:00+00:00')
            await _seed_event(store, event_type='api_error', created_at='2026-08-01T00:00:00+00:00')
            rows = await store.account_events_in_window(
                '2026-07-31T00:00:00+00:00',
                '2026-07-31T23:59:59+00:00',
            )
        assert [r['created_at'] for r in rows] == ['2026-07-31T12:00:00+00:00']

    async def test_event_type_filter_narrows_to_one_type(self, tmp_path: Path) -> None:
        """event_type='api_error' excludes the pre-existing cap_hit/switch rows."""
        async with CostStore(tmp_path / 'costs.db') as store:
            await _seed_event(store, event_type='cap_hit', created_at='2026-07-31T01:00:00+00:00')
            await _seed_event(store, event_type='api_error', created_at='2026-07-31T02:00:00+00:00')
            await _seed_event(store, event_type='switch', created_at='2026-07-31T03:00:00+00:00')
            filtered = await store.account_events_in_window(
                '2026-07-31T00:00:00+00:00',
                '2026-07-31T23:59:59+00:00',
                event_type='api_error',
            )
            unfiltered = await store.account_events_in_window(
                '2026-07-31T00:00:00+00:00',
                '2026-07-31T23:59:59+00:00',
            )
        assert [r['event_type'] for r in filtered] == ['api_error']
        assert [r['event_type'] for r in unfiltered] == ['cap_hit', 'api_error', 'switch']

    async def test_empty_window_returns_empty_list(self, tmp_path: Path) -> None:
        async with CostStore(tmp_path / 'costs.db') as store:
            await _seed_event(store, event_type='api_error', created_at='2026-07-31T12:00:00+00:00')
            rows = await store.account_events_in_window(
                '2026-01-01T00:00:00+00:00',
                '2026-01-02T00:00:00+00:00',
            )
        assert rows == []

    async def test_orders_by_created_at_ascending(self, tmp_path: Path) -> None:
        """The dashboard strip renders a timeline — insertion order must not leak through."""
        async with CostStore(tmp_path / 'costs.db') as store:
            for stamp in (
                '2026-07-31T05:00:00+00:00',
                '2026-07-31T01:00:00+00:00',
                '2026-07-31T03:00:00+00:00',
            ):
                await _seed_event(store, event_type='api_error', created_at=stamp)
            rows = await store.account_events_in_window(
                '2026-07-31T00:00:00+00:00',
                '2026-07-31T23:59:59+00:00',
            )
        assert [r['created_at'] for r in rows] == [
            '2026-07-31T01:00:00+00:00',
            '2026-07-31T03:00:00+00:00',
            '2026-07-31T05:00:00+00:00',
        ]

    async def test_not_opened_raises_runtime_error(self, tmp_path: Path) -> None:
        """A never-opened store raises rather than reporting a healthy empty window."""
        store = CostStore(tmp_path / 'costs.db')
        with pytest.raises(RuntimeError, match='CostStore not opened'):
            await store.account_events_in_window(
                '2026-07-31T00:00:00+00:00',
                '2026-07-31T23:59:59+00:00',
            )
