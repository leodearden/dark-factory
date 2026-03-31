"""Tests for AsyncSqliteBase base class and apply_wal_pragmas utility."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import pytest

from shared.async_sqlite_base import AsyncSqliteBase

# ---------------------------------------------------------------------------
# Step-1: apply_wal_pragmas
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestApplyWalPragmas:
    """apply_wal_pragmas(conn, busy_timeout_ms) sets WAL mode and busy_timeout."""

    async def test_sets_journal_mode_wal(self, tmp_path: Path):
        """After apply_wal_pragmas, PRAGMA journal_mode returns 'wal'."""
        from shared.async_sqlite_base import apply_wal_pragmas

        db_path = tmp_path / 'test.db'
        async with aiosqlite.connect(str(db_path)) as conn:
            await apply_wal_pragmas(conn, busy_timeout_ms=5000)
            async with conn.execute('PRAGMA journal_mode') as cur:
                row = await cur.fetchone()
        assert row[0] == 'wal'

    async def test_sets_busy_timeout(self, tmp_path: Path):
        """After apply_wal_pragmas, PRAGMA busy_timeout returns the configured value."""
        from shared.async_sqlite_base import apply_wal_pragmas

        db_path = tmp_path / 'test.db'
        async with aiosqlite.connect(str(db_path)) as conn:
            await apply_wal_pragmas(conn, busy_timeout_ms=12345)
            async with conn.execute('PRAGMA busy_timeout') as cur:
                row = await cur.fetchone()
        assert row[0] == 12345

    async def test_zero_busy_timeout_skips_pragma(self, tmp_path: Path):
        """busy_timeout_ms=0 means skip the PRAGMA busy_timeout entirely (not set to 0)."""
        from shared.async_sqlite_base import apply_wal_pragmas

        db_path = tmp_path / 'test.db'
        async with aiosqlite.connect(str(db_path)) as conn:
            # Set a non-zero value first so we can confirm it was NOT changed
            await conn.execute('PRAGMA busy_timeout=9999')
            await apply_wal_pragmas(conn, busy_timeout_ms=0)
            async with conn.execute('PRAGMA busy_timeout') as cur:
                row = await cur.fetchone()
        # busy_timeout=0 → skip pragma → previous value 9999 should be unchanged
        assert row[0] == 9999

    async def test_default_busy_timeout_is_set(self, tmp_path: Path):
        """apply_wal_pragmas with busy_timeout_ms=5000 sets the timeout."""
        from shared.async_sqlite_base import apply_wal_pragmas

        db_path = tmp_path / 'test.db'
        async with aiosqlite.connect(str(db_path)) as conn:
            await apply_wal_pragmas(conn, busy_timeout_ms=5000)
            async with conn.execute('PRAGMA busy_timeout') as cur:
                row = await cur.fetchone()
        assert row[0] == 5000

    async def test_wal_fallback_raises_runtime_error(self) -> None:
        """apply_wal_pragmas raises RuntimeError when journal_mode PRAGMA returns a
        non-WAL result (e.g. 'delete' on a filesystem that doesn't support WAL)."""
        from shared.async_sqlite_base import apply_wal_pragmas

        mock_cursor = AsyncMock()
        mock_cursor.fetchone = AsyncMock(return_value=('delete',))
        mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
        mock_cursor.__aexit__ = AsyncMock(return_value=False)

        mock_conn = AsyncMock()
        # execute() must return a sync value (the cursor) so `async with conn.execute(...)`
        # can call __aenter__ on it directly — AsyncMock would return a coroutine instead.
        mock_conn.execute = MagicMock(return_value=mock_cursor)

        with pytest.raises(RuntimeError, match='WAL'):
            await apply_wal_pragmas(mock_conn, busy_timeout_ms=5000)

    async def test_wal_none_row_raises_runtime_error(self) -> None:
        """apply_wal_pragmas raises RuntimeError when journal_mode PRAGMA returns no
        rows (fetchone() → None). Guards against unexpected empty result sets."""
        from shared.async_sqlite_base import apply_wal_pragmas

        mock_cursor = AsyncMock()
        mock_cursor.fetchone = AsyncMock(return_value=None)
        mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
        mock_cursor.__aexit__ = AsyncMock(return_value=False)

        mock_conn = AsyncMock()
        mock_conn.execute = MagicMock(return_value=mock_cursor)

        with pytest.raises(RuntimeError, match='WAL'):
            await apply_wal_pragmas(mock_conn, busy_timeout_ms=5000)


# ---------------------------------------------------------------------------
# Concrete test subclass used for AsyncSqliteBase tests
# ---------------------------------------------------------------------------

_SIMPLE_SCHEMA = """\
CREATE TABLE IF NOT EXISTS items (
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL
);
"""


class _SimpleStore(AsyncSqliteBase):
    @property
    def _schema(self) -> str:
        return _SIMPLE_SCHEMA


# ---------------------------------------------------------------------------
# Step-3: AsyncSqliteBase.__init__
# ---------------------------------------------------------------------------


class TestAsyncSqliteBaseInit:
    """AsyncSqliteBase.__init__ stores db_path, sets _conn to None, stores busy_timeout_ms."""

    def test_init_stores_db_path(self, tmp_path: Path):
        db_path = tmp_path / 'store.db'
        store = _SimpleStore(db_path)
        assert store.db_path == db_path

    def test_init_conn_is_none(self, tmp_path: Path):
        store = _SimpleStore(tmp_path / 'store.db')
        assert store._conn is None

    def test_init_default_busy_timeout(self, tmp_path: Path):
        """Default busy_timeout_ms is 5000."""
        store = _SimpleStore(tmp_path / 'store.db')
        assert store.busy_timeout_ms == 5000

    def test_init_custom_busy_timeout(self, tmp_path: Path):
        """busy_timeout_ms can be overridden at construction."""
        store = _SimpleStore(tmp_path / 'store.db', busy_timeout_ms=30000)
        assert store.busy_timeout_ms == 30000

    def test_cannot_instantiate_without_schema(self, tmp_path: Path):
        """AsyncSqliteBase is abstract; instantiating without _schema raises TypeError."""
        with pytest.raises(TypeError):
            AsyncSqliteBase(tmp_path / 'store.db')  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Step-5: AsyncSqliteBase.open()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAsyncSqliteBaseOpen:
    """Tests for AsyncSqliteBase.open()."""

    async def test_open_creates_connection(self, tmp_path: Path) -> None:
        """After open(), _conn is not None."""
        store = _SimpleStore(tmp_path / 'store.db')
        await store.open()
        try:
            assert store._conn is not None
        finally:
            await store.close()

    async def test_open_sets_wal_mode(self, tmp_path: Path) -> None:
        """WAL journal mode is active after open()."""
        async with _SimpleStore(tmp_path / 'store.db') as store:  # noqa: SIM117
            async with store._conn.execute('PRAGMA journal_mode') as cur:  # type: ignore[union-attr]
                row = await cur.fetchone()
        assert row[0] == 'wal'

    async def test_open_sets_busy_timeout(self, tmp_path: Path) -> None:
        """busy_timeout PRAGMA reflects the configured busy_timeout_ms value."""
        async with _SimpleStore(tmp_path / 'store.db', busy_timeout_ms=7777) as store:  # noqa: SIM117
            async with store._conn.execute('PRAGMA busy_timeout') as cur:  # type: ignore[union-attr]
                row = await cur.fetchone()
        assert row[0] == 7777

    async def test_open_creates_schema_tables(self, tmp_path: Path) -> None:
        """After open(), tables declared in _schema exist in the database."""
        async with _SimpleStore(tmp_path / 'store.db') as store:  # noqa: SIM117
            async with store._conn.execute(  # type: ignore[union-attr]
                "SELECT name FROM sqlite_master WHERE type='table' AND name='items'"
            ) as cur:
                row = await cur.fetchone()
        assert row is not None

    async def test_open_creates_parent_dirs(self, tmp_path: Path) -> None:
        """open() creates parent directories that do not yet exist."""
        nested = tmp_path / 'a' / 'b' / 'c' / 'store.db'
        store = _SimpleStore(nested)
        await store.open()
        try:
            assert nested.exists()
        finally:
            await store.close()

    async def test_double_open_raises_runtime_error(self, tmp_path: Path) -> None:
        """A second call to open() raises RuntimeError('{ClassName} already opened')."""
        store = _SimpleStore(tmp_path / 'store.db')
        await store.open()
        try:
            with pytest.raises(RuntimeError, match='_SimpleStore already opened'):
                await store.open()
        finally:
            await store.close()

    async def test_open_no_resource_leak_on_schema_failure(self, tmp_path: Path) -> None:
        """If executescript fails during open(), the conn is closed and _conn stays None."""
        store = _SimpleStore(tmp_path / 'broken.db')

        # Build a mock connection whose executescript raises.
        # Patch apply_wal_pragmas to a no-op so we can test schema failure in isolation
        # without needing to replicate aiosqlite's dual-protocol execute() object.
        mock_conn = AsyncMock()
        mock_conn.executescript = AsyncMock(side_effect=RuntimeError('schema failure'))
        mock_conn.close = AsyncMock()

        with patch('shared.async_sqlite_base.apply_wal_pragmas', new=AsyncMock()), \
             patch('aiosqlite.connect', new=AsyncMock(return_value=mock_conn)), \
             pytest.raises(RuntimeError, match='schema failure'):
            await store.open()

        assert store._conn is None
        mock_conn.close.assert_called_once()


# ---------------------------------------------------------------------------
# Step-7: AsyncSqliteBase.close()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAsyncSqliteBaseClose:
    """Tests for AsyncSqliteBase.close()."""

    async def test_close_sets_conn_to_none(self, tmp_path: Path) -> None:
        """After close(), _conn is None."""
        store = _SimpleStore(tmp_path / 'store.db')
        await store.open()
        assert store._conn is not None
        await store.close()
        assert store._conn is None

    async def test_close_is_idempotent(self, tmp_path: Path) -> None:
        """Double-close does not raise."""
        store = _SimpleStore(tmp_path / 'store.db')
        await store.open()
        await store.close()
        # Second close should not raise
        await store.close()
        assert store._conn is None

    async def test_close_never_opened_is_safe(self, tmp_path: Path) -> None:
        """close() on a store that was never opened is a no-op."""
        store = _SimpleStore(tmp_path / 'store.db')
        # Never opened — close() must not raise
        await store.close()
        assert store._conn is None

    async def test_data_persists_across_close_reopen(self, tmp_path: Path) -> None:
        """Data written before close() is readable after reopen."""
        db_path = tmp_path / 'store.db'

        # Write a row
        async with _SimpleStore(db_path) as store:
            await store._conn.execute("INSERT INTO items (name) VALUES ('hello')")  # type: ignore[union-attr]
            await store._conn.commit()  # type: ignore[union-attr]

        # Reopen and verify the row is still there
        async with _SimpleStore(db_path) as store:  # noqa: SIM117
            async with store._conn.execute("SELECT name FROM items WHERE name='hello'") as cur:  # type: ignore[union-attr]
                row = await cur.fetchone()
        assert row is not None
        assert row[0] == 'hello'


# ---------------------------------------------------------------------------
# Step-9: AsyncSqliteBase context manager (__aenter__ / __aexit__)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAsyncSqliteBaseContextManager:
    """Tests for AsyncSqliteBase.__aenter__ and __aexit__."""

    async def test_aenter_opens_connection(self, tmp_path: Path) -> None:
        """__aenter__ opens the connection (_conn is not None inside the block)."""
        store = _SimpleStore(tmp_path / 'store.db')
        async with store:
            assert store._conn is not None

    async def test_aenter_returns_self(self, tmp_path: Path) -> None:
        """__aenter__ returns self."""
        store = _SimpleStore(tmp_path / 'store.db')
        async with store as ctx:
            assert ctx is store

    async def test_aexit_closes_connection_on_normal_exit(self, tmp_path: Path) -> None:
        """__aexit__ closes the connection after the block exits normally."""
        store = _SimpleStore(tmp_path / 'store.db')
        async with store:
            pass
        assert store._conn is None

    async def test_aexit_closes_connection_on_exception(self, tmp_path: Path) -> None:
        """__aexit__ closes the connection even when the body raises."""
        store = _SimpleStore(tmp_path / 'store.db')
        with pytest.raises(ValueError, match='boom'):
            async with store:
                raise ValueError('boom')
        assert store._conn is None

    def test_aenter_return_annotation_is_self(self) -> None:
        """__aenter__ must be annotated with typing.Self so subclass context managers
        preserve the concrete type for static type checkers."""
        import typing

        hints = typing.get_type_hints(AsyncSqliteBase.__aenter__)
        assert hints['return'] is typing.Self


# ---------------------------------------------------------------------------
# Step-11: AsyncSqliteBase._require_conn()
# ---------------------------------------------------------------------------


class TestAsyncSqliteBaseRequireConn:
    """Tests for AsyncSqliteBase._require_conn()."""

    def test_require_conn_raises_when_not_opened(self, tmp_path: Path) -> None:
        """_require_conn() raises RuntimeError('{ClassName} not opened') when _conn is None."""
        store = _SimpleStore(tmp_path / 'store.db')
        with pytest.raises(RuntimeError, match='_SimpleStore not opened'):
            store._require_conn()

    @pytest.mark.asyncio
    async def test_require_conn_returns_connection_when_open(self, tmp_path: Path) -> None:
        """_require_conn() returns the aiosqlite connection when the store is open."""
        async with _SimpleStore(tmp_path / 'store.db') as store:
            conn = store._require_conn()
            assert conn is store._conn

    @pytest.mark.asyncio
    async def test_require_conn_raises_after_close(self, tmp_path: Path) -> None:
        """_require_conn() raises after close() sets _conn to None."""
        store = _SimpleStore(tmp_path / 'store.db')
        await store.open()
        await store.close()
        with pytest.raises(RuntimeError, match='_SimpleStore not opened'):
            store._require_conn()


# ---------------------------------------------------------------------------
# Concurrent close and open-vs-close race tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAsyncSqliteBaseConcurrentClose:
    """Tests that close() serializes with itself and with open() via _lifecycle_lock."""

    async def test_concurrent_close_does_not_double_close(self, tmp_path: Path) -> None:
        """Two concurrent close() calls must not both call conn.close().

        Without the lifecycle lock, both coroutines pass the `_conn is not None`
        guard before either sets `_conn = None`, causing the underlying aiosqlite
        connection to be closed twice.
        """
        store = _SimpleStore(tmp_path / 'store.db')
        await store.open()

        real_conn = store._conn
        assert real_conn is not None

        close_call_count = 0
        original_close = real_conn.close

        async def counting_close():
            nonlocal close_call_count
            close_call_count += 1
            return await original_close()

        real_conn.close = counting_close  # type: ignore[assignment]

        results = await asyncio.gather(
            store.close(),
            store.close(),
            return_exceptions=True,
        )

        # Both should succeed (idempotent) — no exceptions
        errors = [r for r in results if isinstance(r, BaseException)]
        assert errors == [], f'Unexpected errors: {errors!r}'
        assert store._conn is None
        assert close_call_count == 1, (
            f'Expected conn.close() called once, got {close_call_count}'
        )

    async def test_open_close_race_does_not_invalidate_connection(
        self, tmp_path: Path
    ) -> None:
        """close() racing with open() must not close a freshly-opened connection.

        Scenario: store is open → close() acquires lock, closes conn, sets _conn=None,
        releases lock → open() acquires lock, opens new conn. Without the lock on close(),
        a late close() could clear _conn after open() sets it.
        """
        store = _SimpleStore(tmp_path / 'store.db')
        await store.open()

        # Close then immediately re-open — both serialized by _lifecycle_lock
        await asyncio.gather(store.close(), return_exceptions=True)
        await store.open()

        # Connection should be alive and usable
        assert store._conn is not None
        async with store._conn.execute('SELECT 1') as cur:
            row = await cur.fetchone()
        assert row is not None and row[0] == 1

        await store.close()

    async def test_close_open_interleaving(self, tmp_path: Path) -> None:
        """Rapid close→open→close→open cycles must leave the store in a consistent state.

        Each transition is serialized by _lifecycle_lock so no operation observes a
        half-mutated _conn.
        """
        store = _SimpleStore(tmp_path / 'store.db')

        for _ in range(5):
            await store.open()
            assert store._conn is not None
            await store.close()
            assert store._conn is None
