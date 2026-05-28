"""Tests for AsyncSqliteBase base class and apply_wal_pragmas utility."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import pytest

from shared.async_sqlite_base import (
    AsyncSqliteBase,
    CheckpointResult,
    apply_full_durability_pragmas,
    connect_daemon,
)

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
        assert row is not None
        assert row[0] == 'wal'

    async def test_sets_busy_timeout(self, tmp_path: Path):
        """After apply_wal_pragmas, PRAGMA busy_timeout returns the configured value."""
        from shared.async_sqlite_base import apply_wal_pragmas

        db_path = tmp_path / 'test.db'
        async with aiosqlite.connect(str(db_path)) as conn:
            await apply_wal_pragmas(conn, busy_timeout_ms=12345)
            async with conn.execute('PRAGMA busy_timeout') as cur:
                row = await cur.fetchone()
        assert row is not None
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
        assert row is not None
        assert row[0] == 9999

    async def test_default_busy_timeout_is_set(self, tmp_path: Path):
        """apply_wal_pragmas with busy_timeout_ms=5000 sets the timeout."""
        from shared.async_sqlite_base import apply_wal_pragmas

        db_path = tmp_path / 'test.db'
        async with aiosqlite.connect(str(db_path)) as conn:
            await apply_wal_pragmas(conn, busy_timeout_ms=5000)
            async with conn.execute('PRAGMA busy_timeout') as cur:
                row = await cur.fetchone()
        assert row is not None
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
# Step-2: apply_full_durability_pragmas
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestApplyFullDurabilityPragmas:
    """apply_full_durability_pragmas delegates to apply_wal_pragmas and adds the
    synchronous/wal_autocheckpoint/journal_size_limit triad."""

    async def test_applies_full_pragma_triad(self, tmp_path: Path):
        """A single call sets journal_mode, busy_timeout, synchronous, wal_autocheckpoint,
        and journal_size_limit — all five PRAGMAs in one connection open.

        Uses busy_timeout_ms=12345 so we can distinguish the configured value
        from any SQLite default (5000 is a plausible default; 12345 is not).
        """
        db_path = tmp_path / 'test.db'
        async with aiosqlite.connect(str(db_path)) as conn:
            await apply_full_durability_pragmas(conn, busy_timeout_ms=12345)
            async with conn.execute('PRAGMA journal_mode') as cur:
                journal_row = await cur.fetchone()
            async with conn.execute('PRAGMA busy_timeout') as cur:
                timeout_row = await cur.fetchone()
            async with conn.execute('PRAGMA synchronous') as cur:
                sync_row = await cur.fetchone()
            async with conn.execute('PRAGMA wal_autocheckpoint') as cur:
                checkpoint_row = await cur.fetchone()
            async with conn.execute('PRAGMA journal_size_limit') as cur:
                size_row = await cur.fetchone()

        assert journal_row is not None and journal_row[0] == 'wal'
        assert timeout_row is not None and timeout_row[0] == 12345
        assert sync_row is not None and sync_row[0] == 2  # 2 == FULL
        assert checkpoint_row is not None and checkpoint_row[0] == 100
        assert size_row is not None and size_row[0] == 67108864


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
        assert row is not None
        assert row[0] == 'wal'

    async def test_open_sets_busy_timeout(self, tmp_path: Path) -> None:
        """busy_timeout PRAGMA reflects the configured busy_timeout_ms value."""
        async with _SimpleStore(tmp_path / 'store.db', busy_timeout_ms=7777) as store:  # noqa: SIM117
            async with store._conn.execute('PRAGMA busy_timeout') as cur:  # type: ignore[union-attr]
                row = await cur.fetchone()
        assert row is not None
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

        with patch('shared.async_sqlite_base.apply_full_durability_pragmas', new=AsyncMock()), \
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
        """close() racing with open() must not corrupt internal state.

        With the lifecycle lock, close() and open() are serialized even when
        launched concurrently.  The final state must be consistent regardless
        of which operation acquires the lock first.
        """
        store = _SimpleStore(tmp_path / 'store.db')
        await store.open()

        # Launch close and open concurrently — the lock serializes them
        results = await asyncio.gather(
            store.close(), store.open(), return_exceptions=True
        )

        errors = [r for r in results if isinstance(r, BaseException)]

        if store._conn is not None:
            # close() won the lock → open() succeeded after → store is open & usable
            async with store._conn.execute('SELECT 1') as cur:
                row = await cur.fetchone()
            assert row is not None and row[0] == 1
            await store.close()
        else:
            # open() won first → raised RuntimeError("already opened"), then close() ran
            assert len(errors) == 1
            assert isinstance(errors[0], RuntimeError)

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


# ---------------------------------------------------------------------------
# Exception-safety tests for close()
# ---------------------------------------------------------------------------


async def _swap_in_failing_close_mock(
    store: AsyncSqliteBase, exc: BaseException
) -> AsyncMock:
    """Close the real aiosqlite connection and inject a mock whose close() raises.

    Both exception-safety tests need to replace store._conn with an AsyncMock
    while keeping the real aiosqlite connection cleanly shut down.  Failing to
    close the real connection first orphans the aiosqlite worker thread; the
    orphan's GC-triggered call_soon_threadsafe raises 'Event loop is closed'
    after test teardown, which pytest re-raises in the next test's setup as
    PytestUnhandledThreadExceptionWarning.

    Safe swap sequence:
      1. Capture and close the real connection — worker thread exits cleanly.
      2. Null out store._conn — prevents an accidental close during transition.
      3. Install the AsyncMock — close() raises exc; tests then assert on _conn.
    """
    real_conn = store._conn
    assert real_conn is not None
    store._conn = None
    await real_conn.close()
    mock_conn = AsyncMock()
    mock_conn.close = AsyncMock(side_effect=exc)
    store._conn = mock_conn  # type: ignore[assignment]
    return mock_conn


# Sentinel used by _ensure_real_conn_closed_at_exit to distinguish a missing
# attribute from an attribute that is explicitly None.  A plain None check
# cannot tell the difference, so we use a unique object as a fallback value.
_MISSING = object()


@contextmanager
def _ensure_real_conn_closed_at_exit(store: AsyncSqliteBase):
    """Capture the current aiosqlite connection and assert it was closed
    before the guard exits.

    Order-independent replacement for the deferred-warning ``_z_`` sentinel:
    a test that replaces ``store._conn`` with a mock WITHOUT first awaiting
    ``real_conn.close()`` leaks the worker thread.  Eventually the orphan's
    GC-driven ``call_soon_threadsafe`` runs on a closed event loop and pytest
    surfaces ``PytestUnhandledThreadExceptionWarning`` in another test's
    setup — fragile under randomized collection.

    This guard captures a strong reference to the original Connection at
    entry (keeping it alive until exit) and inspects ``_connection`` at exit.
    ``Connection.close()`` sets ``_connection = None`` synchronously in its
    ``finally`` block, so the check is race-free: closed → None, leaked → set.
    See ``_swap_in_failing_close_mock`` for the safe-swap pattern this guard
    defends.

    If aiosqlite renames ``_connection``, the guard raises ``AssertionError``
    with a diagnostic message pointing here — so the failure is loud and
    attributable rather than a silent ``AttributeError`` or degraded check.
    """
    real_conn = store._conn
    assert real_conn is not None, (
        'store must be open before entering the leak guard'
    )
    try:
        yield
    finally:
        connection_val = getattr(real_conn, '_connection', _MISSING)
        if connection_val is _MISSING:
            raise AssertionError(
                'aiosqlite.Connection no longer exposes ._connection — '
                '_ensure_real_conn_closed_at_exit must be updated to use '
                'the new attribute name (aiosqlite API change detected).'
            )
        if connection_val is not None:
            raise AssertionError(
                f'Leaked aiosqlite connection: {real_conn!r}._connection '
                'is still set when the guard exited. The real connection '
                'was not closed before being replaced/discarded — its '
                'worker thread will eventually try call_soon_threadsafe on '
                'a closed event loop. See _swap_in_failing_close_mock for '
                'the safe-swap pattern.'
            )


# _ensure_real_conn_closed_at_exit is the primary order-independent leak
# detector for tests in this class: it asserts that the real aiosqlite
# Connection was properly closed before the test exits, regardless of
# collection order.  The filterwarnings marker is kept as defense-in-depth —
# it converts PytestUnhandledThreadExceptionWarning to a hard error for any
# edge-case leak that the guard might miss.
@pytest.mark.filterwarnings('error::pytest.PytestUnhandledThreadExceptionWarning')
@pytest.mark.asyncio
class TestAsyncSqliteBaseCloseExceptionSafety:
    """Tests that close() clears _conn even when conn.close() raises."""

    async def test_close_clears_conn_even_on_exception(self, tmp_path: Path) -> None:
        """_conn is set to None even when conn.close() raises OSError.

        Without try/finally, an OSError from conn.close() leaves _conn pointing
        to the stale object, making the store permanently stuck — neither retrying
        close() nor calling open() can succeed.
        """
        store = _SimpleStore(tmp_path / 'store.db')
        await store.open()
        assert store._conn is not None

        with _ensure_real_conn_closed_at_exit(store):
            # Install the failing mock (closes real conn first to avoid worker-thread leak)
            await _swap_in_failing_close_mock(store, OSError('disk failure'))

            # The OSError must propagate (not be swallowed)
            with pytest.raises(OSError, match='disk failure'):
                await store.close()

        # _conn must be None even though close() raised
        assert store._conn is None

    async def test_open_succeeds_after_failed_close(self, tmp_path: Path) -> None:
        """open() succeeds after a close() that raised.

        Without try/finally, _conn retains the stale reference after a failed
        close(). A subsequent open() then raises RuntimeError('already opened')
        because the non-None guard triggers on the stale reference.
        """
        store = _SimpleStore(tmp_path / 'store.db')
        await store.open()

        with _ensure_real_conn_closed_at_exit(store):
            # Install the failing mock (closes real conn first to avoid worker-thread leak)
            await _swap_in_failing_close_mock(store, OSError('disk failure'))

            # close() raises but must clear _conn
            with pytest.raises(OSError, match='disk failure'):
                await store.close()

        # After the failed close, open() must succeed — not raise RuntimeError
        await store.open()
        assert store._conn is not None

        # Verify the new connection is actually functional
        async with store._conn.execute('SELECT 1') as cur:
            row = await cur.fetchone()
        assert row is not None and row[0] == 1

        await store.close()


# ---------------------------------------------------------------------------
# Daemon-thread guarantee: interpreter shutdown never blocks on a leaked conn
# ---------------------------------------------------------------------------


def _assert_daemon_thread(conn: aiosqlite.Connection, label: str = '') -> None:
    """Assert that an aiosqlite connection's worker thread is alive and daemon-marked.

    Centralises the ``._thread`` access pattern so that a future aiosqlite
    rename is caught in one place rather than in every daemon-related test
    method.  The fused-memory counterpart lives in
    ``fused-memory/tests/test_daemon_connect_consolidation.py``.

    Args:
        conn: An open :class:`aiosqlite.Connection` to inspect.
        label: Optional label prepended to assertion failure messages.
    """
    prefix = f'{label}: ' if label else ''
    thread = conn._thread
    assert thread.is_alive(), f'{prefix}worker thread should be alive while connection is open'
    assert thread.daemon is True, (
        f'{prefix}aiosqlite worker thread must be daemon so a leaked '
        'connection cannot block interpreter shutdown'
    )


@pytest.mark.asyncio
class TestWorkerThreadIsDaemon:
    """AsyncSqliteBase.open() marks aiosqlite's worker thread as daemon.

    Rationale: if graceful shutdown is interrupted (e.g. by a second SIGTERM)
    and close() never runs, a non-daemon worker thread will wedge
    threading._shutdown() forever. Marking it daemon lets the interpreter exit
    cleanly in that case. WAL-mode + per-write commits mean committed data is
    durable regardless.
    """

    async def test_worker_thread_is_daemon_after_open(self, tmp_path: Path):
        store = _SimpleStore(tmp_path / 'daemon_check.db')
        await store.open()
        try:
            assert store._conn is not None
            # aiosqlite Connection stores its worker as ._thread (private).
            # If aiosqlite ever renames this, the guard in open() silently
            # degrades — _assert_daemon_thread fails loudly so we notice.
            _assert_daemon_thread(store._conn, label='AsyncSqliteBase.open()')
        finally:
            await store.close()


# ---------------------------------------------------------------------------
# connect_daemon: module-level helper for daemon-marked connections
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestConnectDaemon:
    """connect_daemon() opens an aiosqlite connection with its worker thread marked daemon.

    Mirrors TestWorkerThreadIsDaemon for AsyncSqliteBase.open(), but tests the
    standalone ``connect_daemon`` helper directly.  This helper is the single
    source-of-truth for the marking mechanism so that hand-rolled connect sites
    across fused-memory stores can share the same guard without subclassing.
    """

    async def test_thread_is_daemon(self, tmp_path: Path):
        """connect_daemon returns a live connection whose worker thread is daemon."""
        conn = await connect_daemon(str(tmp_path / 'x.db'))
        try:
            _assert_daemon_thread(conn, label='connect_daemon')
        finally:
            await conn.close()

    async def test_kwargs_passthrough(self, tmp_path: Path):
        """connect_daemon passes extra kwargs to aiosqlite.connect and marks daemon.

        timeout=30, isolation_level=None are the kwargs used by the override-db
        helpers in server/tools.py.  This test ensures they flow through and that
        the resulting connection is both usable and daemon-marked.
        """
        conn = await connect_daemon(str(tmp_path / 'y.db'), timeout=30, isolation_level=None)
        try:
            _assert_daemon_thread(conn, label='connect_daemon(kwargs)')
            # Connection must be usable with the supplied kwargs
            async with conn.execute('SELECT 1') as cur:
                row = await cur.fetchone()
            assert row is not None and row[0] == 1
        finally:
            await conn.close()


# ---------------------------------------------------------------------------
# Unit tests for _ensure_real_conn_closed_at_exit guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSwapLeakGuard:
    """Unit tests for the _ensure_real_conn_closed_at_exit guard context manager."""

    async def test_guard_raises_when_real_conn_was_not_closed(
        self, tmp_path: Path
    ) -> None:
        """Guard raises AssertionError when the real connection was never closed.

        Simulates a buggy swap: store._conn is replaced with an AsyncMock but
        the original aiosqlite Connection is never awaited-closed first.  The
        guard captures real_conn at entry; at exit it finds _connection is
        still set and raises AssertionError immediately — rather than leaking
        an unfinished worker thread into a later test's setup phase.
        """
        store = _SimpleStore(tmp_path / 'store.db')
        await store.open()
        real_conn = store._conn  # keep a reference for post-test cleanup
        assert real_conn is not None

        with pytest.raises(AssertionError, match=r'Leaked aiosqlite connection'):  # noqa: SIM117
            with _ensure_real_conn_closed_at_exit(store):
                # Buggy swap: replace _conn without closing real_conn first
                store._conn = AsyncMock()  # type: ignore[assignment]
                store._conn.close = AsyncMock(side_effect=OSError('boom'))
                # real_conn is deliberately NOT closed here — guard must fire

        # Clean up the leaked connection so it does not pollute subsequent tests
        await real_conn.close()

    async def test_guard_passes_when_safe_swap_was_used(
        self, tmp_path: Path
    ) -> None:
        """Guard exits silently when the safe-swap helper was used.

        _swap_in_failing_close_mock awaits real_conn.close() before installing
        the mock, so real_conn._connection is None at guard exit.
        """
        store = _SimpleStore(tmp_path / 'store.db')
        await store.open()

        with _ensure_real_conn_closed_at_exit(store):
            await _swap_in_failing_close_mock(store, OSError('boom'))
            with pytest.raises(OSError):
                await store.close()


# ---------------------------------------------------------------------------
# Step-13: AsyncSqliteBase.checkpoint()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAsyncSqliteBaseCheckpoint:
    """Tests for AsyncSqliteBase.checkpoint()."""

    async def test_checkpoint_raises_when_not_opened(self, tmp_path: Path) -> None:
        """checkpoint() raises RuntimeError('{ClassName} not opened') before open()."""
        store = _SimpleStore(tmp_path / 'store.db')
        with pytest.raises(RuntimeError, match='_SimpleStore not opened'):
            await store.checkpoint()

    async def test_checkpoint_returns_checkpoint_result_on_fresh_store(self, tmp_path: Path) -> None:
        """checkpoint() returns a CheckpointResult with busy==0 on a fresh store."""
        async with _SimpleStore(tmp_path / 'store.db') as store:
            result = await store.checkpoint()
        assert isinstance(result, CheckpointResult)
        assert isinstance(result, tuple)  # NamedTuple is a tuple subclass
        assert len(result) == 3
        assert result.busy == 0
        # TRUNCATE moves all WAL frames into the main db; log == checkpointed
        assert result.log == result.checkpointed

    async def test_checkpoint_raises_if_pragma_returns_no_rows(self, tmp_path: Path) -> None:
        """checkpoint() raises RuntimeError when PRAGMA wal_checkpoint returns no rows.

        Defensive guard: SQLite always returns a row for this PRAGMA in practice,
        but the explicit check ensures callers get a loud failure rather than a
        silent sentinel if the behaviour ever changes.
        """
        mock_cursor = AsyncMock()
        mock_cursor.fetchone = AsyncMock(return_value=None)
        mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
        mock_cursor.__aexit__ = AsyncMock(return_value=False)

        mock_conn = AsyncMock()
        # execute() must return a sync value (the cursor) so `async with conn.execute(...)`
        # can call __aenter__ on it directly.
        mock_conn.execute = MagicMock(return_value=mock_cursor)

        store = _SimpleStore(tmp_path / 'store.db')
        store._conn = mock_conn  # type: ignore[assignment]

        with pytest.raises(RuntimeError, match='PRAGMA wal_checkpoint returned no rows'):
            await store.checkpoint()

    async def test_checkpoint_truncates_wal_after_writes(self, tmp_path: Path) -> None:
        """After inserting rows, checkpoint() flushes the WAL with busy==0.

        We assert only what SQLite's documented contract guarantees:
        - ``busy == 0`` when no readers are blocking the checkpoint.
        - The WAL file shrinks to near-zero (≤ 32 bytes, the WAL header size).

        log and checkpointed counters are only sanity-bounded (>= 0) because
        their post-TRUNCATE values differ across SQLite builds and are not
        part of the contractually stable API.
        """
        db_path = tmp_path / 'store.db'
        wal_path = tmp_path / 'store.db-wal'
        async with _SimpleStore(db_path) as store:
            # Insert enough rows to populate WAL frames
            for i in range(10):
                await store._conn.execute(  # type: ignore[union-attr]
                    "INSERT INTO items (name) VALUES (?)", (f'item-{i}',)
                )
            await store._conn.commit()  # type: ignore[union-attr]

            # WAL file should have frames from the inserts
            before_size = wal_path.stat().st_size if wal_path.exists() else 0
            assert before_size > 0, 'Expected WAL frames to exist after commits'

            result = await store.checkpoint()

        assert isinstance(result, CheckpointResult)
        # Contractually guaranteed: no readers blocked the checkpoint
        assert result.busy == 0
        # Non-negative sanity bounds only; exact values are build-specific
        assert result.log >= 0
        assert result.checkpointed >= 0
        # WAL file must be truncated to near-zero (SQLite WAL header = 32 bytes)
        after_size = wal_path.stat().st_size if wal_path.exists() else 0
        assert after_size <= 32, f'WAL should be truncated; got {after_size} bytes'
