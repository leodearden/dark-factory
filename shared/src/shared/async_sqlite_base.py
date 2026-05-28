"""Async SQLite base class and utilities for WAL-mode persistent connections.

Provides:
- apply_wal_pragmas(conn, busy_timeout_ms): standalone utility to configure WAL + busy_timeout
- apply_full_durability_pragmas(conn, busy_timeout_ms): WAL + busy_timeout + Phase 3 triad
- connect_daemon(database, **kwargs): open a connection with worker thread marked daemon
- AsyncSqliteBase: ABC with lifecycle management (open/close/context-manager/guard)
"""

from __future__ import annotations

import abc
import asyncio
import contextlib
from pathlib import Path
from typing import NamedTuple, Self

import aiosqlite

__all__ = ['apply_wal_pragmas', 'apply_full_durability_pragmas', 'connect_daemon', 'CheckpointResult', 'AsyncSqliteBase']


class CheckpointResult(NamedTuple):
    """Result of a ``PRAGMA wal_checkpoint(TRUNCATE)`` call.

    Attributes:
        busy: 1 if one or more WAL frames could not be checkpointed because they
            are in use by a reader, 0 otherwise.
        log: Total number of frames in the WAL file.
        checkpointed: Total number of frames that were successfully checkpointed.
    """

    busy: int
    log: int
    checkpointed: int


async def apply_wal_pragmas(conn: aiosqlite.Connection, *, busy_timeout_ms: int) -> None:
    """Configure WAL journal mode and optional busy_timeout on an open aiosqlite connection.

    Args:
        conn: An open aiosqlite connection.
        busy_timeout_ms: Milliseconds to wait for a locked database.
            Pass 0 to skip setting the busy_timeout pragma entirely.
    """
    async with conn.execute('PRAGMA journal_mode=WAL') as cur:
        row = await cur.fetchone()
    if row is None or row[0] != 'wal':
        got = row[0] if row is not None else None
        raise RuntimeError(
            f'Failed to enable WAL journal mode (got {got!r})'
        )
    if busy_timeout_ms != 0:
        await conn.execute(f'PRAGMA busy_timeout={busy_timeout_ms}')


async def apply_full_durability_pragmas(conn: aiosqlite.Connection, *, busy_timeout_ms: int) -> None:
    """Configure WAL mode, busy_timeout, and the Phase 3 durability triad.

    Delegates to ``apply_wal_pragmas`` for WAL + busy_timeout, then sets the
    three additional PRAGMAs that harden crash durability across all
    fused-memory SQLite stores:

    - ``synchronous=FULL`` (2): fsync per-commit; eliminates corruption on
      unexpected shutdown without relying on WAL-checkpoint timing.
    - ``wal_autocheckpoint=100``: auto-checkpoint after every 100 WAL pages to
      bound WAL growth under normal load.
    - ``journal_size_limit=67108864`` (64 MiB): caps the WAL file size to
      prevent unbounded disk use during high-write bursts.

    See ``docs/task-recovery-2026-05-13/`` for the production incident that
    drove this convention across all fused-memory SQLite stores.

    Args:
        conn: An open aiosqlite connection.
        busy_timeout_ms: Milliseconds to wait for a locked database.
            Pass 0 to skip setting the busy_timeout pragma entirely.
    """
    await apply_wal_pragmas(conn, busy_timeout_ms=busy_timeout_ms)
    # synchronous=FULL: per-commit fsync. Cost is ~1-5ms/commit; the
    # win is crash durability without relying on WAL checkpoints. See
    # docs/task-recovery-2026-05-13/ for the prod incident that drove
    # this change across all fused-memory SQLite stores.
    await conn.execute('PRAGMA synchronous=FULL')
    await conn.execute('PRAGMA wal_autocheckpoint=100')
    await conn.execute('PRAGMA journal_size_limit=67108864')


async def connect_daemon(database: str | Path, **kwargs) -> aiosqlite.Connection:
    """Open an aiosqlite connection with its background worker thread marked daemon.

    The worker thread is marked daemon *before* the thread starts (i.e. before
    ``await``), so a connection that is never closed (e.g. graceful-shutdown
    cleanup aborted by a second SIGTERM, MCP stdio clean-EOF, SIGABRT) cannot
    block interpreter exit in ``threading._shutdown()``.  WAL mode makes this
    safe: committed data is durable; only in-flight uncommitted transactions
    are lost, which is already the contract of forced shutdown.

    This is the single source of truth for the daemon-marking mechanism shared by
    ``AsyncSqliteBase.open()`` and all hand-rolled connect sites across the
    fused-memory stores that do not subclass ``AsyncSqliteBase``.

    Args:
        database: Path to the database file (a :class:`str`, :class:`~pathlib.Path`,
            or the special ``':memory:'`` string) passed straight through to
            ``aiosqlite.connect()``.  Both ``str`` and ``Path`` are accepted because
            ``sqlite3.connect`` — which aiosqlite delegates to — accepts any
            :class:`os.PathLike`, and callers may hold either type.
        **kwargs: Any extra keyword arguments (e.g. ``timeout=30``,
            ``isolation_level=None``) forwarded verbatim to ``aiosqlite.connect()``.

    Returns:
        An open, daemon-thread-backed :class:`aiosqlite.Connection`.
    """
    conn_awaitable = aiosqlite.connect(database, **kwargs)
    # Mark the worker thread as daemon before the thread starts.
    # AttributeError: aiosqlite renamed ._thread (graceful degradation).
    # RuntimeError: thread already started (shouldn't happen, but safe).
    with contextlib.suppress(AttributeError, RuntimeError):
        conn_awaitable._thread.daemon = True
    return await conn_awaitable


class AsyncSqliteBase(abc.ABC):
    """Abstract base class for async SQLite stores with WAL-mode persistent connections.

    Subclasses must implement the ``_schema`` property that returns a DDL string
    (passed to ``executescript()`` during ``open()``).

    Lifecycle::

        store = MyStore(path)
        await store.open()
        try:
            ...
        finally:
            await store.close()

    Or via async context manager::

        async with MyStore(path) as store:
            ...

    **Durability**: ``open()`` calls :func:`apply_full_durability_pragmas` on
    every subclass, applying the Phase 3 triad (``synchronous=FULL``,
    ``wal_autocheckpoint=100``, ``journal_size_limit=64 MiB``) — see
    ``docs/task-recovery-2026-05-13/`` for the production incident that
    mandated this convention.  A future subclass that needs WAL-only semantics
    (e.g. an ephemeral or test store) must override ``open()`` and bypass
    ``apply_full_durability_pragmas``; no class-level opt-out toggle exists.
    """

    def __init__(self, db_path: Path, *, busy_timeout_ms: int = 5000) -> None:
        self.db_path = db_path
        self.busy_timeout_ms = busy_timeout_ms
        self._conn: aiosqlite.Connection | None = None
        # Serializes open() and close(); subclasses must not bypass for lifecycle mutations.
        self._lifecycle_lock = asyncio.Lock()

    @property
    @abc.abstractmethod
    def _schema(self) -> str:
        """DDL string passed to executescript() when the store is opened."""

    async def open(self) -> None:
        """Open persistent connection, set WAL + Phase 3 durability triad, ensure schema."""
        async with self._lifecycle_lock:
            if self._conn is not None:
                raise RuntimeError(f'{type(self).__name__} already opened')
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = await connect_daemon(str(self.db_path))
            try:
                await apply_full_durability_pragmas(conn, busy_timeout_ms=self.busy_timeout_ms)
                await conn.executescript(self._schema)
            except BaseException:
                await conn.close()
                raise
            self._conn = conn

    async def close(self) -> None:
        """Close the connection. Idempotent — safe to call when already closed."""
        async with self._lifecycle_lock:
            if self._conn is not None:
                try:
                    await self._conn.close()
                finally:
                    self._conn = None

    async def __aenter__(self) -> Self:
        await self.open()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        await self.close()

    def _require_conn(self) -> aiosqlite.Connection:
        """Return the open connection or raise RuntimeError."""
        if self._conn is None:
            raise RuntimeError(f'{type(self).__name__} not opened')
        return self._conn

    async def checkpoint(self) -> CheckpointResult:
        """Run ``PRAGMA wal_checkpoint(TRUNCATE)`` and return the result.

        Returns:
            A :class:`CheckpointResult` named-tuple ``(busy, log, checkpointed)`` where:

            - ``busy``: 1 if one or more frames could not be checkpointed because
              they are in use by a reader, 0 otherwise.
            - ``log``: total number of frames in the WAL file.
            - ``checkpointed``: total number of checkpointed frames.

        Raises:
            RuntimeError: If the store has not been opened.
            RuntimeError: If ``PRAGMA wal_checkpoint(TRUNCATE)`` returns no rows
                (unexpected; SQLite always returns a row for this pragma).
        """
        conn = self._require_conn()
        async with conn.execute('PRAGMA wal_checkpoint(TRUNCATE)') as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise RuntimeError('PRAGMA wal_checkpoint returned no rows')
        return CheckpointResult(int(row[0]), int(row[1]), int(row[2]))
