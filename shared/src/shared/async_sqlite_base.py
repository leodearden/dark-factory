"""Async SQLite base class and utilities for WAL-mode persistent connections.

Provides:
- apply_wal_pragmas(conn, busy_timeout_ms): standalone utility to configure WAL + busy_timeout
- AsyncSqliteBase: ABC with lifecycle management (open/close/context-manager/guard)
"""

from __future__ import annotations

import abc
import asyncio
import contextlib
from pathlib import Path
from typing import Self

import aiosqlite

__all__ = ['apply_wal_pragmas', 'AsyncSqliteBase']


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
        """Open persistent connection, set WAL + busy_timeout, ensure schema."""
        async with self._lifecycle_lock:
            if self._conn is not None:
                raise RuntimeError(f'{type(self).__name__} already opened')
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn_awaitable = aiosqlite.connect(str(self.db_path))
            # Mark the worker thread as daemon before it starts, so a
            # leaked connection (e.g. graceful-shutdown cleanup aborted
            # by a second SIGTERM) can never block interpreter exit in
            # threading._shutdown(). WAL mode makes this safe: committed
            # data is durable; only in-flight uncommitted transactions
            # are lost, which is already the contract of forced shutdown.
            with contextlib.suppress(AttributeError, RuntimeError):
                conn_awaitable._thread.daemon = True
            conn = await conn_awaitable
            try:
                await apply_wal_pragmas(conn, busy_timeout_ms=self.busy_timeout_ms)
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
