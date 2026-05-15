"""Persistent read-only connection pool for dashboard SQLite databases.

Instead of opening a fresh ``aiosqlite.connect()`` per request (each spawning
a thread), the :class:`DbPool` maintains one long-lived connection per database
path and reuses it across poll cycles.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TypeVar
from urllib.parse import quote

import aiosqlite

logger = logging.getLogger(__name__)

_T = TypeVar('_T')


class DbPool:
    """Lazy-open pool of read-only aiosqlite connections.

    Call :meth:`get` to obtain a connection for a given path.  The connection
    is created on first access and reused thereafter.  Call :meth:`close_all`
    during shutdown.
    """

    def __init__(self) -> None:
        self._conns: dict[Path, aiosqlite.Connection] = {}
        self._closed: bool = False
        # Per-path open locks — prevents duplicate opens for the same path while
        # allowing disjoint paths to open concurrently (no serialisation between
        # unrelated paths).  Mirrors SqliteTaskBackend._get_connection convention.
        self._open_locks: dict[Path, asyncio.Lock] = {}
        self._open_locks_lock: asyncio.Lock = asyncio.Lock()

    async def get(self, db_path: Path) -> aiosqlite.Connection | None:
        """Return a cached connection, opening one lazily if needed.

        Returns ``None`` when the database file does not exist or cannot be
        opened (e.g. corrupt, locked exclusively).
        """
        resolved = db_path.resolve()
        # Lock-free fast path — common case when connection is already cached.
        if resolved in self._conns:
            return self._conns[resolved]
        # Acquire (or create) the per-path lock.  The meta-lock is held only
        # for the synchronous setdefault call and released before any await.
        async with self._open_locks_lock:
            lock = self._open_locks.setdefault(resolved, asyncio.Lock())
        # Serialize same-path opens; disjoint paths use independent locks.
        async with lock:
            # Re-check after acquiring: a racing coroutine may have opened it.
            if resolved in self._conns:
                return self._conns[resolved]
            # Refuse to open new connections once close_all() has been called.
            # Fully aligns with SqliteTaskBackend._get_connection _closed convention:
            # a concurrent get() that is mid-open when close_all() runs will see
            # this flag after acquiring the per-path lock and abort cleanly.
            if self._closed:
                return None
            try:
                if not resolved.exists():
                    return None
                # safe='/' preserves POSIX path separators; dashboard is Linux-only.
                # For Windows portability use pathlib.PurePath.as_uri() instead.
                conn = await aiosqlite.connect(
                    f'file:{quote(str(resolved), safe="/")}?mode=ro',
                    uri=True,
                )
                conn.row_factory = aiosqlite.Row
                self._conns[resolved] = conn
                return conn
            except (FileNotFoundError, sqlite3.OperationalError, OSError):
                logger.debug('DbPool: cannot open %s', resolved, exc_info=True)
                return None

    @property
    def open_count(self) -> int:
        """Number of currently held connections."""
        return len(self._conns)

    async def close_all(self) -> None:
        """Close every managed connection and clear the pool."""
        # Set before iterating so any concurrent get() that acquires a per-path
        # lock after we clear _conns will see _closed=True and return None rather
        # than re-populating a pool that is supposed to be drained.
        self._closed = True
        for conn in self._conns.values():
            try:
                await conn.close()
            except Exception:
                logger.debug('DbPool: error closing connection', exc_info=True)
        self._conns.clear()
        # Clear per-path locks to prevent unbounded growth across distinct paths.
        self._open_locks.clear()


async def with_db(
    db: aiosqlite.Connection | None,
    fn: Callable[[aiosqlite.Connection], Awaitable[_T]],
    default: _T,
) -> _T:
    """Run *fn* against *db*, returning *default* on ``None`` or error.

    Drop-in replacement for the per-module ``_with_readonly_db`` helpers that
    opened a fresh connection each call.
    """
    if db is None:
        return default
    try:
        return await fn(db)
    except (sqlite3.OperationalError, OSError):
        logger.debug('with_db: query failed', exc_info=True)
        return default
