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
        self._open_lock: asyncio.Lock = asyncio.Lock()

    async def get(self, db_path: Path) -> aiosqlite.Connection | None:
        """Return a cached connection, opening one lazily if needed.

        Returns ``None`` when the database file does not exist or cannot be
        opened (e.g. corrupt, locked exclusively).
        """
        resolved = db_path.resolve()
        # Lock-free fast path — common case when connection is already cached.
        if resolved in self._conns:
            return self._conns[resolved]
        # Serialize same-path opens so only one connection is created.
        async with self._open_lock:
            # Re-check after acquiring: a racing coroutine may have opened it.
            if resolved in self._conns:
                return self._conns[resolved]
            try:
                if not resolved.exists():
                    return None
                # safe='/' preserves POSIX path separators; dashboard is Linux-only.
                # For Windows portability use pathlib.PurePath.as_uri() instead.
                conn = await aiosqlite.connect(
                    f'file:{quote(str(resolved), safe="/")}?mode=ro', uri=True,
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
        for conn in self._conns.values():
            try:
                await conn.close()
            except Exception:
                logger.debug('DbPool: error closing connection', exc_info=True)
        self._conns.clear()


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
