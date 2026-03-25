"""Persistent read-only connection pool for dashboard SQLite databases.

Instead of opening a fresh ``aiosqlite.connect()`` per request (each spawning
a thread), the :class:`DbPool` maintains one long-lived connection per database
path and reuses it across poll cycles.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TypeVar

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

    async def get(self, db_path: Path) -> aiosqlite.Connection | None:
        """Return a cached connection, opening one lazily if needed.

        Returns ``None`` when the database file does not exist or cannot be
        opened (e.g. corrupt, locked exclusively).
        """
        resolved = db_path.resolve()
        if resolved in self._conns:
            return self._conns[resolved]
        if not resolved.exists():
            return None
        try:
            conn = await aiosqlite.connect(
                f'file:{resolved}?mode=ro', uri=True,
            )
            conn.row_factory = aiosqlite.Row
            self._conns[resolved] = conn
            return conn
        except (FileNotFoundError, sqlite3.OperationalError):
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
    except sqlite3.OperationalError:
        logger.debug('with_db: query failed', exc_info=True)
        return default
