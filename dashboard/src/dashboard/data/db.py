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

import aiosqlite

logger = logging.getLogger(__name__)

_T = TypeVar('_T')


class DbPool:
    """Lazy-open pool of read-only aiosqlite connections.

    Call :meth:`get` to obtain a connection for a given path.  The connection
    is created on first access and reused thereafter.  Call :meth:`close_all`
    during shutdown.

    **_open_locks growth is bounded by construction.**
    Every path passed to :meth:`get` is ``root / <fixed-rel-path>`` where
    ``root`` ∈ ``{config.project_root} ∪ config.known_project_roots``.
    Both sets are fixed once at startup (``DashboardConfig.from_env`` reads
    ``DASHBOARD_KNOWN_PROJECT_ROOTS``; ``__post_init__`` resolves them; they
    are never mutated at runtime).  The ``rel-path`` values are fixed literals
    (``data/orchestrator/runs.db``, ``data/burndown/burndown.db``) plus the
    six fixed ``@property`` paths on ``DashboardConfig``.  The ``@property``
    paths (``reconciliation_db``, ``tickets_db``, ``metrics_db``,
    ``write_journal_db``, etc.) are called only with ``config.project_root``,
    never with each root in ``known_project_roots``, so they contribute a
    flat ``+6`` rather than a per-root multiplier.  Therefore::

        |_open_locks| ≤ (1 + len(known_project_roots)) × 2 + 6

    Per-path lock entries are retained for the process lifetime (cleared only
    in :meth:`close_all`) — this is *intentional*: evicting an entry while
    another coroutine holds the lock reintroduces an absent→present race that
    would require refcount machinery to make safe.

    **Guardrail**: any future caller that passes per-run or otherwise-unbounded
    paths to :meth:`get` MUST revisit lock eviction with a refcount-gated
    scheme, because naive per-path deletion is not safe under concurrent access.
    """

    def __init__(self) -> None:
        self._conns: dict[Path, aiosqlite.Connection] = {}
        self._closed: bool = False
        # Per-path open locks — prevents duplicate opens for the same path while
        # allowing disjoint paths to open concurrently (no serialisation between
        # unrelated paths).  Mirrors SqliteTaskBackend._get_connection convention.
        # Growth is bounded; see class docstring for the structural argument.
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
            # Two _closed guards are needed:
            # (a) Pre-connect fast-abort — if close_all() already finished before
            #     we acquired this lock, exit immediately without touching aiosqlite.
            # (b) Post-connect re-check — close_all() does NOT hold the per-path
            #     lock, so it can run to completion while this coroutine is
            #     suspended inside `await aiosqlite.connect()`.  Without (b), the
            #     resumed get() would install a fresh connection into an already-
            #     drained pool, leaking the aiosqlite worker thread indefinitely.
            #     DbPool closes that window here; note the mirrored
            #     SqliteTaskBackend._get_connection convention has the same window
            #     (not closed there — callers must not race close and get).
            if self._closed:
                return None
            try:
                if not resolved.exists():
                    return None
                # as_uri() yields the correctly percent-encoded file: URI (stdlib, POSIX/Windows-aware).
                conn = await aiosqlite.connect(
                    f'{resolved.as_uri()}?mode=ro',
                    uri=True,
                )
                # (b) Post-connect re-check: close_all() may have completed while
                # we were suspended inside aiosqlite.connect() above.
                if self._closed:
                    try:
                        await conn.close()
                    except Exception:
                        logger.debug('DbPool: error closing mid-open conn after pool closed', exc_info=True)
                    return None
                conn.row_factory = aiosqlite.Row
                self._conns[resolved] = conn
                return conn
            except (FileNotFoundError, sqlite3.OperationalError, OSError):
                logger.warning('DbPool: cannot open %s', resolved, exc_info=True)
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
        logger.warning('with_db: query failed', exc_info=True)
        return default
