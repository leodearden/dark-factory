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

# How long :meth:`DbPool.close_all` waits for in-flight ``aiosqlite.connect()``
# calls to land before giving up on them and reporting them.  BOUNDED on
# purpose: an unbounded wait would let one wedged sqlite open hang application
# shutdown forever — the exact failure mode ``_probe_db`` (dashboard/app.py)
# was written to avoid — and would convert a leak into a hang inside pytest
# teardown.  Stragglers are reported LOUDLY instead (see close_all).
_INFLIGHT_DRAIN_TIMEOUT = 5.0


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

    **Ownership invariant: the pool owns every connection it causes to exist,
    including ones still being opened.**
    ``aiosqlite.Connection.__await__`` starts its worker thread BEFORE the
    connect completes, so between ``aiosqlite.connect(...)`` and the value
    landing in ``_conns`` there is a live OS thread that ``close_all()`` cannot
    see.  A caller cancelled in that window (``lifespan()`` cancelling
    ``_metrics_loop`` at shutdown; ``_probe_db`` abandoning a /healthz probe on
    its deadline) would strand that thread permanently — and DbPool connects
    via plain ``aiosqlite.connect``, whose thread is NOT a daemon, so a
    stranded one also blocks process exit.  Every in-flight connect is
    therefore run in its own task tracked in ``_inflight``, awaited under
    :func:`asyncio.shield` so a cancelled caller does not cancel the connect
    itself, and drained by :meth:`close_all` before it returns.
    """

    def __init__(self) -> None:
        self._conns: dict[Path, aiosqlite.Connection] = {}
        self._closed: bool = False
        # In-flight aiosqlite.connect() tasks -> the path each is opening.
        # Populated by get() before it awaits, cleared by each task's own
        # done-callback.  See the ownership invariant in the class docstring.
        self._inflight: dict[asyncio.Task, Path] = {}
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
                async def _do_connect() -> aiosqlite.Connection:
                    # Awaiting INSIDE an `async def` (rather than
                    # asyncio.ensure_future(aiosqlite.connect(...))) is
                    # load-bearing: the real aiosqlite.connect() returns an
                    # awaitable Connection, not a coroutine, while the test
                    # suite patches it with an `async def` wrapper that returns
                    # a coroutine.  `await` handles both shapes identically.
                    return await aiosqlite.connect(
                        f'{resolved.as_uri()}?mode=ro',
                        uri=True,
                    )

                task = asyncio.create_task(_do_connect())
                self._inflight[task] = resolved
                task.add_done_callback(self._forget_inflight)
                # shield(), not a bare `await task`: a bare await propagates
                # THIS caller's cancellation into the connect task, and
                # cancelling an in-flight aiosqlite connect is not a fix —
                # it leaves the same half-built Connection with a live worker
                # thread, just reached by a different route.  shield() lets the
                # connect run to completion so the pool can actually close it.
                conn = await asyncio.shield(task)
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

    def _forget_inflight(self, task: asyncio.Task) -> None:
        """Drop a finished connect task from ``_inflight``.

        Also consumes the task's exception so asyncio does not log "exception
        was never retrieved" for a connect whose only awaiter was cancelled
        before the shield could re-raise it.  Mirrors ``_discard_abandoned_probe``
        in dashboard/app.py.  Consuming it here does NOT hide it from
        :meth:`get`: retrieving an exception only clears the never-retrieved
        warning flag; the shield's own callback still copies it to the awaiting
        future, which lands in get()'s ``except`` funnel.
        """
        self._inflight.pop(task, None)
        if not task.cancelled():
            task.exception()

    @property
    def open_count(self) -> int:
        """Number of currently held connections."""
        return len(self._conns)

    @property
    def inflight_count(self) -> int:
        """Number of ``aiosqlite.connect()`` calls currently in flight."""
        return len(self._inflight)

    async def _drain_inflight(self) -> None:
        """Wait out in-flight connects and close whatever landed.

        Runs while the event loop is STILL RUNNING — that is the whole point.
        aiosqlite's worker thread signals completion with
        ``future.get_loop().call_soon_threadsafe(...)``; once the owning loop is
        closed that call raises ``RuntimeError: Event loop is closed`` OUT of
        ``_connection_worker_thread``, the thread never reaches its STOP
        sentinel, and the escaped exception gets attributed to whatever
        unrelated code happens to be running.  Closing here, before shutdown
        returns, is what lets those threads exit cleanly.
        """
        # Snapshot the mapping, not just the keys: each task's own done-callback
        # pops it from _inflight, so the paths would be gone by the time we
        # want to report on them.
        inflight = dict(self._inflight)
        if not inflight:
            return
        done, _still_pending = await asyncio.wait(
            set(inflight), timeout=_INFLIGHT_DRAIN_TIMEOUT
        )
        for task in done:
            if task.cancelled() or task.exception() is not None:
                # A connect that raised already stopped its own worker thread
                # (aiosqlite's _connect catches BaseException and calls stop()).
                continue
            try:
                await task.result().close()
            except Exception:
                logger.debug(
                    'DbPool: error closing in-flight connection for %s',
                    inflight.get(task),
                    exc_info=True,
                )

    async def close_all(self) -> None:
        """Close every managed connection and clear the pool."""
        # Set before iterating so any concurrent get() that acquires a per-path
        # lock after we clear _conns will see _closed=True and return None rather
        # than re-populating a pool that is supposed to be drained.
        self._closed = True
        # Drain in-flight connects FIRST: a connection still being opened is not
        # in _conns yet, so the loop below cannot reap it.
        await self._drain_inflight()
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
