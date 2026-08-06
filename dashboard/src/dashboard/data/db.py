"""Persistent read-only connection pool for dashboard SQLite databases.

Instead of opening a fresh ``aiosqlite.connect()`` per request (each spawning
a thread), the :class:`DbPool` maintains one long-lived connection per database
path and reuses it across poll cycles.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import sqlite3
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TypeVar

import aiosqlite

logger = logging.getLogger(__name__)

_T = TypeVar('_T')

# Fire-and-forget close() tasks for connections that landed with no owner (see
# DbPool._adopt_or_close).  The event loop holds only a WEAK reference to a
# Task, so an unreferenced one can be garbage-collected mid-flight; this set
# holds the strong reference until each task's own done-callback removes it.
# Mirrors _ABANDONED_PROBES in dashboard/app.py.
_PENDING_CLOSES: set[asyncio.Task] = set()


def _discard_pending_close(task: asyncio.Task) -> None:
    _PENDING_CLOSES.discard(task)
    if not task.cancelled():
        task.exception()  # consume so "exception was never retrieved" isn't logged


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

    Two further invariants keep that ownership single-valued and complete:

    * **Exactly one connection per path survives, even when an adoption races a
      concurrent getter.**  ``_adopt_or_close`` installs into ``_conns`` without
      holding ``_open_locks[path]`` (the cancelled getter released it while
      unwinding), so a getter suspended in its own connect for the same path can
      resume to find the slot already filled.  It keeps the INCUMBENT and closes
      the connection it opened.  A second connection for one path is reachable
      by nothing and would strand its worker thread.
    * **:meth:`close_all` does not return while any close it caused is still in
      flight.**  ``_close_once`` JOINS a concurrent close rather than skipping
      it, and ``close_all`` gathers the fire-and-forget closes
      ``_adopt_or_close`` scheduled.  Returning early would hand callers a pool
      that looks drained while a worker thread is still mid-``close()`` — and a
      loop torn down at that instant strands it, which is the original leak by a
      different route.
    """

    def __init__(self) -> None:
        self._conns: dict[Path, aiosqlite.Connection] = {}
        self._closed: bool = False
        # In-flight aiosqlite.connect() tasks -> the path each is opening.
        # Populated by get() before it awaits, cleared by each task's own
        # done-callback.  See the ownership invariant in the class docstring.
        self._inflight: dict[asyncio.Task, Path] = {}
        # Connections with a close() currently in flight.  Guards against two
        # closers racing on the same Connection: aiosqlite's close() calls
        # stop() in its finally, and two concurrent stop()s queue two STOP
        # sentinels while the worker thread breaks on the first — so the second
        # close() would await a future nothing will ever resolve.  See
        # _close_once.
        self._closing: dict[aiosqlite.Connection, asyncio.Future[None]] = {}
        # Per-path open locks — prevents duplicate opens for the same path while
        # allowing disjoint paths to open concurrently (no serialisation between
        # unrelated paths).  Mirrors SqliteTaskBackend._get_connection convention.
        # Growth is bounded; see class docstring for the structural argument.
        self._open_locks: dict[Path, asyncio.Lock] = {}
        self._open_locks_lock: asyncio.Lock = asyncio.Lock()
        # Fire-and-forget close tasks scheduled by _adopt_or_close, tracked
        # PER INSTANCE as well as in the module-global _PENDING_CLOSES.  The two
        # sets have distinct jobs and neither subsumes the other, so do not
        # "tidy" either away:
        #   * _PENDING_CLOSES (module-global) is what survives if the DbPool
        #     itself is garbage-collected while a close is in flight — the event
        #     loop holds only a WEAK reference to a Task.
        #   * _pending_closes (this one) is the only handle close_all() has on
        #     those tasks, and gathering them is what makes "close_all() has
        #     returned" mean "every close it caused has finished".
        self._pending_closes: set[asyncio.Task] = set()

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
                try:
                    conn = await asyncio.shield(task)
                except asyncio.CancelledError:
                    # THIS caller is gone but the connect is still running, and
                    # its aiosqlite worker thread is already alive.  Hand
                    # ownership to the pool before unwinding — otherwise the
                    # Connection lands with nobody holding it and the (non-
                    # daemon) thread is stranded for the life of the process.
                    #
                    # Callback ORDER is load-bearing: _forget_inflight is
                    # re-attached last so inflight_count — the pool's public
                    # "is anything still landing?" signal — never reads 0 while
                    # a landed connection is still unowned.
                    task.remove_done_callback(self._forget_inflight)
                    task.add_done_callback(
                        functools.partial(self._adopt_or_close, path=resolved)
                    )
                    task.add_done_callback(self._forget_inflight)
                    raise
                # (b) Post-connect re-check: close_all() may have completed while
                # we were suspended inside aiosqlite.connect() above.
                if self._closed:
                    await self._close_once(conn, resolved)
                    return None
                # Post-shield CACHE re-check.  The _closed re-check alone is not
                # enough: _adopt_or_close installs into _conns WITHOUT holding
                # _open_locks[resolved] — the cancelled getter released that lock
                # while unwinding through `async with lock` — so an adopted
                # connection for this very path can appear while this coroutine
                # is suspended in the shield.  Overwriting it would drop it from
                # _conns with nothing else referencing it, putting it in neither
                # _conns nor _inflight: close_all() could never reach it and its
                # (non-daemon) worker thread would outlive the process's useful
                # life.  Keep the incumbent and close what we opened.
                #
                # This is one half of a pair covering both landing orders; the
                # other half is _adopt_or_close's own `path not in self._conns`
                # guard, which handles the reverse (pooled connection first,
                # adoption second).  Neither guard alone is sufficient.
                existing = self._conns.get(resolved)
                if existing is not None and existing is not conn:
                    await self._close_once(conn, resolved)
                    return existing
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

    async def _close_once(self, conn: aiosqlite.Connection, path: Path) -> None:
        """Close *conn*, but never concurrently with another close of *conn*.

        ``aiosqlite.Connection.close()`` is idempotent when called SEQUENTIALLY
        (it returns early once ``_connection is None``), but two closes running
        CONCURRENTLY both get past that check and both call ``stop()`` in their
        ``finally``.  That queues two STOP sentinels while the worker thread
        breaks out of its loop on the first, so the second close awaits a future
        nothing will ever resolve — a permanent hang.  Three call sites can
        legitimately race here (a resuming getter, ``close_all``'s drain, and
        the ``_adopt_or_close`` handoff), so they all funnel through this guard.

        The guard SERIALISES rather than DROPS: a second caller JOINS the
        in-flight close instead of returning while it is still running.
        :meth:`close_all` depends on that, because the first closer is often the
        fire-and-forget task ``_adopt_or_close`` scheduled and ``close_all`` has
        no other handle on it — a skipping guard would let ``close_all`` return
        with a close still in flight, and a loop closed at that moment strands
        the worker thread (the very leak this pool exists to prevent).

        The membership check and insert straddle no ``await``, so this is
        race-free on a single event loop.  The entry is removed afterwards to
        keep the map bounded; a later close then finds ``_connection is None``
        and returns immediately, which is safe precisely because it is
        sequential.
        """
        existing = self._closing.get(conn)
        if existing is not None:
            # shield(), not a bare `await existing`: the future is SHARED, so a
            # bare await would let one cancelled waiter cancel it out from under
            # every other waiter.
            await asyncio.shield(existing)
            return
        finished: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._closing[conn] = finished
        try:
            await conn.close()
        except Exception:
            logger.debug('DbPool: error closing connection for %s', path, exc_info=True)
        finally:
            # Resolved in the `finally`, not after the await: if the FIRST closer
            # is itself cancelled mid-close the joiners must still be released,
            # or the join deadlocks on a future nobody will ever resolve.
            self._closing.pop(conn, None)
            if not finished.done():
                finished.set_result(None)

    def _adopt_or_close(self, task: asyncio.Task, *, path: Path) -> None:
        """Take ownership of a connection whose getter was cancelled mid-connect.

        Attached (from :meth:`get`) only when the awaiting caller is cancelled,
        which is the one case where the landed ``Connection`` would otherwise
        have no owner at all.

        ADOPT when the pool is still open and holds nothing for *path*: the
        connection is exactly what the next ``get(path)`` would have opened, so
        installing it makes the cancellation free rather than merely harmless.
        This is what upgrades an abandoned ``/healthz`` probe (``_probe_db``,
        dashboard/app.py — it cancels its probe task at the deadline and
        deliberately does not await the unwinding) from a permanent thread leak
        into a warm cached connection.

        CLOSE otherwise — the pool is already drained, or a connection for
        *path* won the race — because a second connection for one path is not
        reachable by :meth:`close_all` and would strand its worker thread.

        The ``path not in self._conns`` guard below is one HALF of a pair; the
        other half is the post-shield ``existing is not conn`` re-check in
        :meth:`get`.  They cover the two landing orders of the same race — this
        one handles "a pooled connection already won", ``get`` handles "the
        adoption already won" — and neither is sufficient alone, because
        adoption happens without the per-path lock a getter would otherwise be
        serialised by.
        """
        if task.cancelled() or task.exception() is not None:
            # A connect that raised already stopped its own worker thread:
            # aiosqlite's Connection._connect catches BaseException and calls
            # stop() before re-raising.
            return
        conn = task.result()
        if not self._closed and path not in self._conns:
            conn.row_factory = aiosqlite.Row
            self._conns[path] = conn
            return
        # Nothing is awaiting this connection, so the close has to be scheduled
        # rather than awaited.  _close_once makes it safe for close_all()'s
        # drain to be closing the very same connection concurrently.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop: the close cannot be performed at all, and
            # aiosqlite's own ResourceWarning would be the only trace. Say so.
            logger.warning(
                'DbPool: connect for %s landed with no running event loop; '
                'its aiosqlite worker thread cannot be reaped',
                path,
            )
            return
        close_task = loop.create_task(self._close_once(conn, path))
        # Registered in BOTH sets — see the __init__ comment for why neither is
        # redundant.
        _PENDING_CLOSES.add(close_task)
        self._pending_closes.add(close_task)
        close_task.add_done_callback(_discard_pending_close)
        close_task.add_done_callback(self._discard_pending_close)

    def _discard_pending_close(self, task: asyncio.Task) -> None:
        """Per-instance mirror of the module-level :func:`_discard_pending_close`."""
        self._pending_closes.discard(task)
        if not task.cancelled():
            task.exception()  # consume so "exception was never retrieved" isn't logged

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
        done, still_pending = await asyncio.wait(
            set(inflight), timeout=_INFLIGHT_DRAIN_TIMEOUT
        )
        if still_pending:
            # The budget is the fail-soft; this WARNING is what stops it being a
            # SILENT one.  Nothing downstream can recover these paths, so name
            # them (INV-2, structured-facts-at-failure).
            logger.warning(
                'DbPool.close_all: %d in-flight connect(s) did not land within '
                '%.1fs; their aiosqlite worker threads may outlive this event '
                'loop: %s',
                len(still_pending),
                _INFLIGHT_DRAIN_TIMEOUT,
                ', '.join(sorted(str(inflight[t]) for t in still_pending)),
            )
        # Deliberately NOT cancelled.  Cancelling an in-flight aiosqlite connect
        # re-creates the exact orphan this drain exists to prevent: the worker
        # thread is already started, and a cancel just abandons the half-built
        # Connection by a different route.  The _adopt_or_close callback from
        # get() stays attached instead, so a late arrival is still closed if the
        # loop is still alive.  Do NOT "tidy" this by cancelling still_pending.
        for task in done:
            if task.cancelled() or task.exception() is not None:
                # A connect that raised already stopped its own worker thread
                # (aiosqlite's _connect catches BaseException and calls stop()).
                continue
            await self._close_once(task.result(), inflight[task])

    async def close_all(self) -> None:
        """Close every managed connection and clear the pool.

        **Invariant on return**: every connection this pool caused to exist has
        been CLOSED — including ones that were still being opened, and ones
        adopted from a cancelled getter after the drain budget expired.  The
        sole exception is an in-flight CONNECT that exceeded
        ``_INFLIGHT_DRAIN_TIMEOUT``, which :meth:`_drain_inflight` reports at
        WARNING with its path rather than dropping silently.

        That invariant is what callers rely on to tear the event loop down
        immediately afterwards: aiosqlite's worker signals completion via
        ``call_soon_threadsafe``, so any close still in flight when the loop
        closes leaves its (non-daemon) thread stranded.
        """
        # Set before iterating so any concurrent get() that acquires a per-path
        # lock after we clear _conns will see _closed=True and return None rather
        # than re-populating a pool that is supposed to be drained.
        self._closed = True
        # Drain in-flight connects FIRST: a connection still being opened is not
        # in _conns yet, so the loop below cannot reap it.
        await self._drain_inflight()
        for path, conn in self._conns.items():
            await self._close_once(conn, path)
        self._conns.clear()
        # Clear per-path locks to prevent unbounded growth across distinct paths.
        self._open_locks.clear()
        # Join the closes _adopt_or_close scheduled but nobody awaited.  A
        # straggler connect that missed the drain budget lands in still_pending,
        # not done, so _drain_inflight never awaits it; if it lands while the
        # loop above is running, this is the ONLY handle on its close.
        # return_exceptions=True so one failed close cannot abort the rest of
        # shutdown.
        pending = tuple(self._pending_closes)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


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
