"""Async SQLite base class and utilities for WAL-mode persistent connections.

Provides:
- apply_wal_pragmas(conn, busy_timeout_ms): standalone utility to configure WAL + busy_timeout
- apply_full_durability_pragmas(conn, busy_timeout_ms): WAL + busy_timeout + Phase 3 triad
- connect_daemon(database, **kwargs): open a connection with worker thread marked daemon
- AtomicConnection: per-connection lock making every access an atomic unit
- AsyncSqliteBase: ABC with lifecycle management (open/close/context-manager/guard)
"""

from __future__ import annotations

import abc
import asyncio
import contextlib
from collections.abc import AsyncIterator, Iterable
from pathlib import Path
from typing import Any, NamedTuple, Self

import aiosqlite

__all__ = [
    'apply_wal_pragmas',
    'apply_full_durability_pragmas',
    'connect_daemon',
    'CheckpointResult',
    'AtomicConnection',
    'AsyncSqliteBase',
]


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
        raise RuntimeError(f'Failed to enable WAL journal mode (got {got!r})')
    if busy_timeout_ms != 0:
        await conn.execute(f'PRAGMA busy_timeout={busy_timeout_ms}')


async def apply_full_durability_pragmas(
    conn: aiosqlite.Connection, *, busy_timeout_ms: int
) -> None:
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


class AtomicConnection:
    """Serializes every access on ONE aiosqlite connection into an atomic unit.

    A single aiosqlite connection is shared by every coroutine in a process,
    and aiosqlite funnels its statements through one worker thread.  That makes
    the statements ordered but NOT grouped: another coroutine's statement can
    be queued between any two of yours.  Two failures follow, both observed in
    production against ``data/reconciliation/reconciliation.db`` and diagnosed
    in ``plans/recon-sqlite-database-locked-rca-2026-09-16.md``.

    **Pinned read snapshot.**  ``async with conn.execute(sql) as cur: await
    cur.fetchall()`` is several queued hops, and SQLite pins the read snapshot
    from the execute hop until the statement completes.  A write queued into
    that gap fails immediately with ``SQLITE_BUSY_SNAPSHOT`` once a DIFFERENT
    connection to the same file has committed.  Raising ``busy_timeout`` cannot
    help, because the busy handler is never invoked while a transaction is
    already open.  :meth:`read_all` closes the gap by doing execute and fetch
    in ONE hop; the lock keeps a write from being queued mid-unit.

    **Connection-wide rollback.**  ``rollback()`` is a property of the
    CONNECTION, not of a unit, so a failing unit used to discard another
    coroutine's in-flight write — whose own commit then succeeded silently,
    losing the write with no error anywhere.  Holding the lock across the
    rollback puts that back inside one unit's blast radius.

    Wraps an already-open connection rather than owning connect, pragmas and
    schema: those differ per store in load-bearing ways (an in-memory store
    cannot enable WAL, migration ordering is store-specific) and folding them
    in would cost one flag per store.
    """

    def __init__(self, connection: aiosqlite.Connection) -> None:
        self._connection = connection
        self._lock = asyncio.Lock()
        self._holder: asyncio.Task | None = None

    @property
    def connection(self) -> aiosqlite.Connection:
        """The wrapped connection, for LIFECYCLE and TEST SUPPORT only.

        No store method may issue statements through it outside ``initialize()``
        and ``close()``.  Routing every other access through :meth:`read_all`,
        :meth:`read_one` and :meth:`write` is what makes atomicity structural
        rather than a convention each new call site has to remember.
        """
        return self._connection

    @contextlib.asynccontextmanager
    async def _held(self, what: str) -> AsyncIterator[None]:
        """Hold the connection lock for one access, refusing to nest."""
        if asyncio.current_task() is self._holder:
            raise RuntimeError(
                f'AtomicConnection.{what} called from inside this task\'s own open '
                f'write() unit — nesting two accesses on one connection would '
                f'deadlock the per-connection lock. Pass the unit\'s connection to '
                f'a private helper instead of calling back through the primitive.'
            )
        async with self._lock:
            self._holder = asyncio.current_task()
            try:
                yield
            finally:
                self._holder = None

    async def read_all(
        self, sql: str, params: Iterable[Any] = ()
    ) -> list[aiosqlite.Row]:
        """Run ``sql`` and materialise every row, in ONE queued worker-thread hop.

        ``execute_fetchall`` is a single queued call, so no other coroutine's
        statement can be queued between the execute and the fetch — which is
        the whole read-side fix.  Never spell this as ``execute`` + ``fetch*``.

        Rows honour the wrapped connection's ``row_factory``, so ``row['col']``
        works when it is set to :class:`aiosqlite.Row`.
        """
        async with self._held('read_all()'):
            return list(await self._connection.execute_fetchall(sql, params))

    async def read_one(
        self, sql: str, params: Iterable[Any] = ()
    ) -> aiosqlite.Row | None:
        """Return the first row of ``sql``, or None when it matches nothing.

        The caller MUST bound the query — a primary key, a ``LIMIT`` or an
        aggregate — because :meth:`read_all` materialises the full result set
        before this discards all but the first row.

        Delegates rather than nests: the lock is taken by :meth:`read_all`
        after this method returns control to it, so the re-entrancy guard sees
        one access, not two.
        """
        rows = await self.read_all(sql, params)
        return rows[0] if rows else None

    @contextlib.asynccontextmanager
    async def write(self) -> AsyncIterator[aiosqlite.Connection]:
        """Hold the connection for one write unit: commit on clean exit, else roll back.

        Yields the connection so the unit's statements — including reads it
        needs to batch, such as a ``DELETE ... RETURNING`` drain — run directly
        on it.  Routing those back through :meth:`read_all` would nest and is
        refused.

        ``BaseException`` is caught deliberately: cancellation must roll back
        too, or aiosqlite's implicit transaction stays open holding the writer
        lock against every other coroutine on the connection.

        Residual, and stated as a known BOUND rather than left to look like an
        oversight: a unit that SELECTs before it writes opens a read snapshot
        ahead of its first write, so a commit from a DIFFERENT connection to the
        same file landing between the two statements can still raise
        ``SQLITE_BUSY_SNAPSHOT``.  Four units have that shape today —
        ``EventBuffer.claim_deferred_writes``, ``EventBuffer.release_stale_claims``,
        ``ReconLedgerStore.gc``'s TTL flip, and ``ReconLedgerStore.mark_addressed``.

        The lock removes the in-process, cross-COROUTINE collision, which is the
        failure this primitive owns and the one the incidents were.  Closing the
        remaining cross-CONNECTION window would need ``BEGIN IMMEDIATE`` — which
        the RCA measured still raising, and excludes by name — or a bounded
        retry, which is separate work.  Do not read the four units above as
        sites awaiting conversion: batching their read inside the unit is
        deliberate, because each must see its own uncommitted write.
        """
        async with self._held('write()'):
            try:
                yield self._connection
                await self._connection.commit()
            except BaseException:
                with contextlib.suppress(Exception):
                    await self._connection.rollback()
                raise

    async def checkpoint(self) -> CheckpointResult:
        """Run ``PRAGMA wal_checkpoint(TRUNCATE)`` as one atomic access.

        Returns ``CheckpointResult(-1, -1, -1)`` when the pragma yields no row,
        preserving the contract the reconciliation stores' callers already
        depend on: the checkpoint cycle unpacks the tuple and logs raises
        separately, so turning a benign empty result into an exception would
        report a checkpoint failure on every affected tick.
        """
        async with self._held('checkpoint()'):
            rows = list(
                await self._connection.execute_fetchall('PRAGMA wal_checkpoint(TRUNCATE)')
            )
        if not rows:
            return CheckpointResult(-1, -1, -1)
        row = rows[0]
        return CheckpointResult(int(row[0]), int(row[1]), int(row[2]))

    async def close(self) -> None:
        """Truncate the WAL best-effort, then close the wrapped connection.

        Taking the lock means a unit already in flight commits first rather
        than being cut off mid-transaction.
        """
        async with self._held('close()'):
            with contextlib.suppress(Exception):
                await self._connection.execute_fetchall('PRAGMA wal_checkpoint(TRUNCATE)')
            await self._connection.close()


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
