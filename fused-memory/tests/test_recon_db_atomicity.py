"""Atomicity regressions for the stores that share one ``reconciliation.db`` file.

``ReconciliationJournal``, ``EventBuffer`` and ``ReconLedgerStore`` each hold their
own aiosqlite connection to ``<data_dir>/reconciliation.db``, and every coroutine in
the fused-memory server shares each of those connections.  A store method is
therefore several queued worker-thread hops rather than an atomic unit, and another
coroutine's statement can be queued between any two of them.  Diagnosed in
``plans/recon-sqlite-database-locked-rca-2026-09-16.md``.

These are cross-store concurrency regressions, not per-store unit tests — each arm
needs a SECOND connection to the same file — so they live in one module instead of
being split across test_journal.py / test_event_buffer.py / test_recon_ledger.py.

The COLLISION arms, a write queued into a read's pinned snapshot, are
CONSTRUCTED rather than raced.  ``_StatementGate`` parks the store's own read just
after it pins its snapshot, so each arm is one deterministic run.  Each arm's
read certifies the pin by not seeing the row the foreign connection committed
while it was parked, and each positive arm has a legacy control through the same
helper: the replaced execute + fetchall read, which the same collision kills.

MEASURED BASELINES, taken on this task's base commit (2026-09-17), recorded so the
next reader does not re-derive them:

    journal: checkpoint vs. write        20/20 SQLITE_LOCKED         ->  0/20
    EventBuffer: checkpoint vs. push     20/20 SQLITE_LOCKED         ->  0/20
    ledger: checkpoint vs. upsert        20/20 SQLITE_LOCKED         ->  0/20
    ledger: cancel vs. concurrent write  survivor's row LOST, 3/3   ->  survives

The checkpoint arms are the direct reproduction of the production log line
``checkpoint recon_journal failed: database table is locked``.  Note the direction:
it is the CHECKPOINT that raises when another coroutine has a statement in flight on
the same connection, not the write.  Same defect, same fix; the arms assert neither
side raises.

TWO GAPS, both deliberate, recorded so a later reader does not mistake either for an
oversight:

1. There is no ReconLedgerStore COLLISION arm.  One was attempted and measured at
   0/20 on this base — a multi-hop ``list_suppressions`` read pinned, a foreign
   commit on the journal's connection, then ``upsert``, at 8,000 rows.  No
   reproducing shape was found for this store, and an arm that passes on the
   unfixed code asserts nothing.  The ledger is migrated anyway: it is the third
   connection to the same file, its checkpoint arm below DOES fail 20/20 today, and
   its old transaction wrapper carried the same connection-wide rollback.

2. The FAILING-unit half of the connection-wide-rollback regression (one coroutine's
   failing unit rolls back the CONNECTION, discarding another coroutine's in-flight
   write, whose own commit then succeeds silently) is pinned at the primitive, in
   ``shared/tests/test_async_sqlite_base.py``, not here.  It needs a suspension
   point INSIDE a write unit and no public store method has one; the nearest
   store-level shape lost 0 of 20 writes when measured.  The CANCELLATION half of
   the same defect a store method CAN trigger, and is pinned below at the ledger.
"""

from __future__ import annotations

import asyncio
import contextlib
import enum
import sqlite3
import threading
from collections.abc import Awaitable, Callable, Coroutine
from datetime import UTC, datetime, timedelta
from typing import Any, TypeVar

import aiosqlite
import pytest
import pytest_asyncio
from shared.async_sqlite_base import CheckpointResult, connect_daemon

from fused_memory.models.reconciliation import (
    EventSource,
    EventType,
    ReconciliationEvent,
)
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.reconciliation.journal import ReconciliationJournal
from fused_memory.reconciliation.recon_ledger import ReconLedgerRecord, ReconLedgerStore

PROJECT_ID = 'atomicity'

# The reaper's age cutoff, and a run started far enough past it to be returned.
STALE_CUTOFF_SECONDS = 1800.0
STALE_RUN_AGE_SECONDS = 2000.0

# How long a parked statement waits to be released.  A hang guard only: never
# elapses on a passing run.  It turns a harness bug into a legible failure
# instead of a wedged worker thread.
_GATE_HANG_GUARD_SECS = 30.0

CHECKPOINT_ITERATIONS = 20
# This failure class can present as an unbounded block as well as a raise, and a
# hang at ``-n 8`` kills the whole xdist worker at the 300 s suite timeout instead
# of failing in seconds.  The bound turns that into a legible TimeoutError.
CHECKPOINT_TIMEOUT_SECONDS = 10

SEEDED_EVENTS = 3

_INSERT_RUN = """INSERT INTO runs
    (id, project_id, run_type, trigger_reason, started_at, completed_at, status)
    VALUES (?, ?, ?, ?, ?, ?, ?)"""

_INSERT_EVENT = """INSERT INTO event_buffer
    (id, project_id, event_type, event_source, agent_id, timestamp, payload, status)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)"""


class _ConnectTap:
    """A store's ``connect``: opens through ``connect_daemon`` and keeps the connection.

    The collision arms install their gate on, and issue the legacy controls'
    reads through, the store's OWN connection; this is how they reach it.
    """

    def __init__(self) -> None:
        self.database: str | None = None
        self._connection: aiosqlite.Connection | None = None

    async def __call__(self, database: str) -> aiosqlite.Connection:
        self.database = database
        self._connection = await connect_daemon(database)
        return self._connection

    @property
    def connection(self) -> aiosqlite.Connection:
        if self._connection is None:
            raise RuntimeError('the store has not opened its connection through this tap')
        return self._connection


@pytest.mark.asyncio
async def test_each_collision_store_runs_its_statements_on_its_injected_connection(tmp_path):
    """Each store opens through its ``connect`` and runs its statements on that connection.

    Every collision arm relies on this: a gate installed on the tapped
    connection parks the store's own read, not some other connection's.
    """
    data_dir = tmp_path / 'recon'
    db_path = str(data_dir / 'reconciliation.db')
    journal_tap, buffer_tap = _ConnectTap(), _ConnectTap()
    journal = ReconciliationJournal(data_dir, connect=journal_tap)
    buffer = EventBuffer(db_path=data_dir / 'reconciliation.db', connect=buffer_tap)
    try:
        for name, store, tap, public_read in (
            ('ReconciliationJournal', journal, journal_tap,
             lambda: journal.get_stale_runs(STALE_CUTOFF_SECONDS)),
            ('EventBuffer', buffer, buffer_tap, lambda: buffer.peek_buffered(PROJECT_ID, 10)),
        ):
            await store.initialize()
            traced: list[str] = []
            await tap.connection.set_trace_callback(traced.append)

            await public_read()

            assert tap.database == db_path, (
                f'{name} asked its connect for {tap.database!r}, not {db_path!r}'
            )
            assert traced, f"{name}'s read did not run on the connection its connect opened"
    finally:
        await buffer.close()
        await journal.close()


class _GateState(enum.Enum):
    ARMED = enum.auto()
    STARTED = enum.auto()
    PARKED = enum.auto()


class _StatementGate:
    """Parks the first statement to start on a connection, just after it pins its snapshot.

    Progress callbacks also fire while a statement is being PREPARED, before its
    snapshot exists, so the gate waits for the statement's own trace callback.
    The first progress callback after that is past ``OP_Transaction``.  Both
    callbacks run on the connection's one worker thread, so the state needs no
    lock.

    Single-use: once parked, both callbacks are inert for the rest of the
    connection's life, so the gate is never uninstalled.  aiosqlite's
    annotations do not admit the ``None`` that would clear them.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.parked = asyncio.Event()
        self._loop = loop
        self._released = threading.Event()
        self._state = _GateState.ARMED

    def on_statement(self, sql: str) -> None:
        if self._state is _GateState.ARMED:
            self._state = _GateState.STARTED

    def on_progress(self) -> int:
        if self._state is _GateState.STARTED:
            self._state = _GateState.PARKED
            self._loop.call_soon_threadsafe(self.parked.set)
            if not self._released.wait(timeout=_GATE_HANG_GUARD_SECS):
                raise TimeoutError('the statement gate was never released')
        return 0

    async def install(self, conn: aiosqlite.Connection) -> None:
        await conn.set_trace_callback(self.on_statement)
        await conn.set_progress_handler(self.on_progress, 1)

    def release(self) -> None:
        self._released.set()


async def _sqlite_error_name(op: Callable[[], Awaitable[object]]) -> str | None:
    """Await ``op`` and return its SQLite error name, or None if it raised nothing."""
    try:
        await op()
    except sqlite3.OperationalError as exc:
        return exc.sqlite_errorname
    return None


async def _until_parked(gate: _StatementGate, reader: asyncio.Task[Any]) -> None:
    """Return once ``reader`` is parked in ``gate``; fail if it finished without parking.

    No deadline: a read that neither parks nor finishes is a genuine hang, which
    pytest-timeout reports.
    """
    parked = asyncio.create_task(gate.parked.wait())
    try:
        await asyncio.wait({parked, reader}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        parked.cancel()
    if not gate.parked.is_set():
        raise AssertionError('the read finished without the gate parking it: nothing collided')


_Rows = TypeVar('_Rows')


async def _collide_with_a_pinned_read(
    conn: aiosqlite.Connection,
    *,
    read: Callable[[], Coroutine[Any, Any, _Rows]],
    write: Callable[[], Awaitable[object]],
    foreign_commit: Callable[[], Awaitable[None]],
) -> tuple[_Rows, str | None]:
    """Issue ``write`` into ``read``'s pinned-snapshot window on ``conn``, after a foreign commit.

    Returns the read's result and the write's SQLite error name (None if it
    raised nothing).  The order is constructed, not raced:

    1. ``read`` parks in the gate on ``conn`` with its snapshot pinned.
    2. The writer task is created, then the test suspends on ``foreign_commit``.
       asyncio's FIFO ready queue runs the writer's first step during that
       suspension, which either queues its statement behind the parked hop or
       parks it on :class:`AtomicConnection`'s lock.
    3. The foreign commit lands, so the pinned snapshot is now stale.
    4. Only then is the gate released.
    """
    gate = _StatementGate(asyncio.get_running_loop())
    await gate.install(conn)
    async with asyncio.TaskGroup() as tasks:
        reader = tasks.create_task(read())
        try:
            await _until_parked(gate, reader)
            writer = tasks.create_task(_sqlite_error_name(write))
            await foreign_commit()
        finally:
            gate.release()
    return reader.result(), writer.result()


@pytest.fixture
def journal_tap() -> _ConnectTap:
    return _ConnectTap()


@pytest_asyncio.fixture
async def journal(tmp_path, journal_tap):
    store = ReconciliationJournal(tmp_path / 'recon', connect=journal_tap)
    await store.initialize()
    try:
        yield store
    finally:
        await store.close()


@pytest_asyncio.fixture
async def second_connection(journal):
    """A second raw connection to the journal's own ``reconciliation.db`` file.

    Both the seeder and the foreign writer: the collisions these arms construct need
    a commit from a DIFFERENT connection to the same file, and seeding through it
    keeps a bulk-insert method off ``ReconciliationJournal``'s public interface.

    Closed from teardown, NEVER only after the assertions.  A raw ``aiosqlite``
    connection's worker thread is non-daemon — the stores are immune because they
    open via ``connect_daemon`` — so one still open when an exception escapes HANGS
    interpreter shutdown instead of reporting.  These arms are SUPPOSED to fail
    while the fix is still being written, and verify runs every pytest leg with
    ``-n 8`` where pytest-timeout ``os._exit()``s the worker, so a hang would take
    that worker's unrelated tests down with it.
    """
    conn = await aiosqlite.connect(str(journal.data_dir / 'reconciliation.db'))
    try:
        yield conn
    finally:
        await conn.close()


async def _seed_runs(conn, *, stale_run: bool) -> None:
    """Insert the run rows the arms collide against, in ONE ``executemany``.

    ``live`` is the write target (started now, so the reaper's age filter never
    returns it), and ``long`` — present only when *stale_run* — is the one row
    the reaper's scan matches.
    """
    now = datetime.now(UTC)
    rows = [('live', PROJECT_ID, 'full', 'seed', now.isoformat(), None, 'running')]
    if stale_run:
        started = (now - timedelta(seconds=STALE_RUN_AGE_SECONDS)).isoformat()
        rows.append(('long', PROJECT_ID, 'full', 'seed', started, None, 'running'))
    async with conn.executemany(_INSERT_RUN, rows):
        pass
    await conn.commit()


# The reaper's scan, test-owned: the legacy controls run it in the replaced
# execute + fetchall shape on the journal's own connection.
_LEGACY_REAPER_SCAN = "SELECT * FROM runs WHERE status = 'running' AND started_at < ?"


async def _legacy_multi_hop_read(
    conn: aiosqlite.Connection, sql: str, params: tuple[object, ...]
) -> list[aiosqlite.Row]:
    """The replaced read shape: execute and fetch as separate queued hops."""
    async with conn.execute(sql, params) as cur:
        return list(await cur.fetchall())


async def _file_a_late_stale_run(conn: aiosqlite.Connection) -> None:
    """Commit run ``late``, which the reaper's scan WOULD match, from ``conn``.

    A read pinned before this commit cannot return it, which is how each journal
    collision arm certifies that the collision really happened.
    """
    started = (datetime.now(UTC) - timedelta(seconds=STALE_RUN_AGE_SECONDS)).isoformat()
    async with conn.execute(
        _INSERT_RUN, ('late', PROJECT_ID, 'full', 'seed', started, None, 'running')
    ):
        pass
    await conn.commit()


_COLLIDED_SESSION = 'session-collided'


async def _record_the_collided_session(journal: ReconciliationJournal) -> None:
    await journal.record_run_session(
        'live', session_id=_COLLIDED_SESSION, stage_cursor='stage-collided'
    )


def _reaper_cutoff() -> str:
    return (datetime.now(UTC) - timedelta(seconds=STALE_CUTOFF_SECONDS)).isoformat()


async def _assert_the_collided_write_landed(journal: ReconciliationJournal) -> None:
    live = await journal.get_run('live')
    assert live is not None and live.session_id == _COLLIDED_SESSION, (
        f"the collided write did not land: run 'live' carries "
        f'{None if live is None else live.session_id!r}'
    )


@pytest.mark.asyncio
async def test_a_write_survives_the_stale_run_reapers_read(
    journal, journal_tap, second_connection
):
    """Arm A: the reaper's pinned read snapshot must not kill a concurrent write.

    Constructed, not raced: the production ``get_stale_runs`` is parked with its
    snapshot pinned, a foreign connection files ``late``, and the write is
    queued into that window.

    ``record_run_session`` is the probe because its unit is a SINGLE ``UPDATE`` with
    no preceding SELECT — a pure write opens no read snapshot of its own, so the
    "expect no error" assertion is structurally sound rather than merely lucky.  A
    SELECT-then-write method would be a flaky probe here (see ``AtomicConnection.
    write``'s stated residual).
    """
    await _seed_runs(second_connection, stale_run=True)

    stale, error = await _collide_with_a_pinned_read(
        journal_tap.connection,
        read=lambda: journal.get_stale_runs(STALE_CUTOFF_SECONDS),
        write=lambda: _record_the_collided_session(journal),
        foreign_commit=lambda: _file_a_late_stale_run(second_connection),
    )

    assert error is None, (
        f'the write died while the stale-run reaper held a read snapshot on the '
        f'same connection: {error}'
    )
    assert [run.id for run in stale] == ['long'], (
        f"the reaper returned {[run.id for run in stale]!r}: it was not pinned before "
        "the foreign commit filed 'late', so this run did not collide at all"
    )
    await _assert_the_collided_write_landed(journal)


@pytest.mark.asyncio
async def test_a_legacy_multi_hop_reaper_read_kills_the_write(
    journal, journal_tap, second_connection
):
    """CONTROL: the replaced read shape loses arm A's exact collision.

    Same collision, same production write, same foreign commit; only the read's
    shape is replaced, by an execute + fetchall on the journal's own connection.
    That it kills the write is what proves arm A discriminates rather than
    passing vacuously.
    """
    await _seed_runs(second_connection, stale_run=True)

    rows, error = await _collide_with_a_pinned_read(
        journal_tap.connection,
        read=lambda: _legacy_multi_hop_read(
            journal_tap.connection, _LEGACY_REAPER_SCAN, (_reaper_cutoff(),)
        ),
        write=lambda: _record_the_collided_session(journal),
        foreign_commit=lambda: _file_a_late_stale_run(second_connection),
    )

    assert error == 'SQLITE_BUSY_SNAPSHOT', (
        f'the replaced reaper read did not kill the write ({error!r}), so this '
        'collision no longer reproduces the defect arm A guards'
    )
    assert [row['id'] for row in rows] == ['long'], (
        f"the parked read saw {[row['id'] for row in rows]!r}: it was not pinned "
        "before the foreign commit filed 'late'"
    )


@pytest.mark.asyncio
async def test_no_stale_run_means_no_pinned_read_and_no_failures(
    journal, journal_tap, second_connection
):
    """Arm B, a permanent control: the replaced read shape, with nothing past the cutoff.

    With no row matched, the replaced shape's first step runs to completion and
    releases its snapshot before the write runs, so nothing is held across hops.
    Paired with the legacy reaper control, which matches ``long`` and kills the
    write, this shows SQLITE_BUSY_SNAPSHOT is caused by the HELD row, not by
    the gate or the foreign commit.
    """
    await _seed_runs(second_connection, stale_run=False)

    rows, error = await _collide_with_a_pinned_read(
        journal_tap.connection,
        read=lambda: _legacy_multi_hop_read(
            journal_tap.connection, _LEGACY_REAPER_SCAN, (_reaper_cutoff(),)
        ),
        write=lambda: _record_the_collided_session(journal),
        foreign_commit=lambda: _file_a_late_stale_run(second_connection),
    )

    assert error is None, (
        f'the write failed with no run past the cutoff, so this collision fails for '
        f'a reason the held-row control does not describe: {error}'
    )
    assert rows == [], (
        f"the parked read saw {[row['id'] for row in rows]!r}: it was not pinned "
        "before the foreign commit filed 'late'"
    )
    await _assert_the_collided_write_landed(journal)


@pytest.mark.asyncio
async def test_checkpoint_and_write_do_not_lock_each_other_out(journal, second_connection):
    """Arm C: ``checkpoint()`` concurrent with a write — neither side may raise."""
    await _seed_runs(second_connection, stale_run=False)

    for i in range(CHECKPOINT_ITERATIONS):
        written, checkpointed = await asyncio.wait_for(
            asyncio.gather(
                journal.record_run_session(
                    'live', session_id=f'session-{i}', stage_cursor=f'stage-{i}'
                ),
                journal.checkpoint(),
                return_exceptions=True,
            ),
            timeout=CHECKPOINT_TIMEOUT_SECONDS,
        )
        assert not isinstance(written, BaseException), (
            f'iteration {i}: the write raised alongside a concurrent checkpoint: {written!r}'
        )
        assert not isinstance(checkpointed, BaseException), (
            f'iteration {i}: the checkpoint raised alongside a concurrent write: '
            f'{checkpointed!r}'
        )

    # server/main.py::_run_checkpoint_cycle unpacks the result positionally, so the
    # 3-element shape is part of the contract, not an implementation detail.
    busy, log, pages = await journal.checkpoint()
    assert all(isinstance(value, int) for value in (busy, log, pages)), (
        f'checkpoint() must keep unpacking as (busy, log, checkpointed): {(busy, log, pages)!r}'
    )


@pytest.fixture
def buffer_tap() -> _ConnectTap:
    return _ConnectTap()


@pytest_asyncio.fixture
async def event_buffer(journal, buffer_tap):
    """An EventBuffer on the journal's OWN ``reconciliation.db`` file.

    Several stores, several connections, one file — production's shape.  The
    collision arms' foreign writer is nonetheless ``second_connection``, not the
    journal: the pin proof needs the foreign commit to land a row the peek would
    match, and the journal writes no ``event_buffer`` rows.  Both stores open via
    ``connect_daemon``, so neither carries the shutdown hazard that
    ``second_connection`` documents.
    """
    buffer = EventBuffer(db_path=journal.data_dir / 'reconciliation.db', connect=buffer_tap)
    await buffer.initialize()
    try:
        yield buffer
    finally:
        await buffer.close()


def _event(index: int) -> ReconciliationEvent:
    return ReconciliationEvent(
        id=f'pushed-{index}',
        type=EventType.memory_added,
        source=EventSource.agent,
        project_id=PROJECT_ID,
        timestamp=datetime.now(UTC),
    )


def _seeded_event_ids(count: int = SEEDED_EVENTS) -> list[str]:
    return [f'seeded-{i}' for i in range(count)]


async def _seed_buffered_events(conn, *, count: int) -> None:
    """Fill the buffer with ``count`` ``status='buffered'`` rows, in ONE executemany.

    Distinct, increasing timestamps, so ``peek_buffered``'s ``ORDER BY
    timestamp`` returns them in ``_seeded_event_ids`` order.
    """
    now = datetime.now(UTC)
    rows = [
        (
            event_id,
            PROJECT_ID,
            'memory_added',
            'agent',
            None,
            (now - timedelta(seconds=count - i)).isoformat(),
            '{}',
            'buffered',
        )
        for i, event_id in enumerate(_seeded_event_ids(count))
    ]
    async with conn.executemany(_INSERT_EVENT, rows):
        pass
    await conn.commit()


# The peek's scan, test-owned: the legacy control runs it in the replaced
# execute + fetchall shape on the buffer's own connection.
_LEGACY_PEEK = """SELECT * FROM event_buffer
    WHERE project_id = ? AND status = 'buffered'
    ORDER BY timestamp ASC LIMIT ?"""


async def _buffer_a_late_event(conn: aiosqlite.Connection) -> None:
    """Commit buffered event ``late``, which the peek WOULD match, from ``conn``.

    A peek pinned before this commit cannot return it, which is how each buffer
    collision arm certifies that the collision really happened.
    """
    async with conn.execute(
        _INSERT_EVENT,
        (
            'late',
            PROJECT_ID,
            'memory_added',
            'agent',
            None,
            datetime.now(UTC).isoformat(),
            '{}',
            'buffered',
        ),
    ):
        pass
    await conn.commit()


@pytest.mark.asyncio
async def test_a_heartbeat_survives_a_peek_on_the_same_connection(
    journal, event_buffer, buffer_tap, second_connection
):
    """Arm D: a buffer write must survive a buffer read that is still in flight.

    Constructed, not raced: the production ``peek_buffered`` is parked with its
    snapshot pinned, a foreign connection buffers ``late``, and the heartbeat is
    queued into that window.

    ``heartbeat`` is the probe because its unit is a SINGLE ``UPDATE`` with no
    preceding SELECT — the same property that makes ``record_run_session`` the
    right probe for the journal arm.  Do not substitute another write method
    without re-checking it.
    """
    await _seed_buffered_events(second_connection, count=SEEDED_EVENTS)

    peeked, error = await _collide_with_a_pinned_read(
        buffer_tap.connection,
        read=lambda: event_buffer.peek_buffered(PROJECT_ID, 1_000_000),
        write=lambda: event_buffer.heartbeat(PROJECT_ID),
        foreign_commit=lambda: _buffer_a_late_event(second_connection),
    )

    assert error is None, (
        f'the heartbeat died while peek_buffered held a read snapshot on the same '
        f'connection: {error}'
    )
    assert [event.id for event in peeked] == _seeded_event_ids(), (
        f'the peek returned {[event.id for event in peeked]!r}: it was not pinned '
        "before the foreign commit buffered 'late', so this run did not collide at all"
    )


@pytest.mark.asyncio
async def test_a_legacy_multi_hop_peek_kills_the_heartbeat(
    journal, event_buffer, buffer_tap, second_connection
):
    """CONTROL: the replaced read shape loses arm D's exact collision.

    Same collision, same production heartbeat, same foreign commit; only the
    peek's shape is replaced, by an execute + fetchall on the buffer's own
    connection.  That it kills the heartbeat is what proves arm D discriminates.
    """
    await _seed_buffered_events(second_connection, count=SEEDED_EVENTS)

    rows, error = await _collide_with_a_pinned_read(
        buffer_tap.connection,
        read=lambda: _legacy_multi_hop_read(
            buffer_tap.connection, _LEGACY_PEEK, (PROJECT_ID, 1_000_000)
        ),
        write=lambda: event_buffer.heartbeat(PROJECT_ID),
        foreign_commit=lambda: _buffer_a_late_event(second_connection),
    )

    assert error == 'SQLITE_BUSY_SNAPSHOT', (
        f'the replaced peek did not kill the heartbeat ({error!r}), so this '
        'collision no longer reproduces the defect arm D guards'
    )
    assert [row['id'] for row in rows] == _seeded_event_ids(), (
        f"the parked peek saw {[row['id'] for row in rows]!r}: it was not pinned "
        "before the foreign commit buffered 'late'"
    )


@pytest.mark.asyncio
async def test_event_buffer_checkpoint_and_push_do_not_lock_each_other_out(event_buffer):
    """Arm E: the journal's arm C, at the second of the three shared connections."""
    for i in range(CHECKPOINT_ITERATIONS):
        pushed, checkpointed = await asyncio.wait_for(
            asyncio.gather(
                event_buffer.push(_event(i)),
                event_buffer.checkpoint(),
                return_exceptions=True,
            ),
            timeout=CHECKPOINT_TIMEOUT_SECONDS,
        )
        assert not isinstance(pushed, BaseException), (
            f'iteration {i}: the push raised alongside a concurrent checkpoint: {pushed!r}'
        )
        assert not isinstance(checkpointed, BaseException), (
            f'iteration {i}: the checkpoint raised alongside a concurrent push: '
            f'{checkpointed!r}'
        )


@pytest.mark.asyncio
async def test_checkpoint_on_an_uninitialized_event_buffer_is_a_non_event():
    """An EventBuffer that was never initialized answers the sentinel, not a raise.

    server/main.py's checkpoint cycle unpacks the result and logs raises
    separately, so turning this into an exception would report a checkpoint
    failure on every tick of a Taskmaster-disabled deployment.
    """
    assert await EventBuffer().checkpoint() == CheckpointResult.unavailable()


@pytest_asyncio.fixture
async def ledger(journal):
    """A ReconLedgerStore on the journal's own ``reconciliation.db`` file.

    The third of production's three connections to that one file.
    """
    store = ReconLedgerStore(journal.data_dir / 'reconciliation.db')
    await store.initialize()
    try:
        yield store
    finally:
        await store.close()


def _ledger_record(task_id: str) -> ReconLedgerRecord:
    return ReconLedgerRecord(
        project_id=PROJECT_ID,
        record_kind='marker',
        task_id=task_id,
        payload_json='{}',
        state='open',
        created_at='2026-09-17T00:00:00+00:00',
    )


@pytest.mark.asyncio
async def test_ledger_checkpoint_and_upsert_do_not_lock_each_other_out(ledger):
    """Arm F: the checkpoint arm at the third shared connection."""
    for i in range(CHECKPOINT_ITERATIONS):
        written, checkpointed = await asyncio.wait_for(
            asyncio.gather(
                ledger.upsert(_ledger_record(f'marker-{i}')),
                ledger.checkpoint(),
                return_exceptions=True,
            ),
            timeout=CHECKPOINT_TIMEOUT_SECONDS,
        )
        assert not isinstance(written, BaseException), (
            f'iteration {i}: the upsert raised alongside a concurrent checkpoint: {written!r}'
        )
        assert not isinstance(checkpointed, BaseException), (
            f'iteration {i}: the checkpoint raised alongside a concurrent upsert: '
            f'{checkpointed!r}'
        )

    busy, log, pages = await ledger.checkpoint()
    assert all(isinstance(value, int) for value in (busy, log, pages)), (
        f'checkpoint() must keep unpacking as (busy, log, checkpointed): {(busy, log, pages)!r}'
    )


@pytest.mark.asyncio
async def test_a_cancelled_write_does_not_discard_a_concurrent_one(ledger):
    """Arm G: cancelling one unit must not roll back another's in-flight write.

    Rollback is a property of the CONNECTION, not of a transaction, so a
    cancelled unit that rolls back the shared connection takes any other
    coroutine's uncommitted statements with it — and that coroutine's own commit
    then succeeds, reporting a write that no longer exists.  Silent data loss
    with no error anywhere, which is why it is asserted on the surviving row
    rather than on an exception.

    Both writes are queued before the cancellation lands, so the victim's
    rollback (if it takes one) happens while the survivor's INSERT is still
    uncommitted.
    """
    survivor = asyncio.create_task(ledger.upsert(_ledger_record('survivor')))
    victim = asyncio.create_task(
        ledger.upsert_many([_ledger_record(f'victim-{i}') for i in range(200)])
    )
    await asyncio.sleep(0)
    victim.cancel()

    with contextlib.suppress(asyncio.CancelledError):
        await victim
    await survivor

    assert await ledger.get_by_identity(PROJECT_ID, 'marker', task_id='survivor') is not None, (
        "the surviving write is gone: the cancelled unit's rollback discarded it, and "
        'its own commit then reported success'
    )
    assert await ledger.get_by_identity(PROJECT_ID, 'marker', task_id='victim-0') is None, (
        'the cancelled unit left rows behind: its rollback was partial'
    )


@pytest.mark.asyncio
async def test_every_recon_store_answers_unavailable_after_close(journal, event_buffer, ledger):
    """After ``close()``, each recon store's ``checkpoint()`` answers the sentinel.

    ``server/main.py::_run_checkpoint_cycle`` runs on a timer against stores
    whose shutdown it does not own, so a tick landing after ``close()`` is an
    expected race, not a fault. Every fused-memory AtomicConnection store answers
    it with ``CheckpointResult.unavailable()`` and never raises; these are the
    three that share ``reconciliation.db``.
    """
    await journal.close()
    await event_buffer.close()
    await ledger.close()

    for name, store in (
        ('ReconciliationJournal', journal),
        ('EventBuffer', event_buffer),
        ('ReconLedgerStore', ledger),
    ):
        assert await store.checkpoint() == CheckpointResult.unavailable(), (
            f'{name}.checkpoint() after close must answer the sentinel: the '
            'checkpoint cycle does not own its shutdown'
        )
