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

MEASURED BASELINES, taken on this task's base commit (2026-09-17), recorded so the
next reader does not re-derive them:

    journal: reaper read vs. write       39/40 SQLITE_BUSY_SNAPSHOT  ->  0/40
    journal: same loop, nothing stale     0/40                       ->  0/40
    journal: checkpoint vs. write        20/20 SQLITE_LOCKED         ->  0/20
    EventBuffer: peek vs. heartbeat      ~70% of iterations (28/40)  ->  0/20
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
import sqlite3
from datetime import UTC, datetime, timedelta

import aiosqlite
import pytest
import pytest_asyncio

from fused_memory.models.reconciliation import (
    EventSource,
    EventType,
    ReconciliationEvent,
    Watermark,
)
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.reconciliation.journal import ReconciliationJournal
from fused_memory.reconciliation.recon_ledger import ReconLedgerRecord, ReconLedgerStore

PROJECT_ID = 'atomicity'

# The reaper's age cutoff, and a run started far enough past it to be returned.
STALE_CUTOFF_SECONDS = 1800.0
STALE_RUN_AGE_SECONDS = 2000.0

# Enough completed rows that the reaper's full scan of ``runs`` is still in the
# worker thread when the next statement is queued behind it.  Measured: 0.25 s to
# seed, 0.93 s for the 40-iteration loop.
FILLER_RUNS = 24_000
COLLISION_ITERATIONS = 40

CHECKPOINT_ITERATIONS = 20
# This failure class can present as an unbounded block as well as a raise, and a
# hang at ``-n 8`` kills the whole xdist worker at the 300 s suite timeout instead
# of failing in seconds.  The bound turns that into a legible TimeoutError.
CHECKPOINT_TIMEOUT_SECONDS = 10

# Enough buffered rows that peek_buffered's ORDER BY timestamp scan is still in the
# worker thread when the write is queued behind it.  Do NOT tune these counts down:
# below the reproducing size the arm passes for the wrong reason.  verify appends
# ``-n 8`` to every pytest leg, so this runs 8-way parallel on a loaded machine,
# which WIDENS the window rather than narrowing it.
#
# Two measurements of the pre-fix failure rate at 4,000 rows, both worth keeping
# because they bracket what a future tuner should expect: planning measured 28/40
# iterations (~70%, and 39/40 at 8,000 rows); re-measured here while writing the
# arm, 28 of 80 iterations across four runs (~35%), worst run 4/20 and never a
# clean one.  The cause of the spread was not established.  What the second
# measurement does establish is that 4,000 is comfortably above the reproducing
# size on this tree, which is the property the count has to have.
BUFFERED_EVENTS = 4_000
BUFFER_COLLISION_ITERATIONS = 20

_INSERT_RUN = """INSERT INTO runs
    (id, project_id, run_type, trigger_reason, started_at, completed_at, status)
    VALUES (?, ?, ?, ?, ?, ?, ?)"""

_INSERT_EVENT = """INSERT INTO event_buffer
    (id, project_id, event_type, event_source, agent_id, timestamp, payload, status)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)"""


@pytest_asyncio.fixture
async def journal(tmp_path):
    store = ReconciliationJournal(tmp_path / 'recon')
    await store.initialize()
    try:
        yield store
    finally:
        await store.close()


@pytest_asyncio.fixture
async def second_connection(journal):
    """A second raw connection to the journal's own ``reconciliation.db`` file.

    Both the seeder and the foreign writer: the collisions these arms provoke need
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


async def _seed_runs(conn, *, filler: int, stale_run: bool) -> None:
    """Insert the run rows the arms race against, in ONE ``executemany``.

    ``live`` is the write target (started now, so the reaper's age filter never
    returns it), ``r3`` is the foreign writer's target, and ``long`` — present only
    when *stale_run* — is the one row the reaper's scan actually matches.
    """
    now = datetime.now(UTC)
    # completed_at is NULL for a running run, so the element type must admit None:
    # inferred from the all-`str` filler comprehension alone it would not.
    rows: list[tuple[str, str, str, str, str, str | None, str]] = [
        (
            f'filler-{i}',
            PROJECT_ID,
            'full',
            'seed',
            (now - timedelta(seconds=STALE_RUN_AGE_SECONDS)).isoformat(),
            now.isoformat(),
            'completed',
        )
        for i in range(filler)
    ]
    rows.append(('live', PROJECT_ID, 'full', 'seed', now.isoformat(), None, 'running'))
    rows.append(('r3', PROJECT_ID, 'full', 'seed', now.isoformat(), now.isoformat(), 'completed'))
    if stale_run:
        rows.append(
            (
                'long',
                PROJECT_ID,
                'full',
                'seed',
                (now - timedelta(seconds=STALE_RUN_AGE_SECONDS)).isoformat(),
                None,
                'running',
            )
        )
    async with conn.executemany(_INSERT_RUN, rows):
        pass
    await conn.commit()


async def _write_failures_under_a_pinned_read(journal, foreign) -> list[str]:
    """Race a write against the reaper's read, and report what killed the writes.

    Each iteration puts ``get_stale_runs`` into the connection's worker thread,
    commits on the *foreign* connection so the file's snapshot moves, then queues a
    write behind the pinned read.  The foreign UPDATE writes a CHANGING value —
    a no-op UPDATE does not advance the WAL and the arm would pass vacuously.
    """
    failures: list[str] = []
    for i in range(COLLISION_ITERATIONS):
        reaper = asyncio.create_task(journal.get_stale_runs(STALE_CUTOFF_SECONDS))
        try:
            await asyncio.sleep(0)
            async with foreign.execute(
                'UPDATE runs SET trigger_reason = ? WHERE id = ?', (f'rotate-{i}', 'r3')
            ):
                pass
            await foreign.commit()
            try:
                await journal.record_run_session(
                    'live', session_id=f'session-{i}', stage_cursor=f'stage-{i}'
                )
            except sqlite3.OperationalError as exc:
                failures.append(exc.sqlite_errorname)
        finally:
            await reaper
    return failures


@pytest.mark.asyncio
async def test_a_write_survives_the_stale_run_reapers_read(journal, second_connection):
    """Arm A: the reaper's pinned read snapshot must not kill a concurrent write.

    ``record_run_session`` is the probe because its unit is a SINGLE ``UPDATE`` with
    no preceding SELECT — a pure write opens no read snapshot of its own, so the
    "expect zero" assertion is structurally sound rather than merely lucky.  A
    SELECT-then-write method would be a flaky probe here (see ``AtomicConnection.
    write``'s stated residual).
    """
    await _seed_runs(second_connection, filler=FILLER_RUNS, stale_run=True)

    failures = await _write_failures_under_a_pinned_read(journal, second_connection)

    assert not failures, (
        f'{len(failures)}/{COLLISION_ITERATIONS} writes died while the stale-run '
        f'reaper held a read snapshot on the same connection: {sorted(set(failures))}'
    )


@pytest.mark.asyncio
async def test_no_stale_run_means_no_pinned_read_and_no_failures(journal, second_connection):
    """Arm B, a permanent control: the same loop, with nothing past the cutoff.

    Measured at 0/40 before the fix as well as after.  It is what distinguishes
    "arm A is keyed to the reaper's pinned snapshot" from "arm A is keyed to the
    shape of the loop" — without it, a future regression that broke the seeding
    would turn arm A green and look like a pass.
    """
    await _seed_runs(second_connection, filler=FILLER_RUNS, stale_run=False)

    failures = await _write_failures_under_a_pinned_read(journal, second_connection)

    assert not failures, (
        f'{len(failures)}/{COLLISION_ITERATIONS} writes failed with no run past the '
        f'cutoff, so this loop collides for a reason arm A does not describe: '
        f'{sorted(set(failures))}'
    )


@pytest.mark.asyncio
async def test_checkpoint_and_write_do_not_lock_each_other_out(journal, second_connection):
    """Arm C: ``checkpoint()`` concurrent with a write — neither side may raise."""
    await _seed_runs(second_connection, filler=0, stale_run=False)

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


@pytest_asyncio.fixture
async def event_buffer(journal):
    """An EventBuffer on the journal's OWN ``reconciliation.db`` file.

    Two stores, two connections, one file — production's exact shape, which is
    what makes the journal a genuine foreign writer rather than a stand-in.  Both
    open via ``connect_daemon``, so neither carries the shutdown hazard that
    ``second_connection`` documents.
    """
    buffer = EventBuffer(db_path=journal.data_dir / 'reconciliation.db')
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


async def _seed_buffered_events(conn, *, count: int) -> None:
    """Fill the buffer with ``count`` ``status='buffered'`` rows, in ONE executemany.

    Distinct descending-age timestamps, so ``peek_buffered``'s ``ORDER BY
    timestamp`` has real work to do rather than reading an already-ordered heap.
    """
    now = datetime.now(UTC)
    rows = [
        (
            f'seeded-{i}',
            PROJECT_ID,
            'memory_added',
            'agent',
            None,
            (now - timedelta(seconds=count - i)).isoformat(),
            '{}',
            'buffered',
        )
        for i in range(count)
    ]
    async with conn.executemany(_INSERT_EVENT, rows):
        pass
    await conn.commit()


async def _heartbeat_failures_under_a_pinned_read(buffer, journal) -> list[str]:
    """Race the buffer's heartbeat against its own peek, with the journal writing.

    The journal is the foreign connection here: its ``update_watermark`` commit —
    a CHANGING value each iteration, so the WAL really advances — lands while
    ``peek_buffered`` still holds a read snapshot on the buffer's connection.
    """
    failures: list[str] = []
    for i in range(BUFFER_COLLISION_ITERATIONS):
        reader = asyncio.create_task(buffer.peek_buffered(PROJECT_ID, 1_000_000))
        try:
            await asyncio.sleep(0)
            await journal.update_watermark(
                Watermark(project_id=PROJECT_ID, last_full_run_id=f'run-{i}')
            )
            try:
                await buffer.heartbeat(PROJECT_ID)
            except sqlite3.OperationalError as exc:
                failures.append(exc.sqlite_errorname)
        finally:
            await reader
    return failures


@pytest.mark.asyncio
async def test_a_heartbeat_survives_a_peek_on_the_same_connection(
    journal, event_buffer, second_connection
):
    """Arm D: a buffer write must survive a buffer read that is still in flight.

    ``heartbeat`` is the probe because its unit is a SINGLE ``UPDATE`` with no
    preceding SELECT — the same property that makes ``record_run_session`` the
    right probe for the journal arm.  Do not substitute another write method
    without re-checking it.
    """
    await _seed_buffered_events(second_connection, count=BUFFERED_EVENTS)

    failures = await _heartbeat_failures_under_a_pinned_read(event_buffer, journal)

    assert not failures, (
        f'{len(failures)}/{BUFFER_COLLISION_ITERATIONS} heartbeats died while '
        f'peek_buffered held a read snapshot on the same connection: '
        f'{sorted(set(failures))}'
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
    """An EventBuffer that was never initialized reports (-1, -1, -1), not a raise.

    server/main.py's checkpoint cycle unpacks the result and logs raises
    separately, so turning this into an exception would report a checkpoint
    failure on every tick of a Taskmaster-disabled deployment.
    """
    assert await EventBuffer().checkpoint() == (-1, -1, -1)


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
async def test_the_post_close_checkpoint_contract_of_each_store(journal, event_buffer, ledger):
    """Post-close, ``checkpoint()`` raises for two of the three stores, not the third.

    ``server/main.py``'s checkpoint cycle runs on a timer against stores whose
    shutdown it does not own, so a tick landing after ``close()`` is a real
    production path rather than a hypothetical.  The three stores answer it
    differently: EventBuffer short-circuits a missing connection to the
    ``(-1, -1, -1)`` its callers already had, while the journal and the ledger
    raise 'not initialized' from ``_require_access()``.

    The split is deliberate — preserving each store's existing caller contract is
    what kept ``server/main.py`` edit-free through this migration — but it is
    exactly the kind of near-uniform invariant that drifts unwatched, and the
    only other post-close coverage is the never-initialized EventBuffer above.
    Pinned here so whoever unifies the two contracts (task 5562) changes it
    deliberately and sees both halves at once.
    """
    await journal.close()
    await event_buffer.close()
    await ledger.close()

    assert await event_buffer.checkpoint() == (-1, -1, -1), (
        'EventBuffer must keep answering the sentinel after close: the checkpoint '
        'cycle unpacks the result and logs raises separately'
    )
    with pytest.raises(RuntimeError, match='not initialized'):
        await journal.checkpoint()
    with pytest.raises(RuntimeError, match='not initialized'):
        await ledger.checkpoint()
