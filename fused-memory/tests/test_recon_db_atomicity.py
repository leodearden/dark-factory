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

    journal collision (arm A)     39/40 SQLITE_BUSY_SNAPSHOT   ->  0/40
    cutoff control    (arm B)      0/40                        ->  0/40
    checkpoint + write (arm C)    20/20 SQLITE_LOCKED          ->  0/20

Arm C is the direct reproduction of the production log line ``checkpoint
recon_journal failed: database table is locked``.  Note the direction: it is the
CHECKPOINT that raises when another coroutine has a statement in flight on the same
connection, not the write.  Same defect, same fix; the arm asserts neither side
raises.

NOT HERE, deliberately, so a later reader does not mistake the gap for an oversight:
the connection-wide-rollback regression (one coroutine's failing unit rolls back the
CONNECTION, discarding another coroutine's in-flight write, whose own commit then
succeeds silently).  Reproducing it needs a suspension point INSIDE a write unit and
no public store method has one; the nearest store-level shape lost 0 of 20 writes
when measured on this base, so an assertion here would be green for reasons nobody
checked.  It is pinned at the primitive instead, in
``shared/tests/test_async_sqlite_base.py``.
"""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime, timedelta

import aiosqlite
import pytest
import pytest_asyncio

from fused_memory.reconciliation.journal import ReconciliationJournal

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

_INSERT_RUN = """INSERT INTO runs
    (id, project_id, run_type, trigger_reason, started_at, completed_at, status)
    VALUES (?, ?, ?, ?, ?, ?, ?)"""


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
    rows = [
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
