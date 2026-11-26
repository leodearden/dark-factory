"""Atomicity regressions for WriteJournal's one shared aiosqlite connection.

Every coroutine in the fused-memory server shares the journal's single
connection, so a store method is several queued worker-thread hops rather than
an atomic unit, and another coroutine's statement can be queued between any two
of them.  A rollback is a property of the CONNECTION, so a cancelled unit used to
discard another coroutine's in-flight write, whose own commit then reported
success: ``plans/recon-sqlite-database-locked-rca-2026-09-16.md`` RC9.

MEASURED BASELINES, taken on this task's base commit (2026-10-08), recorded so the
next reader does not re-derive them:

    checkpoint vs. log_write_op             20/20 SQLITE_LOCKED      ->  0/20
    cancelled referent batch vs. terminal   survivor ABSENT, 3/3     ->  survives
      outcome                               (call returned True)
    checkpoint after close()                raises on a dead handle  ->  sentinel

The checkpoint arm is the direct reproduction of the production log line
``checkpoint write_journal failed: database table is locked`` (2026-08-26,
2026-09-14).  It is the CHECKPOINT that raises when a write is in flight on the
same connection; the write swallows its own failures, so its side is asserted
through the public drop counter.

A sibling module rather than more of ``test_write_journal.py``: that file is past
the 2,000-line alarm, and task 5405 is editing it.
"""

from __future__ import annotations

import asyncio

import pytest
import pytest_asyncio
from shared.async_sqlite_base import CheckpointResult

from fused_memory.services.write_journal import WriteJournal

CHECKPOINT_ITERATIONS = 20
TIMEOUT_SECONDS = 10


@pytest_asyncio.fixture
async def journal(tmp_path):
    store = WriteJournal(tmp_path / 'journal')
    await store.initialize()
    try:
        yield store
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_checkpoint_and_write_do_not_lock_each_other_out(journal):
    for i in range(CHECKPOINT_ITERATIONS):
        _, checkpointed = await asyncio.wait_for(
            asyncio.gather(
                journal.log_write_op(write_op_id=f'w{i}', operation='add_memory'),
                journal.checkpoint(),
                return_exceptions=True,
            ),
            TIMEOUT_SECONDS,
        )
        assert not isinstance(checkpointed, BaseException), (
            f'iteration {i}: the checkpoint raised alongside a concurrent write: '
            f'{checkpointed!r}'
        )

    assert journal.journal_drop_stats()['dropped_total'] == 0, (
        'a write was dropped alongside a concurrent checkpoint'
    )


@pytest.mark.asyncio
async def test_a_cancelled_write_does_not_discard_a_concurrent_terminal_outcome(journal):
    """Cancelling one unit must not roll back another's in-flight write.

    ``record_terminal_outcome`` returns whether the outcome was durably recorded,
    so the defect is a method reporting True for a row a foreign rollback had
    already discarded — asserted on the row, not on an exception.
    """
    victim = asyncio.create_task(
        journal.log_referent_findings(
            [{'k': i} for i in range(50)], group_id='g', episode_uuid='e'
        )
    )
    survivor = asyncio.create_task(
        journal.record_terminal_outcome(write_op_id='survivor', terminal_status='completed')
    )
    await asyncio.sleep(0)
    victim.cancel()

    _, recorded = await asyncio.wait_for(
        asyncio.gather(victim, survivor, return_exceptions=True), TIMEOUT_SECONDS
    )

    assert recorded is True, recorded
    assert await journal.get_write_op('survivor') is not None, (
        "the terminal outcome is gone: the cancelled unit's rollback discarded it, "
        'and record_terminal_outcome still reported it durable'
    )
    assert await journal.get_referent_findings() == [], (
        'the cancelled batch left rows behind: its rollback was partial'
    )


@pytest.mark.asyncio
async def test_checkpoint_after_close_answers_unavailable(journal):
    """A checkpoint tick landing after ``close()`` answers the sentinel, never a raise.

    ``server/main.py::_run_checkpoint_cycle`` runs on a timer against stores whose
    shutdown it does not own, so this is an expected race.
    """
    await journal.close()

    assert await journal.checkpoint() == CheckpointResult.unavailable()
