"""The single restore policy for one reconciliation run's drained events.

A run's events are re-buffered unless the journal records that run durably
completed: a completed run's events were consumed, and restoring them would
double-process them (task 6453). restore_on_raise is the caller-span form of
the same policy, for the span a caller that minted a run id runs it in.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fused_memory.reconciliation.sqlite_failures import await_best_effort

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from fused_memory.reconciliation.event_buffer import EventBuffer
    from fused_memory.reconciliation.journal import ReconciliationJournal

logger = logging.getLogger(__name__)


async def restore_unless_completed(
    journal: ReconciliationJournal,
    buffer: EventBuffer,
    project_id: str,
    run_id: str,
) -> int:
    if await journal.is_run_completed(run_id):
        return 0
    return await buffer.restore_drained(project_id, run_id=run_id)


async def restore_unless_completed_best_effort(
    journal: ReconciliationJournal,
    buffer: EventBuffer,
    project_id: str,
    run_id: str,
) -> None:
    """The form for a caller already handling a run's failure or deadline: a
    failing restore is logged as ``restore_drained failed: …`` at ERROR, never
    raised over what is being handled. Cancellation propagates."""
    await await_best_effort(
        restore_unless_completed(journal, buffer, project_id, run_id),
        what='restore_drained', run_id=run_id,
        logger=logger, level=logging.ERROR,
    )


@asynccontextmanager
async def restore_on_raise(
    journal: ReconciliationJournal,
    buffer: EventBuffer,
    project_id: str,
    run_id: str,
) -> AsyncIterator[None]:
    """The span a caller that minted *run_id* runs that run in. Any raise, a
    deadline or shutdown cancellation included, re-buffers the run's drained
    events unless the journal records it completed, then propagates. The
    restore is shielded, so a second cancellation cannot abort it (task 6695)."""
    try:
        yield
    except (Exception, asyncio.CancelledError):
        await asyncio.shield(
            restore_unless_completed_best_effort(journal, buffer, project_id, run_id),
        )
        raise
