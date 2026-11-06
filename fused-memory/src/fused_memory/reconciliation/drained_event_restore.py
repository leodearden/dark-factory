"""The single restore policy for one reconciliation run's drained events.

A run's events are re-buffered unless the journal records that run durably
completed: a completed run's events were consumed, and restoring them would
double-process them (task 6453).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fused_memory.reconciliation.sqlite_failures import await_best_effort

if TYPE_CHECKING:
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
