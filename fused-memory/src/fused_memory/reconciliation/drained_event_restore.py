"""The single restore policy for one reconciliation run's drained events.

A run's events are re-buffered unless the journal records that run durably
completed: a completed run's events were consumed, and restoring them would
double-process them (task 6453).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from fused_memory.models.reconciliation import RunStatus

if TYPE_CHECKING:
    from fused_memory.reconciliation.event_buffer import EventBuffer
    from fused_memory.reconciliation.journal import ReconciliationJournal


async def restore_unless_completed(
    journal: ReconciliationJournal,
    buffer: EventBuffer,
    project_id: str,
    run_id: str,
) -> int:
    run = await journal.get_run(run_id)
    if run is not None and run.status == RunStatus.completed:
        return 0
    return await buffer.restore_drained(project_id, run_id=run_id)
