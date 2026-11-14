"""Startup recovery of events a dead predecessor drained but never wrote a run
row for (task 9655). Task 9757 writes the run row before the drain; once it
lands, this pass is a safety net.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fused_memory.reconciliation.event_buffer import DrainedRun, EventBuffer
    from fused_memory.reconciliation.journal import ReconciliationJournal

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecoveredDrain:
    drained: DrainedRun
    restored: int
    lock_released: bool


async def recover_predecessor_drains(
    journal: ReconciliationJournal, buffer: EventBuffer,
) -> list[RecoveredDrain]:
    """Re-buffer every run-less drain another instance left, releasing its lock if it still holds it.

    Call once at startup, before any project loop spawns, so nothing of this
    process can be mid-drain. The premise is the one
    ``ReconciliationHarness._recover_predecessor_runs`` relies on: fm is
    single-instance, so a foreign instance seen at startup is a dead
    predecessor, and a project lock is released only when its holder is the
    drain's own instance. A drained run that has a journal row is left to the
    journal-row recovery passes; a completed run's events were consumed. One
    drained run's failure is logged and does not stop the others.
    """
    recovered: list[RecoveredDrain] = []
    for drained in await buffer.drained_runs_of_other_instances():
        try:
            outcome = await _recover_one_drain(journal, buffer, drained)
        except Exception:
            logger.warning(
                'reconciliation.predecessor_drain_recovery_failed',
                exc_info=True,
                extra=_drained_fields(drained),
            )
            continue
        if outcome is not None:
            recovered.append(outcome)
    return recovered


async def _recover_one_drain(
    journal: ReconciliationJournal, buffer: EventBuffer, drained: DrainedRun,
) -> RecoveredDrain | None:
    if await journal.get_run(drained.run_id) is not None:
        return None
    lock_holder, _ = await buffer.get_lock_status(drained.project_id)
    restored = await buffer.restore_drained(drained.project_id, run_id=drained.run_id)
    lock_released = (
        lock_holder == drained.instance_id
        and await buffer.mark_run_complete(
            drained.project_id, instance_id=drained.instance_id,
        ) > 0
    )
    logger.info(
        'reconciliation.predecessor_drain_recovered',
        extra={
            **_drained_fields(drained),
            'restored': restored,
            'lock_released': lock_released,
        },
    )
    return RecoveredDrain(drained, restored, lock_released)


def _drained_fields(drained: DrainedRun) -> dict[str, str]:
    return {
        'project_id': drained.project_id,
        'run_id': drained.run_id,
        'instance_id': drained.instance_id,
    }
