"""A scheduler-filed L1 keyed by a synthetic sentinel task id, one open per episode.

The shape a scheduler storm escape files (PRD INV-4).  ``has_open_l1`` keeps
one open L1 per sentinel, so a condition re-checked every tick never stacks
duplicates.  The filer that sees its condition clear resolves the L1 with
``resolve_sentinel_l1``.  A human may resolve it too; if the condition still
holds, the next check files a fresh one.  Filing and resolving are both
best-effort: a missing queue (bare-Scheduler unit tests) is a warning on
filing and a no-op on resolving, and any queue failure is logged and
swallowed, so neither can abort the tick that noticed the condition.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from escalation.models import Escalation

if TYPE_CHECKING:
    from escalation.queue import EscalationQueue

logger = logging.getLogger(__name__)

SENTINEL_L1_ROLE: str = 'orchestrator-scheduler'


@dataclass(frozen=True)
class SentinelL1:
    sentinel: str
    category: str
    summary: str
    detail: str
    suggested_action: str


def file_sentinel_l1(queue: EscalationQueue | None, l1: SentinelL1) -> None:
    """File *l1* at level 1 unless one is already open for its sentinel; never raises."""
    if queue is None:
        logger.warning('%s: %s — no escalation_queue is wired, not filing', l1.sentinel, l1.summary)
        return
    try:
        if queue.has_open_l1(l1.sentinel):
            return
        esc = Escalation(
            id=queue.make_id(l1.sentinel),
            task_id=l1.sentinel,
            agent_role=SENTINEL_L1_ROLE,
            severity='blocking',
            category=l1.category,
            summary=l1.summary[:200],
            detail=l1.detail,
            suggested_action=l1.suggested_action,
            level=1,
        )
        queue.submit(esc)
        logger.warning('Filed L1 %s escalation %s', l1.sentinel, esc.id)
    except Exception:
        logger.warning('Failed to file %s escalation', l1.sentinel, exc_info=True)


def resolve_sentinel_l1(queue: EscalationQueue | None, sentinel: str, resolution: str) -> None:
    """Resolve each open L1 filed under *sentinel*, its condition cleared; never raises."""
    if queue is None:
        return
    try:
        for esc in queue.get_by_task(sentinel, status='pending', level=1, agent_role=SENTINEL_L1_ROLE):
            queue.resolve(esc.id, resolution, resolved_by=SENTINEL_L1_ROLE)
            logger.info('Auto-resolved L1 %s escalation %s', sentinel, esc.id)
    except Exception:
        logger.warning('Failed to resolve %s escalation', sentinel, exc_info=True)
