"""File an orchestrator-detected condition as one deduped born-at-L2 escalation.

A born-at-L2 escalation goes straight to a human: it is filed at level 2 with
severity ``urgent`` under an ``orchestrator-`` role, so the agent-role
downgrade gate exempts it. One pending L2 per root cause is the dedup policy,
and a filing failure never reaches the caller.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from escalation.categories import EscalationCategory, require_registered_category
from escalation.models import Escalation

from orchestrator.event_store import EventType

if TYPE_CHECKING:
    from escalation.queue import EscalationQueue

    from orchestrator.event_store import EventStore

logger = logging.getLogger(__name__)

_SUMMARY_LIMIT = 200


@dataclass(frozen=True)
class BornAtL2Alarm:
    """What stays the same each time one kind of alarm is filed."""

    root_cause: str
    sentinel_task: str
    role: str
    category: EscalationCategory
    suggested_action: str
    created_reason: str

    def __post_init__(self) -> None:
        require_registered_category(self.category)


def file_deduped_born_at_l2(
    queue: EscalationQueue | None,
    alarm: BornAtL2Alarm,
    *,
    summary: str,
    detail: str,
    event_store: EventStore | None,
    filing_claimant_run_id: str | None,
) -> Escalation | None:
    """File *alarm* unless an L2 with its root cause is already pending.

    Returns the filed escalation, or ``None`` when there is no queue, one was
    already pending, or filing failed. A failure is logged, never raised.
    """
    if queue is None:
        return None
    try:
        if queue.find_pending_l2_by_root_cause(alarm.root_cause) is not None:
            return None
        escalation = _escalation(queue, alarm, summary, detail, filing_claimant_run_id)
        queue.submit(escalation)
    except Exception:
        logger.warning('%s: failed to file the born-at-L2', alarm.root_cause, exc_info=True)
        return None
    _emit_created(event_store, alarm, escalation)
    logger.warning('%s: born-at-L2 %s filed', alarm.root_cause, escalation.id)
    return escalation


def _escalation(
    queue: EscalationQueue,
    alarm: BornAtL2Alarm,
    summary: str,
    detail: str,
    filing_claimant_run_id: str | None,
) -> Escalation:
    return Escalation(
        id=queue.make_id(alarm.sentinel_task),
        task_id=alarm.sentinel_task,
        agent_role=alarm.role,
        severity='urgent',
        category=alarm.category,
        level=2,
        root_cause=alarm.root_cause,
        summary=summary[:_SUMMARY_LIMIT],
        detail=detail,
        suggested_action=alarm.suggested_action,
        filing_claimant_run_id=filing_claimant_run_id,
    )


def _emit_created(
    event_store: EventStore | None, alarm: BornAtL2Alarm, escalation: Escalation,
) -> None:
    if event_store is None:
        return
    try:
        event_store.emit(
            EventType.escalation_created,
            task_id=alarm.sentinel_task,
            data={
                'escalation_id': escalation.id,
                'category': escalation.category,
                'severity': escalation.severity,
                'level': escalation.level,
                'reason': alarm.created_reason,
            },
        )
    except Exception:
        logger.warning(
            '%s: born-at-L2 %s filed but its escalation_created emit failed',
            alarm.root_cause, escalation.id, exc_info=True,
        )
