"""Keep at most one pending escalation per sentinel task id and level: file it once, then refresh it in place."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from escalation.categories import EscalationCategory, require_registered_category
from escalation.models import Escalation

if TYPE_CHECKING:
    from escalation.queue import EscalationQueue

__all__ = ['RefreshableAlarm', 'file_or_refresh']


@dataclass(frozen=True)
class RefreshableAlarm:
    """What stays the same each time one sentinel alarm is filed or refreshed."""

    task_id: str
    level: int
    agent_role: str
    severity: str
    category: EscalationCategory
    suggested_action: str

    def __post_init__(self) -> None:
        require_registered_category(self.category)


def file_or_refresh(
    queue: EscalationQueue,
    alarm: RefreshableAlarm,
    *,
    summary: str,
    detail: str,
    filing_claimant_run_id: str | None,
) -> Escalation:
    """File *alarm*, or refresh its pending record at the alarm's level; return what was submitted.

    A refresh rewrites summary, detail and timestamp, counts the repeat in
    dedupe_count, and keeps the id and every other field. That includes
    filing_claimant_run_id, which names the incarnation that FILED the record
    (task 3550); a refresh is not a filing.

    Records at another level, and records no longer pending, are never
    refreshed. The level is part of the key because has_open_l1 only sees
    level 1.

    Failures propagate, and the caller decides what a failed filing costs.
    """
    pending = queue.get_by_task(alarm.task_id, status='pending', level=alarm.level)
    if pending:
        escalation = dataclasses.replace(
            pending[0],
            summary=summary,
            detail=detail,
            timestamp=datetime.now(UTC).isoformat(),
            dedupe_count=pending[0].dedupe_count + 1,
        )
    else:
        escalation = Escalation(
            id=queue.make_id(alarm.task_id),
            task_id=alarm.task_id,
            agent_role=alarm.agent_role,
            severity=alarm.severity,
            level=alarm.level,
            category=alarm.category,
            summary=summary,
            detail=detail,
            suggested_action=alarm.suggested_action,
            filing_claimant_run_id=filing_claimant_run_id,
        )
    queue.submit(escalation)
    return escalation
