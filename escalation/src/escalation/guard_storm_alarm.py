"""File a boundary guard's burst alarm into an escalation queue, at most once per open burst.

A guard's storm escape (``shared.boundary_storm_escape``) fires at most once
per window and leaves dedup to its sink, because only the sink can see the
queue. A defect running for hours would otherwise file one record per hour.
:func:`file_storm_once` folds every fire into the single open alarm until an
operator resolves it (INV-4). A :class:`GuardStormAlarm` names whose alarm it
is, so each guard on a server shares this one filer.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from escalation.categories import EscalationCategory
from escalation.models import Escalation
from escalation.queue import EscalationQueue

__all__ = ['GuardStormAlarm', 'file_storm_once']

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GuardStormAlarm:
    """One guard's burst alarm: where it is filed, who it is, and what it says.

    *category* and *agent_role* together are the alarm's identity, the pair the
    dedup predicate matches. *condition* closes the summary line, and
    *suggested_action* tells the reader where to start.
    """

    anchor_task_id: str
    category: EscalationCategory
    agent_role: str
    condition: str
    suggested_action: str


def file_storm_once(
    queue: EscalationQueue, alarm: GuardStormAlarm, record: Mapping[str, Any],
) -> str | None:
    """File *alarm* for the burst *record* describes, unless it is already open.

    Returns the id of the open alarm, newly filed or folded into.

    The open-record test is "is MY alarm already open", never "is anything
    pending on this anchor". The anchor is a synthetic id that nothing
    reserves, and ``fused_memory/server/markup_tripwire.py`` records the cost
    of the category-blind test: it "filed nothing 2026-08-16..2026-08-19 while
    41 rejections occurred" because another producer held its anchor open.
    Category and filer are both checked because either can differ on its own;
    ``getattr`` keeps an older record shape lacking either field from raising.

    A failed read falls THROUGH to filing: losing duplicate suppression is
    strictly better than losing the alarm for an active defect.

    Level 1 and 'blocking': a rate alarm about a condition, not a hold on any
    one caller's payload. Unstamped by design (task 3550): no task-workflow
    incarnation filed it.
    """
    try:
        existing = queue.get_by_task(alarm.anchor_task_id, status='pending')
    except Exception:
        logger.exception(
            '%s: could not check for an open storm record; filing a new one rather '
            'than dropping the alarm', alarm.agent_role,
        )
        existing = []
    mine = [
        esc for esc in existing
        if getattr(esc, 'category', None) == alarm.category
        and getattr(esc, 'agent_role', None) == alarm.agent_role
    ]
    if mine:
        # The lowest id, not ``mine[0]``: get_by_task's order comes from a
        # directory glob, and ids carry a monotonic sequence, so the lowest is
        # the record that opened the alarm.
        open_alarm = min(mine, key=lambda esc: esc.id)
        logger.info(
            '%s: %s already open (storm %r now); not duplicating',
            alarm.agent_role, open_alarm.id, dict(record),
        )
        return open_alarm.id

    return queue.submit(Escalation(
        id=queue.make_id(alarm.anchor_task_id),
        task_id=alarm.anchor_task_id,
        agent_role=alarm.agent_role,
        severity='blocking',
        category=alarm.category,
        summary=(
            f'{record.get("count")} {record.get("outcome")} MCP call(s) in '
            f'{record.get("window_seconds")}s for project={record.get("project")!r} '
            f'— {alarm.condition}'
        ),
        detail='\n'.join(f'{key}={record[key]!r}' for key in sorted(record)),
        suggested_action=alarm.suggested_action,
        level=1,
    ))
