"""File a boundary guard's burst alarm into an escalation queue, at most once per open burst.

A guard's storm escape (``shared.boundary_storm_escape``) fires at most once
per window and leaves dedup to its sink, because only the sink can see the
queue. A defect running for hours would otherwise file one record per hour.
:func:`file_storm_once` folds every fire into the single open alarm until an
operator resolves it (INV-4), noting on it any outcome it does not yet name. A
:class:`GuardStormAlarm` names whose alarm it is, so each guard on a server
shares this one filer.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from shared.boundary_storm_escape import storm_outcome_label

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
    strictly better than losing the alarm for an active defect. So does a
    burst of a new outcome that cannot be noted on the open alarm
    (:func:`_folded`).

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
        if _folded(queue, alarm, open_alarm, record):
            return open_alarm.id

    return queue.submit(Escalation(
        id=queue.make_id(alarm.anchor_task_id),
        task_id=alarm.anchor_task_id,
        agent_role=alarm.agent_role,
        severity='blocking',
        category=alarm.category,
        summary=f'{_burst_line(record)} — {alarm.condition}',
        detail='\n'.join(f'{key}={record[key]!r}' for key in sorted(record)),
        suggested_action=alarm.suggested_action,
        level=1,
    ))


def _outcome_phrase(record: Mapping[str, Any]) -> str:
    """How an alarm's text names a burst's outcome: one rendering, to write it and to find it."""
    return f' {storm_outcome_label(record)} MCP call(s) '


def _burst_line(record: Mapping[str, Any]) -> str:
    return (
        f'{record.get("count")}{_outcome_phrase(record)}in {record.get("window_seconds")}s '
        f'for project={record.get("project")!r}'
    )


def _folded(
    queue: EscalationQueue, alarm: GuardStormAlarm, open_alarm: Escalation,
    record: Mapping[str, Any],
) -> bool:
    """Fold *record* into *open_alarm*; False when its new outcome could not be noted there.

    The open alarm's summary names the outcome it was filed for, and that is
    what an operator acts on. A burst of ANOTHER outcome (a resolver that went
    from slow to down) folded in silently would leave the alarm naming only the
    first, so it is appended to the alarm as an amendment and logged at
    WARNING. A burst whose outcome the alarm already names folds at INFO.

    A note that cannot land (the amend raised, or the alarm closed after it was
    read) returns False, so the burst is filed as an alarm of its own.
    """
    phrase = _outcome_phrase(record)
    named = [open_alarm.summary, *(note.get('summary', '') for note in open_alarm.amendments)]
    if any(phrase in text for text in named):
        logger.info(
            '%s: %s already open (storm %r now); not duplicating',
            alarm.agent_role, open_alarm.id, dict(record),
        )
        return True
    logger.warning(
        '%s: %s is open naming another outcome; noting this %s burst on it (storm %r)',
        alarm.agent_role, open_alarm.id, storm_outcome_label(record), dict(record),
    )
    try:
        noted = queue.amend(
            open_alarm.id, summary=f'also folded in: {_burst_line(record)}',
            agent_role=alarm.agent_role,
        )
    except Exception:
        logger.exception(
            '%s: could not note the burst on %s; filing it as an alarm of its own',
            alarm.agent_role, open_alarm.id,
        )
        return False
    return noted['status'] in ('amended', 'repeat_framing')
