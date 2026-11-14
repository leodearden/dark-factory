"""Discharge the L2 wrappers of a sentinel-filed monitor alarm by MEMBERSHIP.

**This docstring is the canonical statement of the rule; its callers point here.**

Some detectors file a monitor alarm against a synthetic sentinel task id
(``recovery_emission.RECOVERY_VETO_STREAK_SENTINEL_PREFIX``,
``zero_progress_requeue.ZERO_PROGRESS_SENTINEL_PREFIX``).  The auto-watcher
may cluster those alarms into L2 WRAPPERS whose task_id and category an LLM
chose.  When a detector stands its alarm down, it must stand those wrappers
down too, or a stale blocking L2 outlives the condition it reports.

A pending L2 is discharged when ALL of these hold:

* every member resolves to a record filed under *alarm_role*;
* none of those members is still pending;
* at least one of them was filed under *sentinel*;
* no blocking declared pin (``escalation.declared_pins.blocking_pin_declarations``)
  rides on the wrapper or its members.

Neither the wrapper's category nor its task_id is consulted.  Every
uncertainty — an unreadable member, an ordinary one, an alarm of another
class, another task's still-live alarm — leaves the wrapper pending: a stale
wrapper is visible and hand-closable, while closing a cluster that still
holds another task's live alarm would resolve that alarm silently through
the L2 cascade.  The LAST recovering member-task therefore closes a shared
wrapper.

Caller precondition: resolve your own alarms FIRST.  They then keep the
detector's ``resolved_by`` rather than ``l2-cascade:<id>``, and "none still
pending" already holds for this task's members when a wrapper is judged.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ['discharge_sentinel_wrappers']


def discharge_sentinel_wrappers(
    escalation_queue: Any,
    pending: Iterable[Any],
    *,
    alarm_role: str,
    sentinel: str,
    resolution: str,
) -> int:
    """Resolve every L2 in *pending* that only wraps settled *alarm_role* alarms, one of them *sentinel*'s.

    Each qualifying wrapper is resolved with ``resolved_by=alarm_role``.
    Returns how many were.  Wholly fail-open: on any exception it logs a
    WARNING and returns the count so far.  It carries its own guard because
    the caller's would report an own-alarm resolve that already happened on
    disk as "nothing cleared".
    """
    discharged = 0
    try:
        from escalation.declared_pins import (  # noqa: PLC0415 — optional dep
            blocking_pin_declarations,
        )

        for wrapper in pending:
            if wrapper.level < 2 or not wrapper.members:
                continue
            members = _settled_alarms(escalation_queue, wrapper.members, alarm_role)
            if not members or not any(m.task_id == sentinel for m in members):
                continue
            if blocking_pin_declarations([wrapper, *members]):
                continue
            escalation_queue.resolve(wrapper.id, resolution, resolved_by=alarm_role)
            discharged += 1
    except Exception as exc:  # noqa: BLE001 — fail-open backstop
        logger.warning(
            '%s wrappers for %s could not be discharged (non-fatal): %s',
            alarm_role, sentinel, exc,
        )
    return discharged


def _settled_alarms(
    escalation_queue: Any, member_ids: list[str], alarm_role: str,
) -> list[Any] | None:
    """Read *member_ids*; ``None`` at the first that is unreadable, not an *alarm_role* alarm, or still pending."""
    members = []
    for member_id in dict.fromkeys(member_ids):
        member = escalation_queue.get(member_id)
        if member is None or member.agent_role != alarm_role or member.status == 'pending':
            return None
        members.append(member)
    return members
