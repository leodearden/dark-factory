"""Closing an alarm filed on a sentinel task id.

An alarm is an escalation filed on a sentinel task id, one per watched
condition, and closed when the condition clears. Closing is one policy for
every such alarm: resolve each PENDING record on the sentinel as benign, at
any level, because a promoted L2 carries the sentinel too and closes with its
L1 (task 5290).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from escalation.models import Escalation


class PendingAlarmQueue(Protocol):
    """The escalation-queue calls that closing an alarm makes."""

    def get_by_task(self, task_id: str, status: str | None = None) -> Sequence[Escalation]: ...

    def resolve(
        self, escalation_id: str, resolution: str, *,
        resolved_by: str | None = None, resolution_class: str | None = None,
    ) -> object: ...


def resolve_pending_alarm(
    queue: PendingAlarmQueue, sentinel: str, resolution: str, *, resolved_by: str,
) -> tuple[str, ...]:
    """Resolve every pending record on *sentinel* as benign, by *resolved_by*; return their ids.

    The one read is both the set resolved and the answer to "was an alarm
    open", so the two cannot disagree.
    """
    pending = queue.get_by_task(sentinel, status='pending')
    for esc in pending:
        queue.resolve(esc.id, resolution, resolved_by=resolved_by, resolution_class='benign')
    return tuple(esc.id for esc in pending)
