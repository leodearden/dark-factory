"""Pending level-1 escalation records for tests of the workflow's open-L1 hand-off gates."""

from __future__ import annotations

from escalation.models import Escalation
from escalation.queue import EscalationQueue


def submit_l1(
    queue: EscalationQueue, task_id: str, severity: str, *, agent_role: str,
) -> Escalation:
    """Submit and return a pending ``design_concern`` L1 on *task_id* at *severity*.

    *agent_role* names the filer: ``'steward'`` for a hand-off to a human,
    ``'harness-info-l0-router'`` for
    ``orchestrator/src/orchestrator/escalation_router.py::_promote``.
    """
    esc = Escalation(
        id=queue.make_id(task_id), task_id=task_id, agent_role=agent_role,
        severity=severity, category='design_concern',
        summary=f'a {severity!r} L1 filed by {agent_role!r}', level=1,
    )
    queue.submit(esc)
    return esc
