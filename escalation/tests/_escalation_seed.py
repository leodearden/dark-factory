"""The single construction site for a directly-submitted pending ``Escalation``.

Callers wrap ``seed_escalation`` to supply their own default *summary*: tests
that share one module-scoped queue rely on it to name the module that seeded a
record. *task_id* doubles as the ``queue.make_id`` namespace key only as a test
convenience; ``escalation/src/escalation/queue.py::make_id`` explains why the
two differ in production.
"""

from __future__ import annotations

from typing import Any

from escalation.models import Escalation
from escalation.queue import EscalationQueue


def seed_escalation(
    queue: EscalationQueue,
    *,
    level: int,
    task_id: str,
    agent_role: str = 'implementer',
    summary: str | None = None,
    **kw: Any,
) -> Escalation:
    """Build a pending escalation at *level*, submit it to *queue*, and return it."""
    if summary is None:
        summary = f'seeded test escalation (level={level})'
    kw.setdefault('severity', 'blocking')
    kw.setdefault('category', 'scope_violation')
    esc = Escalation(
        id=queue.make_id(task_id),
        task_id=task_id,
        agent_role=agent_role,
        level=level,
        summary=summary,
        **kw,
    )
    queue.submit(esc)
    return esc
