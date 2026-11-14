"""The reconciler's escalation-queue seam.

Owns the optional ``escalation`` package import and its installed default
queue factory, filing one reconciler-authored L1, and reading the pending
queue. Every function here is synchronous blocking filesystem I/O that
coroutine callers run under ``asyncio.to_thread`` (INV-8).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from shared.project_layout import escalation_queue_dir

from fused_memory.models.scope import ProjectRoot

__all__ = [
    'INSTALLED_ESCALATION_QUEUE_FACTORY',
    'EscalationQueueFactory',
    'file_reconciler_l1',
    'pending_escalations',
]

EscalationQueueFactory = Callable[[Path], 'EscalationQueue']

# `escalation` is an optional workspace package. When it is absent the default
# factory is None and every escalation path degrades to its logged no-op or veto.
try:
    from escalation.models import Escalation  # type: ignore[import-untyped]
    from escalation.queue import EscalationQueue  # type: ignore[import-untyped]
    INSTALLED_ESCALATION_QUEUE_FACTORY: EscalationQueueFactory | None = EscalationQueue
except ImportError:  # pragma: no cover — exercised only in minimal envs
    INSTALLED_ESCALATION_QUEUE_FACTORY = None


def file_reconciler_l1(
    queue_factory: EscalationQueueFactory,
    project_root: ProjectRoot,
    *,
    task_id: str,
    severity: str,
    category: str,
    summary: str,
    detail: str,
    suggested_action: str,
) -> str:
    """File one reconciler-authored L1 escalation and return its id.

    Blocking filesystem I/O: callers run it under ``asyncio.to_thread``
    (INV-8). The queue is built here, only when a filing actually happens,
    because ``EscalationQueue.__init__`` mkdirs: an eager build would leave
    ``data/escalations/`` in every target project. Raises whatever the queue
    raised; each caller owns its containment.
    """
    queue = queue_factory(escalation_queue_dir(project_root))
    return queue.submit(Escalation(
        id=queue.make_id(task_id),
        task_id=task_id,
        agent_role='reconciler',
        severity=severity,
        category=category,
        summary=summary,
        detail=detail,
        suggested_action=suggested_action,
        level=1,
    ))


def pending_escalations(
    queue_factory: EscalationQueueFactory, project_root: ProjectRoot,
) -> list[Escalation]:
    """Every pending escalation in project_root's queue; blocking I/O, raises whatever the queue raised."""
    return queue_factory(escalation_queue_dir(project_root)).get_pending()
