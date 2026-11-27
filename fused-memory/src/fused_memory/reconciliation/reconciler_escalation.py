"""The reconciler's escalation-queue seam.

Holds its guarded import of the optional ``escalation`` package and the
installed default queue factory that import yields, filing one
reconciler-authored L1, and deciding which pending records pin a task. Other
modules, ``reconciliation/harness.py`` among them, keep their own guarded
imports for the names they use. Every function here is synchronous blocking
filesystem I/O that coroutine callers run under ``asyncio.to_thread`` (INV-8).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from shared.project_layout import escalation_queue_dir

from fused_memory.models.scope import ProjectRoot

__all__ = [
    'INSTALLED_ESCALATION_QUEUE_FACTORY',
    'EscalationQueueFactory',
    'file_reconciler_l1',
    'pinning_escalation_ids_by_task',
]

EscalationQueueFactory = Callable[[Path], 'EscalationQueue']

# `escalation` is an optional workspace package. When it is absent the default
# factory is None and every escalation path degrades to its logged no-op or veto.
try:
    from escalation.models import Escalation  # type: ignore[import-untyped]
    from escalation.pins import is_queue_handoff
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


def pinning_escalation_ids_by_task(
    queue_factory: EscalationQueueFactory, project_root: ProjectRoot,
) -> dict[str, tuple[str, ...]]:
    """Map each task to the ids of its pending escalations that PIN it.

    Blocking I/O; raises whatever the queue raised, like its sibling. One
    ``EscalationQueue.get_pending_by_task`` scan serves every task, and a
    record pins exactly when ``escalation.pins.is_queue_handoff`` says so.
    """
    queue = queue_factory(escalation_queue_dir(project_root))
    pinning_ids: dict[str, list[str]] = defaultdict(list)
    for raw_task_id, records in queue.get_pending_by_task().items():
        # str(): an int-typed task_id must still match the str dependent id.
        task_id = str(raw_task_id or '')
        ids = [esc.id for esc in records if is_queue_handoff(esc)]
        if task_id and ids:
            pinning_ids[task_id].extend(ids)
    return {task_id: tuple(ids) for task_id, ids in pinning_ids.items()}
