"""Runs a resolved held pure gate's DeterministicRunner at resolution time, not in a slot.

The run closes the gate or re-holds it. Rationale: docs/deterministic-runner.md
§3 "Resuming a pure gate" (task 3638).
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Any

from shared.task_fields import dependency_ids
from shared.task_metadata import GATE_ESCALATED_AT_KEY, Milestone

from orchestrator.dependency_verdict import deps_verdict
from orchestrator.scheduler import TaskAssignment

if TYPE_CHECKING:
    from orchestrator.deterministic_runner import DeterministicRunner
    from orchestrator.scheduler import Scheduler
    from orchestrator.workflow import WorkflowOutcome

__all__ = ['is_held_pure_gate', 'resume_resolved_pure_gate']


def is_held_pure_gate(task: Mapping[str, Any]) -> bool:
    """Whether *task* is a blocked pure gate whose gate escalation has been filed."""
    metadata = task.get('metadata')
    return (
        task.get('status') == 'blocked'
        and isinstance(metadata, dict)
        and metadata.get('task_kind') == 'deterministic'
        and metadata.get('before_done') is None
        and bool(metadata.get('always_escalates'))
        and bool(metadata.get(GATE_ESCALATED_AT_KEY))
    )


async def resume_resolved_pure_gate(
    scheduler: Scheduler,
    runner: DeterministicRunner,
    task_id: str,
    row: dict,
    *,
    now: datetime,
) -> WorkflowOutcome | None:
    """Run *row*'s gate now; ``None`` when it is no held pure gate or dispatch would make it wait.

    The holds mirror ``orchestrator/scheduler.py::Scheduler._eligible_for_dispatch``'s
    ``_deferred_watch_gated``, ``_milestone_time_gated`` and ``_deps_satisfied``.
    """
    if not is_held_pure_gate(row):
        return None
    metadata = row['metadata']
    if _deferred_watch_holds(metadata) or not _milestone_fired(metadata, now):
        return None
    if not await _dependencies_satisfied(scheduler, row):
        return None
    return await runner.run(TaskAssignment(task_id=task_id, task=row, modules=[]))


def _deferred_watch_holds(metadata: dict) -> bool:
    return bool(metadata.get('deferred_watch')) and not metadata.get('trigger_met')


def _milestone_fired(metadata: dict, now: datetime) -> bool:
    raw = metadata.get('milestone')
    if not raw:
        return True
    if not isinstance(raw, dict):
        return False
    try:
        return Milestone.model_validate(raw).fired(metadata.get('milestone_deps_satisfied_at'), now)
    except ValueError:
        return False


async def _dependencies_satisfied(scheduler: Scheduler, row: dict) -> bool:
    """``deps_verdict`` with empty caches, so an external or delivered-check dependency waits."""
    dep_ids = dependency_ids(row)
    deps = await asyncio.gather(*(scheduler.get_task(dep_id) for dep_id in dep_ids))
    tasks_by_id = {
        dep_id: dep for dep_id, dep in zip(dep_ids, deps, strict=True) if dep is not None
    }
    status_map = {dep_id: str(dep.get('status')) for dep_id, dep in tasks_by_id.items()}
    return deps_verdict(
        row, status_map, tasks_by_id, external_status_cache={}, delivered_check_cache={},
    ).satisfied
