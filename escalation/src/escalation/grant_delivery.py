"""Whether a ``resolve_issue`` scope grant (``granted_files``) reaches its task.

This module is the only home for the granted-files delivery vocabulary
(INV-9): ``escalation/server.py::resolve_issue`` reports a
:class:`GrantDelivery` on every success return, and
``orchestrator/src/orchestrator/harness.py::Harness`` probes for and records
the same values when its asynchronous delivery falls short. Both take their
wording from :func:`shortfall_reason`.

It imports nothing from ``orchestrator``: ``orchestrator`` already depends on
``escalation``, and the reverse direction would be a cycle. Task 5650.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import StrEnum
from typing import assert_never

from escalation.action_effects import WORKFLOW_RESUME

__all__ = [
    'GRANT_NOT_DELIVERED_MARKER',
    'GrantDelivery',
    'grant_delivery_for',
    'shortfall_reason',
]

GRANT_NOT_DELIVERED_MARKER = (
    '[!! SCOPE GRANT NOT DELIVERED: granted_files DID NOT REACH THE TASK SCOPE !!]'
)


class GrantDelivery(StrEnum):
    """What became of a resolution's ``granted_files``.

    ``SCHEDULED`` means ROUTED to the harness's delivery path, not landed:
    delivery is asynchronous, and a shortfall found there (``NO_PLAN_TO_WIDEN``,
    ``NEEDS_UNHELD_LOCK``, ``WRITE_FAILED``) is appended to the record's
    resolution under :data:`GRANT_NOT_DELIVERED_MARKER`. Every value other than
    ``NONE_GRANTED`` and ``SCHEDULED`` is a shortfall.
    """

    NONE_GRANTED = 'none_granted'
    SCHEDULED = 'scheduled'
    ACTION_DOES_NOT_APPLY_GRANTS = 'action_does_not_apply_grants'
    RECORD_NOT_OPEN = 'record_not_open'
    NO_ORCHESTRATOR = 'no_orchestrator'
    NO_PLAN_TO_WIDEN = 'no_plan_to_widen'
    NEEDS_UNHELD_LOCK = 'needs_unheld_lock'
    WRITE_FAILED = 'write_failed'


def grant_delivery_for(
    *,
    granted_files: Sequence[str] | None,
    workflow_disposition: str,
    applied: bool,
    probe_blocker: Callable[[], GrantDelivery | None],
) -> GrantDelivery:
    """Decide what a resolution does with its grant, consulting the harness last.

    *workflow_disposition* is the Table B
    (``escalation.action_effects``) disposition the caller already resolved
    from ``(action, level, category)``. *applied* is False when the record was
    already closed, in which case ``EscalationQueue.resolve`` neither stamps
    the grant nor fires the resolve callback. *probe_blocker* is called only
    when the grant would otherwise be ``SCHEDULED``.
    """
    if not granted_files:
        return GrantDelivery.NONE_GRANTED
    if workflow_disposition != WORKFLOW_RESUME:
        return GrantDelivery.ACTION_DOES_NOT_APPLY_GRANTS
    if not applied:
        return GrantDelivery.RECORD_NOT_OPEN
    blocker = probe_blocker()
    return GrantDelivery.SCHEDULED if blocker is None else blocker


def shortfall_reason(
    delivery: GrantDelivery, *, task_id: str, granted_files: Sequence[str],
) -> str | None:
    """Explain a shortfall to the resolver, or return None when there is none."""
    paths = ', '.join(granted_files)
    resume_recourse = (
        f"resolve an open escalation for task {task_id} with action='resume' "
        f'and these granted_files to widen its scope'
    )
    match delivery:
        case GrantDelivery.NONE_GRANTED | GrantDelivery.SCHEDULED:
            return None
        case GrantDelivery.ACTION_DOES_NOT_APPLY_GRANTS:
            return (
                f'granted_files {paths} did not widen task {task_id}: only '
                f"action='resume' delivers a grant (park discards it outright), "
                f'so {resume_recourse}.'
            )
        case GrantDelivery.RECORD_NOT_OPEN:
            return (
                f'granted_files {paths} did not widen task {task_id}: the record '
                f'was already closed, so the grant was neither recorded nor '
                f'routed; {resume_recourse}.'
            )
        case GrantDelivery.NO_ORCHESTRATOR:
            return (
                f'granted_files {paths} were recorded but not routed for task '
                f'{task_id}: no orchestrator is attached to this escalation '
                f"server, so nothing writes them into the task's plan.json or "
                f'metadata.files; widen both by hand if the scope must change.'
            )
        case GrantDelivery.NO_PLAN_TO_WIDEN:
            return (
                f'granted_files {paths} were recorded but did not widen task '
                f'{task_id}: it has no plan.json to widen (it never reached the '
                f'architect, or its worktree was reclaimed); add them to its '
                f'metadata.files by hand if the scope must change before it is '
                f'next planned.'
            )
        case GrantDelivery.NEEDS_UNHELD_LOCK:
            return (
                f'granted_files {paths} were recorded but not widened into task '
                f'{task_id}: its workflow is running without the module lock '
                f'they need, and widening its live plan would let it edit '
                f'unlocked code. They land when that workflow next resumes from '
                f'its own level-0 escalation, which acquires the lock; '
                f'otherwise, once the task is no longer running, '
                f'{resume_recourse}.'
            )
        case GrantDelivery.WRITE_FAILED:
            return (
                f'granted_files {paths} were recorded but may not have widened '
                f"task {task_id}: writing them into its plan.json or "
                f'metadata.files failed (see the orchestrator log); check both '
                f'and widen by hand, or expect the agent to re-escalate.'
            )
        case _:
            assert_never(delivery)
