"""The rule for which buffered task events fold together and how their payloads merge.

Recon lever 3 (task 4278): a task that changes status or is edited many times
before the next drain needs only one buffered row, because reconciliation reads
the task's live state when it assembles.  ``coalesce_key`` names the events that
may share a row; ``event_buffer.EventBuffer.push`` performs the fold.

Dependency operations never fold.  Each ``add_dependency`` / ``remove_dependency``
event's ``depends_on`` is a distinct fact: on 2026-10-19, 113 live such events
covered 111 distinct edges.

The key is a pure function of the event, so it is stored even when coalescing is
switched off; turning it back on then folds into rows written meanwhile.
Rationale and sizing: ``plans/recon-throughput-capacity.md`` (Lever 3).
"""

from __future__ import annotations

from fused_memory.models.reconciliation import EventType, ReconciliationEvent

__all__ = [
    'coalesce_key',
    'task_id_key',
]

_COALESCIBLE_TASK_MODIFIED_OPERATIONS = frozenset({'update_task'})


def task_id_key(value: object) -> str | None:
    """A payload ``task_id`` as its canonical string, or None when it names no task.

    Writers emit int or str ids, so ``7`` and ``'7'`` are one task.  Anything
    else is rejected rather than stringified: ``str()`` of a bool or a float
    would mint a key (``'True'``, ``'7.0'``) that never matches the real id.
    """
    if isinstance(value, bool) or not isinstance(value, int | str):
        return None
    key = str(value)
    return key or None


def coalesce_key(event: ReconciliationEvent) -> str | None:
    """The key ``event`` folds under in the buffer, or None when it never folds."""
    task_id = task_id_key(event.payload.get('task_id'))
    if task_id is None:
        return None
    if event.type is EventType.task_status_changed:
        return f'{event.type.value}:{task_id}'
    operation = event.payload.get('operation')
    if (
        event.type is EventType.task_modified
        and operation in _COALESCIBLE_TASK_MODIFIED_OPERATIONS
    ):
        return f'{event.type.value}:{task_id}:{operation}'
    return None
