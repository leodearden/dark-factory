"""A task's live scheduling priority, as a narrow read-only lookup.

The scheduler owns the facts (``orchestrator/src/orchestrator/scheduler.py::Scheduler.task_priority``);
consumers that must not import the scheduler — the merge lane today, verify
admission (task 8261) and the merge-pool pick next — are handed a
:class:`TaskPriorityLookup` and read it through :func:`read_task_priority`,
which never raises.
"""
from __future__ import annotations

import dataclasses
from typing import Protocol


@dataclasses.dataclass(frozen=True)
class TaskPriority:
    """The scheduler's effective tier for a task and whether it is pinned.

    ``None`` in either field means unknown, never a default.
    """

    tier: str | None = None
    pinned: bool | None = None


UNKNOWN_PRIORITY = TaskPriority()


class TaskPriorityLookup(Protocol):
    def __call__(self, task_id: str, /) -> TaskPriority: ...


def read_task_priority(lookup: TaskPriorityLookup | None, task_id: object) -> TaskPriority:
    """*task_id*'s priority through *lookup*, or :data:`UNKNOWN_PRIORITY`.

    The single place a task id is normalised (to ``str``) before a lookup, and
    the single guard that keeps a lookup from raising into its caller: a
    missing lookup, a raising one, or one answering anything but a
    ``TaskPriority`` all read as unknown, and so does a field of the wrong type.
    """
    if lookup is None:
        return UNKNOWN_PRIORITY
    try:
        found = lookup(str(task_id))
    except Exception:  # noqa: BLE001
        return UNKNOWN_PRIORITY
    if not isinstance(found, TaskPriority):
        return UNKNOWN_PRIORITY
    return TaskPriority(
        tier=found.tier if isinstance(found.tier, str) else None,
        pinned=found.pinned if isinstance(found.pinned, bool) else None,
    )
