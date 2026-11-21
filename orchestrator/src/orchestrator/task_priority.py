"""A task's live scheduling priority, as a narrow read-only lookup.

The scheduler owns the facts (``orchestrator/src/orchestrator/scheduler.py::Scheduler.task_priority``);
consumers that must not import the scheduler — the merge lane today, verify
admission (task 8261) and the merge-pool pick next — are handed a
:class:`TaskPriorityLookup` and read it through :func:`read_task_priority`,
which never raises.
"""
from __future__ import annotations

import dataclasses
from datetime import datetime
from typing import Protocol


@dataclasses.dataclass(frozen=True)
class TaskPriority:
    """A task's priority as the later gates see it; ``None`` always means unknown.

    ``tier`` is the effective tier, ``PINNED`` exactly when ``pin_order`` is
    set, and ``base_tier`` the tier with pins ignored. ``pinned`` says the
    task's own override row is pinned; ``age_anchor`` is when the task first
    became eligible.
    """

    tier: str | None = None
    pinned: bool | None = None
    base_tier: str | None = dataclasses.field(default=None, kw_only=True)
    pin_order: int | None = dataclasses.field(default=None, kw_only=True)
    age_anchor: datetime | None = dataclasses.field(default=None, kw_only=True)


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
    except Exception:
        return UNKNOWN_PRIORITY
    if not isinstance(found, TaskPriority):
        return UNKNOWN_PRIORITY
    return TaskPriority(
        tier=found.tier if isinstance(found.tier, str) else None,
        pinned=found.pinned if isinstance(found.pinned, bool) else None,
        base_tier=found.base_tier if isinstance(found.base_tier, str) else None,
        pin_order=found.pin_order if _is_plain_int(found.pin_order) else None,
        age_anchor=found.age_anchor if _is_aware_datetime(found.age_anchor) else None,
    )


def _is_plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_aware_datetime(value: object) -> bool:
    return isinstance(value, datetime) and value.utcoffset() is not None
