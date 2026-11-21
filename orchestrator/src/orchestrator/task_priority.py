"""A task's live scheduling priority, the lookup that answers it, and the one gate order.

The scheduler owns the facts (``orchestrator/src/orchestrator/scheduler.py::Scheduler.task_priority``);
consumers that must not import the scheduler — the merge lane today, verify
admission (task 8261) and the merge-pool pick next — are handed a
:class:`TaskPriorityLookup` and read it through :func:`read_task_priority`,
which never raises. :func:`order_key` is the only place a later gate's sort
key is built. A leaf: it imports nothing from the orchestrator, so the config
and every gate can import it.
"""
from __future__ import annotations

import dataclasses
import math
from datetime import UTC, datetime
from typing import Protocol

PRIORITY_TIERS: tuple[str, ...] = ('critical', 'high', 'medium', 'low', 'polish')
PINNED = 'pinned'
GATE_RANK: tuple[str, ...] = (PINNED, *PRIORITY_TIERS)
_RANK_OF: dict[str, int] = {tier: rank for rank, tier in enumerate(GATE_RANK)}
_HIGH_RANK = _RANK_OF['high']
_UNKNOWN_TIER_RANK = _RANK_OF['medium']


@dataclasses.dataclass(frozen=True)
class TaskPriority:
    """A task's priority as the later gates see it; ``None`` always means unknown.

    ``tier`` is the effective tier and ``base_tier`` the tier with pins
    ignored; :func:`read_task_priority` answers ``tier`` ``PINNED`` exactly
    when ``pin_order`` is set. ``pinned`` says the task's own override row is
    pinned; ``age_anchor`` is when the task first became eligible.
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
    A ``PINNED`` tier with no well-typed order reads as order 0, and an order
    without the ``PINNED`` tier is dropped.
    """
    if lookup is None:
        return UNKNOWN_PRIORITY
    try:
        found = lookup(str(task_id))
    except Exception:
        return UNKNOWN_PRIORITY
    if not isinstance(found, TaskPriority):
        return UNKNOWN_PRIORITY
    tier = found.tier if isinstance(found.tier, str) else None
    return TaskPriority(
        tier=tier,
        pinned=found.pinned if isinstance(found.pinned, bool) else None,
        base_tier=found.base_tier if isinstance(found.base_tier, str) else None,
        pin_order=_pin_order_of(tier, found.pin_order),
        age_anchor=found.age_anchor if _is_aware_datetime(found.age_anchor) else None,
    )


def _pin_order_of(tier: str | None, pin_order: int | None) -> int | None:
    if tier != PINNED:
        return None
    return pin_order if _is_plain_int(pin_order) else 0


def _is_plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_aware_datetime(value: object) -> bool:
    return isinstance(value, datetime) and value.utcoffset() is not None


def gate_rank(tier: object) -> int | None:
    """*tier*'s place in :data:`GATE_RANK`, or ``None`` when it is not a known tier."""
    return _RANK_OF.get(tier) if isinstance(tier, str) else None


@dataclasses.dataclass(frozen=True)
class TaskAge:
    secs: float
    from_anchor: bool


def task_age(p: TaskPriority, *, now: datetime, fallback_anchor: datetime | None) -> TaskAge:
    """How long *p* has waited: from its ``age_anchor``, else *fallback_anchor*, else 0."""
    from_anchor = p.age_anchor is not None
    start = p.age_anchor if from_anchor else fallback_anchor
    if start is None:
        return TaskAge(0.0, from_anchor=False)
    try:
        secs = (now - start).total_seconds()
    except (TypeError, OverflowError):
        secs = 0.0
    return TaskAge(secs, from_anchor=from_anchor)


def anchor_from_epoch(epoch: float) -> datetime | None:
    """The aware UTC instant *epoch* seconds names, or ``None`` when it names none
    (NaN, an infinity, an out-of-range or millisecond epoch); never raises."""
    try:
        return datetime.fromtimestamp(epoch, UTC)
    except (OverflowError, ValueError, OSError):
        return None


def order_key(
    p: TaskPriority, *, now: datetime, waited_secs: float,
    promote_after_secs: float | None, fallback_anchor: datetime | None,
) -> tuple[int, int, float]:
    """*p*'s sort key at a later gate; smaller sorts first. Never raises."""
    rank = gate_rank(p.tier)
    promoted = _promoted_rank(
        _UNKNOWN_TIER_RANK if rank is None else rank, waited_secs, promote_after_secs,
    )
    age = task_age(p, now=now, fallback_anchor=fallback_anchor)
    return (promoted, p.pin_order or 0, -age.secs)


def _promoted_rank(rank: int, waited_secs: float, promote_after_secs: float | None) -> int:
    if rank <= _HIGH_RANK or promote_after_secs is None:
        return rank
    if not (promote_after_secs > 0 and math.isfinite(promote_after_secs)):
        return rank
    if not (math.isfinite(waited_secs) and waited_secs >= 0):
        return rank
    periods = min(waited_secs / promote_after_secs, rank - _HIGH_RANK)
    return rank - math.floor(periods)
