"""Pin reservations: the head lock-blocked pin's claim on its modules (task 6040).

An operator pin decides WHICH task runs next, but before this module a pinned
task blocked by a module lock earned nothing while it waited: every module
that freed up could be taken by whoever scored next, so the pin could starve
indefinitely on a hot file.  The head lock-blocked pin(s) now hold a PIN
RESERVATION — an ordinary entry on the lock table's park stacks, at a rank
band that sits strictly above every priority tier and is ordered by
pin_order.  Everything else falls out of the existing park machinery:

- a pin reservation SHADOWS any fairness park, critical included, and the
  shadowed park is restored (task 1865's LIFO stacks) when the pin dispatches
  or is released;
- the pin owner's OWN fairness park on a key stays beneath its pin
  reservation rather than being replaced by it, so releasing the pin
  reservation hands the key back to exactly the fairness park it had;
- among pins, the lower pin_order outranks the higher one on a shared key;
- a HELD lock is never preempted: ``try_acquire``'s live held-lock gate runs
  before the park gate and knows nothing of pins.

This is the bounded EASY-backfill head exception to
plans/scheduler-dispatch-scoring-and-lock-layer-prd.md §7 row 1 ("below-rank-1
park installation"), NOT a reversal of it.  §7 measured GENERAL reservation
for every starving task as harmful: many automatic reservations gridlock on
each other, and each holds an idle footprint.  A pin reservation is different
on every count the PRD measured: at most ``pin_reservation_max_active`` exist,
all operator-chosen; one reservation cannot gridlock with itself; its idle cost
is one task's footprint; and C7 backfill still borrows through it exactly as
it borrows through a fairness park.

This module holds the vocabulary and policy only — rank, source, blocker
naming, release reasons, the ``pin_blocked`` cadence — and imports nothing
from the scheduler.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from orchestrator.config import PRIORITY_RANK, PRIORITY_TIERS, coerce_tier

#: Width of the pin rank band.  Pin ranks occupy ``[-_PIN_RANK_SPAN, -1]``,
#: strictly below the best tier rank (0).
_PIN_RANK_SPAN = 1_000_000

_BEST_TIER_RANK = PRIORITY_RANK[PRIORITY_TIERS[0]]


@dataclass(frozen=True)
class PinOrder:
    """An operator pin's position in the pin queue, as a park priority.

    ``rank`` maps it into the pin band: every pin rank is better (lower) than
    every tier rank, and a lower pin_order is a better rank.  A negative
    pin_order clamps to 0, and pin orders at or past ``_PIN_RANK_SPAN - 1``
    all tie at rank -1 — a tie only matters if two such pins reserve one key
    at once, where the earlier install then blocks the later one (INV-3).
    """

    value: int

    @property
    def rank(self) -> int:
        return -_PIN_RANK_SPAN + min(max(self.value, 0), _PIN_RANK_SPAN - 1)


#: What a park is installed AT: a priority tier name, or a pin's order.
ParkPriority = str | PinOrder


def park_rank(priority: ParkPriority) -> int:
    """The park-stack rank for *priority* — the single rank function.

    Lower is stronger (INV-3).  Tier names resolve through ``coerce_tier``, so
    an unknown tier falls back exactly as it always has.
    """
    if isinstance(priority, PinOrder):
        return priority.rank
    return PRIORITY_RANK[coerce_tier(priority)]


def is_pin_rank(rank: int) -> bool:
    """True iff *rank* lies in the pin band, i.e. beats every priority tier."""
    return rank < _BEST_TIER_RANK


def priority_payload(
    priority: ParkPriority, *, tier_key: str = 'priority'
) -> dict[str, str | int]:
    """How a ``reservation_*`` event names the priority a park was installed at.

    A tier park reports its tier under *tier_key*.  A pin reservation reports
    ``pin_order`` and no tier at all: its rank lies above every tier, so the
    owner's tier would misdescribe the park it holds.
    """
    if isinstance(priority, PinOrder):
        return {'pin_order': priority.value}
    return {tier_key: priority}


class ReservationSource(StrEnum):
    """Why a park exists: an operator pin, or the automatic fairness machinery.

    Carried as ``data.source`` on every ``reservation_*`` event.  It is DERIVED
    from the park's rank, never stored beside it, so it cannot disagree with
    the lock table.  ``reserve_now`` parks are tier-ranked and so read as
    FAIRNESS.
    """

    PIN = 'pin'
    FAIRNESS = 'fairness'

    @classmethod
    def source_of_rank(cls, rank: int) -> ReservationSource:
        return cls.PIN if is_pin_rank(rank) else cls.FAIRNESS


def rank_payload(rank: int) -> dict[str, str | int]:
    """How a park-stack entry names the priority behind its *rank*.

    The inverse of :func:`priority_payload` for a stored rank: always a
    ``source``, plus ``pin_order`` for a pin rank or ``tier`` for a tier rank.
    A rank past the last tier clamps to the last tier.
    """
    source = ReservationSource.source_of_rank(rank)
    if source is ReservationSource.PIN:
        return {'source': source.value, 'pin_order': rank + _PIN_RANK_SPAN}
    tier = PRIORITY_TIERS[min(rank, len(PRIORITY_TIERS) - 1)]
    return {'source': source.value, 'tier': tier}


class BlockerKind(StrEnum):
    """How a module refuses a requester: a live holder, or an active park."""

    HELD = 'held'
    PARKED = 'parked'


@dataclass(frozen=True, order=True)
class Blocker:
    """One reason a requester cannot take one module right now."""

    module: str
    owner: str
    kind: BlockerKind

    def as_payload(self) -> dict[str, str]:
        return {'module': self.module, 'owner': self.owner, 'kind': self.kind.value}


class PinReleaseReason(StrEnum):
    """Why the pin phase released a pin reservation: ``reservation_expired``'s ``reason``.

    Owner-state releases (terminal, missing, deps unsatisfied) belong to park
    GC and keep its vocabulary; these are the ones only the pin phase can see.
    """

    UNPINNED = 'unpinned'
    GATED = 'gated'
    INELIGIBLE = 'ineligible'
    DETERMINISTIC = 'deterministic'
    DISPLACED = 'pin_displaced'
    DISABLED = 'pin_reservations_disabled'


def pin_release_reason(
    *,
    enabled: bool,
    reservable: bool,
    pinned: bool,
    gated: bool,
    deterministic: bool,
) -> PinReleaseReason:
    """Why a pin reservation the pin phase is not keeping gets released.

    *reservable* means the owner is still a non-deterministic entry of this
    tick's pin queue, so a reservable owner being released was pushed out by
    the ``pin_reservation_max_active`` cap.  The first matching fact wins:
    disabled, displaced, unpinned, gated, deterministic, else ineligible.
    """
    if not enabled:
        return PinReleaseReason.DISABLED
    if reservable:
        return PinReleaseReason.DISPLACED
    if not pinned:
        return PinReleaseReason.UNPINNED
    if gated:
        return PinReleaseReason.GATED
    if deterministic:
        return PinReleaseReason.DETERMINISTIC
    return PinReleaseReason.INELIGIBLE


class PinBlockedLimiter:
    """The per-pin cadence of ``pin_blocked``: on the transition, then per interval.

    A pin's first blocked observation is due at once; later ones are due only
    once *interval* seconds have passed since the last emission.  The scheduler
    calls :meth:`forget` when the pin dispatches and :meth:`retain` with the
    pin queue each tick, so dispatching or leaving the queue ends the episode
    and the next block is a new transition.  Times are monotonic seconds.
    """

    def __init__(self) -> None:
        self._last_emit: dict[str, float] = {}

    def due(self, task_id: str, *, now: float, interval: float) -> bool:
        """True iff *task_id* should emit now; a True answer records the emission."""
        last = self._last_emit.get(task_id)
        if last is not None and now - last < interval:
            return False
        self._last_emit[task_id] = now
        return True

    def forget(self, task_id: str) -> None:
        """End *task_id*'s blocked episode."""
        self._last_emit.pop(task_id, None)

    def retain(self, task_ids: Iterable[str]) -> None:
        """End the episode of every tracked pin not in *task_ids*."""
        keep = set(task_ids)
        for task_id in self._last_emit.keys() - keep:
            del self._last_emit[task_id]
