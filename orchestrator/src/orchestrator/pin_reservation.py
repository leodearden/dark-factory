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

This module holds the vocabulary and pure policy only — rank, source, blocker
naming, release reasons — and imports nothing from the scheduler.
"""

from __future__ import annotations

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
