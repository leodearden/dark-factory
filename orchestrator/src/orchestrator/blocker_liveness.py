"""Whether a lock blocker's holder is live, following parks to their owner's own blockers.

A HELD module's owner must be a slot this scheduler process has dispatched and
not yet released; a lock outside every slot is a leak.  An in-flight owner
awaiting its claim or its release is live: terminal, pending with no claimant
(dispatched, not yet claimed), or gone from the task read.  Any other in-flight
owner is live while someone alive holds it, by
``shared.task_claimant.is_stranded_any_status``: the dispatch gate's
claimant/heartbeat liveness at its TTL, plus the legacy ``metadata.infra_hold``
exemption.  A workflow heartbeats for its whole slot, merge phase included, so
an owner no one holds is non-live: ``blocked`` when blocked, else
``no_live_claimant``.  A pending owner still carrying a stale claimant is a
slot whose requeue exit hung past the TTL.  A PARKED module's owner is a
pending task waiting for its own modules; it is followed to those modules'
blockers, at most ``PARK_FOLLOW_DEPTH`` parks deep.  A park owner whose modules
are all free waits only for a run slot, which is not a wedge.

plans/scheduler-dispatch-scoring-and-lock-layer-prd.md C4 "Cap L1" records why
the walk stops where it does.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

from shared.task_claimant import carries_claimant, is_stranded_any_status, is_stranded_blocked
from shared.task_statuses import TERMINAL, TaskStatus

from orchestrator.pin_reservation import Blocker, BlockerKind

PARK_FOLLOW_DEPTH = 3


class HolderState(StrEnum):
    """Why a blocker's holder is not live."""

    NOT_ACTIVE = 'not_active'
    BLOCKED = 'blocked'
    NO_LIVE_CLAIMANT = 'no_live_claimant'
    PARK_CYCLE = 'park_cycle'
    PARK_TOO_DEEP = 'park_too_deep'
    LOCK_WITHOUT_SLOT = 'lock_without_slot'


@dataclass(frozen=True)
class LockWorld:
    """One tick's view: active task records, each task's blockers, and the slots in flight."""

    tasks: Mapping[str, Mapping[str, Any]]
    blockers_of: Callable[[str], Sequence[Blocker]]
    now: datetime
    claimant_ttl: timedelta
    in_flight: frozenset[str]


@dataclass(frozen=True)
class NonLiveHolder:
    """The blocker whose owner is not live, and why."""

    blocker: Blocker
    reason: HolderState


def non_live_holder(blocker: Blocker, world: LockWorld) -> NonLiveHolder | None:
    """The first non-live holder behind *blocker*, or None when its holder is live."""
    return _walk(blocker, world, frozenset(), PARK_FOLLOW_DEPTH)


def _walk(
    blocker: Blocker, world: LockWorld, path: frozenset[str], depth: int,
) -> NonLiveHolder | None:
    owner = blocker.owner
    record = world.tasks.get(owner)
    if blocker.kind is BlockerKind.HELD:
        reason = _held_owner_state(owner, record, world)
        return None if reason is None else NonLiveHolder(blocker, reason)
    if owner in path:
        return NonLiveHolder(blocker, HolderState.PARK_CYCLE)
    if depth == 0:
        return NonLiveHolder(blocker, HolderState.PARK_TOO_DEEP)
    if not _active(record):
        return NonLiveHolder(blocker, HolderState.NOT_ACTIVE)
    for behind in world.blockers_of(owner):
        found = _walk(behind, world, path | {owner}, depth - 1)
        if found is not None:
            return found
    return None


def _held_owner_state(
    owner: str, record: Mapping[str, Any] | None, world: LockWorld,
) -> HolderState | None:
    if owner not in world.in_flight:
        return HolderState.LOCK_WITHOUT_SLOT
    if record is None or _awaiting_claim_or_release(record):
        return None
    if is_stranded_blocked(record, world.now, world.claimant_ttl):
        return HolderState.BLOCKED
    if is_stranded_any_status(record, world.now, world.claimant_ttl):
        return HolderState.NO_LIVE_CLAIMANT
    return None


def _awaiting_claim_or_release(record: Mapping[str, Any]) -> bool:
    """Terminal, or pending with no claimant: dispatched and not yet claimed, or unwinding to its slot release."""
    status = record.get('status')
    return status in TERMINAL or (status == TaskStatus.PENDING and not carries_claimant(record))


def _active(record: Mapping[str, Any] | None) -> bool:
    return record is not None and record.get('status') not in TERMINAL
