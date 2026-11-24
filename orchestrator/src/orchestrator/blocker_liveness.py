"""Whether a lock blocker's holder is live, following parks to their owner's own blockers.

plans/scheduler-dispatch-scoring-and-lock-layer-prd.md C4 "Cap L1" *Liveness*
states the rule and records why the walk stops where it does.
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
