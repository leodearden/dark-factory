"""Whether a lock blocker's holder is live, following parks to their owner's own blockers.

A HELD module's owner is live when it is in the tick's active task read, not
terminal, not blocked, and has a live claimant: the scheduler's own
claimant/heartbeat liveness at the dispatch gate's TTL.  A workflow heartbeats
for its whole slot, merge phase included, so a holder that is merely slow
reads live.  A PARKED module's owner is a pending task waiting for its own
modules; it is followed to those modules' blockers, at most
``PARK_FOLLOW_DEPTH`` parks deep.  A park owner whose modules are all free
waits only for a run slot, which is not a wedge.

plans/scheduler-dispatch-scoring-and-lock-layer-prd.md C4 Bounds records why
the walk stops where it does.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

from shared.task_claimant import has_live_claimant
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


@dataclass(frozen=True)
class LockWorld:
    """One tick's view: the active task records and each task's current blockers."""

    tasks: Mapping[str, Mapping[str, Any]]
    blockers_of: Callable[[str], Sequence[Blocker]]
    now: datetime
    claimant_ttl: timedelta


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
        reason = _held_owner_state(record, world)
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


def _held_owner_state(record: Mapping[str, Any] | None, world: LockWorld) -> HolderState | None:
    if record is None or not _active(record):
        return HolderState.NOT_ACTIVE
    if record.get('status') == TaskStatus.BLOCKED:
        return HolderState.BLOCKED
    if not has_live_claimant(record, world.now, world.claimant_ttl):
        return HolderState.NO_LIVE_CLAIMANT
    return None


def _active(record: Mapping[str, Any] | None) -> bool:
    return record is not None and record.get('status') not in TERMINAL
