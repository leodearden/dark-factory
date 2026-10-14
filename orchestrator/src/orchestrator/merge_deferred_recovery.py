"""Find merge-deferred members nothing will ever move again, and recover them (task 4737).

A member parks merge-deferred to wait for a train.  It stays owned while a
train the merge lane holds lists it, a live workflow holds it, a live claimant
or a pinning escalation holds it, or a peer of its declared train is still
active.  A member with none of these owners is abandoned.  The operator's view
of the sweep, including how to hold a member by hand, is OPERATIONS.md
§ "Abandoned merge-deferred members (task 4737)".
"""

from __future__ import annotations

import enum
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from shared.task_claimant import is_stranded_any_status

from orchestrator.scheduler import train_order_key
from orchestrator.task_status import TaskStatus


class ParkedOwner(enum.StrEnum):
    TRAIN_CUSTODY = 'train_custody'
    WORKFLOW = 'workflow'
    EXTERNAL_HOLD = 'external_hold'
    TRAIN_PEER_ACTIVE = 'train_peer_active'
    NONE = 'none'


def _task_id(task: Mapping[str, Any]) -> str | None:
    raw = task.get('id')
    return None if raw is None or raw == '' else str(raw)


@dataclass(frozen=True)
class DeclaredTrainRef:
    train_id: str
    order_key: tuple[int, int]

    @classmethod
    def from_task(cls, task: Mapping[str, Any]) -> DeclaredTrainRef | None:
        metadata = task.get('metadata')
        train = metadata.get('train') if isinstance(metadata, Mapping) else None
        if not isinstance(train, dict) or not train.get('id'):
            return None
        return cls(str(train['id']), train_order_key(task))


@dataclass(frozen=True)
class ParkedMember:
    task_id: str
    declared_train: DeclaredTrainRef | None
    claimant_live: bool

    @classmethod
    def from_task(
        cls, task: Mapping[str, Any], *, now: datetime, claimant_ttl: timedelta,
    ) -> ParkedMember | None:
        task_id = _task_id(task)
        if task_id is None:
            return None
        return cls(
            task_id=task_id,
            declared_train=DeclaredTrainRef.from_task(task),
            claimant_live=not is_stranded_any_status(task, now, claimant_ttl),
        )


@dataclass(frozen=True)
class TrainPeer:
    task_id: str
    status: str
    train: DeclaredTrainRef

    @classmethod
    def from_task(cls, task: Mapping[str, Any]) -> TrainPeer | None:
        task_id, train = _task_id(task), DeclaredTrainRef.from_task(task)
        if task_id is None or train is None:
            return None
        return cls(task_id=task_id, status=str(task.get('status')), train=train)


@dataclass(frozen=True)
class OwnershipView:
    """Every ownership signal, read at one instant."""

    custody: frozenset[str]
    workflow_held: frozenset[str]
    externally_held: frozenset[str]
    declared_trains: Mapping[str, tuple[TrainPeer, ...]]

    def holds(self, task_id: str) -> bool:
        return (
            task_id in self.custody
            or task_id in self.workflow_held
            or task_id in self.externally_held
        )


@dataclass(frozen=True)
class RecoveryTarget:
    redrive_id: str
    members: tuple[str, ...]
    label: str
    strip_unlanded_base: bool
    declared_train_id: str | None


def owner_of(member: ParkedMember, view: OwnershipView) -> ParkedOwner:
    if member.task_id in view.custody:
        return ParkedOwner.TRAIN_CUSTODY
    if member.task_id in view.workflow_held:
        return ParkedOwner.WORKFLOW
    if member.task_id in view.externally_held:
        return ParkedOwner.EXTERNAL_HOLD
    if member.declared_train is not None and _train_peer_active(
        member.task_id, member.declared_train, view,
    ):
        return ParkedOwner.TRAIN_PEER_ACTIVE
    return ParkedOwner.NONE


def _train_peer_active(task_id: str, train: DeclaredTrainRef, view: OwnershipView) -> bool:
    return any(
        peer.status != TaskStatus.MERGE_DEFERRED or view.holds(peer.task_id)
        for peer in view.declared_trains.get(train.train_id, ())
        if peer.task_id != task_id
    )


def recovery_targets(
    abandoned: Iterable[ParkedMember], view: OwnershipView,
) -> tuple[RecoveryTarget, ...]:
    """A member of no declared train is re-driven alone, un-stacked.  An abandoned
    declared train is re-driven through its tip alone, never un-stacked: only
    the tip's branch carries every member's commits.  A train is a target only
    when every peer the view lists is among *abandoned*."""
    solos: list[RecoveryTarget] = []
    trains: dict[str, list[tuple[tuple[int, int], str]]] = {}
    for member in abandoned:
        train = member.declared_train
        if train is None:
            solos.append(RecoveryTarget(
                redrive_id=member.task_id, members=(member.task_id,),
                label=f'abandoned-{member.task_id}', strip_unlanded_base=True,
                declared_train_id=None,
            ))
        else:
            trains.setdefault(train.train_id, []).append((train.order_key, member.task_id))
    train_targets = (
        _train_target(train_id, entries, view) for train_id, entries in trains.items()
    )
    return (*solos, *(target for target in train_targets if target is not None))


def _train_target(
    train_id: str, entries: Sequence[tuple[tuple[int, int], str]], view: OwnershipView,
) -> RecoveryTarget | None:
    members = tuple(task_id for _, task_id in sorted(entries, key=lambda entry: entry[0]))
    if any(peer.task_id not in members for peer in view.declared_trains.get(train_id, ())):
        return None
    return RecoveryTarget(
        redrive_id=members[-1], members=members, label=f'abandoned-{train_id}',
        strip_unlanded_base=False, declared_train_id=train_id,
    )
