"""Find merge-deferred members nothing will ever move again, and recover them (task 4737).

A member parks merge-deferred to wait for a train.  It stays owned while a
train the merge lane holds lists it, a live single merge request was submitted
under its id (task 7022), a live workflow holds it, a live claimant or a
pinning escalation holds it, or a peer of its declared train is still active.
A member with none of these owners is abandoned.  The operator's view
of the sweep, including how to hold a member by hand, is OPERATIONS.md
§ "Abandoned merge-deferred members (task 4737)".
"""

from __future__ import annotations

import enum
import logging
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal, Protocol

from escalation.models import Escalation
from shared.task_claimant import is_stranded_any_status
from shared.task_statuses import ACTIVE

from orchestrator.merge_lane import MemberRedriveOutcome
from orchestrator.recovery_pins import records_pin_recovery
from orchestrator.scheduler import train_order_key
from orchestrator.task_status import TaskStatus

if TYPE_CHECKING:
    from escalation.queue import EscalationQueue

logger = logging.getLogger(__name__)

MERGE_DEFERRED_ABANDONED_CATEGORY = 'merge_deferred_abandoned'

_AGENT_ROLE = 'harness-merge-deferred-recovery'
_RUNBOOK = 'OPERATIONS.md § "Abandoned merge-deferred members (task 4737)"'
_ABSENT_OWNERS = (
    'no train in merge-lane custody, no live single merge request, no live workflow, '
    'no live claimant, no pinning escalation and no active declared-train peer'
)
_LANDED: frozenset[MemberRedriveOutcome] = frozenset({
    MemberRedriveOutcome.ON_MAIN_ATTRIBUTED,
    MemberRedriveOutcome.ON_MAIN_UNATTRIBUTED,
})
_RECOVERED: frozenset[MemberRedriveOutcome] = _LANDED | {MemberRedriveOutcome.NOT_ON_MAIN}
_NOT_ACTED_ON: frozenset[MemberRedriveOutcome] = frozenset({
    MemberRedriveOutcome.NOT_PARKED,
    MemberRedriveOutcome.HELD_BY_LIVE_TRAIN,
})


class ParkedOwner(enum.StrEnum):
    TRAIN_CUSTODY = 'train_custody'
    SINGLE_REQUEST = 'single_request'
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
    single_requests: frozenset[str]
    workflow_held: frozenset[str]
    externally_held: frozenset[str]
    declared_trains: Mapping[str, tuple[TrainPeer, ...]]

    def holds(self, task_id: str) -> bool:
        return (
            task_id in self.custody
            or task_id in self.single_requests
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

    @property
    def landed_with_tip(self) -> tuple[str, ...]:
        """The members whose commits a landed tip carried onto main."""
        return tuple(mid for mid in self.members if mid != self.redrive_id)


def owner_of(member: ParkedMember, view: OwnershipView) -> ParkedOwner:
    if member.task_id in view.custody:
        return ParkedOwner.TRAIN_CUSTODY
    if member.task_id in view.single_requests:
        return ParkedOwner.SINGLE_REQUEST
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
    declared train is re-driven through its tip, never un-stacked: only the
    tip's branch carries every member's commits.  A train is a target only
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


class ParkedMemberLane(Protocol):
    """What the sweep needs of the merge lane; SpeculativeMergeWorker is one."""

    def train_member_ids_in_custody(self) -> frozenset[str]: ...

    async def redrive_abandoned_members(
        self, label: str, member_ids: Sequence[str], *, strip_unlanded_base: bool,
    ) -> Mapping[str, MemberRedriveOutcome]: ...


class LiveMergeRequests(Protocol):
    """The task ids live merge requests were submitted under; InFlightMergeRegistry is one."""

    def task_ids_in_flight(self) -> frozenset[str]: ...


class ParkedTaskStore(Protocol):
    """What the sweep needs of the task store; the Scheduler is one."""

    async def get_tasks(
        self, *, statuses: Iterable[str], distinguish_failure: Literal[True],
    ) -> list[dict] | None: ...

    def is_actively_held(self, tid: str) -> bool: ...


@dataclass(frozen=True)
class SweepReport:
    owners: Mapping[ParkedOwner, int]
    suspects: frozenset[str]
    outcomes: Mapping[str, MemberRedriveOutcome]


_NOTHING_SWEPT = SweepReport(owners={}, suspects=frozenset(), outcomes={})


@dataclass(frozen=True)
class _Snapshot:
    members: tuple[ParkedMember, ...]
    peers: tuple[TrainPeer, ...]


class AbandonedMemberSweep:
    """Re-drives each merge-deferred member found abandoned on two consecutive
    passes, and files one escalation per target it acted on."""

    def __init__(self) -> None:
        self._suspects: frozenset[str] = frozenset()

    async def run_pass(
        self,
        *,
        lane: ParkedMemberLane | None,
        requests: LiveMergeRequests,
        tasks: ParkedTaskStore,
        escalations: EscalationQueue | None,
        claimant_ttl: timedelta,
    ) -> SweepReport:
        previous, self._suspects = self._suspects, frozenset()
        if lane is None:
            return _NOTHING_SWEPT
        snapshot = await _snapshot(tasks, claimant_ttl)
        if snapshot is None:
            return _NOTHING_SWEPT
        view = _ownership_view(snapshot, lane, requests, tasks, escalations)
        owners = {member.task_id: owner_of(member, view) for member in snapshot.members}
        abandoned = [m for m in snapshot.members if owners[m.task_id] is ParkedOwner.NONE]
        self._suspects = frozenset(member.task_id for member in abandoned)
        outcomes: dict[str, MemberRedriveOutcome] = {}
        for target in recovery_targets(abandoned, view):
            if target.redrive_id in previous and not _submitted_since_view(requests, target):
                target_outcomes = await _recover(lane, target)
                outcomes.update(target_outcomes)
                _file_escalation(escalations, target, target_outcomes)
        counts = Counter(owners.values())
        report = SweepReport(
            owners={owner: counts[owner] for owner in ParkedOwner},
            suspects=self._suspects,
            outcomes=outcomes,
        )
        logger.info(
            'Abandoned merge-deferred sweep: owners %s; suspects %s; re-driven %s',
            {str(owner): n for owner, n in report.owners.items() if n},
            sorted(report.suspects), {tid: str(o) for tid, o in outcomes.items()},
        )
        return report


def _submitted_since_view(requests: LiveMergeRequests, target: RecoveryTarget) -> bool:
    """A re-drive of an earlier target yields the loop, so a merge request for
    this one may have been submitted since the view was read."""
    submitted = requests.task_ids_in_flight().intersection(target.members)
    if submitted:
        logger.info(
            'Abandoned merge-deferred sweep: %s submitted a merge request since this '
            'pass read its owners — not re-driving %s', ', '.join(sorted(submitted)),
            target.label,
        )
    return bool(submitted)


async def _snapshot(tasks: ParkedTaskStore, claimant_ttl: timedelta) -> _Snapshot | None:
    """Every non-terminal row, infra-hold included: an infra-held peer keeps
    its declared train active."""
    rows = await tasks.get_tasks(statuses=ACTIVE, distinguish_failure=True)
    if rows is None:
        logger.warning(
            'Abandoned merge-deferred sweep: the task listing failed — skipping this '
            'pass; every member needs two fresh consecutive sightings again',
        )
        return None
    now = datetime.now(UTC)
    parked = (row for row in rows if row.get('status') == TaskStatus.MERGE_DEFERRED)
    members = (ParkedMember.from_task(row, now=now, claimant_ttl=claimant_ttl) for row in parked)
    peers = (TrainPeer.from_task(row) for row in rows)
    return _Snapshot(
        members=tuple(m for m in members if m is not None),
        peers=tuple(p for p in peers if p is not None),
    )


def _ownership_view(
    snapshot: _Snapshot,
    lane: ParkedMemberLane,
    requests: LiveMergeRequests,
    tasks: ParkedTaskStore,
    escalations: EscalationQueue | None,
) -> OwnershipView:
    trains: dict[str, list[TrainPeer]] = {}
    for peer in snapshot.peers:
        trains.setdefault(peer.train.train_id, []).append(peer)
    task_ids = {m.task_id for m in snapshot.members} | {p.task_id for p in snapshot.peers}
    return OwnershipView(
        custody=lane.train_member_ids_in_custody(),
        single_requests=requests.task_ids_in_flight(),
        workflow_held=frozenset(tid for tid in task_ids if tasks.is_actively_held(tid)),
        externally_held=frozenset(
            m.task_id for m in snapshot.members if _externally_held(m, escalations)
        ),
        declared_trains={train_id: tuple(peers) for train_id, peers in trains.items()},
    )


def _externally_held(member: ParkedMember, escalations: EscalationQueue | None) -> bool:
    if member.claimant_live:
        return True
    records = _pending_records(escalations, member.task_id)
    return records_pin_recovery(member.task_id, records, live_claimant=False)


def _pending_records(
    escalations: EscalationQueue | None, task_id: str,
) -> Sequence[Escalation] | None:
    """The task's open escalations; ``None`` (which pins) when they cannot be read."""
    if escalations is None:
        return ()
    try:
        return escalations.get_by_task(task_id, status='pending')
    except Exception:
        logger.warning(
            'Abandoned merge-deferred sweep: escalations of %s unreadable — '
            'treating it as held', task_id, exc_info=True,
        )
        return None


async def _recover(
    lane: ParkedMemberLane, target: RecoveryTarget,
) -> dict[str, MemberRedriveOutcome]:
    """Re-drive *target*'s tip.  A tip found on main carried the rest of its
    declared train onto main, so they are re-driven in the same pass."""
    outcomes = await _redrive(
        lane, target.label, (target.redrive_id,),
        strip_unlanded_base=target.strip_unlanded_base,
    )
    if outcomes[target.redrive_id] in _LANDED and target.landed_with_tip:
        outcomes |= await _redrive(
            lane, target.label, target.landed_with_tip, strip_unlanded_base=False,
        )
    return outcomes


async def _redrive(
    lane: ParkedMemberLane, label: str, member_ids: Sequence[str], *, strip_unlanded_base: bool,
) -> dict[str, MemberRedriveOutcome]:
    try:
        outcomes = await lane.redrive_abandoned_members(
            label, list(member_ids), strip_unlanded_base=strip_unlanded_base,
        )
        return {mid: outcomes[mid] for mid in member_ids}
    except Exception:
        logger.exception(
            'Abandoned merge-deferred sweep: re-driving %s (%s) raised — '
            'continuing with the remaining targets', ', '.join(member_ids), label,
        )
        return dict.fromkeys(member_ids, MemberRedriveOutcome.ERRORED)


@dataclass(frozen=True)
class _Filing:
    level: int
    severity: str
    suggested_action: str
    verdict: str


_RECEIPT = _Filing(
    level=0, severity='info', suggested_action='close_only', verdict='recovered',
)
_STRANDED = _Filing(
    level=1, severity='blocking', suggested_action='investigate_stranded_merge_deferred',
    verdict='NOT recovered',
)


def _file_escalation(
    escalations: EscalationQueue | None,
    target: RecoveryTarget,
    outcomes: Mapping[str, MemberRedriveOutcome],
) -> None:
    acted = {mid: outcome for mid, outcome in outcomes.items() if outcome not in _NOT_ACTED_ON}
    if not acted:
        return
    filing = _RECEIPT if all(o in _RECOVERED for o in acted.values()) else _STRANDED
    task_id = target.redrive_id
    if escalations is None:
        logger.warning(
            'Abandoned merge-deferred sweep: no escalation queue — %s %s (%s) '
            'with nothing filed', task_id, filing.verdict, _render(acted),
        )
        return
    try:
        if _already_filed(escalations, task_id, filing):
            return
        escalations.submit(_escalation(escalations.make_id(task_id), target, acted, filing))
    except Exception:
        logger.warning(
            'Abandoned merge-deferred sweep: filing the escalation for %s failed',
            task_id, exc_info=True,
        )


def _already_filed(escalations: EscalationQueue, task_id: str, filing: _Filing) -> bool:
    """An open record of the same severity, so a receipt never hides a failure."""
    return any(
        esc.category == MERGE_DEFERRED_ABANDONED_CATEGORY and esc.severity == filing.severity
        for esc in escalations.get_by_task(task_id, status='pending')
    )


def _render(outcomes: Mapping[str, MemberRedriveOutcome]) -> str:
    return ', '.join(f'{mid} {outcome}' for mid, outcome in outcomes.items())


def _escalation(
    escalation_id: str,
    target: RecoveryTarget,
    outcomes: Mapping[str, MemberRedriveOutcome],
    filing: _Filing,
) -> Escalation:
    train = (
        f' (declared train {target.declared_train_id}: re-driven through its tip '
        f'{target.redrive_id}, not un-stacked)'
        if target.declared_train_id is not None else ''
    )
    return Escalation(
        id=escalation_id,
        task_id=target.redrive_id,
        agent_role=_AGENT_ROLE,
        category=MERGE_DEFERRED_ABANDONED_CATEGORY,
        level=filing.level,
        severity=filing.severity,
        suggested_action=filing.suggested_action,
        summary=(
            f'Abandoned merge-deferred {target.redrive_id} {filing.verdict}: '
            f'{", ".join(sorted(set(outcomes.values())))}'
        ),
        detail='\n'.join((
            f'Members: {", ".join(target.members)}{train}',
            f'Outcomes: {_render(outcomes)}',
            f'Found merge-deferred on two consecutive passes with {_ABSENT_OWNERS}.',
            f'Runbook: {_RUNBOOK}.',
        )),
    )
