"""Recovery of abandoned merge-deferred members (task 4737).

Part 1 pins the pure ownership discriminator in
``orchestrator/src/orchestrator/merge_deferred_recovery.py``: who, if anyone,
still owns a parked member, and what recovering an abandoned one targets.
Part 2 drives ``AbandonedMemberSweep.run_pass`` against a fake lane and task
store and a real escalation queue.  Part 3 is the regression for members a
restart left parked: real git, a real merge worker and the real train
callbacks.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from _live_merge_worker import running_merge_worker
from _workflow_helpers import FakeScheduler
from escalation.models import Escalation
from escalation.queue import EscalationQueue
from test_git_ops_unstack import _commit_files, _init_repo, _make_git_ops, _rev_parse
from test_merge_lane_member_redrive import add_unlanded, land_attributed
from test_merge_lane_parked_custody import held_train_request, until_custody

from orchestrator.git_ops import GitOps
from orchestrator.harness import build_train_callback_factory
from orchestrator.merge_deferred_recovery import (
    MERGE_DEFERRED_ABANDONED_CATEGORY,
    AbandonedMemberSweep,
    DeclaredTrainRef,
    OwnershipView,
    ParkedMember,
    ParkedOwner,
    RecoveryTarget,
    TrainPeer,
    owner_of,
    recovery_targets,
)
from orchestrator.merge_lane import MemberRedriveOutcome, SpeculativeMergeWorker
from orchestrator.scheduler import train_order_key

NOW = datetime(2026, 10, 14, 12, 0, tzinfo=UTC)
TTL = timedelta(seconds=300)


def _row(tid: Any, *, status: str = 'merge-deferred', train: Any = None, **fields: Any) -> dict:
    metadata: dict[str, Any] = dict(fields.pop('metadata', {}))
    if train is not None:
        metadata['train'] = train
    return {'id': tid, 'status': status, 'metadata': metadata, **fields}


def _parse(row: Mapping[str, Any]) -> ParkedMember | None:
    return ParkedMember.from_task(row, now=NOW, claimant_ttl=TTL)


def _member(tid: str, train_id: str | None = None, order: int = 0) -> ParkedMember:
    train = None if train_id is None else DeclaredTrainRef(train_id, (0, order))
    return ParkedMember(task_id=tid, declared_train=train, claimant_live=False)


def _peer(tid: str, train_id: str, order: int, status: str = 'merge-deferred') -> TrainPeer:
    return TrainPeer(task_id=tid, status=status, train=DeclaredTrainRef(train_id, (0, order)))


def _view(
    *,
    custody: Iterable[str] = (),
    workflow: Iterable[str] = (),
    external: Iterable[str] = (),
    peers: Iterable[TrainPeer] = (),
) -> OwnershipView:
    trains: dict[str, list[TrainPeer]] = {}
    for peer in peers:
        trains.setdefault(peer.train.train_id, []).append(peer)
    return OwnershipView(
        custody=frozenset(custody),
        workflow_held=frozenset(workflow),
        externally_held=frozenset(external),
        declared_trains={tid: tuple(members) for tid, members in trains.items()},
    )


class TestParkedMemberFromTask:
    def test_the_id_is_a_string(self) -> None:
        member = _parse(_row(42))
        assert member == ParkedMember(task_id='42', declared_train=None, claimant_live=False)

    def test_a_train_dict_is_a_declared_train(self) -> None:
        row = _row('a', train={'id': 'T', 'order': 1})
        member = _parse(row)
        assert member is not None
        assert member.declared_train == DeclaredTrainRef('T', train_order_key(row))
        assert member.declared_train == DeclaredTrainRef('T', (0, 1))

    @pytest.mark.parametrize('train', ['T', ['T'], {'order': 0}, {'id': '', 'order': 0}])
    def test_anything_but_a_train_dict_with_an_id_is_no_train(self, train: Any) -> None:
        member = _parse(_row('a', train=train))
        assert member is not None
        assert member.declared_train is None

    def test_no_train_metadata_is_no_train(self) -> None:
        member = _parse(_row('a'))
        assert member is not None
        assert member.declared_train is None

    def test_a_fresh_heartbeat_is_a_live_claimant(self) -> None:
        member = _parse(_row(
            'a', claimant_run_id='run-1', heartbeat_at=(NOW - timedelta(seconds=10)).isoformat(),
        ))
        assert member is not None
        assert member.claimant_live

    def test_the_legacy_infra_hold_flag_is_a_live_claimant(self) -> None:
        member = _parse(_row('a', metadata={'infra_hold': True}))
        assert member is not None
        assert member.claimant_live

    @pytest.mark.parametrize('fields', [
        {},
        {'claimant_run_id': '  ', 'heartbeat_at': NOW.isoformat()},
        {'claimant_run_id': 'run-1', 'heartbeat_at': (NOW - timedelta(hours=1)).isoformat()},
        {'claimant_run_id': 'run-1'},
    ])
    def test_a_stale_or_absent_claimant_is_not_live(self, fields: dict[str, str]) -> None:
        member = _parse(_row('a', **fields))
        assert member is not None
        assert not member.claimant_live

    @pytest.mark.parametrize('tid', [None, ''])
    def test_a_row_without_an_id_is_skipped(self, tid: Any) -> None:
        assert _parse(_row(tid)) is None


class TestTrainPeerFromTask:
    def test_a_train_row_is_a_peer_carrying_the_shared_order_key(self) -> None:
        row = _row(7, status='pending', train={'id': 'T', 'order': 'tip'})
        assert TrainPeer.from_task(row) == TrainPeer(
            task_id='7', status='pending', train=DeclaredTrainRef('T', train_order_key(row)),
        )

    def test_a_row_without_a_train_is_no_peer(self) -> None:
        assert TrainPeer.from_task(_row('a')) is None


class TestOwnerOf:
    def test_custody(self) -> None:
        assert owner_of(_member('a'), _view(custody=['a'])) is ParkedOwner.TRAIN_CUSTODY

    def test_workflow(self) -> None:
        assert owner_of(_member('a'), _view(workflow=['a'])) is ParkedOwner.WORKFLOW

    def test_external_hold(self) -> None:
        assert owner_of(_member('a'), _view(external=['a'])) is ParkedOwner.EXTERNAL_HOLD

    @pytest.mark.parametrize('status', ['pending', 'in-progress', 'blocked'])
    def test_an_unparked_peer_keeps_the_train_active(self, status: str) -> None:
        view = _view(peers=[_peer('d0', 'T', 0, status), _peer('d1', 'T', 1)])
        assert owner_of(_member('d1', 'T', 1), view) is ParkedOwner.TRAIN_PEER_ACTIVE

    @pytest.mark.parametrize('holder', ['custody', 'workflow', 'external'])
    def test_an_owned_parked_peer_keeps_the_train_active(self, holder: str) -> None:
        view = _view(peers=[_peer('d0', 'T', 0), _peer('d1', 'T', 1)], **{holder: ['d0']})
        assert owner_of(_member('d1', 'T', 1), view) is ParkedOwner.TRAIN_PEER_ACTIVE

    def test_a_train_whose_peers_are_all_parked_and_unowned_is_abandoned(self) -> None:
        view = _view(peers=[_peer('d0', 'T', 0), _peer('d1', 'T', 1)])
        assert owner_of(_member('d1', 'T', 1), view) is ParkedOwner.NONE

    def test_a_member_with_no_owner_is_abandoned(self) -> None:
        assert owner_of(_member('a'), _view()) is ParkedOwner.NONE

    def test_custody_wins_over_every_other_signal(self) -> None:
        view = _view(
            custody=['d1'], workflow=['d1'], external=['d1'],
            peers=[_peer('d0', 'T', 0, 'pending'), _peer('d1', 'T', 1)],
        )
        assert owner_of(_member('d1', 'T', 1), view) is ParkedOwner.TRAIN_CUSTODY


class TestRecoveryTargets:
    def test_a_member_of_no_train_is_its_own_target(self) -> None:
        (target,) = recovery_targets([_member('a')], _view())
        assert target.redrive_id == 'a'
        assert target.members == ('a',)
        assert target.strip_unlanded_base
        assert target.declared_train_id is None
        assert 'a' in target.label

    def test_an_abandoned_declared_train_is_one_target_its_tip_unstripped(self) -> None:
        peers = [_peer('d2', 'T', 2), _peer('d0', 'T', 0), _peer('d1', 'T', 1)]
        members = [_member(p.task_id, 'T', p.train.order_key[1]) for p in peers]

        targets = recovery_targets(members, _view(peers=peers))

        assert targets == (RecoveryTarget(
            redrive_id='d2', members=('d0', 'd1', 'd2'), label=targets[0].label,
            strip_unlanded_base=False, declared_train_id='T',
        ),)
        assert 'T' in targets[0].label

    def test_a_member_without_an_integer_order_is_the_tip_as_the_scheduler_sorts_it(
        self,
    ) -> None:
        rows = [
            _row('d0', train={'id': 'T', 'order': 0}),
            _row('dx', train={'id': 'T'}),
            _row('d1', train={'id': 'T', 'order': 1}),
        ]
        assert [train_order_key(row) for row in rows] == [(0, 0), (1, 0), (0, 1)]
        peers = [peer for row in rows if (peer := TrainPeer.from_task(row)) is not None]
        members = [m for row in rows if (m := _parse(row)) is not None]

        (target,) = recovery_targets(members, _view(peers=peers))

        assert target.redrive_id == 'dx'
        assert target.members == ('d0', 'd1', 'dx')


# ---------------------------------------------------------------------------
# Part 2: AbandonedMemberSweep.run_pass
# ---------------------------------------------------------------------------


@dataclass
class _FakeLane:
    custody: set[str] = field(default_factory=set)
    outcomes: dict[str, MemberRedriveOutcome] = field(default_factory=dict)
    raises_for: set[str] = field(default_factory=set)
    calls: list[tuple[list[str], bool]] = field(default_factory=list)

    def train_member_ids_in_custody(self) -> frozenset[str]:
        return frozenset(self.custody)

    async def redrive_abandoned_members(
        self, label: str, member_ids: Sequence[str], *, strip_unlanded_base: bool,
    ) -> dict[str, MemberRedriveOutcome]:
        self.calls.append((list(member_ids), strip_unlanded_base))
        if self.raises_for & set(member_ids):
            raise RuntimeError('lane exploded')
        return {
            mid: self.outcomes.get(mid, MemberRedriveOutcome.NOT_ON_MAIN) for mid in member_ids
        }


@dataclass
class _FakeTasks:
    rows: list[dict] | None = field(default_factory=list)
    held: set[str] = field(default_factory=set)
    listings: int = 0

    async def get_tasks(
        self, *, statuses: Iterable[str], distinguish_failure: bool,
    ) -> list[dict] | None:
        assert distinguish_failure
        assert 'merge-deferred' in set(statuses)
        self.listings += 1
        wanted = set(statuses)
        if self.rows is None:
            return None
        return [dict(row) for row in self.rows if row['status'] in wanted]

    def is_actively_held(self, tid: str) -> bool:
        return tid in self.held


class _UnreadableQueue(EscalationQueue):
    def get_by_task(self, task_id: str, status: str | None = None, level: int | None = None,
                    agent_role: str | None = None) -> list[Escalation]:
        raise OSError('escalation store unreadable')


@dataclass
class _Sweep:
    lane: _FakeLane
    tasks: _FakeTasks
    queue: EscalationQueue | None
    sweep: AbandonedMemberSweep = field(default_factory=AbandonedMemberSweep)

    async def run(self, *, lane: Any = ...):
        return await self.sweep.run_pass(
            lane=self.lane if lane is ... else lane,
            tasks=self.tasks,
            escalations=self.queue,
            claimant_ttl=TTL,
        )

    def filed(self, tid: str) -> list[Escalation]:
        assert self.queue is not None
        return [
            esc for esc in self.queue.get_by_task(tid, status='pending')
            if esc.category == MERGE_DEFERRED_ABANDONED_CATEGORY
        ]

    def redriven(self) -> list[tuple[list[str], bool]]:
        return self.lane.calls


@pytest.fixture
def sweep(tmp_path: Path) -> _Sweep:
    return _Sweep(
        lane=_FakeLane(),
        tasks=_FakeTasks(rows=[_row('a')]),
        queue=EscalationQueue(tmp_path / 'esc'),
    )


def _file(
    queue: EscalationQueue, tid: str, *, category: str, severity: str, level: int = 1,
) -> None:
    queue.submit(Escalation(
        id=queue.make_id(tid), task_id=tid, agent_role='test', severity=severity,
        category=category, summary='pre-existing', level=level,
    ))


@pytest.mark.asyncio
class TestDebounce:
    async def test_the_first_sighting_only_marks_a_suspect(self, sweep: _Sweep) -> None:
        report = await sweep.run()

        assert report.suspects == frozenset({'a'})
        assert sweep.redriven() == []
        assert sweep.filed('a') == []

    async def test_the_second_consecutive_sighting_redrives(self, sweep: _Sweep) -> None:
        await sweep.run()
        report = await sweep.run()

        assert sweep.redriven() == [(['a'], True)]
        assert report.outcomes == {'a': MemberRedriveOutcome.NOT_ON_MAIN}

    @pytest.mark.parametrize('owner', ['workflow', 'custody'])
    async def test_an_owner_between_passes_restarts_the_count(
        self, sweep: _Sweep, owner: str,
    ) -> None:
        holders = sweep.tasks.held if owner == 'workflow' else sweep.lane.custody
        await sweep.run()
        holders.add('a')
        await sweep.run()
        holders.discard('a')
        await sweep.run()
        assert sweep.redriven() == []

        await sweep.run()

        assert sweep.redriven() == [(['a'], True)]

    async def test_a_failed_listing_aborts_and_restarts_the_count(self, sweep: _Sweep) -> None:
        rows = sweep.tasks.rows
        await sweep.run()
        sweep.tasks.rows = None
        await sweep.run()
        sweep.tasks.rows = rows
        await sweep.run()
        assert sweep.redriven() == []

        await sweep.run()

        assert sweep.redriven() == [(['a'], True)]

    async def test_no_lane_is_a_no_op(self, sweep: _Sweep) -> None:
        await sweep.run(lane=None)
        await sweep.run(lane=None)

        assert sweep.tasks.listings == 0
        assert sweep.redriven() == []
        assert sweep.filed('a') == []

    async def test_a_pass_without_a_lane_restarts_the_count(self, sweep: _Sweep) -> None:
        await sweep.run()
        await sweep.run(lane=None)
        await sweep.run()
        assert sweep.redriven() == []

        await sweep.run()

        assert sweep.redriven() == [(['a'], True)]


@pytest.mark.asyncio
class TestEscalation:
    async def test_a_recovery_files_one_info_l0_receipt_naming_the_absent_owners(
        self, sweep: _Sweep,
    ) -> None:
        await sweep.run()
        await sweep.run()

        (esc,) = sweep.filed('a')
        assert (esc.level, esc.severity, esc.suggested_action) == (0, 'info', 'close_only')
        assert MemberRedriveOutcome.NOT_ON_MAIN in esc.summary
        for signal in ('custody', 'workflow', 'claimant', 'escalation', 'peer'):
            assert signal in esc.detail

    @pytest.mark.parametrize('outcome', [
        MemberRedriveOutcome.ERRORED,
        MemberRedriveOutcome.STALE_EVIDENCE,
        MemberRedriveOutcome.PROVENANCE_CONFLICT_MEMOIZED,
    ])
    async def test_a_failed_recovery_files_blocking_and_then_pins_the_member(
        self, sweep: _Sweep, outcome: MemberRedriveOutcome,
    ) -> None:
        sweep.lane.outcomes['a'] = outcome
        await sweep.run()
        await sweep.run()

        (esc,) = sweep.filed('a')
        assert (esc.level, esc.severity, esc.suggested_action) == (
            1, 'blocking', 'investigate_stranded_merge_deferred',
        )

        report = await sweep.run()
        await sweep.run()

        assert report.owners[ParkedOwner.EXTERNAL_HOLD] == 1
        assert sweep.redriven() == [(['a'], True)]

    @pytest.mark.parametrize('outcome', [
        MemberRedriveOutcome.NOT_PARKED, MemberRedriveOutcome.HELD_BY_LIVE_TRAIN,
    ])
    async def test_a_member_the_lane_did_not_act_on_files_nothing(
        self, sweep: _Sweep, outcome: MemberRedriveOutcome,
    ) -> None:
        sweep.lane.outcomes['a'] = outcome
        await sweep.run()
        await sweep.run()

        assert sweep.redriven() == [(['a'], True)]
        assert sweep.filed('a') == []

    async def test_an_open_record_of_the_same_severity_suppresses_a_second_filing(
        self, sweep: _Sweep,
    ) -> None:
        assert sweep.queue is not None
        _file(
            sweep.queue, 'a', category=MERGE_DEFERRED_ABANDONED_CATEGORY,
            severity='info', level=0,
        )
        await sweep.run()
        await sweep.run()

        assert sweep.redriven() == [(['a'], True)]
        assert [esc.summary for esc in sweep.filed('a')] == ['pre-existing']

    @pytest.mark.parametrize('receipt_level', [0, 1])
    async def test_an_open_receipt_never_suppresses_a_failure_which_then_pins(
        self, sweep: _Sweep, receipt_level: int,
    ) -> None:
        assert sweep.queue is not None
        _file(
            sweep.queue, 'a', category=MERGE_DEFERRED_ABANDONED_CATEGORY,
            severity='info', level=receipt_level,
        )
        sweep.lane.outcomes['a'] = MemberRedriveOutcome.ERRORED
        await sweep.run()
        await sweep.run()

        assert {(esc.level, esc.severity) for esc in sweep.filed('a')} == {
            (1, 'blocking'), (receipt_level, 'info'),
        }

        report = await sweep.run()
        await sweep.run()

        assert report.owners[ParkedOwner.EXTERNAL_HOLD] == 1
        assert sweep.redriven() == [(['a'], True)]

    async def test_a_lane_failure_on_one_target_does_not_stop_the_next(
        self, sweep: _Sweep,
    ) -> None:
        sweep.tasks.rows = [_row('a'), _row('b'), _row('c')]
        sweep.lane.raises_for.add('b')
        await sweep.run()
        report = await sweep.run()

        assert sorted(ids for ids, _ in sweep.redriven()) == [['a'], ['b'], ['c']]
        assert report.outcomes['b'] is MemberRedriveOutcome.ERRORED
        assert [esc.severity for esc in sweep.filed('b')] == ['blocking']
        assert [esc.severity for esc in sweep.filed('c')] == ['info']

    async def test_no_queue_still_recovers(self, sweep: _Sweep) -> None:
        sweep.queue = None
        await sweep.run()
        await sweep.run()

        assert sweep.redriven() == [(['a'], True)]


@pytest.mark.asyncio
class TestOwnership:
    async def test_a_pending_blocking_escalation_holds_the_member(self, sweep: _Sweep) -> None:
        assert sweep.queue is not None
        _file(sweep.queue, 'a', category='design_concern', severity='blocking')
        await sweep.run()
        report = await sweep.run()

        assert report.owners[ParkedOwner.EXTERNAL_HOLD] == 1
        assert sweep.redriven() == []

    async def test_an_unreadable_escalation_store_holds_the_member(self, tmp_path: Path) -> None:
        sweep = _Sweep(
            lane=_FakeLane(), tasks=_FakeTasks(rows=[_row('a')]),
            queue=_UnreadableQueue(tmp_path / 'esc'),
        )
        await sweep.run()
        report = await sweep.run()

        assert report.owners[ParkedOwner.EXTERNAL_HOLD] == 1
        assert sweep.redriven() == []

    async def test_a_live_claimant_holds_the_member(self, sweep: _Sweep) -> None:
        sweep.tasks.rows = [_row(
            'a', claimant_run_id='run-1', heartbeat_at=datetime.now(UTC).isoformat(),
        )]
        await sweep.run()
        report = await sweep.run()

        assert report.owners[ParkedOwner.EXTERNAL_HOLD] == 1
        assert sweep.redriven() == []

    async def test_an_abandoned_declared_train_is_redriven_through_its_tip(
        self, sweep: _Sweep,
    ) -> None:
        sweep.tasks.rows = [
            _row(f'd{order}', train={'id': 'T', 'order': order}) for order in (0, 1, 2)
        ]
        await sweep.run()
        await sweep.run()

        assert sweep.redriven() == [(['d2'], False)]
        (esc,) = sweep.filed('d2')
        for mid in ('d0', 'd1', 'd2'):
            assert mid in esc.detail
        assert sweep.filed('d0') == sweep.filed('d1') == []

    @pytest.mark.parametrize('landed', [
        MemberRedriveOutcome.ON_MAIN_ATTRIBUTED, MemberRedriveOutcome.ON_MAIN_UNATTRIBUTED,
    ])
    async def test_a_tip_found_on_main_carries_the_rest_of_its_train_in_one_pass(
        self, sweep: _Sweep, landed: MemberRedriveOutcome,
    ) -> None:
        sweep.tasks.rows = [
            _row(f'd{order}', train={'id': 'T', 'order': order}) for order in (0, 1, 2)
        ]
        sweep.lane.outcomes.update(d0=landed, d1=landed, d2=landed)
        await sweep.run()
        report = await sweep.run()

        assert sweep.redriven() == [(['d2'], False), (['d0', 'd1'], False)]
        assert report.outcomes == {'d0': landed, 'd1': landed, 'd2': landed}
        (esc,) = sweep.filed('d2')
        assert (esc.level, esc.severity) == (0, 'info')
        for mid in ('d0', 'd1', 'd2'):
            assert f'{mid} {landed}' in esc.detail
        assert sweep.filed('d0') == sweep.filed('d1') == []

    async def test_a_landed_tip_whose_peer_fails_files_one_blocking_record(
        self, sweep: _Sweep,
    ) -> None:
        sweep.tasks.rows = [
            _row(f'd{order}', train={'id': 'T', 'order': order}) for order in (0, 1)
        ]
        sweep.lane.outcomes.update(
            d0=MemberRedriveOutcome.ERRORED, d1=MemberRedriveOutcome.ON_MAIN_ATTRIBUTED,
        )
        await sweep.run()
        await sweep.run()

        assert sweep.redriven() == [(['d1'], False), (['d0'], False)]
        (esc,) = sweep.filed('d1')
        assert (esc.level, esc.severity) == (1, 'blocking')
        assert 'd0 errored' in esc.detail

    async def test_an_infra_held_peer_keeps_its_train_active(self, sweep: _Sweep) -> None:
        sweep.tasks.rows = [
            _row('d0', train={'id': 'T', 'order': 0}),
            _row('d1', train={'id': 'T', 'order': 1}),
            _row('d2', status='infra-hold', train={'id': 'T', 'order': 2}),
        ]
        await sweep.run()
        report = await sweep.run()

        assert report.owners[ParkedOwner.TRAIN_PEER_ACTIVE] == 2
        assert sweep.redriven() == []


# ---------------------------------------------------------------------------
# Part 3: members a restart left parked, against real collaborators
# ---------------------------------------------------------------------------


class ParkedScheduler(FakeScheduler):
    """FakeScheduler plus the two reads the abandoned-member sweep makes."""

    def __init__(self) -> None:
        super().__init__()
        self.held: set[str] = set()

    async def get_tasks(
        self, *, statuses: Iterable[str] | None = None, distinguish_failure: bool = False,
    ) -> list[dict]:
        wanted = None if statuses is None else set(statuses)
        return [
            {
                'id': tid,
                'status': history[-1],
                'metadata': dict((self.task_data.get(tid) or {}).get('metadata') or {}),
            }
            for tid, history in self.statuses.items()
            if history and (wanted is None or history[-1] in wanted)
        ]

    def is_actively_held(self, tid: str) -> bool:
        return tid in self.held

    async def park(self, *tids: str, train_id: str | None = None) -> None:
        for order, tid in enumerate(tids):
            if train_id is not None:
                self.task_data[tid] = {
                    'id': tid, 'metadata': {'train': {'id': train_id, 'order': order}},
                }
            await self.set_task_status(tid, 'merge-deferred')


@pytest_asyncio.fixture
async def git_ops(tmp_path: Path) -> GitOps:
    root = tmp_path / 'repo'
    root.mkdir()
    await _init_repo(root)
    await _commit_files(root, {'shared.txt': 'v0\n'}, 'initial')
    return _make_git_ops(root)


def _abandoned_records(queue: EscalationQueue, tid: str) -> list[tuple[int, str]]:
    return [
        (esc.level, esc.severity) for esc in queue.get_by_task(tid, status='pending')
        if esc.category == MERGE_DEFERRED_ABANDONED_CATEGORY
    ]


@pytest.mark.asyncio
class TestRecoveryAfterARestart:
    async def test_abandoned_members_recover_and_owned_ones_stay_parked(
        self, git_ops: GitOps, tmp_path: Path,
    ) -> None:
        root = git_ops.project_root
        await land_attributed(git_ops, 'landed')
        for mid in ('unlanded', 'stacked', 'd0', 'd1'):
            await add_unlanded(git_ops, mid)
        stacked = await git_ops.stack_train_branches(['unlanded', 'stacked'])
        declared = await git_ops.stack_train_branches(['d0', 'd1'])
        assert (stacked.survivors, declared.survivors) == (['unlanded', 'stacked'], ['d0', 'd1'])
        unlanded_commit = await _rev_parse(root, 'task/unlanded')
        d1_tip = await _rev_parse(root, 'task/d1')

        sched = ParkedScheduler()
        await sched.park('landed', 'unlanded', 'stacked', 'live1', 'live2')
        await sched.park('d0', 'd1', train_id='train-d0-4737')
        queue = EscalationQueue(tmp_path / 'esc')
        intake: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(
            git_ops, intake, train_callback_factory=build_train_callback_factory(sched),
        )
        sweep = AbandonedMemberSweep()

        async def run_pass():
            return await sweep.run_pass(
                lane=worker, tasks=sched, escalations=queue, claimant_ttl=TTL,
            )

        worker.halt_for_wip('test')
        await intake.put(held_train_request(git_ops, sched, 'train-live-4737', ['live1', 'live2']))
        parked = {tid: list(history) for tid, history in sched.statuses.items()}
        async with running_merge_worker(worker):
            await until_custody(worker, frozenset({'live1', 'live2'}))

            await run_pass()
            assert sched.statuses == parked
            assert queue.get_pending() == []

            await run_pass()

            latest = {tid: history[-1] for tid, history in sched.statuses.items()}
            assert latest == {
                'landed': 'done', 'unlanded': 'pending', 'stacked': 'pending',
                'live1': 'merge-deferred', 'live2': 'merge-deferred',
                'd0': 'merge-deferred', 'd1': 'pending',
            }
            assert sched.provenance['landed']['kind'] == 'found_on_main'
            assert not await git_ops.is_ancestor(unlanded_commit, 'task/stacked')
            assert await _rev_parse(root, 'task/d1') == d1_tip
            for tid in ('live1', 'live2', 'd0'):
                assert sched.statuses[tid] == parked[tid]
            for tid in ('landed', 'unlanded', 'stacked', 'd1'):
                assert _abandoned_records(queue, tid) == [(0, 'info')]
            for tid in ('d0', 'live1', 'live2'):
                assert queue.get_by_task(tid, status='pending') == []

            third = await run_pass()

        assert third.owners[ParkedOwner.TRAIN_PEER_ACTIVE] == 1
        assert third.outcomes == {}
        assert sched.statuses['d0'] == parked['d0']
