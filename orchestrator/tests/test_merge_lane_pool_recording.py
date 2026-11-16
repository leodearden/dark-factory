"""The facts the merge lane records for the merge-pool calibration (task 9908).

Every key is additive: ``merge_queued`` names the request, its lane, tier, pin,
head and age; ``merge_dequeued`` names the pick and everything still waiting at
that instant; ``merge_verify`` names the requests the verify covered. Each
scene drives a real lane over a real tmp repo and reads the rows a real
``EventStore`` wrote. Recording must never change what the lane does, so the
scenes without a lookup, or with a raising one, must land exactly as the
scene with one does.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from _git_fixtures import seed_repo
from _live_merge_worker import REAL_GIT_MERGE_RESULT_TIMEOUT
from _merge_lane_fakes import (
    FakeVerifier,
    VerifyScript,
    hangs_until,
    lane_scene_config,
    passes,
    running_lane,
)
from _merge_worker_drive import HEAVY_BARRIER_TEST_TIMEOUT
from _orch_helpers import wait_responsive
from _poll_responsive import poll_responsive

from orchestrator.config import GitConfig, MergeDeepConfig
from orchestrator.event_store import EventStore
from orchestrator.git_ops import GitOps, _run
from orchestrator.merge_lane import (
    MergeLane,
    MergeOutcome,
    MergeRequest,
    QueuedBranch,
    enqueue_merge_request,
)
from orchestrator.task_priority import TaskPriority, TaskPriorityLookup

pytestmark = [pytest.mark.asyncio, pytest.mark.timeout(HEAVY_BARRIER_TEST_TIMEOUT)]

_GIT = GitConfig(
    main_branch='main', branch_prefix='task/', remote='origin',
    worktree_dir='.worktrees', push_after_advance=False,
)
_FIRST_SUBMITTED = 1000.0
_SHARED = ''.join(f'line{i}\n' for i in range(1, 21))

_NEW_QUEUED_KEYS = ('request_id', 'lane', 'tier', 'pinned', 'head_sha', 'first_enqueued_at')
_NEW_DEQUEUED_KEYS = (
    'request_id', 'lane', 'tier', 'pinned', 'waiting', 'waiting_dropped',
    'halted_lanes', 'textual_conflict_pairs', 'footprint_pair_count',
)


class _Scene:
    def __init__(self, root: Path, *, chain_cap: int = 0) -> None:
        self.repo = seed_repo(root / 'repo')
        self.git_ops = GitOps(_GIT, self.repo)
        self.config = lane_scene_config(
            self.repo, _GIT, merge_deep=MergeDeepConfig(chain_cap=chain_cap),
            merge_train_coalesce_enabled=False, verify_host_policy='prefer_local',
        )
        self.db_path = root / 'events.db'
        self.store = EventStore(self.db_path, 'run-9908')
        self.queue: asyncio.Queue[MergeRequest] = asyncio.Queue()

    async def commit_shared(self) -> None:
        (self.repo / 'shared.txt').write_text(_SHARED)
        await _run(['git', 'add', 'shared.txt'], cwd=self.repo)
        await _run(['git', 'commit', '-m', 'seed shared.txt'], cwd=self.repo)

    def lane(self, verifier: FakeVerifier, task_priority: TaskPriorityLookup | None) -> MergeLane:
        return MergeLane(
            self.git_ops, self.queue, event_store=self.store, verifier=verifier,
            task_priority=task_priority,
        )

    async def request(
        self, task_id: str, *, filename: str | None = None, content: str | None = None,
        snapshot_tip: bool = False,
    ) -> MergeRequest:
        worktree = (await self.git_ops.create_worktree(task_id)).path
        (worktree / (filename or f'{task_id}.py')).write_text(content or f'v_{task_id} = 1\n')
        await self.git_ops.commit(worktree, f'{task_id}: own work')
        _, head, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=worktree)
        return MergeRequest(
            task_id=task_id,
            branch=QueuedBranch.parse(f'task/{task_id}', _GIT.branch_prefix),
            worktree=worktree,
            pre_rebased=False,
            task_files=None,
            module_configs=[],
            config=self.config,
            result=asyncio.get_running_loop().create_future(),
            merge_first_enqueued_at=_FIRST_SUBMITTED,
            snapshot_tip=head.strip() if snapshot_tip else None,
        )

    async def submit(self, req: MergeRequest, task_priority: TaskPriorityLookup | None) -> None:
        await enqueue_merge_request(self.queue, req, self.store, task_priority=task_priority)

    def rows(self, event_type: str) -> list[tuple[str, dict[str, Any]]]:
        conn = sqlite3.connect(str(self.db_path))
        try:
            found = conn.execute(
                'SELECT task_id, data FROM events WHERE event_type = ? ORDER BY id', (event_type,),
            ).fetchall()
        finally:
            conn.close()
        return [(task_id, json.loads(data)) for task_id, data in found]

    def data_of(self, event_type: str, task_id: str) -> list[dict[str, Any]]:
        return [data for tid, data in self.rows(event_type) if tid == task_id]


def _lookup(table: dict[str, TaskPriority]) -> TaskPriorityLookup:
    return lambda task_id: table.get(task_id, TaskPriority())


def _raising(task_id: str) -> TaskPriority:
    raise RuntimeError(f'scheduler gone for {task_id}')


async def _outcomes(requests: list[MergeRequest]) -> dict[str, MergeOutcome]:
    return {
        req.task_id: await wait_responsive(
            req.result, timeout=REAL_GIT_MERGE_RESULT_TIMEOUT, label=f'{req.task_id} outcome',
        )
        for req in requests
    }


async def _two_sequential_merges(
    root: Path, task_priority: TaskPriorityLookup | None,
) -> tuple[_Scene, dict[str, MergeOutcome]]:
    scene = _Scene(root)
    reqs = [await scene.request('pa'), await scene.request('pb')]
    async with running_lane(scene.lane(FakeVerifier(), task_priority)):
        for req in reqs:
            await scene.submit(req, task_priority)
        outcomes = await _outcomes(reqs)
    return scene, outcomes


class TestSingleRequest:

    async def test_every_new_key_is_recorded_for_one_request(self, tmp_path: Path) -> None:
        scene = _Scene(tmp_path)
        req = await scene.request('solo', snapshot_tip=True)
        priority = _lookup({'solo': TaskPriority(tier='high', pinned=True)})

        async with running_lane(scene.lane(FakeVerifier(), priority)):
            await scene.submit(req, priority)
            outcome = (await _outcomes([req]))['solo']

        assert outcome.status == 'done', outcome
        (queued,) = scene.data_of('merge_queued', 'solo')
        assert {k: queued[k] for k in _NEW_QUEUED_KEYS} == {
            'request_id': req.request_id, 'lane': 'normal', 'tier': 'high', 'pinned': True,
            'head_sha': req.snapshot_tip, 'first_enqueued_at': _FIRST_SUBMITTED,
        }
        assert queued['branch'] == 'solo'
        (dequeued,) = scene.data_of('merge_dequeued', 'solo')
        assert {k: dequeued[k] for k in _NEW_DEQUEUED_KEYS} == {
            'request_id': req.request_id, 'lane': 'normal', 'tier': 'high', 'pinned': True,
            'waiting': [], 'waiting_dropped': 0, 'halted_lanes': [],
            'textual_conflict_pairs': [], 'footprint_pair_count': 0,
        }
        assert dequeued['branch'] == 'solo' and 'queue_depth' in dequeued
        verifies = scene.data_of('merge_verify', 'solo')
        assert verifies, 'the request was verified'
        for row in verifies:
            assert row['chain_items'] == 1
            assert row['chain_request_ids'] == [req.request_id]
            assert row['chain_task_ids'] == ['solo']

    async def test_an_unchanged_resubmission_carries_the_same_head_and_first_submission(
        self, tmp_path: Path,
    ) -> None:
        scene = _Scene(tmp_path)
        first = await scene.request('again', snapshot_tip=True)
        resubmitted = MergeRequest(
            task_id=first.task_id, branch=first.branch, worktree=first.worktree,
            pre_rebased=False, task_files=None, module_configs=[], config=scene.config,
            result=asyncio.get_running_loop().create_future(),
            merge_first_enqueued_at=first.merge_first_enqueued_at,
            snapshot_tip=first.snapshot_tip,
        )

        await scene.submit(first, None)
        await scene.submit(resubmitted, None)

        rows = scene.data_of('merge_queued', 'again')
        assert [row['request_id'] for row in rows] == [first.request_id, resubmitted.request_id]
        assert rows[0]['head_sha'] == rows[1]['head_sha'] == first.snapshot_tip
        assert rows[0]['first_enqueued_at'] == rows[1]['first_enqueued_at'] == _FIRST_SUBMITTED

    async def test_first_enqueued_at_falls_back_to_this_submission(self, tmp_path: Path) -> None:
        scene = _Scene(tmp_path)
        req = await scene.request('fresh')
        req.merge_first_enqueued_at = None

        await scene.submit(req, None)

        (row,) = scene.data_of('merge_queued', 'fresh')
        assert row['first_enqueued_at'] == req.enqueued_at
        assert row['head_sha'] is None


class TestLookupFailureLeavesTheLaneAlone:

    @pytest.mark.parametrize('broken', [None, _raising], ids=['missing', 'raising'])
    async def test_keys_are_null_and_the_lane_lands_as_it_does_with_a_lookup(
        self, tmp_path: Path, broken: TaskPriorityLookup | None,
    ) -> None:
        good = _lookup({'pa': TaskPriority('high', False), 'pb': TaskPriority('low', True)})
        with_lookup, landed = await _two_sequential_merges(tmp_path / 'with', good)
        without, landed_without = await _two_sequential_merges(tmp_path / 'without', broken)

        assert {t: o.status for t, o in landed_without.items()} == {
            t: o.status for t, o in landed.items()
        } == {'pa': 'done', 'pb': 'done'}
        for kind in ('merge_queued', 'merge_dequeued', 'merge_verify', 'merge_finalized'):
            assert [t for t, _ in without.rows(kind)] == [t for t, _ in with_lookup.rows(kind)], kind
        for _, row in without.rows('merge_queued') + without.rows('merge_dequeued'):
            assert row['tier'] is None and row['pinned'] is None
        assert [row['tier'] for _, row in with_lookup.rows('merge_dequeued')] == ['high', 'low']
        assert [row['pinned'] for _, row in with_lookup.rows('merge_queued')] == [False, True]


class TestWaitingSetAtThePick:

    async def test_the_pick_lists_what_waits_in_scan_order_and_its_true_conflicts(
        self, tmp_path: Path,
    ) -> None:
        scene = _Scene(tmp_path)
        await scene.commit_shared()
        head = await scene.request('wx')
        left = await scene.request(
            'wb', filename='shared.txt', content=_SHARED.replace('line5\n', 'from-wb\n'),
        )
        right = await scene.request(
            'wc', filename='shared.txt', content=_SHARED.replace('line5\n', 'from-wc\n'),
        )
        loner = await scene.request('wd')
        loner.lane = 'high'
        priority = _lookup({'wb': TaskPriority('medium', False), 'wd': TaskPriority('high', True)})
        release, entered = asyncio.Event(), asyncio.Event()
        verifier = FakeVerifier(scripts={'wd': VerifyScript(
            result=passes().result, release=release, entered=entered,
        )})
        for req in (head, left, right, loner):
            await scene.submit(req, priority)

        async with running_lane(scene.lane(verifier, priority)):
            try:
                await wait_responsive(entered.wait(), label='wd verify entry')
            finally:
                release.set()

        first_task, first = scene.rows('merge_dequeued')[0]
        assert first_task == 'wd', 'the high lane picks first'
        assert [(w['task_id'], w['lane']) for w in first['waiting']] == [
            ('wx', 'normal'), ('wb', 'normal'), ('wc', 'normal'),
        ]
        by_task = {w['task_id']: w for w in first['waiting']}
        assert by_task['wb'] == {
            'request_id': left.request_id, 'task_id': 'wb', 'lane': 'normal',
            'tier': 'medium', 'pinned': False, 'first_enqueued_at': _FIRST_SUBMITTED,
            'reserved': False, 'in_graph': True,
        }
        assert by_task['wx']['tier'] is None
        assert first['in_graph'] and all(w['in_graph'] for w in first['waiting'])
        assert (first['tier'], first['pinned'], first['lane']) == ('high', True, 'high')
        assert first['textual_conflict_pairs'] == [sorted([left.request_id, right.request_id])]
        assert first['footprint_pair_count'] == 1
        assert first['waiting_dropped'] == 0


class TestBatchMembersOnMergeVerify:

    async def _burst(self, root: Path, *, chain_cap: int) -> tuple[_Scene, list[MergeRequest]]:
        """A head whose verify is held, then three followers arriving in one burst."""
        scene = _Scene(root, chain_cap=chain_cap)
        head = await scene.request('bh')
        followers = [await scene.request(t) for t in ('b1', 'b2', 'b3')]
        release, entered = asyncio.Event(), asyncio.Event()
        verifier = FakeVerifier(sequence=[hangs_until(release, entered=entered)])
        async with running_lane(scene.lane(verifier, None)):
            try:
                await scene.submit(head, None)
                await wait_responsive(entered.wait(), label='head verify entry')
                for req in followers:
                    await scene.submit(req, None)
                await poll_responsive(
                    lambda: any(t in {'b1', 'b2', 'b3'} for t, _ in scene.rows('merge_dequeued')),
                    timeout=REAL_GIT_MERGE_RESULT_TIMEOUT, label='a follower to be dequeued',
                )
            finally:
                release.set()
            outcomes = await _outcomes([head, *followers])
        assert {t: o.status for t, o in outcomes.items()} == dict.fromkeys(
            ('bh', 'b1', 'b2', 'b3'), 'done',
        )
        return scene, followers

    @staticmethod
    def _verify_rows(scene: _Scene) -> list[dict[str, Any]]:
        rows = [row for _, row in scene.rows('merge_verify')]
        for row in rows:
            assert len(row['chain_request_ids']) == len(row['chain_task_ids']) == row['chain_items']
        return rows

    async def test_a_chain_of_three_lists_its_members_as_built(self, tmp_path: Path) -> None:
        scene, followers = await self._burst(tmp_path, chain_cap=6)

        (deep,) = [row for row in self._verify_rows(scene) if row['chain_items'] > 1]
        assert deep['chain_items'] == 3
        assert deep['chain_request_ids'] == [req.request_id for req in followers]
        assert deep['chain_task_ids'] == ['b1', 'b2', 'b3']

    async def test_chain_cap_zero_verifies_one_request_at_a_time(self, tmp_path: Path) -> None:
        scene, _ = await self._burst(tmp_path, chain_cap=0)

        rows = self._verify_rows(scene)
        assert {tid for tid, _ in scene.rows('merge_verify')} == {'bh', 'b1', 'b2', 'b3'}
        for (task_id, _), row in zip(scene.rows('merge_verify'), rows, strict=True):
            assert row['chain_items'] == 1
            assert row['chain_task_ids'] == [task_id]
