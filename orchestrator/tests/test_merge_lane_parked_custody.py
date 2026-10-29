"""The merge lane's custody of parked train members, and its re-drive of
abandoned ones (task 4737).

``SpeculativeMergeWorker.train_member_ids_in_custody`` names every member of a
train the lane still holds; ``redrive_abandoned_members`` hands parked members
no train holds back through the worker's own train callbacks, and refuses any
member a live train still holds.  Driven through public seams only.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest
import pytest_asyncio
from _live_merge_worker import running_merge_worker
from _workflow_helpers import FakeScheduler
from test_git_ops_unstack import _commit_files, _init_repo, _make_git_ops
from test_merge_lane_member_redrive import add_unlanded, land_attributed

from orchestrator.config import OrchestratorConfig
from orchestrator.git_ops import GitOps
from orchestrator.harness import build_train_callback_factory
from orchestrator.merge_lane import (
    GroupMergeRequest,
    MemberRedriveOutcome,
    MergeOutcome,
    QueuedBranch,
)
from orchestrator.merge_lane.worker import SpeculativeMergeWorker

_CUSTODY_WAIT_S = 5.0


def held_train_request(
    git_ops: GitOps, sched: FakeScheduler, train_id: str, member_ids: Sequence[str],
) -> GroupMergeRequest:
    """A train over *member_ids* wired to *sched* through the real callback factory."""
    callbacks = build_train_callback_factory(sched)(train_id)
    tip = member_ids[-1]
    prefix = git_ops.config.branch_prefix
    result: asyncio.Future[MergeOutcome] = asyncio.get_running_loop().create_future()
    return GroupMergeRequest(
        task_id=tip,
        branch=QueuedBranch.parse(f'{prefix}{tip}', prefix),
        worktree=git_ops.worktree_base / tip,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=OrchestratorConfig(project_root=git_ops.project_root, git=git_ops.config),
        result=result,
        train_id=train_id,
        member_task_ids=list(member_ids),
        tip_branch=QueuedBranch.parse(f'{prefix}{tip}', prefix),
        tip_task_id=tip,
        status_check=callbacks.status_check,
        mark_member_done=callbacks.mark_member_done,
        redrive_member=callbacks.redrive_member,
    )


@dataclass(frozen=True)
class _Lane:
    git_ops: GitOps
    sched: FakeScheduler
    queue: asyncio.Queue

    def worker(self) -> SpeculativeMergeWorker:
        return SpeculativeMergeWorker(
            self.git_ops, self.queue,
            train_callback_factory=build_train_callback_factory(self.sched),
        )

    def declared_train(self, train_id: str, member_ids: Sequence[str]) -> GroupMergeRequest:
        return held_train_request(self.git_ops, self.sched, train_id, member_ids)

    async def park(self, *mids: str) -> None:
        for mid in mids:
            await self.sched.set_task_status(mid, 'merge-deferred')


@pytest_asyncio.fixture
async def lane(tmp_path: Path) -> _Lane:
    root = tmp_path / 'repo'
    root.mkdir()
    await _init_repo(root)
    await _commit_files(root, {'shared.txt': 'v0\n'}, 'initial')
    return _Lane(git_ops=_make_git_ops(root), sched=FakeScheduler(), queue=asyncio.Queue())


async def until_custody(worker: SpeculativeMergeWorker, expected: frozenset[str]) -> None:
    deadline = time.monotonic() + _CUSTODY_WAIT_S
    while worker.train_member_ids_in_custody() != expected:
        assert time.monotonic() < deadline, (
            f'custody never became {sorted(expected)}; '
            f'last saw {sorted(worker.train_member_ids_in_custody())}'
        )
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
class TestCustody:
    async def test_a_held_train_is_in_custody_until_the_worker_stops(self, lane: _Lane) -> None:
        worker = lane.worker()
        worker.halt_for_wip('test')
        await lane.queue.put(lane.declared_train('train-p1-held', ['p1', 'p2']))

        async with running_merge_worker(worker):
            await until_custody(worker, frozenset({'p1', 'p2'}))

        assert worker.train_member_ids_in_custody() == frozenset()

    async def test_a_worker_holding_nothing_has_nothing_in_custody(self, lane: _Lane) -> None:
        assert lane.worker().train_member_ids_in_custody() == frozenset()


@pytest.mark.asyncio
class TestRedriveAbandonedMembers:
    async def test_abandoned_members_are_redriven_through_the_train_callbacks(
        self, lane: _Lane,
    ) -> None:
        await land_attributed(lane.git_ops, 'r-on')
        await add_unlanded(lane.git_ops, 'r-off')
        await lane.park('r-on', 'r-off')

        outcomes = await lane.worker().redrive_abandoned_members(
            'abandoned-x', ['r-on', 'r-off'], strip_unlanded_base=True,
        )

        assert outcomes == {
            'r-on': MemberRedriveOutcome.ON_MAIN_ATTRIBUTED,
            'r-off': MemberRedriveOutcome.NOT_ON_MAIN,
        }
        assert lane.sched.statuses['r-on'][-1] == 'done'
        assert lane.sched.provenance['r-on']['kind'] == 'found_on_main'
        assert lane.sched.statuses['r-off'][-1] == 'pending'

    async def test_a_member_a_live_train_holds_is_refused(self, lane: _Lane) -> None:
        await lane.park('p1', 'p2')
        history_before = list(lane.sched.statuses['p1'])
        worker = lane.worker()
        worker.halt_for_wip('test')
        await lane.queue.put(lane.declared_train('train-p1-held', ['p1', 'p2']))

        async with running_merge_worker(worker):
            await until_custody(worker, frozenset({'p1', 'p2'}))
            outcomes = await worker.redrive_abandoned_members(
                'abandoned-p1', ['p1'], strip_unlanded_base=True,
            )

        assert outcomes == {'p1': MemberRedriveOutcome.HELD_BY_LIVE_TRAIN}
        assert lane.sched.statuses['p1'] == history_before

    async def test_a_worker_without_train_callbacks_refuses_loudly(self, lane: _Lane) -> None:
        worker = SpeculativeMergeWorker(lane.git_ops, lane.queue)

        with pytest.raises(RuntimeError, match='train_callback_factory'):
            await worker.redrive_abandoned_members(
                'abandoned-x', ['a'], strip_unlanded_base=True,
            )
