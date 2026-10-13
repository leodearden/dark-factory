"""A train's verify reaches the lane's injected ``VerifyPort`` and runs on a
leased verify host (task 6160).

Every scene drives a stacked train through the lane's own queue over a real
tmp git repo, with the verify scripted by ``FakeVerifier``.
"""

from __future__ import annotations

import asyncio
import dataclasses
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from _git_fixtures import RepoSeed, seed_repo
from _merge_lane_fakes import (
    FakeVerifier,
    fails,
    lane_scene_config,
    make_lane,
    merge_through_lane,
)

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps
from orchestrator.merge_lane import (
    TRAIN_VERIFY_FAILED_REASON_PREFIX,
    GroupMergeRequest,
    QueuedBranch,
)

pytestmark = pytest.mark.asyncio

_GIT = GitConfig(
    main_branch='main', branch_prefix='task/', worktree_dir='.worktrees',
    push_after_advance=False,
)
_SEED = RepoSeed(files=(('README.md', '# Test\n'),), message='Initial commit')


@dataclasses.dataclass(frozen=True)
class _Scene:
    repo: Path
    git_ops: GitOps
    config: OrchestratorConfig


@pytest.fixture
def scene(tmp_path: Path) -> _Scene:
    repo = seed_repo(tmp_path / 'repo', _SEED)
    return _Scene(repo=repo, git_ops=GitOps(_GIT, repo), config=lane_scene_config(repo, _GIT))


async def _branch(scene: _Scene, name: str) -> Path:
    worktree = (await scene.git_ops.create_worktree(name)).path
    (worktree / f'{name}.txt').write_text(f'{name}\n')
    await scene.git_ops.commit(worktree, f'{name}: own work')
    return worktree


@dataclasses.dataclass(frozen=True)
class _Train:
    """A stacked train's request and the member callbacks it was built with."""

    request: GroupMergeRequest
    mark_member_done: AsyncMock
    redrive_member: AsyncMock

    @property
    def tip(self) -> str:
        return self.request.tip_task_id


async def _stacked_train(
    scene: _Scene, members: int, *, train_id: str = 'train-6160',
) -> _Train:
    names = [f'tr-{index}' for index in range(members)]
    worktrees = [await _branch(scene, name) for name in names]
    stacked = await scene.git_ops.stack_train_branches(names)
    assert stacked.survivors == names, stacked
    tip = QueuedBranch.parse(names[-1], _GIT.branch_prefix)
    mark_member_done = AsyncMock()
    redrive_member = AsyncMock()
    request = GroupMergeRequest(
        task_id=names[-1],
        branch=tip,
        worktree=worktrees[-1],
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=scene.config,
        result=asyncio.get_running_loop().create_future(),
        train_id=train_id,
        member_task_ids=names,
        tip_branch=tip,
        tip_task_id=names[-1],
        status_check=AsyncMock(return_value=dict.fromkeys(names, 'merge-deferred')),
        mark_member_done=mark_member_done,
        redrive_member=redrive_member,
    )
    return _Train(request, mark_member_done, redrive_member)


# ─── The train verify obeys the lane's injected verifier ────────────────────


async def test_a_red_verify_on_the_lanes_verifier_derails_the_train(scene: _Scene) -> None:
    train = await _stacked_train(scene, 2)
    verifier = FakeVerifier(default=fails(category='pytest', summary='red train verify'))
    queue: asyncio.Queue[Any] = asyncio.Queue()
    lane = make_lane(scene.git_ops, queue, verifier=verifier)
    main_before = await scene.git_ops.get_main_sha()

    outcome = await merge_through_lane(lane, queue, train.request)

    assert outcome.status == 'blocked', outcome
    assert outcome.reason.startswith(TRAIN_VERIFY_FAILED_REASON_PREFIX), outcome
    assert outcome.failure_category == 'pytest'
    assert await scene.git_ops.get_main_sha() == main_before
    assert verifier.verified == [train.tip]
    train.mark_member_done.assert_not_awaited()


async def test_the_lanes_disk_guard_skips_the_train_verify(scene: _Scene) -> None:
    train = await _stacked_train(scene, 2)
    verifier = FakeVerifier(disk_reason='disk guard: test reason')
    queue: asyncio.Queue[Any] = asyncio.Queue()
    lane = make_lane(scene.git_ops, queue, verifier=verifier)
    main_before = await scene.git_ops.get_main_sha()

    outcome = await merge_through_lane(lane, queue, train.request)

    assert outcome.status == 'blocked', outcome
    assert outcome.reason == 'disk guard: test reason'
    assert outcome.verify_skipped is True
    assert verifier.verified == []
    assert await scene.git_ops.get_main_sha() == main_before
