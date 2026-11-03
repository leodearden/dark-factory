"""What a coalesce train that does not land hands back (task 4953).

Forming a coalesce train rebases each successor onto its predecessor. When
the train does not land, every member goes back on the tip it had before
stacking. That happens BEFORE the re-drive flips the member to pending, so a
re-dispatched agent never meets the stacked tree. A branch a live agent may
own is never reset, including one the scheduler claims while another member
is being handed back.

Driven through the lane's public surface over a real git repo. Every request
is enqueued before the lane starts, so the pass runs on the merger's first
iteration. Main always gains a commit after the member branches exist.
Otherwise the task-5618 re-drive un-stack (``git rebase --onto main <cut>``)
could recreate a pre-coalesce commit byte for byte within one committer
second, and these tests would pass without any restore.
"""

from __future__ import annotations

import asyncio
import dataclasses
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from _git_fixtures import RepoSeed, seed_repo
from _merge_lane_fakes import lane_scene_config, make_lane, running_lane
from _orch_helpers import git_env_with_ceiling, wait_responsive
from df_pytest_isolation import git_redirect_env

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.git_ops import GitOps
from orchestrator.merge_lane import MergeRequest, QueuedBranch
from orchestrator.merge_lane.types import TrainCallbacks

pytestmark = pytest.mark.asyncio

_GIT = GitConfig(
    main_branch='main', branch_prefix='task/', worktree_dir='.worktrees',
    push_after_advance=False,
)
_SEED = RepoSeed(files=(('README.md', '# Test\n'),), message='Initial commit')
_ANCHOR, _MID, _TIP = 'uw-anchor', 'uw-mid', 'uw-tip'
_MEMBERS = (_ANCHOR, _TIP)


def _git(cwd: Path, *args: str) -> str:
    env = git_env_with_ceiling(cwd)
    for key in git_redirect_env(env):
        del env[key]
    return subprocess.run(
        ['git', *args], cwd=cwd, env=env, check=True, capture_output=True, text=True,
    ).stdout.strip()


@dataclasses.dataclass(frozen=True)
class _Scene:
    repo: Path
    git_ops: GitOps
    events: EventStore
    config: OrchestratorConfig

    def tip_of(self, member: str) -> str:
        return _git(self.repo, 'rev-parse', '--verify', f'{_GIT.branch_prefix}{member}')

    def commit_on_main(self, relpath: str, text: str) -> None:
        (self.repo / relpath).write_text(text)
        _git(self.repo, 'add', '--', relpath)
        _git(self.repo, 'commit', '-m', f'main: {relpath}')


@pytest.fixture
def scene(tmp_path: Path) -> _Scene:
    repo = seed_repo(tmp_path / 'repo', _SEED)
    return _Scene(
        repo=repo,
        git_ops=GitOps(_GIT, repo),
        events=EventStore(db_path=tmp_path / 'events.db', run_id='coalesce-unwind'),
        config=lane_scene_config(repo, _GIT, merge_train_coalesce_enabled=True),
    )


async def _workflow_request(scene: _Scene, name: str) -> MergeRequest:
    """A workflow-awaited single for a fresh branch adding ``<name>.txt``."""
    worktree = (await scene.git_ops.create_worktree(name)).path
    (worktree / f'{name}.txt').write_text(f'{name}\n')
    await scene.git_ops.commit(worktree, f'{name}: own work')
    return MergeRequest(
        task_id=name,
        branch=QueuedBranch.parse(name, _GIT.branch_prefix),
        worktree=worktree,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=scene.config,
        result=asyncio.get_running_loop().create_future(),
        awaited_by_workflow=True,
    )


@dataclasses.dataclass(frozen=True)
class _Redrive:
    member: str
    on_main: bool
    sha: str | None
    branch_tip: str | None
    """The member's branch tip at the moment its re-drive was called."""


class _UnwindHooks:
    """Train callbacks reporting fixed member *statuses*, recording each re-drive."""

    def __init__(self, git_ops: GitOps, statuses: dict[str, str], *, redrives: int) -> None:
        self._git_ops = git_ops
        self._statuses = statuses
        self._expected = redrives
        self._all_redriven = asyncio.Event()
        self.redriven: list[_Redrive] = []

    def factory(self, train_id: str) -> TrainCallbacks:
        return TrainCallbacks(
            status_check=self.status_check,
            mark_member_done=AsyncMock(),
            redrive_member=self._redrive,
        )

    async def status_check(self, ids: list[str]) -> dict[str, str]:
        return {i: self._statuses[i] for i in ids}

    async def _redrive(self, member: str, on_main: bool, sha: str | None) -> None:
        tip = await self._git_ops.resolve_branch_sha(f'{_GIT.branch_prefix}{member}')
        self.redriven.append(_Redrive(member, on_main, sha, tip))
        if len(self.redriven) >= self._expected:
            self._all_redriven.set()

    async def wait(self) -> None:
        await wait_responsive(self._all_redriven.wait(), label='members re-driven')


class _ClaimDuringUnwindHooks(_UnwindHooks):
    """Reports *claimed* in-progress from the moment *restored* is back on its
    pre-stack tip: the scheduler claims a pending member while the unwind is
    still handing back another."""

    def __init__(
        self, git_ops: GitOps, statuses: dict[str, str], *, redrives: int,
        claimed: str, restored: str, pre_stack_tip: str,
    ) -> None:
        super().__init__(git_ops, statuses, redrives=redrives)
        self._claimed = claimed
        self._restored_branch = f'{_GIT.branch_prefix}{restored}'
        self._pre_stack_tip = pre_stack_tip

    async def status_check(self, ids: list[str]) -> dict[str, str]:
        statuses = await super().status_check(ids)
        restored = await self._git_ops.resolve_branch_sha(self._restored_branch)
        if restored == self._pre_stack_tip and self._claimed in statuses:
            statuses[self._claimed] = 'in-progress'
        return statuses


async def _coalesce_and_unwind(
    scene: _Scene, requests: list[MergeRequest], hooks: _UnwindHooks,
) -> None:
    """Coalesce *requests* into a train, run the lane until *hooks* has seen
    every re-drive it expects, then stop the lane."""
    queue: asyncio.Queue[Any] = asyncio.Queue()
    for request in requests:
        queue.put_nowait(request)
    lane = make_lane(
        scene.git_ops, queue, event_store=scene.events, train_callback_factory=hooks.factory,
    )
    async with running_lane(lane):
        await hooks.wait()
    assert [e['data']['member_task_ids'] for e in scene.events.fetch_events_by_type(
        EventType.train_coalesced,
    )] == [[r.task_id for r in requests]]


async def test_a_derailed_train_hands_every_member_back_before_its_redrive(
    scene: _Scene,
) -> None:
    requests = [await _workflow_request(scene, name) for name in _MEMBERS]
    scene.commit_on_main(f'{_TIP}.txt', 'main wrote this first\n')
    before = {name: scene.tip_of(name) for name in _MEMBERS}
    hooks = _UnwindHooks(
        scene.git_ops, dict.fromkeys(_MEMBERS, 'merge-deferred'), redrives=2,
    )

    await _coalesce_and_unwind(scene, requests, hooks)

    assert len(scene.events.fetch_events_by_type(EventType.train_derailed)) == 1
    assert {name: scene.tip_of(name) for name in _MEMBERS} == before
    tip_worktree = requests[-1].worktree
    assert _git(tip_worktree, 'rev-parse', 'HEAD') == before[_TIP]
    assert _git(tip_worktree, 'status', '--porcelain') == ''
    assert {r.member: r.branch_tip for r in hooks.redriven} == before


async def test_an_incomplete_train_hands_back_its_pending_tip(scene: _Scene) -> None:
    requests = [await _workflow_request(scene, name) for name in _MEMBERS]
    scene.commit_on_main('drift.txt', 'main moved\n')
    before = scene.tip_of(_TIP)
    hooks = _UnwindHooks(
        scene.git_ops, {_ANCHOR: 'merge-deferred', _TIP: 'pending'}, redrives=1,
    )

    await _coalesce_and_unwind(scene, requests, hooks)

    assert [r.member for r in hooks.redriven] == [_ANCHOR]
    assert scene.tip_of(_TIP) == before


async def test_a_tip_a_live_agent_may_own_is_left_stacked(scene: _Scene) -> None:
    requests = [await _workflow_request(scene, name) for name in _MEMBERS]
    scene.commit_on_main('drift.txt', 'main moved\n')
    hooks = _UnwindHooks(
        scene.git_ops, {_ANCHOR: 'merge-deferred', _TIP: 'in-progress'}, redrives=1,
    )

    await _coalesce_and_unwind(scene, requests, hooks)

    assert [r.member for r in hooks.redriven] == [_ANCHOR]
    assert await scene.git_ops.is_ancestor(f'task/{_ANCHOR}', f'task/{_TIP}')


async def test_a_member_claimed_while_another_is_handed_back_is_left_stacked(
    scene: _Scene,
) -> None:
    members = (_ANCHOR, _MID, _TIP)
    requests = [await _workflow_request(scene, name) for name in members]
    scene.commit_on_main('drift.txt', 'main moved\n')
    before = {name: scene.tip_of(name) for name in members}
    hooks = _ClaimDuringUnwindHooks(
        scene.git_ops,
        {_ANCHOR: 'merge-deferred', _MID: 'merge-deferred', _TIP: 'pending'},
        redrives=2, claimed=_TIP, restored=_MID, pre_stack_tip=before[_MID],
    )

    await _coalesce_and_unwind(scene, requests, hooks)

    assert [r.member for r in hooks.redriven] == [_ANCHOR, _MID]
    assert scene.tip_of(_MID) == before[_MID]
    assert scene.tip_of(_TIP) != before[_TIP]
    assert await scene.git_ops.is_ancestor(f'task/{_ANCHOR}', f'task/{_TIP}')
