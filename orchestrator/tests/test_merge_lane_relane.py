"""An operator re-lanes a merge request after it was enqueued (task 5246).

A request's lane is decided once, at enqueue. ``relane`` is the lever on a
request that is already waiting: it moves the request, or the coalesce train
carrying a member, into the other lane, and refuses one the merger has already
taken.

Driven through the lane's public surface over a real git repo. The normal lane
is halted, so the only way a request lands is by being moved into the high
lane. Observed through the request futures, ``snapshot()``, and what
``relane`` returns.
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
    hangs_until,
    lane_scene_config,
    make_lane,
    running_lane,
)
from _orch_helpers import wait_responsive

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps
from orchestrator.merge_lane import MergeRequest, QueuedBranch
from orchestrator.merge_lane.types import TrainCallbacks
from orchestrator.merge_lane.worker import SpeculativeMergeWorker

pytestmark = pytest.mark.asyncio

_GIT = GitConfig(
    main_branch='main', branch_prefix='task/', worktree_dir='.worktrees',
    push_after_advance=False,
)
_SEED = RepoSeed(files=(('README.md', '# Test\n'),), message='Initial commit')
_LANDS_WITHIN = 30.0


@dataclasses.dataclass(frozen=True)
class _Scene:
    git_ops: GitOps
    config: OrchestratorConfig


@pytest.fixture
def scene(tmp_path: Path) -> _Scene:
    repo = seed_repo(tmp_path / 'repo', _SEED)
    return _Scene(
        git_ops=GitOps(_GIT, repo),
        config=lane_scene_config(repo, _GIT, merge_train_coalesce_enabled=True),
    )


async def _single(scene: _Scene, name: str, **flags: Any) -> MergeRequest:
    """A normal-lane request for a fresh branch that adds one file of its own."""
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
        **flags,
    )


class _ParkedMembers:
    """Train callbacks under which every member reports merge-deferred, and
    which record each member the landed train marks done."""

    def __init__(self) -> None:
        self.marked_done: list[str] = []
        self._marked = asyncio.Event()

    def factory(self, train_id: str) -> TrainCallbacks:
        return TrainCallbacks(
            status_check=self._statuses,
            mark_member_done=self._mark_done,
            redrive_member=AsyncMock(),
        )

    async def _statuses(self, ids: list[str]) -> dict[str, str]:
        return dict.fromkeys(ids, 'merge-deferred')

    async def _mark_done(self, task_id: str, _sha: str) -> None:
        self.marked_done.append(task_id)
        self._marked.set()

    async def until_landed(self, *task_ids: str) -> None:
        while not set(task_ids) <= set(self.marked_done):
            self._marked.clear()
            await self._marked.wait()


def _halted_normal_lane(
    scene: _Scene, requests: list[MergeRequest], **lane_kwargs: Any,
) -> tuple[SpeculativeMergeWorker, asyncio.Queue[Any]]:
    """A lane whose normal lane is halted, with *requests* already enqueued,
    so the merger's first pass buffers them all and can pick none."""
    queue: asyncio.Queue[Any] = asyncio.Queue()
    for request in requests:
        queue.put_nowait(request)
    lane = make_lane(scene.git_ops, queue, **lane_kwargs)
    lane.halt_lane('normal', 'test: only a re-laned request may land')
    return lane, queue


async def _until_buffered(queue: asyncio.Queue[Any]) -> None:
    """Wait for the merger to take everything off the queue into its buffers."""
    while not queue.empty():
        await asyncio.sleep(0.01)


def _queued_lanes(lane: SpeculativeMergeWorker) -> list[tuple[str, str]]:
    return [
        (entry['task_id'], entry['lane'])
        for entry in lane.snapshot()['entries'] if entry['state'] == 'queued'
    ]


async def test_a_relaned_request_lands_ahead_of_the_halted_normal_lane(scene: _Scene) -> None:
    older = await _single(scene, 'older')
    urgent = await _single(scene, 'urgent')
    lane, queue = _halted_normal_lane(scene, [older, urgent])

    async with running_lane(lane) as run:
        await wait_responsive(_until_buffered(queue), label='both requests buffered')

        moved = lane.relane('high', request_id=urgent.request_id)

        assert (moved.relaned, moved.from_lane, moved.lane) == (True, 'normal', 'high')
        assert _queued_lanes(lane)[0] == ('urgent', 'high')
        outcome = await run.outcome(urgent, timeout=_LANDS_WITHIN)
        assert outcome.status == 'done', outcome
        assert not older.result.done()
        lane.unhalt_lane('normal')
        assert (await run.outcome(older, timeout=_LANDS_WITHIN)).status == 'done'


async def test_relaning_a_coalesced_member_moves_its_whole_train(scene: _Scene) -> None:
    members = [await _single(scene, name, awaited_by_workflow=True) for name in ('m-a', 'm-b')]
    hooks = _ParkedMembers()
    lane, _queue = _halted_normal_lane(scene, members, train_callback_factory=hooks.factory)

    async with running_lane(lane) as run:
        absorbed = await run.outcome(members[0], timeout=_LANDS_WITHIN)
        assert absorbed.status == 'superseded', absorbed
        train_id = absorbed.superseded_by

        moved = lane.relane('high', task_id='m-a')

        assert (moved.relaned, moved.via_train) == (True, train_id)
        await wait_responsive(hooks.until_landed('m-a', 'm-b'), label='the re-laned train landed')


async def test_a_member_relaned_while_its_train_forms_takes_the_train_with_it(
    scene: _Scene, monkeypatch: pytest.MonkeyPatch,
) -> None:
    members = [await _single(scene, name, awaited_by_workflow=True) for name in ('r-a', 'r-b')]
    stacking, resume = asyncio.Event(), asyncio.Event()
    stack = scene.git_ops.stack_train_branches

    async def _stack_after_resume(*args: Any, **kwargs: Any) -> Any:
        stacking.set()
        await resume.wait()
        return await stack(*args, **kwargs)

    monkeypatch.setattr(scene.git_ops, 'stack_train_branches', _stack_after_resume)
    hooks = _ParkedMembers()
    lane, _queue = _halted_normal_lane(scene, members, train_callback_factory=hooks.factory)

    async with running_lane(lane) as run:
        await wait_responsive(stacking.wait(), label='the coalescing pass is stacking')

        moved = lane.relane('high', task_id='r-a')
        resume.set()

        assert (moved.relaned, moved.from_lane) == (True, 'normal')
        assert (await run.outcome(members[0], timeout=_LANDS_WITHIN)).status == 'superseded'
        await wait_responsive(
            hooks.until_landed('r-a', 'r-b'), label='the train carrying r-a landed',
        )


async def test_a_request_the_merger_has_taken_is_refused(scene: _Scene) -> None:
    release, entered = asyncio.Event(), asyncio.Event()
    taken = await _single(scene, 'taken')
    queue: asyncio.Queue[Any] = asyncio.Queue()
    lane = make_lane(
        scene.git_ops, queue,
        verifier=FakeVerifier(default=hangs_until(release, entered=entered)),
    )

    async with running_lane(lane) as run:
        await queue.put(taken)
        await wait_responsive(entered.wait(), label='taken is verifying')

        refused = lane.relane('high', task_id='taken')

        assert (refused.relaned, refused.refusal) == (False, 'already_picked')
        assert refused.request_id == taken.request_id
        assert refused.state is not None
        release.set()
        assert (await run.outcome(taken, timeout=_LANDS_WITHIN)).status == 'done'


async def test_an_unknown_request_is_refused_as_not_found(scene: _Scene) -> None:
    lane = make_lane(scene.git_ops)

    refused = lane.relane('high', request_id='mr-nothere')

    assert (refused.relaned, refused.refusal, refused.request_id) == (False, 'not_found', None)
