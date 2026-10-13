"""Which waiting singles the retroactive coalescing pass may absorb (task 4953).

A coalesce train lands only once every member is merge-deferred, and only a
TaskWorkflow awaiting its request parks a task there. So the pass absorbs only
a request built with ``awaited_by_workflow=True``. One built without it -- as
the MCP ``merge_request``, the stranded-branch reaper and journal recovery
build theirs -- keeps its own handle and merges solo, whatever an injected
risk predicate says.

Driven through the lane's public surface over a real git repo. Every request
is enqueued before the lane starts, so the pass runs on the merger's first
iteration. Observed through the request futures and the ``train_coalesced``
events only. A train that forms verifies through the suite's autouse stub
(``orchestrator/tests/conftest.py::_mock_merge_queue_verification``), so it lands.
"""

from __future__ import annotations

import asyncio
import dataclasses
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from _git_fixtures import RepoSeed, seed_repo
from _merge_lane_fakes import lane_scene_config, make_lane, running_lane
from _orch_helpers import wait_responsive

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.git_ops import GitOps
from orchestrator.merge_lane import MergeOutcome, MergeRequest, QueuedBranch
from orchestrator.merge_lane.types import MergeReadyPredicate, TrainCallbacks

pytestmark = pytest.mark.asyncio

_GIT = GitConfig(
    main_branch='main', branch_prefix='task/', worktree_dir='.worktrees',
    push_after_advance=False,
)
_SEED = RepoSeed(files=(('README.md', '# Test\n'),), message='Initial commit')


@dataclasses.dataclass(frozen=True)
class _Scene:
    git_ops: GitOps
    events: EventStore
    config: OrchestratorConfig


@pytest.fixture
def scene(tmp_path: Path) -> _Scene:
    repo = seed_repo(tmp_path / 'repo', _SEED)
    return _Scene(
        git_ops=GitOps(_GIT, repo),
        events=EventStore(db_path=tmp_path / 'events.db', run_id='coalesce-admission'),
        config=lane_scene_config(repo, _GIT, merge_train_coalesce_enabled=True),
    )


async def _single(scene: _Scene, name: str, **flags: bool) -> MergeRequest:
    """A single request for a fresh branch that adds one file of its own."""
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


class _TrainHooks:
    """Train callbacks: the *parked* tasks report merge-deferred, as a workflow
    that received 'superseded' leaves them; every other task is still in progress."""

    def __init__(self, parked: set[str]) -> None:
        self._parked = parked
        self.marked_done: list[str] = []
        self._all_marked = asyncio.Event()

    def factory(self, train_id: str) -> TrainCallbacks:
        return TrainCallbacks(
            status_check=self._statuses,
            mark_member_done=self._mark_done,
            redrive_member=AsyncMock(),
        )

    async def _statuses(self, ids: list[str]) -> dict[str, str]:
        return {i: 'merge-deferred' if i in self._parked else 'in-progress' for i in ids}

    async def _mark_done(self, task_id: str, _sha: str) -> None:
        self.marked_done.append(task_id)
        if set(self.marked_done) >= self._parked:
            self._all_marked.set()

    async def wait_landed(self) -> None:
        """Wait for every parked task to be marked done; with none parked, no
        train can land, so there is nothing to wait for."""
        if self._parked:
            await wait_responsive(self._all_marked.wait(), label='coalesce train landed')


def _coalesced(scene: _Scene) -> list[dict[str, Any]]:
    return [event['data'] for event in scene.events.fetch_events_by_type(EventType.train_coalesced)]


async def _merge_all(
    scene: _Scene, requests: list[MergeRequest], hooks: _TrainHooks,
    predicate: MergeReadyPredicate | None = None,
) -> list[MergeOutcome]:
    queue: asyncio.Queue[Any] = asyncio.Queue()
    for request in requests:
        queue.put_nowait(request)
    lane = make_lane(
        scene.git_ops, queue,
        event_store=scene.events,
        train_callback_factory=hooks.factory,
        merge_ready_predicate=predicate,
    )
    async with running_lane(lane) as run:
        outcomes = [await run.outcome(request) for request in requests]
        await hooks.wait_landed()
    return outcomes


async def test_workflow_less_singles_form_no_train(scene: _Scene) -> None:
    singles = [await _single(scene, name) for name in ('mcp-a', 'mcp-b')]

    outcomes = await _merge_all(scene, singles, _TrainHooks(parked=set()))

    assert [o.status for o in outcomes] == ['done', 'done'], outcomes
    assert _coalesced(scene) == []


async def test_workflow_awaited_singles_coalesce_into_one_train(scene: _Scene) -> None:
    names = ('wf-a', 'wf-b')
    singles = [await _single(scene, name, awaited_by_workflow=True) for name in names]
    hooks = _TrainHooks(parked=set(names))

    outcomes = await _merge_all(scene, singles, hooks)

    (train,) = _coalesced(scene)
    assert train['member_task_ids'] == list(names)
    assert [(o.status, o.superseded_by) for o in outcomes] == [
        ('superseded', train['train_id']), ('superseded', train['train_id']),
    ]
    assert sorted(hooks.marked_done) == sorted(names)


async def test_a_workflow_less_single_between_two_awaited_ones_is_excluded(
    scene: _Scene,
) -> None:
    first = await _single(scene, 'wf-a', awaited_by_workflow=True)
    mcp = await _single(scene, 'mcp-m')
    last = await _single(scene, 'wf-b', awaited_by_workflow=True)
    hooks = _TrainHooks(parked={'wf-a', 'wf-b'})

    first_outcome, mcp_outcome, last_outcome = await _merge_all(
        scene, [first, mcp, last], hooks,
    )

    (train,) = _coalesced(scene)
    assert train['member_task_ids'] == ['wf-a', 'wf-b']
    assert {'request_id': mcp.request_id, 'reason': 'no_awaiting_workflow'} in train['exclusions']
    assert mcp_outcome.status == 'done', mcp_outcome
    assert (first_outcome.status, last_outcome.status) == ('superseded', 'superseded')


async def test_an_injected_predicate_cannot_readmit_workflow_less_singles(
    scene: _Scene,
) -> None:
    singles = [await _single(scene, name) for name in ('mcp-a', 'mcp-b')]

    outcomes = await _merge_all(
        scene, singles, _TrainHooks(parked=set()), predicate=lambda _req: None,
    )

    assert [o.status for o in outcomes] == ['done', 'done'], outcomes
    assert _coalesced(scene) == []
