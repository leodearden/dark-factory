"""A fresh merge built while the queue head verifies its own merge stacks on
the frozen-prefix tip, so its verify never starts on main at depth >= 1
(task 10851).

Live merger and verifier loops over a real tmp git repo. Head H verifies on
the remote 'head-host'; the request under test verifies on 'local' while H
still does. Every observation is public: request futures, ``snapshot()``,
the recorded escalations and the worktrees the local host was handed.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any

import pytest
from _git_fixtures import seed_repo
from _merge_lane_fakes import (
    FakeRemoteRunner,
    FakeVerifier,
    RecordingEscalations,
    hangs_until,
    healthy_verify_clock,
    lane_scene_config,
    make_lane,
    raises,
    running_lane,
)
from _orch_helpers import MERGE_GATE_BARRIER_TIMEOUT, wait_responsive

from orchestrator.config import GitConfig, MergeDeepConfig, OrchestratorConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.git_ops import GitOps, MergeVerifyLeaseHeld
from orchestrator.merge_lane import MergeLane, MergeRequest, QueuedBranch
from orchestrator.merge_lane.verify_base_detector import VERIFY_BASE_DIVERGENCE_SENTINEL

pytestmark = pytest.mark.asyncio

_GIT = GitConfig(
    main_branch='main', branch_prefix='task/', worktree_dir='.worktrees',
    push_after_advance=False,
)
_HEAD_HOST = 'head-host'

Tree = frozenset[str]


@dataclasses.dataclass(frozen=True)
class _Scene:
    git_ops: GitOps
    config: OrchestratorConfig
    events: EventStore
    escalations: RecordingEscalations

    @property
    def repo(self) -> Path:
        return self.git_ops.project_root


def _scene(tmp_path: Path) -> _Scene:
    repo = seed_repo(tmp_path / 'repo')
    config = lane_scene_config(
        repo, _GIT, verify_host_policy='prefer_remote', verify_cross_check_remote_green=False,
        merge_deep=MergeDeepConfig(chain_cap=0),
    )
    assert config.merge_deep.chain_cap == 0, (
        f'the operational yaml switched deep chains on in the scene: {config.merge_deep!r}'
    )
    return _Scene(
        git_ops=GitOps(_GIT, repo),
        config=config,
        events=EventStore(db_path=tmp_path / 'events.db', run_id='fresh-merge-on-live-tip'),
        escalations=RecordingEscalations(),
    )


async def _request(scene: _Scene, name: str, files: Mapping[str, str]) -> MergeRequest:
    """The request merging task *name*'s branch, one commit writing *files*."""
    worktree = (await scene.git_ops.create_worktree(name)).path
    for filename, content in files.items():
        (worktree / filename).write_text(content)
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
        lane='normal',
    )


def _task_files(names: list[str]) -> Tree:
    return frozenset(name for name in names if name.endswith('.txt'))


async def _main_task_files(scene: _Scene) -> Tree:
    """The task files (``*.txt``) in main's tree."""
    proc = await asyncio.create_subprocess_exec(
        'git', 'ls-tree', '--name-only', 'main', cwd=scene.repo,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
    )
    stdout, _ = await proc.communicate()
    assert proc.returncode == 0, f'git ls-tree main exited {proc.returncode}'
    return _task_files(stdout.decode().split())


async def _entered(event: asyncio.Event, label: str) -> None:
    await wait_responsive(event.wait(), timeout=MERGE_GATE_BARRIER_TIMEOUT, label=label)


async def _status(req: MergeRequest) -> str:
    outcome = await wait_responsive(req.result, label=f'{req.task_id} outcome')
    return outcome.status


class _SignalsDequeueOf(EventStore):
    """An event store that sets ``dequeued`` on *task_id*'s ``merge_dequeued``
    row, which the merger writes once its fresh-dequeue decision is made."""

    def __init__(self, db_path: Path, run_id: str, *, task_id: str) -> None:
        super().__init__(db_path, run_id)
        self._task_id = task_id
        self.dequeued = asyncio.Event()

    def emit(self, event_type: EventType, **fields: Any) -> int | None:
        if event_type is EventType.merge_dequeued and fields.get('task_id') == self._task_id:
            self.dequeued.set()
        return super().emit(event_type, **fields)


@dataclasses.dataclass
class _Run:
    """A running lane: H's verify on 'head-host' is held until ``release_h``."""

    scene: _Scene
    lane: MergeLane
    queue: asyncio.Queue[Any]
    local: FakeVerifier
    release_h: asyncio.Event

    def put(self, req: MergeRequest) -> None:
        self.queue.put_nowait(req)

    def tree_at_call(self, i: int) -> Tree:
        """The task files in the worktree the *i*-th local verify was handed,
        read while that verify still holds it."""
        return _task_files([p.name for p in self.local.verify_calls[i].worktree.glob('*.txt')])

    def divergence_alarms(self) -> list[Any]:
        return [
            esc for esc in self.scene.escalations.filed
            if esc.task_id == VERIFY_BASE_DIVERGENCE_SENTINEL
        ]

    def recorded_divergences(self) -> list[Any]:
        return self.lane.snapshot()['verify_base_divergences']


@contextlib.asynccontextmanager
async def _running(
    scene: _Scene, head: MergeRequest, local: FakeVerifier,
) -> AsyncIterator[_Run]:
    """*head* merged on main and verifying on 'head-host', held there."""
    h_entered, release_h = asyncio.Event(), asyncio.Event()
    queue: asyncio.Queue[Any] = asyncio.Queue()
    lane = make_lane(
        scene.git_ops, queue, verifier=local, clock=healthy_verify_clock(),
        speculation_depth=2,
        remote_runners=[FakeRemoteRunner(_HEAD_HOST, release=release_h, entered=h_entered)],
        event_store=scene.events, escalation_queue=scene.escalations,
    )
    async with running_lane(lane):
        run = _Run(scene, lane, queue, local, release_h)
        run.put(head)
        await _entered(h_entered, f'{head.task_id} verifying on {_HEAD_HOST}')
        yield run


class TestARequeuedRequestRemergesOnTheLiveHead:
    """A request its verify gave back to the queue re-merges on the head still verifying."""

    async def test_a_request_requeued_from_verify_restacks_on_the_live_head(
        self, tmp_path: Path,
    ) -> None:
        """S, attached to H and verifying on local, finds the merge-verify
        lane held and defers, so its request is requeued while H still
        verifies. S's re-merge stacks on H's commit again, not on main."""
        scene = _scene(tmp_path)
        h = await _request(scene, 'H', {'H.txt': 'H\n'})
        s = await _request(scene, 'S', {'S.txt': 'S\n'})
        s_entered, release_s = asyncio.Event(), asyncio.Event()
        s2_entered, release_s2 = asyncio.Event(), asyncio.Event()
        lane_held = raises(MergeVerifyLeaseHeld(scene.repo / '_merge-verify', 4242))
        local = FakeVerifier(sequence=[
            dataclasses.replace(lane_held, release=release_s, entered=s_entered),
            hangs_until(release_s2, entered=s2_entered),
        ])
        async with _running(scene, h, local) as run:
            run.put(s)
            await _entered(s_entered, 'S verifying on local behind H')

            release_s.set()
            await _entered(s2_entered, "S's re-merge verifying on local")
            assert run.tree_at_call(1) == {'H.txt', 'S.txt'}, (
                "S's re-merge did not stack on H's live commit"
            )
            assert run.divergence_alarms() == []
            assert run.recorded_divergences() == []

            run.release_h.set()
            release_s2.set()
            statuses = {req.task_id: await _status(req) for req in (h, s)}
            assert statuses == {'H': 'done', 'S': 'done'}, statuses
            assert await _main_task_files(scene) == {'H.txt', 'S.txt'}
            assert run.divergence_alarms() == []


class TestAFreshDequeueAfterARequestThatBuiltNothing:
    """A speculative request that consumed the merger's stack tip and built
    nothing leaves the next fresh dequeue stacking on the head still verifying."""

    async def test_a_fresh_arrival_after_an_already_merged_speculative_request_stacks_on_the_live_head(
        self, tmp_path: Path,
    ) -> None:
        """Y, a branch with no work of its own, attaches to H's commit and
        is found already merged, so it builds nothing. X, dequeued after it
        while H still verifies, stacks on H.

        Y is not a conflict with H: the η suffix bounce
        (orchestrator/suffix_graph.py::SuffixConflictTracker.bounce_conflicting_suffix_items)
        escalates a request conflicting with the frozen tip before the merger
        ever dequeues it."""
        events = _SignalsDequeueOf(
            tmp_path / 'events-dequeue.db', 'fresh-after-nothing', task_id='Y',
        )
        scene = dataclasses.replace(_scene(tmp_path), events=events)
        h = await _request(scene, 'H', {'H.txt': 'H\n'})
        y = await _request(scene, 'Y', {})
        x = await _request(scene, 'X', {'X.txt': 'X\n'})
        x_entered, release_x = asyncio.Event(), asyncio.Event()
        local = FakeVerifier(scripts={'X': hangs_until(release_x, entered=x_entered)})
        async with _running(scene, h, local) as run:
            run.put(y)
            await _entered(events.dequeued, 'Y dequeued behind H')

            run.put(x)
            await _entered(x_entered, 'X verifying on local')
            assert run.tree_at_call(0) == {'H.txt', 'X.txt'}, (
                "X did not stack on H's live commit"
            )
            assert run.divergence_alarms() == []
            assert run.recorded_divergences() == []

            run.release_h.set()
            release_x.set()
            statuses = {req.task_id: await _status(req) for req in (h, x, y)}
            assert statuses == {'H': 'done', 'X': 'done', 'Y': 'already_merged'}, statuses
            assert await _main_task_files(scene) == {'H.txt', 'X.txt'}
            assert run.divergence_alarms() == []
