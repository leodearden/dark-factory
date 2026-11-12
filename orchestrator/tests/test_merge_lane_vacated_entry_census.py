"""A vacated in-flight entry is invisible to the lane's census (task 4582),
driven through public submission over a real tmp git repo (task 6342).
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any, Literal, assert_never

import pytest
from _git_fixtures import RepoSeed, seed_repo
from _merge_lane_fakes import (
    FakeRemoteRunner,
    FakeVerifier,
    hangs_until,
    healthy_verify_clock,
    lane_entry,
    lane_scene_config,
    lane_state,
    make_lane,
    running_lane,
    slot_states,
)
from _orch_helpers import wait_responsive

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps
from orchestrator.merge_lane import MergeLane, MergeRequest, QueuedBranch

pytestmark = pytest.mark.asyncio

_GIT = GitConfig(
    main_branch='main', branch_prefix='task/', worktree_dir='.worktrees',
    push_after_advance=False,
)
_SEED = RepoSeed(files=(('README.md', '# Test\n'),), message='Initial commit')
_HEAD_HOST = 'head-host'
_VerifyExit = Literal['no_progress', 'abandon']
_REQUEUE_TRIGGERS: tuple[_VerifyExit, ...] = ('no_progress',)
_POLL_SECS = 0.005
_SETTLE_POLLS = 200


@dataclasses.dataclass(frozen=True)
class _Scene:
    git_ops: GitOps
    config: OrchestratorConfig


@pytest.fixture
def scene(tmp_path: Path) -> _Scene:
    repo = seed_repo(tmp_path / 'repo', _SEED)
    config = lane_scene_config(
        repo, _GIT, verify_host_policy='prefer_remote', verify_cross_check_remote_green=False,
    )
    return _Scene(git_ops=GitOps(_GIT, repo), config=config)


async def _request(scene: _Scene, name: str) -> MergeRequest:
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
        lane='normal',
    )


async def _until(predicate: Callable[[], bool]) -> None:
    while not predicate():
        await asyncio.sleep(_POLL_SECS)


async def _settle(predicate: Callable[[], bool]) -> None:
    """Give the lane up to ``_SETTLE_POLLS`` short turns to make *predicate*
    hold, and return either way: the caller's assertion, not a timeout,
    then reports what is still there.
    """
    for _ in range(_SETTLE_POLLS):
        if predicate():
            return
        await asyncio.sleep(_POLL_SECS)


@dataclasses.dataclass(frozen=True)
class _VacatedScene:
    """A running lane whose head H still verifies, after R's verify exited."""

    lane: MergeLane
    head: MergeRequest
    req: MergeRequest
    main_sha: str


@contextlib.asynccontextmanager
async def _vacated_scene(scene: _Scene, trigger: _VerifyExit) -> AsyncIterator[_VacatedScene]:
    """Run a lane until R's verify has exited through *trigger*.

    'no_progress' requeues R, and ``halt_admission()`` (taken before R's
    verify exits) holds it in its lane buffer; 'abandon' cancels R's outcome,
    and the lane drops it. 'lane_contended' is not driven here, because its
    requeue can land in the merger's post-halt window, where
    worker.py::SpeculativeMergeWorker._acquire_next_request loses it.

    After an abandon the scene waits for R's 'local' lease to come back, read
    from the hosts census rather than from the filter under test, and only
    then lets the retire that follows settle. A vacated entry that still
    renders therefore fails the test's own assertion, which names it, rather
    than timing out this wait.
    """
    head = await _request(scene, 'head')
    req = await _request(scene, 'req')
    main_sha = await scene.git_ops.get_main_sha()
    clock = healthy_verify_clock()
    head_entered = asyncio.Event()
    remote = FakeRemoteRunner(_HEAD_HOST, release=asyncio.Event(), entered=head_entered)
    r_entered = asyncio.Event()
    verifier = FakeVerifier(scripts={req.task_id: hangs_until(asyncio.Event(), entered=r_entered)})
    queue: asyncio.Queue[Any] = asyncio.Queue()
    lane = make_lane(
        scene.git_ops, queue,
        verifier=verifier, clock=clock, speculation_depth=2, remote_runners=[remote],
    )

    async with running_lane(lane):
        await queue.put(head)
        await wait_responsive(head_entered.wait(), label='head verify under way on the remote')
        await queue.put(req)
        await wait_responsive(r_entered.wait(), label='R verify under way on local')
        assert slot_states(lane).get('local') == 'busy', lane.snapshot()['hosts']
        if trigger == 'abandon':
            req.result.cancel()
            await wait_responsive(
                _until(lambda: slot_states(lane).get('local') == 'free'),
                label="abandoned R's verify handed its local lease back",
            )
            await _settle(lambda: lane_state(lane, req.request_id) is None)
        elif trigger == 'no_progress':
            lane.halt_admission('test: hold the requeued request')
            clock.content_tick = 0.0
            await wait_responsive(
                _until(lambda: req.request_id in lane.unfrozen_suffix()),
                label='R requeued by no_progress into its lane buffer',
            )
        else:
            assert_never(trigger)
        yield _VacatedScene(lane=lane, head=head, req=req, main_sha=main_sha)


class TestVacatedInflightEntryRendersAtMostOnce:
    """R's vacated entry is invisible to every in-flight reader: a requeued R
    renders exactly once, as queued, and never in the frozen prefix; a
    dropped R does not render at all (task 4582).

    Scene: head H verifies without end on the remote 'head-host'. R, merged
    speculatively behind it (``speculation_depth=2``), verifies on 'local',
    so R's entry sits behind H in the pipeline and cannot reach head of line
    while H verifies. R's verify then exits, leaving its vacated entry there.

    ``snapshot()`` keeps one row per request_id, the head-of-line copy
    (worker.py::_first_copy_per_request), so a leaked vacated entry never
    shows as a second row. It shows as R's one row carrying the corpse's
    in-flight state instead of 'queued', which is what the requeue test's
    state assertion catches.

    Read through the public seams only: ``snapshot()`` (entries, depth,
    occupancy, verify_in_progress, frozen_prefix, the hosts census),
    ``frozen_prefix()``, ``unfrozen_suffix()`` and
    ``check_frozen_prefix_invariant()``.
    """

    @pytest.mark.parametrize('trigger', _REQUEUE_TRIGGERS)
    async def test_requeued_request_renders_once_as_queued(
        self, scene: _Scene, trigger: _VerifyExit,
    ) -> None:
        async with _vacated_scene(scene, trigger) as vacated:
            lane = vacated.lane
            rid = vacated.req.request_id
            head = vacated.head
            snap = lane.snapshot()
            entries = snap['entries']

            assert [e['state'] for e in entries if e['request_id'] == rid] == ['queued'], (
                f'the requeued request must render once, as its lane-buffer copy, '
                f'not as its vacated in-flight entry: {entries!r}'
            )
            assert rid not in lane.frozen_prefix(), (
                f'a vacated entry is not frozen: {lane.frozen_prefix()!r} {entries!r}'
            )
            assert rid in lane.unfrozen_suffix(), f'{lane.unfrozen_suffix()!r} {entries!r}'
            assert lane.check_frozen_prefix_invariant(vacated.main_sha) == [], (
                f'{lane.check_frozen_prefix_invariant(vacated.main_sha)!r} {entries!r}'
            )
            assert snap['depth'] == 2 == len(entries), (
                f'depth must count the two live requests: depth={snap["depth"]} {entries!r}'
            )
            assert snap['occupancy']['inflight_by_host'] == {_HEAD_HOST: [head.task_id]}, (
                f'{snap["occupancy"]!r} {entries!r}'
            )
            assert snap['occupancy']['inflight_total'] == 1, f'{snap["occupancy"]!r} {entries!r}'
            assert snap['verify_in_progress'] is not None, f'{entries!r}'
            assert snap['verify_in_progress']['task_id'] == head.task_id, (
                f'{snap["verify_in_progress"]!r} {entries!r}'
            )
            assert snap['frozen_prefix']['request_ids'] == [head.request_id], (
                f'{snap["frozen_prefix"]!r} {entries!r}'
            )
            assert slot_states(lane).get('local') == 'free', (
                f'the vacate must hand R\'s local lease back: {snap["hosts"]!r} {entries!r}'
            )

    async def test_dropped_request_does_not_render(self, scene: _Scene) -> None:
        async with _vacated_scene(scene, 'abandon') as vacated:
            lane = vacated.lane
            rid = vacated.req.request_id
            snap = lane.snapshot()
            entries = snap['entries']

            assert lane_entry(lane, rid) is None, (
                f'a dropped request must not render as its vacated entry: {entries!r}'
            )
            assert rid not in lane.frozen_prefix(), f'{lane.frozen_prefix()!r} {entries!r}'
            assert snap['depth'] == 1, f'{entries!r}'
            assert snap['occupancy']['inflight_by_host'] == {_HEAD_HOST: [vacated.head.task_id]}, (
                f'{snap["occupancy"]!r} {entries!r}'
            )
            assert snap['verify_in_progress'] is not None, f'{entries!r}'
            assert snap['verify_in_progress']['task_id'] == vacated.head.task_id, (
                f'{snap["verify_in_progress"]!r} {entries!r}'
            )
            assert slot_states(lane).get('local') == 'free', (
                f'the drop must hand R\'s local lease back: {snap["hosts"]!r} {entries!r}'
            )
            assert lane.check_frozen_prefix_invariant(vacated.main_sha) == [], (
                f'{lane.check_frozen_prefix_invariant(vacated.main_sha)!r} {entries!r}'
            )
