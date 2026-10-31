"""A vacated in-flight entry is invisible to the lane's census (task 4582),
driven through public submission over a real tmp git repo (task 6342).
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest
from _git_fixtures import RepoSeed, seed_repo
from _merge_lane_fakes import (
    FakeRemoteRunner,
    FakeVerifier,
    VerifyScript,
    hangs_until,
    healthy_verify_clock,
    lane_scene_config,
    lane_state,
    make_lane,
    raises_after,
    running_lane,
)
from _orch_helpers import wait_responsive

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, MergeVerifyLeaseContended
from orchestrator.merge_lane import MergeLane, MergeRequest, QueuedBranch

pytestmark = pytest.mark.asyncio

_GIT = GitConfig(
    main_branch='main', branch_prefix='task/', worktree_dir='.worktrees',
    push_after_advance=False,
)
_SEED = RepoSeed(files=(('README.md', '# Test\n'),), message='Initial commit')
_HEAD_HOST = 'head-host'
_REQUEUE_TRIGGERS = ('lane_contended', 'no_progress')


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
        await asyncio.sleep(0.005)


@dataclasses.dataclass(frozen=True)
class _VacatedScene:
    """A running lane whose head H still verifies, after R's verify exited."""

    lane: MergeLane
    head: MergeRequest
    req: MergeRequest
    main_sha: str


def _r_script(trigger: str, release: asyncio.Event, entered: asyncio.Event) -> VerifyScript:
    if trigger == 'lane_contended':
        contended = MergeVerifyLeaseContended(Path('/x/_merge-verify.lock'), 300.0)
        return raises_after(release, contended, entered=entered)
    return hangs_until(release, entered=entered)


@contextlib.asynccontextmanager
async def _vacated_scene(scene: _Scene, trigger: str) -> AsyncIterator[_VacatedScene]:
    """Run a lane until R's verify has exited through *trigger*.

    'lane_contended' and 'no_progress' requeue R, and ``halt_admission()``
    (taken before R's verify exits) holds it in its lane buffer; 'abandon'
    cancels R's outcome, and the lane drops it.
    """
    head = await _request(scene, 'head')
    req = await _request(scene, 'req')
    main_sha = await scene.git_ops.get_main_sha()
    clock = healthy_verify_clock()
    head_entered = asyncio.Event()
    remote = FakeRemoteRunner(_HEAD_HOST, release=asyncio.Event(), entered=head_entered)
    r_release = asyncio.Event()
    r_entered = asyncio.Event()
    verifier = FakeVerifier(scripts={req.task_id: _r_script(trigger, r_release, r_entered)})
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
        if trigger == 'abandon':
            req.result.cancel()
            await wait_responsive(
                _until(lambda: lane_state(lane, req.request_id) is None),
                label='abandoned R dropped',
            )
        else:
            lane.halt_admission('test: hold the requeued request')
            if trigger == 'lane_contended':
                r_release.set()
            else:
                clock.content_tick = 0.0
            await wait_responsive(
                _until(lambda: req.request_id in lane.unfrozen_suffix()),
                label=f'R requeued by {trigger} into its lane buffer',
            )
        yield _VacatedScene(lane=lane, head=head, req=req, main_sha=main_sha)


def _rendered_ids(snap: dict[str, Any]) -> list[str]:
    return [entry['request_id'] for entry in snap['entries']]


def _slot_states(snap: dict[str, Any]) -> dict[str, str]:
    return {host['name']: host['slot_state'] for host in snap['hosts']}


class TestVacatedInflightEntryRendersAtMostOnce:
    """R's vacated entry is invisible to every in-flight reader: a requeued R
    renders exactly once, as queued, and never in the frozen prefix; a
    dropped R does not render at all (task 4582).

    Scene: head H verifies without end on the remote 'head-host'. R, merged
    speculatively behind it (``speculation_depth=2``), verifies on 'local',
    so R's entry sits behind H in the pipeline and cannot reach head of line
    while H verifies. R's verify then exits, leaving its vacated entry there.

    Read through the public seams only: ``snapshot()`` (entries, depth,
    occupancy, verify_in_progress, frozen_prefix, the hosts census),
    ``frozen_prefix()``, ``unfrozen_suffix()`` and
    ``check_frozen_prefix_invariant()``.
    """

    @pytest.mark.parametrize('trigger', _REQUEUE_TRIGGERS)
    async def test_requeued_request_renders_once_as_queued(
        self, scene: _Scene, trigger: str,
    ) -> None:
        async with _vacated_scene(scene, trigger) as vacated:
            lane = vacated.lane
            rid = vacated.req.request_id
            head = vacated.head
            snap = lane.snapshot()
            rendered = _rendered_ids(snap)
            entries = snap['entries']

            assert rendered.count(rid) == 1, (
                f'the requeued request must render exactly once, not also as '
                f'its vacated in-flight entry: {entries!r}'
            )
            assert [e['state'] for e in entries if e['request_id'] == rid] == ['queued'], (
                f'{entries!r}'
            )
            assert rid not in lane.frozen_prefix(), (
                f'a vacated entry is not frozen: {lane.frozen_prefix()!r} {entries!r}'
            )
            assert rid in lane.unfrozen_suffix(), f'{lane.unfrozen_suffix()!r} {entries!r}'
            assert lane.check_frozen_prefix_invariant(vacated.main_sha) == [], (
                f'{lane.check_frozen_prefix_invariant(vacated.main_sha)!r} {entries!r}'
            )
            assert snap['depth'] == 2 == len(set(rendered)), (
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
            assert _slot_states(snap).get('local') == 'free', (
                f'the vacate must hand R\'s local lease back: {snap["hosts"]!r} {entries!r}'
            )

    async def test_dropped_request_does_not_render(self, scene: _Scene) -> None:
        async with _vacated_scene(scene, 'abandon') as vacated:
            lane = vacated.lane
            rid = vacated.req.request_id
            snap = lane.snapshot()
            entries = snap['entries']

            assert _rendered_ids(snap).count(rid) == 0, (
                f'a dropped request must not render as its vacated entry: {entries!r}'
            )
            assert rid not in lane.frozen_prefix(), f'{lane.frozen_prefix()!r} {entries!r}'
            assert snap['depth'] == 1, f'{entries!r}'
            assert snap['occupancy']['inflight_by_host'] == {_HEAD_HOST: [vacated.head.task_id]}, (
                f'{snap["occupancy"]!r} {entries!r}'
            )
            assert lane.check_frozen_prefix_invariant(vacated.main_sha) == [], (
                f'{lane.check_frozen_prefix_invariant(vacated.main_sha)!r} {entries!r}'
            )
