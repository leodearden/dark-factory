"""The merge lane's half of the restart drain (task 5371).

A drained restart halts merge ADMISSION -- nothing new is merged or verified --
while everything already in flight runs to its verdict, and reports what is
in flight so the restart can wait for exactly that.  Driven through the lane's
public surface (``halt_admission``/``resume_admission``/``snapshot``) over a
real git repo, with the verify scripted by ``FakeVerifier``.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from _git_fixtures import RepoSeed, seed_repo
from _merge_lane_fakes import (
    FakeClock,
    FakeVerifier,
    hangs_until,
    lane_scene_config,
    make_lane,
    running_lane,
)
from _orch_helpers import wait_responsive

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps
from orchestrator.merge_lane import GroupMergeRequest, MergeLane, MergeRequest, QueuedBranch
from orchestrator.verify import VerifyResult, merge_verify_command_budget_secs

pytestmark = pytest.mark.asyncio

_GIT = GitConfig(
    main_branch='main', branch_prefix='task/', worktree_dir='.worktrees',
    push_after_advance=False,
)
_SEED = RepoSeed(files=(('README.md', '# Test\n'),), message='Initial commit')
_TRAIN_VERIFY_TARGET = 'orchestrator.merge_lane.worker.run_scoped_verification'


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


async def _request(scene: _Scene, name: str) -> MergeRequest:
    return MergeRequest(
        task_id=name,
        branch=QueuedBranch.parse(name, _GIT.branch_prefix),
        worktree=await _branch(scene, name),
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=scene.config,
        result=asyncio.get_running_loop().create_future(),
    )


def _writing_clock() -> FakeClock:
    """A fake clock on which a held verify keeps writing, so the lane's
    no-progress watchdog never mistakes it for a dead one."""
    return FakeClock(content_mtime=1.0, content_tick=1.0)


async def _until(predicate: Callable[[], bool]) -> None:
    while not predicate():
        await asyncio.sleep(0)


def _drain_view(lane: MergeLane) -> dict[str, Any]:
    return lane.snapshot()['restart_drain']


def _assert_one_in_flight(
    records: list[dict[str, Any]], *, task_id: str, host: str, kind: str,
    budget: float, not_after: float,
) -> None:
    """One record, for *task_id*, started by *not_after*, due exactly *budget* later."""
    (record,) = records
    assert (record['task_id'], record['host'], record['kind']) == (task_id, host, kind)
    assert record['started_ts'] <= not_after
    assert record['deadline_ts'] - record['started_ts'] == budget


def _state(lane: MergeLane, request: MergeRequest) -> str | None:
    for entry in lane.snapshot()['entries']:
        if entry['request_id'] == request.request_id:
            return entry['state']
    return None


async def test_a_halt_starts_nothing_new_and_lets_the_inflight_verify_land(
    scene: _Scene,
) -> None:
    release_first = asyncio.Event()
    clock = _writing_clock()
    verifier = FakeVerifier(scripts={'first': hangs_until(release_first)})
    queue: asyncio.Queue[Any] = asyncio.Queue()
    lane = make_lane(scene.git_ops, queue, verifier=verifier, clock=clock)
    first = await _request(scene, 'first')
    second = await _request(scene, 'second')

    async with running_lane(lane) as run:
        await queue.put(first)
        await wait_responsive(verifier.await_entry(1), label='first verify under way')

        lane.halt_admission('restart drain test')
        await queue.put(second)
        await wait_responsive(
            _until(lambda: second.request_id in lane.unfrozen_suffix()),
            label='second request buffered behind the admission halt',
        )
        view = _drain_view(lane)
        assert view['admission_halted'] is True
        assert view['admission_halt_reason'] == 'restart drain test'
        _assert_one_in_flight(
            view['verifies_in_flight'], task_id='first', host='local', kind='verify',
            budget=merge_verify_command_budget_secs(scene.config, []), not_after=clock.time,
        )
        assert lane.snapshot()['is_wip_halted'] is False

        release_first.set()
        first_outcome = await run.outcome(first)
        assert first_outcome.status == 'done', first_outcome
        await _until(lambda: _drain_view(lane)['verifies_in_flight'] == [])
        for _ in range(50):
            await asyncio.sleep(0)
        assert verifier.verified == ['first']
        assert _state(lane, second) == 'queued'

        lane.resume_admission()
        second_outcome = await run.outcome(second)

    assert second_outcome.status == 'done', second_outcome
    assert verifier.verified == ['first', 'second']
    assert _drain_view(lane)['admission_halted'] is False


async def test_a_merged_item_reaching_dispatch_under_a_halt_is_requeued_unverified(
    scene: _Scene,
) -> None:
    """A speculative merge built before the halt waits for the one host; at
    dispatch it goes back to the queue instead of starting its verify."""
    release_first = asyncio.Event()
    verifier = FakeVerifier(scripts={'first': hangs_until(release_first)})
    queue: asyncio.Queue[Any] = asyncio.Queue()
    lane = make_lane(
        scene.git_ops, queue, verifier=verifier, clock=_writing_clock(), speculation_depth=2,
    )
    first = await _request(scene, 'first')
    second = await _request(scene, 'second')

    async with running_lane(lane) as run:
        await queue.put(first)
        await wait_responsive(verifier.await_entry(1), label='first verify under way')
        await queue.put(second)
        await wait_responsive(
            _until(lambda: _state(lane, second) in {'awaiting_verify', 'awaiting_host'}),
            label='second merged speculatively, waiting for the host',
        )

        lane.halt_admission('restart drain test')
        release_first.set()
        assert (await run.outcome(first)).status == 'done'
        await wait_responsive(
            _until(lambda: _state(lane, second) == 'queued'),
            label='second requeued at dispatch',
        )
        assert verifier.verified == ['first']

        lane.resume_admission()
        second_outcome = await run.outcome(second)

    assert second_outcome.status == 'done', second_outcome
    assert verifier.verified == ['first', 'second']


async def test_halt_and_resume_are_idempotent_and_leave_lane_halts_alone(
    scene: _Scene,
) -> None:
    lane = make_lane(scene.git_ops)

    lane.resume_admission()
    lane.halt_admission('first reason')
    lane.halt_admission('second reason')
    assert lane.is_admission_halted is True
    assert lane.snapshot()['restart_drain']['admission_halt_reason'] == 'second reason'
    assert lane.is_wip_halted is False
    assert lane.halt_owner_esc_id is None

    lane.resume_admission()
    lane.resume_admission()
    assert lane.snapshot()['restart_drain'] == {
        'admission_halted': False, 'admission_halt_reason': None, 'verifies_in_flight': [],
    }


# ─── A train verifies inline in the merger, with no in-flight entry ─────────

_MEMBERS = ('tr-a', 'tr-b')


async def _stacked_train(scene: _Scene) -> GroupMergeRequest:
    worktrees = [await _branch(scene, member) for member in _MEMBERS]
    stacked = await scene.git_ops.stack_train_branches(list(_MEMBERS))
    assert stacked.survivors == list(_MEMBERS), stacked
    tip = QueuedBranch.parse(_MEMBERS[-1], _GIT.branch_prefix)
    return GroupMergeRequest(
        task_id=_MEMBERS[-1],
        branch=tip,
        worktree=worktrees[-1],
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=scene.config,
        result=asyncio.get_running_loop().create_future(),
        train_id='train-5371',
        member_task_ids=list(_MEMBERS),
        tip_branch=tip,
        tip_task_id=_MEMBERS[-1],
        status_check=AsyncMock(return_value=dict.fromkeys(_MEMBERS, 'merge-deferred')),
        mark_member_done=AsyncMock(),
        redrive_member=AsyncMock(),
    )


async def test_a_train_verifying_in_the_merger_is_in_flight(scene: _Scene) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def _held_verify(*_args: Any, **_kwargs: Any) -> VerifyResult:
        entered.set()
        await release.wait()
        return VerifyResult(
            passed=True, test_output='', lint_output='', type_output='', summary='ok',
        )

    clock = _writing_clock()
    queue: asyncio.Queue[Any] = asyncio.Queue()
    lane = make_lane(scene.git_ops, queue, clock=clock)
    train = await _stacked_train(scene)

    with patch(_TRAIN_VERIFY_TARGET, side_effect=_held_verify):
        async with running_lane(lane) as run:
            await queue.put(train)
            await wait_responsive(entered.wait(), label='train verify under way')

            lane.halt_admission('restart drain test')
            in_flight = _drain_view(lane)['verifies_in_flight']
            captured_at = clock.time
            occupancy = lane.snapshot()['occupancy']

            release.set()
            outcome = await run.outcome(train)

    _assert_one_in_flight(
        in_flight, task_id=_MEMBERS[-1], host='local', kind='train',
        budget=merge_verify_command_budget_secs(scene.config, []), not_after=captured_at,
    )
    assert occupancy['inflight_total'] == 0
    assert outcome.status == 'done', outcome
    assert _drain_view(lane)['verifies_in_flight'] == []


async def test_the_budget_is_the_longest_merge_command_timeout(tmp_path: Path) -> None:
    config = lane_scene_config(
        tmp_path, _GIT,
        verify_command_timeout_secs=7200.0,
        merge_verify_cold_command_timeout_secs=10800.0,
    )
    assert merge_verify_command_budget_secs(config, []) == 10800.0
    warm_heavy = lane_scene_config(
        tmp_path, _GIT,
        verify_command_timeout_secs=9000.0,
        merge_verify_cold_command_timeout_secs=3600.0,
    )
    assert merge_verify_command_budget_secs(warm_heavy, []) == 9000.0

