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
import subprocess
from collections.abc import Callable, Coroutine
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
from _orch_helpers import git_env_with_ceiling, wait_responsive
from df_pytest_isolation import git_redirect_env

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps
from orchestrator.merge_lane import GroupMergeRequest, MergeLane, MergeRequest, QueuedBranch
from orchestrator.merge_lane.types import TrainCallbacks
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


async def _request(scene: _Scene, name: str, *, lane: str = 'normal') -> MergeRequest:
    return MergeRequest(
        task_id=name,
        branch=QueuedBranch.parse(name, _GIT.branch_prefix),
        worktree=await _branch(scene, name),
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=scene.config,
        result=asyncio.get_running_loop().create_future(),
        lane='high' if lane == 'high' else 'normal',
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



def _green() -> VerifyResult:
    return VerifyResult(
        passed=True, test_output='', lint_output='', type_output='', summary='ok',
    )


class _HeldVerify:
    """``run_scoped_verification`` stand-in for the verifies the lane runs
    outside its injected ``VerifyPort`` (a train's, a gate re-verify): counts
    its calls and holds each until released."""

    def __init__(self) -> None:
        self.calls = 0
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def run(self, *_args: Any, **_kwargs: Any) -> VerifyResult:
        self.calls += 1
        self.entered.set()
        await self.release.wait()
        return _green()


async def test_a_train_parked_behind_its_predecessor_does_not_start_its_verify_under_a_halt(
    scene: _Scene,
) -> None:
    """R1: the train's own verify is a verify start the merger makes past the
    pick-time gate; a halt while it is parked sends it back to the queue."""
    release_pred = asyncio.Event()
    verifier = FakeVerifier(scripts={'pred': hangs_until(release_pred)})
    queue: asyncio.Queue[Any] = asyncio.Queue()
    # The production clock: the park polls every 50 ms against a ~2 h
    # deadline, which a fake clock's free sleeps would exhaust at once.
    lane = MergeLane(scene.git_ops, queue, verifier=verifier)
    pred = await _request(scene, 'pred')
    train = await _stacked_train(scene)
    train_verify = _HeldVerify()
    train_verify.release.set()

    with patch(_TRAIN_VERIFY_TARGET, side_effect=train_verify.run):
        async with running_lane(lane) as run:
            await queue.put(pred)
            await wait_responsive(verifier.await_entry(1), label='predecessor verify under way')
            await queue.put(train)
            await wait_responsive(
                _until(lambda: _state(lane, train) == 'merging'),
                label='train dequeued and parked behind its predecessor',
            )

            lane.halt_admission('restart drain test')
            await wait_responsive(
                _until(lambda: _state(lane, train) == 'queued'),
                label='train requeued instead of verifying',
            )
            assert train_verify.calls == 0
            in_flight = _drain_view(lane)['verifies_in_flight']
            assert [(v['task_id'], v['kind']) for v in in_flight] == [('pred', 'verify')]

            release_pred.set()
            assert (await run.outcome(pred)).status == 'done'
            assert train_verify.calls == 0

            lane.resume_admission()
            outcome = await run.outcome(train)

    assert outcome.status == 'done', outcome
    assert train_verify.calls == 1


def _git(cwd: Path, *args: str) -> str:
    env = git_env_with_ceiling(cwd)
    for key in git_redirect_env(env):
        del env[key]
    return subprocess.run(
        ['git', *args], cwd=cwd, env=env, check=True, capture_output=True, text=True,
    ).stdout.strip()


async def test_a_gate_reverify_deadline_runs_from_the_reverify_start(tmp_path: Path) -> None:
    """R2: a main that moves under a head's verify costs a gate re-verify, a
    whole new verify; its deadline is measured from ITS start, not the first
    verify's dispatch."""
    repo = seed_repo(tmp_path / 'repo', _SEED)
    scene = _Scene(
        repo=repo, git_ops=GitOps(_GIT, repo),
        config=lane_scene_config(repo, _GIT, merge_verify_breadth='full'),
    )
    release_first = asyncio.Event()
    clock = _writing_clock()
    verifier = FakeVerifier(scripts={'head': hangs_until(release_first)})
    queue: asyncio.Queue[Any] = asyncio.Queue()
    lane = make_lane(scene.git_ops, queue, verifier=verifier, clock=clock)
    head = await _request(scene, 'head')
    budget = merge_verify_command_budget_secs(scene.config, [])
    reverify = _HeldVerify()

    with patch(_TRAIN_VERIFY_TARGET, side_effect=reverify.run):
        async with running_lane(lane) as run:
            await queue.put(head)
            await wait_responsive(verifier.await_entry(1), label='head verify under way')
            (first,) = _drain_view(lane)['verifies_in_flight']
            (repo / 'drift.txt').write_text('moved under the verify\n')
            _git(repo, 'add', '--', 'drift.txt')
            _git(repo, 'commit', '-m', 'drift: direct commit')
            clock.time += 5_000.0

            release_first.set()
            await wait_responsive(reverify.entered.wait(), label='gate re-verify under way')
            (gate,) = _drain_view(lane)['verifies_in_flight']

            reverify.release.set()
            outcome = await run.outcome(head)

    assert first['kind'] == 'verify'
    assert gate['kind'] == 'gate_reverify'
    assert gate['started_ts'] >= first['started_ts'] + 5_000.0
    assert gate['deadline_ts'] - gate['started_ts'] == budget
    assert outcome.status == 'done', outcome
    assert reverify.calls == 1


# ─── R6: a requeued predecessor is not waited for ───────────────────────────


async def test_a_train_does_not_park_on_a_predecessor_the_halt_sent_back_to_the_queue(
    scene: _Scene,
) -> None:
    """A halt requeues both a parked train T and the predecessor A it waited
    on, T ahead of A. Once admission resumes T must not park on A, which can
    only move after T: that would stall the merger for the whole park."""
    release_head = asyncio.Event()
    verifier = FakeVerifier(scripts={'head': hangs_until(release_head)})
    queue: asyncio.Queue[Any] = asyncio.Queue()
    lane = MergeLane(scene.git_ops, queue, verifier=verifier, speculation_depth=2)
    train = await _stacked_train(scene)
    head = await _request(scene, 'head')
    pred = await _request(scene, 'pred')
    train_verify = _HeldVerify()
    train_verify.release.set()

    with patch(_TRAIN_VERIFY_TARGET, side_effect=train_verify.run):
        async with running_lane(lane) as run:
            await queue.put(head)
            await wait_responsive(verifier.await_entry(1), label='head verify under way')
            await queue.put(pred)
            await wait_responsive(
                _until(lambda: _state(lane, pred) in {'awaiting_verify', 'awaiting_host'}),
                label='predecessor merged, waiting for the one host',
            )
            await queue.put(train)
            await wait_responsive(
                _until(lambda: _state(lane, train) == 'merging'),
                label='train parked behind the predecessor',
            )

            lane.halt_admission('restart drain test')
            release_head.set()
            assert (await run.outcome(head)).status == 'done'
            await wait_responsive(
                _until(lambda: (_state(lane, train), _state(lane, pred)) == ('queued', 'queued')),
                label='train and predecessor both requeued',
            )

            lane.resume_admission()
            train_outcome = await wait_responsive(
                asyncio.shield(train.result), label='train lands without parking on pred',
            )
            pred_outcome = await run.outcome(pred)

    assert train_outcome.status == 'done', train_outcome
    assert pred_outcome.status == 'done', pred_outcome
    assert train_verify.calls == 1


# ─── R7: the drain never strands a coalesce train's members ─────────────────


class _CoalesceHooks:
    """The train callbacks a coalesce train is built with, recording re-drives."""

    def __init__(self) -> None:
        self.redriven: list[tuple[str, bool, str | None]] = []

    def factory(self, train_id: str) -> TrainCallbacks:
        async def _all_deferred(ids: list[str]) -> dict[str, str]:
            return dict.fromkeys(ids, 'merge-deferred')

        return TrainCallbacks(
            status_check=_all_deferred,
            mark_member_done=AsyncMock(),
            redrive_member=self._redrive,
        )

    async def _redrive(self, member: str, on_main: bool, sha: str | None) -> None:
        self.redriven.append((member, on_main, sha))


_SINGLES = ('cs-a', 'cs-b')


@pytest.fixture
def coalesce_scene(tmp_path: Path) -> _Scene:
    repo = seed_repo(tmp_path / 'repo', _SEED)
    return _Scene(
        repo=repo, git_ops=GitOps(_GIT, repo),
        config=lane_scene_config(repo, _GIT, merge_train_coalesce_enabled=True),
    )


def _entries(lane: MergeLane) -> list[tuple[str, str]]:
    return [(e['task_id'], e['state']) for e in lane.snapshot()['entries']]


async def test_a_halt_forms_no_coalesce_train(coalesce_scene: _Scene) -> None:
    """The coalescing pass runs at the top of every merger iteration, the
    first one included; under a halt it forms nothing, so after resume the
    singles merge solo."""
    hooks = _CoalesceHooks()
    queue: asyncio.Queue[Any] = asyncio.Queue()
    lane = make_lane(coalesce_scene.git_ops, queue, train_callback_factory=hooks.factory)
    singles = [await _request(coalesce_scene, name) for name in _SINGLES]
    lane.halt_admission('restart drain test')
    for single in singles:
        await queue.put(single)

    async with running_lane(lane) as run:
        await wait_responsive(
            _until(lambda: lane.unfrozen_suffix() == tuple(r.request_id for r in singles)),
            label='both singles buffered',
        )
        lane.resume_admission()
        outcomes = [await run.outcome(single) for single in singles]

    assert [o.status for o in outcomes] == ['done', 'done'], outcomes
    assert hooks.redriven == []


async def test_a_halt_dissolves_a_coalesce_train_waiting_in_its_buffer(
    coalesce_scene: _Scene,
) -> None:
    hooks = _CoalesceHooks()
    queue: asyncio.Queue[Any] = asyncio.Queue()
    lane = make_lane(coalesce_scene.git_ops, queue, train_callback_factory=hooks.factory)
    singles = [await _request(coalesce_scene, name) for name in _SINGLES]
    lane.halt_lane('normal', 'hold the formed train in its buffer')
    for single in singles:
        await queue.put(single)

    async with running_lane(lane):
        await wait_responsive(
            _until(lambda: all(single.result.done() for single in singles)),
            label='singles coalesced into a buffered train',
        )
        assert _entries(lane) == [(_SINGLES[-1], 'queued')]

        lane.halt_admission('restart drain test')
        await wait_responsive(
            _until(lambda: len(hooks.redriven) == len(_SINGLES)),
            label='train dissolved and its members re-driven',
        )
        assert lane.unfrozen_suffix() == ()
        assert _entries(lane) == []

    assert sorted(hooks.redriven) == [(name, False, None) for name in _SINGLES]


class _WaitWatchingQueue(asyncio.Queue[Any]):
    """The lane's input queue, noting when the merger starts its blocking get.

    The merger calls ``get()`` only to block on the next arrival, right before
    its wait with no await in between, so once ``getting`` is set and the test
    runs again the merger is parked in that wait.
    """

    def __init__(self) -> None:
        super().__init__()
        self.getting = asyncio.Event()

    def get(self) -> Coroutine[Any, Any, Any]:  # type: ignore[override]
        self.getting.set()
        return super().get()


async def test_a_halt_wakes_an_idle_merger_to_dissolve_its_buffered_coalesce_train(
    coalesce_scene: _Scene,
) -> None:
    """R9: the heartbeat halts an IDLE merger, parked waiting for an arrival;
    the halt itself must wake it, or the buffered train survives until the
    next arrival and a restart arriving first strands its members."""
    hooks = _CoalesceHooks()
    queue = _WaitWatchingQueue()
    lane = make_lane(coalesce_scene.git_ops, queue, train_callback_factory=hooks.factory)
    singles = [await _request(coalesce_scene, name) for name in _SINGLES]
    lane.halt_lane('normal', 'hold the formed train in its buffer')
    for single in singles:
        await queue.put(single)

    async with running_lane(lane):
        await wait_responsive(queue.getting.wait(), label='merger parked in its wait')
        assert all(single.result.done() for single in singles)
        assert _entries(lane) == [(_SINGLES[-1], 'queued')]

        lane.halt_admission('restart drain test')
        await wait_responsive(
            _until(lambda: len(hooks.redriven) == len(_SINGLES)),
            label='the halt woke the merger to dissolve the buffered train',
        )
        assert _entries(lane) == []

    assert sorted(hooks.redriven) == [(name, False, None) for name in _SINGLES]


async def test_a_halt_dissolves_a_dequeued_coalesce_train_instead_of_requeuing_it(
    coalesce_scene: _Scene,
) -> None:
    release_head = asyncio.Event()
    verifier = FakeVerifier(scripts={'head': hangs_until(release_head)})
    hooks = _CoalesceHooks()
    queue: asyncio.Queue[Any] = asyncio.Queue()
    # The production clock, as for R1: the train parks behind the head.
    lane = MergeLane(
        coalesce_scene.git_ops, queue, verifier=verifier, train_callback_factory=hooks.factory,
    )
    head = await _request(coalesce_scene, 'head', lane='high')
    singles = [await _request(coalesce_scene, name) for name in _SINGLES]
    train_verify = _HeldVerify()
    train_verify.release.set()

    with patch(_TRAIN_VERIFY_TARGET, side_effect=train_verify.run):
        async with running_lane(lane) as run:
            for request in (head, *singles):
                await queue.put(request)
            await wait_responsive(verifier.await_entry(1), label='head verify under way')
            await wait_responsive(
                _until(lambda: (_SINGLES[-1], 'merging') in _entries(lane)),
                label='coalesce train dequeued and parked behind the head',
            )

            lane.halt_admission('restart drain test')
            await wait_responsive(
                _until(lambda: len(hooks.redriven) == len(_SINGLES)),
                label='dequeued train dissolved and its members re-driven',
            )
            assert _entries(lane) == [('head', 'verifying')]
            assert lane.unfrozen_suffix() == ()

            release_head.set()
            assert (await run.outcome(head)).status == 'done'

    assert sorted(hooks.redriven) == [(name, False, None) for name in _SINGLES]
    assert train_verify.calls == 0
