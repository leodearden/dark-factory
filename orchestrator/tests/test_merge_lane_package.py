"""The merge lane's façade and ports (PRD ``plans/merge-lane-quality-prd.md`` tasks ζ1 and β).

The façade is pinned by what it exports and by importing cleanly from
either side of its cycle with ``orchestrator.merge_queue``. The ports are
pinned by driving a real ``MergeLane`` over a real git repository with the
fakes from ``_merge_lane_fakes`` injected: the injected verifier decides
whether a branch lands, and the injected clock is the one the worker ages
its worktrees by. Every observation is a public one -- a ``MergeOutcome``,
``main``'s tip, a reap report -- never a worker attribute.
"""
from __future__ import annotations

import asyncio
import subprocess
import sys
import time
from pathlib import Path

import pytest
from _merge_lane_fakes import FakeClock, FakeVerifier, fails
from _orch_helpers import make_placeholder_future, wait_responsive

import orchestrator.merge_lane as merge_lane
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.merge_lane import (
    DiscardingEscalations,
    EscalationRecord,
    MergeLane,
    MergeOutcome,
    MergeRequest,
    ProductionEscalations,
    QueuedBranch,
)
from orchestrator.merge_lane.ports import escalation_port
from orchestrator.merge_queue import (
    SpeculativeMergeWorker,
    _resolve_dispatch_time_merge_base,
)

# ---------------------------------------------------------------------------
# The façade
# ---------------------------------------------------------------------------

REASON_CONSTANTS = 17


def test_merge_lane_is_the_worker() -> None:
    assert MergeLane is SpeculativeMergeWorker
    assert merge_lane.resolve_dispatch_time_merge_base is _resolve_dispatch_time_merge_base


def test_every_export_resolves_and_the_reason_constants_are_strings() -> None:
    resolved = {name: getattr(merge_lane, name) for name in merge_lane.__all__}
    reasons = [
        name for name in resolved
        if name.endswith('_REASON') or name.endswith('_REASON_PREFIX')
    ]
    assert len(reasons) == REASON_CONSTANTS, reasons
    assert all(isinstance(resolved[name], str) for name in reasons)
    assert set(merge_lane.__all__) <= set(dir(merge_lane))


def test_a_name_outside_the_facade_is_an_attribute_error() -> None:
    assert not hasattr(merge_lane, 'not_an_export')
    with pytest.raises(AttributeError, match='not_an_export'):
        _ = merge_lane.not_an_export


@pytest.mark.parametrize(
    'first, second',
    [
        ('orchestrator.merge_queue', 'orchestrator.merge_lane'),
        ('orchestrator.merge_lane', 'orchestrator.merge_queue'),
    ],
)
def test_facade_imports_cleanly_from_either_side_of_its_cycle(
    first: str, second: str,
) -> None:
    program = (
        f'import {first}; import {second}; '
        'from orchestrator.merge_lane import MergeLane; '
        'from orchestrator.merge_queue import SpeculativeMergeWorker; '
        'assert MergeLane is SpeculativeMergeWorker'
    )
    completed = subprocess.run(
        [sys.executable, '-c', program], capture_output=True, text=True, timeout=120,
    )
    assert completed.returncode == 0, completed.stderr


# ---------------------------------------------------------------------------
# The escalation port
# ---------------------------------------------------------------------------


class _RecordingQueue:
    """The two calls the merge lane makes on an escalation queue."""

    def __init__(self) -> None:
        self.submitted: list = []

    def make_id(self, task_id: str) -> str:
        return f'esc-{task_id}-{len(self.submitted) + 1}'

    def submit(self, escalation) -> None:
        self.submitted.append(escalation)


def _record() -> EscalationRecord:
    return EscalationRecord(
        task_id='loop:merger', agent_role='orchestrator-merge-worker-supervisor',
        severity='blocking', level=1, category='infra_issue',
        summary='merge_worker_loop_died: merger', detail='trace', suggested_action='restart',
    )


def test_production_escalations_file_through_the_queue() -> None:
    queue = _RecordingQueue()
    filed = ProductionEscalations(queue).file(_record())
    assert filed == 'esc-loop:merger-1'
    (escalation,) = queue.submitted
    assert (escalation.id, escalation.task_id, escalation.level, escalation.summary) == (
        'esc-loop:merger-1', 'loop:merger', 1, 'merge_worker_loop_died: merger',
    )


def test_a_worker_without_a_queue_discards() -> None:
    assert DiscardingEscalations().file(_record()) is None
    assert isinstance(escalation_port(None), DiscardingEscalations)
    assert isinstance(escalation_port(_RecordingQueue()), ProductionEscalations)


# ---------------------------------------------------------------------------
# The verify and clock ports, through a real MergeLane on a real repository
# ---------------------------------------------------------------------------


async def _init_repo(repo: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def git_config() -> GitConfig:
    return GitConfig(
        main_branch='main', branch_prefix='task/', remote='origin',
        worktree_dir='.worktrees', push_after_advance=False,
    )


@pytest.fixture
def git_ops(git_config: GitConfig, tmp_path: Path) -> GitOps:
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_init_repo(repo))
    return GitOps(git_config, repo)


@pytest.fixture
def config(git_ops: GitOps, git_config: GitConfig) -> OrchestratorConfig:
    return OrchestratorConfig(project_root=git_ops.project_root, git=git_config)


async def _branch_with_file(git_ops: GitOps, branch: str, filename: str) -> Path:
    worktree = (await git_ops.create_worktree(branch)).path
    (worktree / filename).write_text(f'{branch}\n')
    await git_ops.commit(worktree, f'Add {filename}')
    return worktree


def _request(task_id: str, worktree: Path, config: OrchestratorConfig) -> MergeRequest:
    try:
        future: asyncio.Future[MergeOutcome] = asyncio.get_running_loop().create_future()
    except RuntimeError:
        future = make_placeholder_future()
    return MergeRequest(
        task_id=task_id,
        branch=QueuedBranch.parse(task_id, config.git.branch_prefix),
        worktree=worktree,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=future,
    )


async def _main_tip(git_ops: GitOps) -> str:
    _, out, _ = await _run(['git', 'rev-parse', 'main'], cwd=git_ops.project_root)
    return out.strip()


@pytest.mark.asyncio
async def test_the_injected_verifier_decides_whether_a_branch_lands(
    git_ops: GitOps, config: OrchestratorConfig,
) -> None:
    red_wt = await _branch_with_file(git_ops, 'red', 'red.py')
    green_wt = await _branch_with_file(git_ops, 'green', 'green.py')
    verifier = FakeVerifier(scripts={
        'red': fails(category='test_failure', summary='fake red: 1 test failed'),
    })
    queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
    worker = MergeLane(git_ops, queue, verifier=verifier)
    worker_task = asyncio.create_task(worker.run())
    try:
        before = await _main_tip(git_ops)
        red = _request('red', red_wt, config)
        await queue.put(red)
        red_outcome = await wait_responsive(red.result, label='red merge outcome')
        assert red_outcome.status == 'blocked', red_outcome
        assert 'fake red: 1 test failed' in red_outcome.reason
        assert await _main_tip(git_ops) == before

        green = _request('green', green_wt, config)
        await queue.put(green)
        green_outcome = await wait_responsive(green.result, label='green merge outcome')
        assert green_outcome.status == 'done', green_outcome
        assert await _main_tip(git_ops) == green_outcome.merge_sha != before
        assert verifier.verified == ['red', 'green']
    finally:
        await worker.stop()
        await worker_task


@pytest.mark.asyncio
async def test_the_injected_clock_is_what_the_worker_ages_worktrees_by(
    git_ops: GitOps,
) -> None:
    stray = git_ops.worktree_base / '_merge-stray'
    stray.mkdir(parents=True)
    queue: asyncio.Queue[MergeRequest] = asyncio.Queue()

    today = FakeClock(time=time.time())
    report = await MergeLane(git_ops, queue, clock=today).reap_orphaned_merge_worktrees()
    assert report['reaped'] == [] and stray.is_dir()

    next_week = FakeClock(time=time.time() + 7 * 24 * 3600)
    report = await MergeLane(git_ops, queue, clock=next_week).reap_orphaned_merge_worktrees()
    assert report['reaped'] == [str(stray)]
    assert not stray.exists()
