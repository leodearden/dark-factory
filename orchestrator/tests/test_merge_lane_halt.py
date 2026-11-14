"""The merge lane's halt state (PRD ``plans/merge-lane-quality-prd.md`` task θ).

``HaltState`` is driven only through its public surface: which lanes are
halted, who owns each halt, whether the operator halt is up, and how often
it asks the worker to wake. The lane tuple and the resume callback are
injected, exactly as the worker's composition root hands them over.

Boundary row 7 then drives the same contract through a real ``MergeLane``
on a real repository: an operator halt aborts the in-flight verify and
requeues its merge, and un-halting lands it.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest
from _git_fixtures import seed_repo
from _merge_lane_fakes import (
    FakeVerifier,
    hangs_until,
    healthy_verify_clock,
    lane_scene_config,
    lane_state,
    make_lane,
    passes,
    running_lane,
)
from _orch_helpers import wait_responsive

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.merge_lane import MergeLane, MergeOutcome, MergeRequest, QueuedBranch
from orchestrator.merge_lane.halt import HaltState

LANES = ('high', 'normal')


class ResumeRecorder:
    """The ``on_resume`` callback, counting how often it was asked to wake."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> None:
        self.calls += 1


@pytest.fixture
def resumes() -> ResumeRecorder:
    return ResumeRecorder()


@pytest.fixture
def halt(resumes: ResumeRecorder) -> HaltState:
    return HaltState(LANES, on_resume=resumes)


def halted_lanes(halt: HaltState) -> list[str]:
    return [lane for lane in LANES if halt.is_lane_halted(lane)]


def test_a_fresh_state_halts_nothing(halt: HaltState, resumes: ResumeRecorder) -> None:
    assert halted_lanes(halt) == []
    assert halt.is_wip_halted is False
    assert halt.halt_owner_esc_id is None
    assert halt.is_operator_halted is False
    assert resumes.calls == 0


def test_halting_one_lane_records_its_owner(halt: HaltState) -> None:
    halt.halt_lane('normal', 'wip conflict', owner_esc_id='e1')

    assert halted_lanes(halt) == ['normal']
    assert halt.is_wip_halted is True
    assert halt.lane_owned_by('e1') == 'normal'
    assert halt.is_halt_owner('e1') is True
    assert halt.halt_owner_esc_id == 'e1'
    assert halt.is_operator_halted is False


def test_unhalting_a_lane_clears_halt_and_owner_and_wakes_once(
    halt: HaltState, resumes: ResumeRecorder,
) -> None:
    halt.halt_lane('normal', 'wip conflict', owner_esc_id='e1')

    halt.unhalt_lane('normal', 'resolved')

    assert halted_lanes(halt) == []
    assert halt.lane_owned_by('e1') is None
    assert halt.is_halt_owner('e1') is False
    assert resumes.calls == 1


def test_a_lane_owner_is_never_overwritten_by_a_different_one(halt: HaltState) -> None:
    halt.halt_lane('normal', 'wip conflict')
    halt.set_lane_halt_owner('normal', 'e1')
    halt.set_lane_halt_owner('normal', 'e1')

    with pytest.raises(AssertionError) as refused:
        halt.set_lane_halt_owner('normal', 'e2')

    message = str(refused.value)
    assert "'normal'" in message
    assert "'e1'" in message
    assert "'e2'" in message
    assert halt.lane_owned_by('e1') == 'normal'


def test_the_wip_halt_stops_every_lane_and_takes_one_owner(halt: HaltState) -> None:
    halt.halt_for_wip('wip conflict')

    assert halted_lanes(halt) == list(LANES)
    assert halt.halt_owner_esc_id is None

    halt.set_halt_owner('e2')

    assert halt.halt_owner_esc_id == 'e2'
    assert halt.is_halt_owner('e2') is True
    with pytest.raises(AssertionError, match="already set to 'e2'"):
        halt.set_halt_owner('e2')
    assert halt.unhalt_lanes_owned_by('e2') == list(LANES)


def test_an_owner_needs_a_halt_to_own(halt: HaltState) -> None:
    with pytest.raises(AssertionError, match='no lane is halted'):
        halt.set_halt_owner('e2')


def test_only_the_operator_halt_raises_the_operator_signal(halt: HaltState) -> None:
    halt.halt_for_wip('wip conflict')
    assert halt.is_operator_halted is False

    halt.operator_halt('operator says stop')

    assert halt.is_operator_halted is True
    assert halted_lanes(halt) == list(LANES)
    assert halt.halt_owner_esc_id is None


def test_resume_all_clears_lanes_owners_and_the_operator_halt(
    halt: HaltState, resumes: ResumeRecorder,
) -> None:
    halt.halt_lane('high', 'wip conflict', owner_esc_id='e1')
    halt.operator_halt('operator says stop')

    halt.unhalt_all_lanes('operator resumed')

    assert halted_lanes(halt) == []
    assert halt.lane_owned_by('e1') is None
    assert halt.is_operator_halted is False
    assert halt.is_wip_halted is False
    assert resumes.calls == 1


def test_resuming_by_owner_reports_exactly_the_lanes_it_resumed(
    halt: HaltState, resumes: ResumeRecorder,
) -> None:
    halt.halt_lane('normal', 'wip conflict', owner_esc_id='e1')
    halt.halt_lane('high', 'other conflict', owner_esc_id='e9')

    assert halt.unhalt_lanes_owned_by('e1') == ['normal']
    assert halted_lanes(halt) == ['high']
    assert resumes.calls == 1

    assert halt.unhalt_lanes_owned_by('e1') == []
    assert resumes.calls == 1


def test_shutdown_release_runs_every_lane_but_keeps_owners_and_stays_quiet(
    halt: HaltState, resumes: ResumeRecorder,
) -> None:
    halt.halt_lane('normal', 'wip conflict', owner_esc_id='e1')
    halt.halt_for_wip('wip conflict')

    halt.release_lanes_for_shutdown()

    assert halted_lanes(halt) == []
    assert halt.is_wip_halted is False
    assert resumes.calls == 0


def test_shutdown_release_leaves_a_lane_owner_in_place(halt: HaltState) -> None:
    halt.halt_lane('normal', 'wip conflict', owner_esc_id='e1')

    halt.release_lanes_for_shutdown()

    assert halt.lane_owned_by('e1') == 'normal'


def test_halt_log_lines_are_unchanged(
    halt: HaltState, caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger='orchestrator.merge_lane'):
        halt.halt_lane('normal', 'wip conflict')
        halt.operator_halt('operator says stop')

    warnings = [
        record.getMessage() for record in caplog.records
        if record.name == 'orchestrator.merge_lane' and record.levelno == logging.WARNING
    ]
    assert warnings == [
        "Merge queue lane 'normal' halted: wip conflict",
        'Merge queue operator-halted: operator says stop',
    ]


# ---------------------------------------------------------------------------
# Boundary row 7: the operator halt through a running lane
# ---------------------------------------------------------------------------


@pytest.fixture
def git_config() -> GitConfig:
    return GitConfig(
        main_branch='main', branch_prefix='task/', remote='origin',
        worktree_dir='.worktrees', push_after_advance=False,
    )


@pytest.fixture
def git_ops(git_config: GitConfig, tmp_path: Path) -> GitOps:
    return GitOps(git_config, seed_repo(tmp_path / 'repo'))


@pytest.fixture
def config(git_ops: GitOps, git_config: GitConfig) -> OrchestratorConfig:
    return lane_scene_config(git_ops.project_root, git_config)


async def branch_with_file(git_ops: GitOps, branch: str, filename: str) -> Path:
    worktree = (await git_ops.create_worktree(branch)).path
    (worktree / filename).write_text(f'{branch}\n')
    await git_ops.commit(worktree, f'Add {filename}')
    return worktree


def merge_request(task_id: str, worktree: Path, config: OrchestratorConfig) -> MergeRequest:
    future: asyncio.Future[MergeOutcome] = asyncio.get_running_loop().create_future()
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


async def main_tip(git_ops: GitOps) -> str:
    _, out, _ = await _run(['git', 'rev-parse', 'main'], cwd=git_ops.project_root)
    return out.strip()


async def left_verifying(lane: MergeLane, request_id: str) -> str | None:
    """Wait until the lane stops reporting *request_id* as verifying; return its state."""
    while lane_state(lane, request_id) == 'verifying':
        await asyncio.sleep(0.01)
    return lane_state(lane, request_id)


@pytest.mark.asyncio
async def test_an_operator_halt_requeues_the_verify_and_unhalting_lands_it(
    git_ops: GitOps, config: OrchestratorConfig,
) -> None:
    worktree = await branch_with_file(git_ops, 'row7', 'row7.py')
    first_verify = asyncio.Event()
    verifier = FakeVerifier(sequence=[hangs_until(first_verify), passes()])
    queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
    lane = make_lane(git_ops, queue, verifier=verifier, clock=healthy_verify_clock())
    before = await main_tip(git_ops)

    async with running_lane(lane) as run:
        request = merge_request('row7', worktree, config)
        await queue.put(request)
        await wait_responsive(verifier.await_entry(1), label='first verify entered')
        assert lane_state(lane, request.request_id) == 'verifying'

        lane.operator_halt('row 7')

        assert lane.is_operator_halted is True
        assert lane.is_wip_halted is True
        assert lane.halt_owner_esc_id is None
        requeued = await wait_responsive(
            left_verifying(lane, request.request_id), label='verify aborted by the halt',
        )
        assert requeued == 'queued'
        assert not request.result.done()

        lane.unhalt_wip()

        outcome = await run.outcome(request)
        await wait_responsive(verifier.await_entry(2), label='second verify entered')

    assert outcome.status == 'done', outcome
    assert lane.is_operator_halted is False
    assert await main_tip(git_ops) == outcome.merge_sha != before
    assert verifier.verified == ['row7', 'row7']
