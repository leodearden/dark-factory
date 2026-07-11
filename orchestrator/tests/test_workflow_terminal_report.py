"""Tests for ``TerminalReport``: the typed RETURN value that replaces the
``_last_block_*`` side channel on the workflow↔harness terminal contract
(W9-γ).

PRD ``plans/workflow-state-machine-prd.md`` §5.5 / §8.1 / invariants TR-1 & SM-2.
"""

from __future__ import annotations

import asyncio
import dataclasses
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

# Cross-module reuse — conftest.py injects orchestrator/tests onto sys.path
# (see test_repend_state_machine.py for the same precedent).
#
# ``_derive_meta_root_like_production`` is imported (not just the plain
# helpers) because it is an autouse fixture in test_workflow_e2e.py — autouse
# only auto-applies within a module where pytest can SEE the fixture, and a
# plain `from test_workflow_e2e import AgentStub, ...` does not pull that in.
# Without it, AgentStub's legacy-only TaskArtifacts writes (_architect/
# _implementer) are invisible to the workflow's relocated meta_root, so
# get_pending_steps() never observes completion and any full-execute-loop
# test (e.g. the clean-DONE case below) spuriously blocks with "Execution
# iterations exhausted" instead of reaching DONE.
from test_workflow_e2e import (
    AgentStub,
    _build_workflow,
    _derive_meta_root_like_production,  # noqa: F401  autouse fixture, see above
    _init_repo,
)
from test_workflow_warm_lane_requeue import _make_workflow as _make_warmlane_workflow

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps
from orchestrator.harness import TaskReport
from orchestrator.scheduler import TaskAssignment
from orchestrator.verify import VerifyResult
from orchestrator.verify_categories import FailureCategory
from orchestrator.workflow_types import (
    TerminalReport,
    WorkflowOutcome,
    WorkflowState,
    outcome_allows_status,
)


class TestTerminalReportValueType:
    """Pure-unit tests for the ``TerminalReport`` dataclass — no ``TaskWorkflow``."""

    def test_field_round_trip(self):
        report = TerminalReport(
            outcome=WorkflowOutcome.BLOCKED,
            reason='r',
            phase=WorkflowState.VERIFY,
            detail='d',
            category=None,
        )
        assert report.outcome == WorkflowOutcome.BLOCKED
        assert report.reason == 'r'
        assert report.phase == WorkflowState.VERIFY
        assert report.detail == 'd'
        assert report.category is None

    def test_field_round_trip_with_category(self):
        report = TerminalReport(
            outcome=WorkflowOutcome.BLOCKED,
            reason='r',
            phase=WorkflowState.VERIFY,
            detail='d',
            category=FailureCategory.TEST_FAILURE,
        )
        assert report.category == FailureCategory.TEST_FAILURE

    def test_is_frozen(self):
        """Mutating any field raises ``dataclasses.FrozenInstanceError``."""
        report = TerminalReport(
            outcome=WorkflowOutcome.DONE,
            reason='',
            phase=WorkflowState.DONE,
            detail='',
            category=None,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            report.outcome = WorkflowOutcome.BLOCKED  # type: ignore[misc]
        with pytest.raises(dataclasses.FrozenInstanceError):
            report.reason = 'x'  # type: ignore[misc]
        with pytest.raises(dataclasses.FrozenInstanceError):
            report.phase = WorkflowState.BLOCKED  # type: ignore[misc]
        with pytest.raises(dataclasses.FrozenInstanceError):
            report.detail = 'x'  # type: ignore[misc]
        with pytest.raises(dataclasses.FrozenInstanceError):
            report.category = FailureCategory.TEST_FAILURE  # type: ignore[misc]

    def test_reexported_from_workflow_module(self):
        """``from orchestrator.workflow import TerminalReport`` resolves too —
        mirrors the WorkflowState/WorkflowOutcome re-export shim precedent.
        """
        from orchestrator.workflow import TerminalReport as ShimTerminalReport

        assert ShimTerminalReport is TerminalReport

    def test_blocked_from_phase_round_trips_and_defaults_to_none(self):
        """REVIEW-CYCLE-1 fix: ``blocked_from_phase`` accepts a WorkflowState
        and round-trips; defaults to None when omitted — additive to every
        pre-existing/synthesized construction site (e.g. the four tests
        above, which never pass it)."""
        report = TerminalReport(
            outcome=WorkflowOutcome.BLOCKED,
            reason='r',
            phase=WorkflowState.BLOCKED,
            detail='d',
            category=None,
            blocked_from_phase=WorkflowState.VERIFY,
        )
        assert report.blocked_from_phase == WorkflowState.VERIFY

        report_omitted = TerminalReport(
            outcome=WorkflowOutcome.DONE,
            reason='',
            phase=WorkflowState.DONE,
            detail='',
            category=None,
        )
        assert report_omitted.blocked_from_phase is None


# ---------------------------------------------------------------------------
# Fixtures for the e2e-style block paths (mirrors test_workflow_e2e.py /
# test_repend_state_machine.py verbatim).
# ---------------------------------------------------------------------------


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A bare-minimum git repo with an initial commit (mirrors test_workflow_e2e.py)."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_init_repo(repo))
    return repo


@pytest.fixture
def config(git_repo: Path) -> OrchestratorConfig:
    return OrchestratorConfig(
        project_root=git_repo,
        max_concurrent_tasks=1,
        max_execute_iterations=5,
        max_verify_attempts=3,
        max_review_cycles=2,
        git=GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
        ),
    )


@pytest.fixture
def git_ops(config: OrchestratorConfig) -> GitOps:
    return GitOps(config.git, config.project_root)


@pytest.fixture
def task_assignment() -> TaskAssignment:
    return TaskAssignment(
        task_id='42',
        task={
            'id': '42',
            'title': 'Add farewell function',
            'description': 'Add a farewell(name) function to lib.py with tests',
            'status': 'pending',
            'metadata': {'files': ['lib']},
            'dependencies': [],
        },
        modules=['lib'],
    )


@pytest.mark.asyncio
class TestTerminalReportPopulatedAtBlockPaths:
    """``_mark_blocked`` and the two non-``_mark_blocked`` block paths
    populate an atomic ``self._terminal_report`` (still additive — ``run()``
    is unchanged and still returns ``WorkflowOutcome`` through this step).
    """

    async def test_mark_blocked_populates_terminal_report(
        self, config, git_ops, task_assignment, monkeypatch,
    ):
        """Generic block via the AllAccountsCappedException path (e2e:5638)."""
        from shared.cli_invoke import AllAccountsCappedException

        stub = AgentStub()
        workflow, scheduler = _build_workflow(config, git_ops, task_assignment, stub)

        async def raise_cap_exc(*args, **kwargs):
            raise AllAccountsCappedException(
                retries=3, elapsed_secs=120.0, label='Task 42 [architect]'
            )

        monkeypatch.setattr('orchestrator.workflow.invoke_agent', raise_cap_exc)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(side_effect=AssertionError('run_scoped_verification must not be called')),
        )

        outcome = (await workflow.run()).outcome

        assert outcome == WorkflowOutcome.BLOCKED
        report = workflow._terminal_report
        assert isinstance(report, TerminalReport)
        assert report.outcome == WorkflowOutcome.BLOCKED
        assert report.reason.lower().startswith('all accounts capped')
        assert report.phase == workflow.machine.state
        assert report.detail
        assert report.category is None

    async def test_warm_lane_requeue_populates_terminal_report(self, tmp_path: Path):
        """Warm-lane requeue (reuses test_workflow_warm_lane_requeue setup)."""
        from orchestrator.git_ops import WarmLanePoolHardDown

        wf = _make_warmlane_workflow(tmp_path=tmp_path)
        wf.git_ops.create_worktree = AsyncMock(
            side_effect=WarmLanePoolHardDown(
                "warm-lane base absent (host-scoped pool hard-down) for branch "
                "'1859'; requeue"
            )
        )
        mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
        wf._mark_blocked = mark_blocked  # type: ignore[method-assign]

        outcome = (await wf.run()).outcome

        assert outcome == WorkflowOutcome.REQUEUED
        mark_blocked.assert_not_awaited()
        report = wf._terminal_report
        assert isinstance(report, TerminalReport)
        assert report.outcome == WorkflowOutcome.REQUEUED
        assert report.reason == 'warm_lane_pool_hard_down'
        assert report.phase == wf.machine.state
        assert report.detail
        assert report.category is None


@pytest.mark.asyncio
class TestBlockedFromPhaseField:
    """REVIEW-CYCLE-1 fix (reviewer_comprehensive, correctness_regression):
    ``TerminalReport.blocked_from_phase`` preserves the PRE-block WORKING
    phase (e.g. VERIFY), distinct from ``phase`` (the terminal
    ``machine.state`` kept for SM-2 — BLOCKED after ``_mark_blocked``'s
    ``_enter_phase(BLOCKED)`` transition). Without this field, the harness's
    ``block_phase`` derived solely from ``phase`` collapses to ``'blocked'``
    for every ``_mark_blocked`` exit, silently disabling the optimistic-path
    auto-eval redo (Lever B/C recovery).
    """

    async def test_mark_blocked_captures_pre_block_working_phase(
        self, config, git_ops, task_assignment,
    ):
        """A ``_mark_blocked`` block driven from a WORKING phase (VERIFY)
        stashes that phase in ``blocked_from_phase`` while ``phase`` itself
        becomes the terminal BLOCKED state (SM-2's
        ``report.phase == machine.state`` invariant)."""
        stub = AgentStub()
        workflow, scheduler = _build_workflow(config, git_ops, task_assignment, stub)

        workflow.state = WorkflowState.VERIFY
        outcome = await workflow._mark_blocked('verify failed', detail='d')

        assert outcome == WorkflowOutcome.BLOCKED
        report = workflow._terminal_report
        assert isinstance(report, TerminalReport)
        assert report.blocked_from_phase == WorkflowState.VERIFY
        assert report.phase == WorkflowState.BLOCKED
        assert report.phase == workflow.machine.state

    async def test_warm_lane_requeue_sets_blocked_from_phase_to_working_phase(
        self, tmp_path: Path,
    ):
        """Warm-lane requeue never transitions to BLOCKED, so
        ``blocked_from_phase`` equals the (unchanged) working phase — the
        same value as ``phase`` (reuses ``_make_warmlane_workflow`` setup)."""
        from orchestrator.git_ops import WarmLanePoolHardDown

        wf = _make_warmlane_workflow(tmp_path=tmp_path)
        wf.git_ops.create_worktree = AsyncMock(
            side_effect=WarmLanePoolHardDown(
                "warm-lane base absent (host-scoped pool hard-down) for branch "
                "'1859'; requeue"
            )
        )
        mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
        wf._mark_blocked = mark_blocked  # type: ignore[method-assign]

        outcome = (await wf.run()).outcome

        assert outcome == WorkflowOutcome.REQUEUED
        report = wf._terminal_report
        assert isinstance(report, TerminalReport)
        assert report.blocked_from_phase == wf.machine.state
        assert report.phase == wf.machine.state


@pytest.mark.asyncio
class TestRunReturnsTerminalReport:
    """``TaskWorkflow.run()`` RETURNS a ``TerminalReport`` (TR-1, boundary row 7).

    Still RED at this step — ``run()`` is not flipped until step-6.  Each
    assertion here checks ``isinstance(report, TerminalReport)`` FIRST so the
    pre-step-6 failure is a clean ``AssertionError`` rather than an
    ``AttributeError`` from probing a ``WorkflowOutcome`` enum member for
    dataclass fields it doesn't have.
    """

    async def test_run_returns_terminal_report_on_done(
        self, config, git_ops, task_assignment, monkeypatch,
    ):
        """Clean merged-DONE run (mirrors test_workflow_e2e.TestHappyPath)."""
        stub = AgentStub()
        workflow, scheduler = _build_workflow(config, git_ops, task_assignment, stub)

        monkeypatch.setattr('orchestrator.workflow.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(return_value=VerifyResult(
                passed=True, test_output='OK', lint_output='',
                type_output='', summary='All checks passed',
            )),
        )

        report = await workflow.run()

        assert isinstance(report, TerminalReport)
        assert not isinstance(report, WorkflowOutcome)
        assert report.outcome == WorkflowOutcome.DONE
        assert report.phase == WorkflowState.DONE

    async def test_run_returns_terminal_report_on_blocked(
        self, config, git_ops, task_assignment, monkeypatch,
    ):
        """Blocked run via the AllAccountsCappedException path (e2e:5638)."""
        from shared.cli_invoke import AllAccountsCappedException

        stub = AgentStub()
        workflow, scheduler = _build_workflow(config, git_ops, task_assignment, stub)

        async def raise_cap_exc(*args, **kwargs):
            raise AllAccountsCappedException(
                retries=3, elapsed_secs=120.0, label='Task 42 [architect]'
            )

        monkeypatch.setattr('orchestrator.workflow.invoke_agent', raise_cap_exc)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(side_effect=AssertionError('run_scoped_verification must not be called')),
        )

        report = await workflow.run()

        assert isinstance(report, TerminalReport)
        assert not isinstance(report, WorkflowOutcome)
        assert report.outcome == WorkflowOutcome.BLOCKED
        assert report.reason.lower().startswith('all accounts capped')
        assert report.phase == workflow.machine.state

    async def test_harness_task_report_maps_from_terminal_report(
        self, config, git_ops, task_assignment, monkeypatch,
    ):
        """Harness-facing: TaskReport.block_* map straight off the returned
        report (report.reason/detail/phase.value) for a blocked workflow.
        """
        from shared.cli_invoke import AllAccountsCappedException

        stub = AgentStub()
        workflow, scheduler = _build_workflow(config, git_ops, task_assignment, stub)

        async def raise_cap_exc(*args, **kwargs):
            raise AllAccountsCappedException(
                retries=3, elapsed_secs=120.0, label='Task 42 [architect]'
            )

        monkeypatch.setattr('orchestrator.workflow.invoke_agent', raise_cap_exc)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(side_effect=AssertionError('run_scoped_verification must not be called')),
        )

        report = await workflow.run()
        assert isinstance(report, TerminalReport)

        task_report = TaskReport(
            task_id='42',
            title='Add farewell function',
            outcome=report.outcome,
            block_reason=report.reason,
            block_detail=report.detail,
            block_phase=report.phase.value,
        )

        assert task_report.block_reason == report.reason
        assert task_report.block_reason.lower().startswith('all accounts capped')
        assert task_report.block_detail == report.detail
        assert task_report.block_detail
        assert task_report.block_phase == report.phase.value
        assert task_report.block_phase == workflow.machine.state.value


@pytest.mark.asyncio
class TestRunExitConsistencyAssert:
    """SM-2: the run()-exit consistency assert (boundary row 6).

    Positive: a DONE run, a BLOCKED run, and a REQUEUED (warm-lane) run each
    complete WITHOUT raising, and the resulting report is consistent with
    both the last persisted DB status (``outcome_allows_status``) and
    ``workflow.machine.state`` (covers the 'blocked'->blocked and
    'requeued'->pending rows of ``_OUTCOME_ALLOWED``).

    Negative: an injected outcome<->status divergence (DB row says 'done'
    while the actual exit is BLOCKED) must make ``run()`` raise loudly
    rather than silently return the mismatched report — this is the
    assertion that is RED until step-8 wires SM-2 into the ``run()``
    wrapper.

    Guard: a ``None`` (unreadable) or out-of-vocabulary status row must not
    crash a normal run — SM-2 fails safe rather than exploding on a
    transient/garbled ``get_status`` read.
    """

    async def test_done_run_is_consistent_with_last_status(
        self, config, git_ops, task_assignment, monkeypatch,
    ):
        """Clean merged-DONE run (mirrors test_workflow_e2e.TestHappyPath)."""
        stub = AgentStub()
        workflow, scheduler = _build_workflow(config, git_ops, task_assignment, stub)
        monkeypatch.setattr('orchestrator.workflow.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(return_value=VerifyResult(
                passed=True, test_output='OK', lint_output='',
                type_output='', summary='All checks passed',
            )),
        )

        report = await workflow.run()

        last_status = await scheduler.get_status(workflow.task_id)
        assert last_status == 'done'
        assert outcome_allows_status(report.outcome, last_status)
        assert report.phase == workflow.machine.state

    async def test_blocked_run_is_consistent_with_last_status(
        self, config, git_ops, task_assignment, monkeypatch,
    ):
        """Blocked run via the AllAccountsCappedException path (e2e:5638)."""
        from shared.cli_invoke import AllAccountsCappedException

        stub = AgentStub()
        workflow, scheduler = _build_workflow(config, git_ops, task_assignment, stub)

        async def raise_cap_exc(*args, **kwargs):
            raise AllAccountsCappedException(
                retries=3, elapsed_secs=120.0, label='Task 42 [architect]'
            )

        monkeypatch.setattr('orchestrator.workflow.invoke_agent', raise_cap_exc)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(side_effect=AssertionError('run_scoped_verification must not be called')),
        )

        report = await workflow.run()

        last_status = await scheduler.get_status(workflow.task_id)
        assert last_status == 'blocked'
        assert outcome_allows_status(report.outcome, last_status)
        assert report.phase == workflow.machine.state

    async def test_requeued_run_is_consistent_with_last_status(self, tmp_path: Path):
        """Warm-lane requeue (reuses test_workflow_warm_lane_requeue setup)."""
        from orchestrator.git_ops import WarmLanePoolHardDown

        wf = _make_warmlane_workflow(tmp_path=tmp_path)
        wf.git_ops.create_worktree = AsyncMock(
            side_effect=WarmLanePoolHardDown(
                "warm-lane base absent (host-scoped pool hard-down) for branch "
                "'1859'; requeue"
            )
        )
        mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
        wf._mark_blocked = mark_blocked  # type: ignore[method-assign]

        report = await wf.run()

        last_status = await wf.scheduler.get_status(wf.task_id)
        assert last_status == 'pending'
        assert outcome_allows_status(report.outcome, last_status)
        assert report.phase == wf.machine.state

    async def test_outcome_status_mismatch_raises_loudly(
        self, config, git_ops, task_assignment, monkeypatch,
    ):
        """Negative: DB row says 'done' while the actual exit is BLOCKED.

        The pre-empt check at the top of ``_drive()`` (workflow.py ~1697)
        also calls ``scheduler.get_status`` — it must see a NON-terminal
        status there, or the task exits CANCELLED before ever reaching the
        AllAccountsCapped block path. Only the SECOND call (SM-2's own read,
        after ``_drive()`` returns) sees the injected 'done', simulating a
        stale/incorrect DB row at the terminal boundary.
        """
        from shared.cli_invoke import AllAccountsCappedException

        stub = AgentStub()
        workflow, scheduler = _build_workflow(config, git_ops, task_assignment, stub)

        calls = {'n': 0}

        async def fake_get_status(task_id):
            calls['n'] += 1
            return 'pending' if calls['n'] == 1 else 'done'

        monkeypatch.setattr(scheduler, 'get_status', fake_get_status)

        async def raise_cap_exc(*args, **kwargs):
            raise AllAccountsCappedException(
                retries=3, elapsed_secs=120.0, label='Task 42 [architect]'
            )

        monkeypatch.setattr('orchestrator.workflow.invoke_agent', raise_cap_exc)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(side_effect=AssertionError('run_scoped_verification must not be called')),
        )

        with pytest.raises((AssertionError, ValueError)) as exc_info:
            await workflow.run()

        message = str(exc_info.value).lower()
        assert 'blocked' in message
        assert 'done' in message

    async def test_guard_skips_when_status_is_none(
        self, config, git_ops, task_assignment, monkeypatch,
    ):
        """A None get_status (transient read failure) must not crash run()."""
        stub = AgentStub()
        workflow, scheduler = _build_workflow(config, git_ops, task_assignment, stub)
        monkeypatch.setattr('orchestrator.workflow.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(return_value=VerifyResult(
                passed=True, test_output='OK', lint_output='',
                type_output='', summary='All checks passed',
            )),
        )
        monkeypatch.setattr(scheduler, 'get_status', AsyncMock(return_value=None))

        report = await workflow.run()

        assert report.outcome == WorkflowOutcome.DONE

    async def test_guard_skips_when_status_is_out_of_vocabulary(
        self, config, git_ops, task_assignment, monkeypatch,
    ):
        """An unrecognized status string must not crash run() either.

        ``outcome_allows_status`` raises ``ValueError`` on an out-of-vocab
        status (task_transitions.py) — the run()-exit guard must catch that
        rather than let it escape as an unhandled crash.
        """
        stub = AgentStub()
        workflow, scheduler = _build_workflow(config, git_ops, task_assignment, stub)
        monkeypatch.setattr('orchestrator.workflow.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(return_value=VerifyResult(
                passed=True, test_output='OK', lint_output='',
                type_output='', summary='All checks passed',
            )),
        )
        monkeypatch.setattr(
            scheduler, 'get_status', AsyncMock(return_value='not-a-real-status'),
        )

        report = await workflow.run()

        assert report.outcome == WorkflowOutcome.DONE


@pytest.mark.asyncio
class TestLastBlockSideChannelDeleted:
    """TR-1 / boundary row 7: the ``_last_block_*`` side channel is GONE.

    This is a behavioral pin, not a grep/docstring check: after a blocked
    run (or a warm-lane requeue), ``TaskWorkflow`` instances must carry NO
    ``_last_block_reason``/``_last_block_detail``/``_last_block_phase``
    attributes at all, and the full block context must still be reachable
    exclusively through the returned ``TerminalReport``.
    """

    async def test_mark_blocked_path_has_no_last_block_attrs(
        self, config, git_ops, task_assignment, monkeypatch,
    ):
        """Generic block via the AllAccountsCappedException path (e2e:5638)."""
        from shared.cli_invoke import AllAccountsCappedException

        stub = AgentStub()
        workflow, scheduler = _build_workflow(config, git_ops, task_assignment, stub)

        async def raise_cap_exc(*args, **kwargs):
            raise AllAccountsCappedException(
                retries=3, elapsed_secs=120.0, label='Task 42 [architect]'
            )

        monkeypatch.setattr('orchestrator.workflow.invoke_agent', raise_cap_exc)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(side_effect=AssertionError('run_scoped_verification must not be called')),
        )

        report = await workflow.run()

        assert report.outcome == WorkflowOutcome.BLOCKED
        assert not hasattr(workflow, '_last_block_reason')
        assert not hasattr(workflow, '_last_block_detail')
        assert not hasattr(workflow, '_last_block_phase')
        # Full block context still reachable via the returned report alone.
        assert report.reason.lower().startswith('all accounts capped')
        assert report.detail
        assert report.phase == workflow.machine.state

    async def test_warm_lane_requeue_has_no_last_block_attrs(self, tmp_path: Path):
        """Warm-lane requeue (reuses test_workflow_warm_lane_requeue setup)."""
        from orchestrator.git_ops import WarmLanePoolHardDown

        wf = _make_warmlane_workflow(tmp_path=tmp_path)
        wf.git_ops.create_worktree = AsyncMock(
            side_effect=WarmLanePoolHardDown(
                "warm-lane base absent (host-scoped pool hard-down) for branch "
                "'1859'; requeue"
            )
        )
        mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
        wf._mark_blocked = mark_blocked  # type: ignore[method-assign]

        report = await wf.run()

        assert report.outcome == WorkflowOutcome.REQUEUED
        assert not hasattr(wf, '_last_block_reason')
        assert not hasattr(wf, '_last_block_detail')
        assert not hasattr(wf, '_last_block_phase')
        assert report.reason == 'warm_lane_pool_hard_down'
