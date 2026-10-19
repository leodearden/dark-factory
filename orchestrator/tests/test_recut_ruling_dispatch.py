"""Dispatch wiring for re-cut rulings (task 3154).

Harness._run_slot executes a pending ruling before any workflow exists, or
blocks the task with a design_concern L1; the workflow hands the same
dispatch-time ruling to create_worktree as the redundant second check.

Scaffolding (_make_slot_harness/_make_mock_workflow) is copied from
test_cross_repo_gate.py, matching this repo's per-test-file helper convention.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _workflow_helpers import _make_warmlane_workflow
from escalation.models import Escalation
from escalation.queue import EscalationQueue

from orchestrator.recut_ruling import (
    RECUT_EXECUTED_KEY,
    RECUT_RULING_KEY,
    ParkedIncarnation,
    RecutRuling,
)
from orchestrator.scheduler import TaskAssignment
from orchestrator.workflow import WorkflowOutcome

TID = '2896'
RULING_ID = 'esc-2896-3/D11-A'
PENDING = {RECUT_RULING_KEY: {'ruling_id': RULING_ID, 'ruled_at': '2026-08-30T00:00:00Z'}}
PARKED = ParkedIncarnation(
    parked_branch='task/2896-20260901T123000Z',
    parked_tip='c' * 40,
    quarantined_worktree=None,
    relocated_meta_roots=(),
)
RECOVERED_PLAN = {'task_id': TID, 'steps': [{'id': 'step-1', 'status': 'done'}]}
RECOVERED_SESSION = {'session_id': 'sess-pre-ruling', 'role': 'implementer'}


def _make_slot_harness(tmp_path: Path):
    from orchestrator.config import OrchestratorConfig
    from orchestrator.harness import Harness

    config = OrchestratorConfig(project_root=tmp_path, max_per_module=1)
    with (
        patch('orchestrator.harness.McpLifecycle'),
        patch('orchestrator.harness.OverrideStore'),
        patch('orchestrator.harness.BriefingAssembler'),
    ):
        h = Harness(config)

    h.scheduler = MagicMock()
    h.scheduler.set_task_status = AsyncMock()
    h.scheduler.is_deterministic = MagicMock(return_value=False)
    h.scheduler.update_task = AsyncMock(return_value=True)
    h.scheduler.release = MagicMock()
    h.scheduler._dispatched = set()

    h.git_ops = MagicMock()
    h.git_ops.resolve_branch_sha = AsyncMock(return_value='deadbeef' * 5)
    h.git_ops.worktree_base = tmp_path / '.worktrees'
    h.git_ops.project_root = tmp_path
    h.git_ops.prune_worktrees = AsyncMock(return_value=None)
    h.git_ops.park_prior_incarnation = AsyncMock(return_value=PARKED)
    h.git_ops.prior_incarnation_evidence = AsyncMock(return_value=())

    h._escalation_queue = EscalationQueue(tmp_path / 'esc')
    h._run_id = None
    h.event_store = None
    h._recovered_plans[TID] = RECOVERED_PLAN
    h._recovered_sessions[TID] = RECOVERED_SESSION
    h._session_resume_reasons = MagicMock(return_value=frozenset())
    return h


def _make_mock_workflow():
    from orchestrator.workflow import TerminalReport, WorkflowState

    wf = AsyncMock()
    wf.run = AsyncMock(return_value=TerminalReport(
        outcome=WorkflowOutcome.DONE, reason='', phase=WorkflowState.DONE,
        detail='', category=None,
    ))
    wf.metrics = MagicMock(
        total_cost_usd=0.0,
        total_duration_ms=0,
        agent_invocations=0,
        execute_iterations=0,
        verify_attempts=0,
        review_cycles=0,
    )
    wf._steward = None
    return wf


def _assignment(metadata: Any) -> TaskAssignment:
    return TaskAssignment(
        task_id=TID, task={'id': TID, 'title': 'T', 'metadata': metadata}, modules=[],
    )


async def _run(h, metadata: Any, calls: list[str] | None = None):
    """Run one slot; return (report, build_workflow mock)."""
    sem = MagicMock()
    sem.release = MagicMock()
    with patch('orchestrator.harness.build_workflow') as build:
        def _build(**_kwargs):
            if calls is not None:
                calls.append('build_workflow')
            return _make_mock_workflow()

        build.side_effect = _build
        report = await h._run_slot(_assignment(metadata), sem)
    return report, build


def _record(calls: list[str], name: str, result: object) -> AsyncMock:
    async def _call(*_args, **_kwargs):
        calls.append(name)
        return result

    return AsyncMock(side_effect=_call)


def _pending_l1s(h) -> list[Escalation]:
    return [e for e in h._escalation_queue.get_pending() if e.task_id == TID and e.level == 1]


def _assert_refused(h, report, build) -> Escalation:
    assert not build.called
    assert report is not None
    assert report.outcome == WorkflowOutcome.BLOCKED
    assert report.block_reason == 'recut_ruling_refused'
    h.scheduler.set_task_status.assert_any_await(TID, 'blocked')
    assert h.scheduler.release.call_args.kwargs.get('requeued') is True
    l1s = _pending_l1s(h)
    assert len(l1s) == 1, l1s
    assert l1s[0].category == 'design_concern'
    assert l1s[0].summary.startswith('RECUT_RULING_REFUSED')
    return l1s[0]


@pytest.mark.asyncio
class TestRunSlotRecutGate:
    async def test_no_marker_never_enters_the_gate(self, tmp_path: Path):
        h = _make_slot_harness(tmp_path)

        _, build = await _run(h, {'files': ['a.py']})

        h.git_ops.park_prior_incarnation.assert_not_awaited()
        kwargs = build.call_args.kwargs
        assert kwargs['initial_plan'] == RECOVERED_PLAN
        assert kwargs['resume_session_id'] == RECOVERED_SESSION

    async def test_pending_ruling_executes_before_the_workflow_exists(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ):
        h = _make_slot_harness(tmp_path)
        calls: list[str] = []
        h.git_ops.park_prior_incarnation = _record(calls, 'park', PARKED)
        h.git_ops.prior_incarnation_evidence = _record(calls, 'evidence', ())
        h.scheduler.update_task = _record(calls, 'update_task', True)

        with caplog.at_level(logging.WARNING, logger='orchestrator.harness'):
            _, build = await _run(h, PENDING, calls)

        assert calls == ['park', 'evidence', 'update_task', 'build_workflow']
        update_args = h.scheduler.update_task.await_args.args
        assert update_args[0] == TID
        assert update_args[1][RECUT_EXECUTED_KEY]['ruling_id'] == RULING_ID
        kwargs = build.call_args.kwargs
        assert kwargs['initial_plan'] is None
        assert kwargs['resume_session_id'] is None
        h._session_resume_reasons.assert_not_called()
        assert RULING_ID in caplog.text
        assert PARKED.parked_branch is not None
        assert PARKED.parked_branch in caplog.text

    async def test_satisfied_ruling_dispatches_without_parking(self, tmp_path: Path):
        h = _make_slot_harness(tmp_path)
        satisfied = {**PENDING, RECUT_EXECUTED_KEY: {'ruling_id': RULING_ID}}

        _, build = await _run(h, satisfied)

        h.git_ops.park_prior_incarnation.assert_not_awaited()
        assert build.called

    async def test_malformed_marker_blocks_and_files_l1(self, tmp_path: Path):
        h = _make_slot_harness(tmp_path)

        report, build = await _run(h, {RECUT_RULING_KEY: {'ruling_id': RULING_ID}})

        _assert_refused(h, report, build)
        h.git_ops.park_prior_incarnation.assert_not_awaited()

    async def test_park_failure_blocks_naming_the_ruling(self, tmp_path: Path):
        h = _make_slot_harness(tmp_path)
        h.git_ops.park_prior_incarnation = AsyncMock(side_effect=RuntimeError('move failed'))

        report, build = await _run(h, PENDING)

        l1 = _assert_refused(h, report, build)
        assert RULING_ID in l1.summary + l1.detail
        h.scheduler.update_task.assert_not_awaited()

    async def test_failed_record_blocks_naming_the_parked_ref(self, tmp_path: Path):
        h = _make_slot_harness(tmp_path)
        h.scheduler.update_task = AsyncMock(return_value=False)

        report, build = await _run(h, PENDING)

        l1 = _assert_refused(h, report, build)
        assert PARKED.parked_branch is not None
        assert PARKED.parked_branch in l1.detail

    async def test_open_design_concern_l1_is_not_duplicated(self, tmp_path: Path):
        h = _make_slot_harness(tmp_path)
        h._escalation_queue.submit(Escalation(
            id=h._escalation_queue.make_id(TID),
            task_id=TID,
            agent_role='orchestrator-scheduler',
            severity='blocking',
            category='design_concern',
            summary='an earlier design concern',
            detail='',
            suggested_action='manual_intervention',
            level=1,
        ))

        report, build = await _run(h, {RECUT_RULING_KEY: 'garbage'})

        assert not build.called
        assert report is not None and report.block_reason == 'recut_ruling_refused'
        h.scheduler.set_task_status.assert_any_await(TID, 'blocked')
        assert len(_pending_l1s(h)) == 1

    async def test_unreadable_metadata_skips_loudly_and_dispatches(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ):
        h = _make_slot_harness(tmp_path)

        with caplog.at_level(logging.WARNING, logger='orchestrator.recut_ruling'):
            _, build = await _run(h, 'not json')

        assert 'no ruling was read' in caplog.text
        h.git_ops.park_prior_incarnation.assert_not_awaited()
        assert build.called


class _ProvisioningStopped(RuntimeError):
    """Raised by the fake create_worktree so a run stops right after provisioning."""


async def _run_to_provisioning(tmp_path: Path, metadata: Any) -> tuple[AsyncMock, Any]:
    """Drive TaskWorkflow.run() up to create_worktree; return (create_worktree, report)."""
    wf = _make_warmlane_workflow(tmp_path=tmp_path, task_id=TID)
    if metadata is not None:
        wf.task['metadata'] = metadata
    create_worktree = AsyncMock(side_effect=_ProvisioningStopped('stop after provisioning'))
    wf.git_ops.create_worktree = create_worktree
    statuses = {TID: 'pending'}

    async def _set_status(task_id: str, status: str, **_kwargs: Any) -> None:
        statuses[task_id] = status

    async def _get_status(task_id: str) -> str | None:
        return statuses.get(task_id)

    async def _mark_blocked(*_args: Any, **_kwargs: Any) -> WorkflowOutcome:
        await _set_status(TID, 'blocked')
        return WorkflowOutcome.BLOCKED

    wf.scheduler.set_task_status = AsyncMock(side_effect=_set_status)
    wf.scheduler.get_status = AsyncMock(side_effect=_get_status)
    wf._mark_blocked = AsyncMock(side_effect=_mark_blocked)  # type: ignore[method-assign]
    return create_worktree, await wf.run()


@pytest.mark.asyncio
class TestWorkflowPassesRulingToProvisioning:
    async def test_pending_ruling_reaches_create_worktree(self, tmp_path: Path):
        create_worktree, _ = await _run_to_provisioning(tmp_path, PENDING)

        assert create_worktree.await_args.kwargs['recut_ruling'] == RecutRuling(
            RULING_ID, datetime(2026, 8, 30, tzinfo=UTC),
        )

    @pytest.mark.parametrize('metadata', [
        pytest.param(None, id='no-marker'),
        pytest.param({**PENDING, RECUT_EXECUTED_KEY: {'ruling_id': RULING_ID}}, id='satisfied'),
    ])
    async def test_nothing_pending_passes_none(self, tmp_path: Path, metadata: Any):
        create_worktree, _ = await _run_to_provisioning(tmp_path, metadata)

        assert create_worktree.await_args.kwargs['recut_ruling'] is None

    async def test_malformed_ruling_blocks_before_provisioning(self, tmp_path: Path):
        create_worktree, report = await _run_to_provisioning(
            tmp_path, {RECUT_RULING_KEY: {'ruling_id': RULING_ID}},
        )

        create_worktree.assert_not_awaited()
        assert report.outcome == WorkflowOutcome.BLOCKED
