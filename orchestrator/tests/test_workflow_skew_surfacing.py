"""Tests for task 2383 β: the ``EventType.workflow_verify`` producer emit at
the ``_enter_phase`` VERIFY→REVIEW edge, and the merge-skew surfacing wired
into ``TaskWorkflow._submit_to_merge_queue``.

This is β's anti-orphan obligation FOR alpha (task 2381): without this
producer emit, ``merge_disposition._branch_pre_merge_verify_green`` has no
row to read on the orchestrator pathway and the whole merge-skew-attribution
PRD degrades to INDETERMINATE everywhere (a G6 no-op).

Model: ``_make_workflow`` mirrors the trimmed builder in
``test_workflow_state_machine.py`` — no worktree/artifacts are needed since
``_enter_phase`` only touches ``self.state``/``self.machine``/
``self.event_store``/``self.metrics``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from _orch_helpers import pydantic_spec

from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.git_ops import GitOps
from orchestrator.merge_disposition import MergeFailureDisposition
from orchestrator.merge_queue import MergeOutcome
from orchestrator.scheduler import TaskAssignment
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome, WorkflowState


def _make_workflow(
    *, task_id: str = '2383', event_store: EventStore | None = None,
) -> TaskWorkflow:
    """Minimal ``TaskWorkflow`` construction, mirroring ``_make_workflow`` in
    ``test_workflow_state_machine.py`` — no worktree/artifacts are needed for
    the ``_enter_phase`` producer-emit surface."""
    assignment = MagicMock()
    assignment.task_id = task_id
    assignment.task = {'id': task_id, 'title': 'T', 'description': 'd'}
    assignment.modules = []

    config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))

    scheduler = MagicMock()
    scheduler.set_task_status = AsyncMock()

    return TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=MagicMock(),
        scheduler=scheduler,
        briefing=MagicMock(),
        mcp=MagicMock(),
        event_store=event_store,
    )


class TestWorkflowVerifyProducerEmit:
    """The VERIFY→REVIEW edge inside ``_enter_phase`` must emit exactly one
    ``EventType.workflow_verify`` row keyed by ``task_id``, carrying
    ``{'passed': True, 'base_sha': self._base_commit, 'branch': self.task_id}``.
    REVIEW is reachable from VERIFY only on a passing verify (a failing
    verify routes to BLOCKED/ESCALATED before ``_enter_phase(REVIEW)``), so
    this edge is a reliable "branch verified green pre-merge" signal.
    """

    def test_verify_to_review_emits_exactly_one_workflow_verify_row(self, tmp_path: Path):
        store = EventStore(tmp_path / 'runs.db', run_id='run-test')
        wf = _make_workflow(event_store=store)
        wf._base_commit = 'deadbeef'
        wf.state = WorkflowState.VERIFY

        wf._enter_phase(WorkflowState.REVIEW)

        rows = store.fetch_events_by_type(EventType.workflow_verify)
        assert len(rows) == 1
        assert rows[0]['task_id'] == wf.task_id
        assert rows[0]['data'] == {
            'passed': True,
            'base_sha': 'deadbeef',
            'branch': wf.task_id,
        }

    def test_execute_to_verify_emits_no_workflow_verify_row(self, tmp_path: Path):
        """A non-VERIFY→REVIEW edge (EXECUTE→VERIFY) must not emit."""
        store = EventStore(tmp_path / 'runs.db', run_id='run-test')
        wf = _make_workflow(event_store=store)
        wf._base_commit = 'deadbeef'
        wf.state = WorkflowState.EXECUTE

        wf._enter_phase(WorkflowState.VERIFY)

        assert store.fetch_events_by_type(EventType.workflow_verify) == []

    def test_verify_to_blocked_emits_no_workflow_verify_row(self, tmp_path: Path):
        """A non-VERIFY→REVIEW edge (VERIFY→BLOCKED, the failing-verify path)
        must not emit — only a passing verify produces the green marker."""
        store = EventStore(tmp_path / 'runs.db', run_id='run-test')
        wf = _make_workflow(event_store=store)
        wf._base_commit = 'deadbeef'
        wf.state = WorkflowState.VERIFY

        wf._enter_phase(WorkflowState.BLOCKED)

        assert store.fetch_events_by_type(EventType.workflow_verify) == []


# ---------------------------------------------------------------------------
# Step-13: _submit_to_merge_queue routes an INTEGRATION_SKEW outcome to a
# dedicated L1 escalation, mirroring test_workflow_main_health_routing.py's
# seam for the MAIN_HEALTH_RED short-circuit.
# ---------------------------------------------------------------------------

MAIN_SHA = 'deadbeef1234567890'
SKEW_SHA = 'abc123deadbeefcafe'
SKEW_FILES = 'src/module/foo.py, src/module/bar.py'
SKEW_TESTS = 'tests/test_foo.py::test_bar'


def _make_routing_config(tmp_path: Path) -> OrchestratorConfig:
    return OrchestratorConfig(
        project_root=tmp_path,
        max_concurrent_tasks=1,
        git=GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
        ),
    )


def _make_routing_workflow(config: OrchestratorConfig, worktree: Path) -> TaskWorkflow:
    """Mirrors ``_make_workflow`` in test_workflow_main_health_routing.py —
    a full TaskWorkflow capable of driving ``_submit_to_merge_queue``."""
    assignment = TaskAssignment(
        task_id='42',
        task={
            'id': '42', 'title': 'X', 'description': '',
            'status': 'pending', 'metadata': {'files': ['lib']},
            'dependencies': [],
        },
        modules=['lib'],
    )
    git_ops = MagicMock(spec=GitOps)
    git_ops.get_main_sha = AsyncMock(return_value=MAIN_SHA)
    # task-1923: _submit_to_merge_queue awaits rebind_branch_to_head before
    # enqueue and reads git_ops.config.branch_prefix to name the ref.  spec=GitOps
    # hides the __init__-assigned ``config`` attr, so set it explicitly.
    git_ops.config = MagicMock(branch_prefix='task/', main_branch='main')
    git_ops.rebind_branch_to_head = AsyncMock(return_value=True)
    scheduler = MagicMock()
    scheduler.set_task_status = AsyncMock()
    workflow = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,
        briefing=MagicMock(),
        mcp=MagicMock(),
    )
    workflow.worktree = worktree
    artifacts = TaskArtifacts(worktree)
    artifacts.init('42', 'X', 'desc', base_commit='base-sha')
    workflow.artifacts = artifacts
    workflow.plan = {'task_id': '42', 'steps': []}
    return workflow


class TestSubmitToMergeQueueIntegrationSkewRouting:
    """_submit_to_merge_queue must route an INTEGRATION_SKEW blocked outcome
    to _mark_blocked(escalate_to_human=True, category='integration_skew',
    suggested_action='port_landed_change'), with the implicated sha + overlap
    files reaching the L1 body verbatim, and must NOT touch this path for a
    non-skew (INDETERMINATE-default) blocked outcome — falls through
    byte-identical to today's generic path.
    """

    def test_integration_skew_routes_to_l1_escalation(self, tmp_path: Path) -> None:
        config = _make_routing_config(tmp_path)
        worktree = tmp_path / 'task-wt'
        worktree.mkdir()
        workflow = _make_routing_workflow(config, worktree)

        skew_reason = (
            f'Post-merge verification failed: pytest failed '
            f'[category: test_failure]\n\n'
            f'...failure details...\n\n'
            f'integration_skew: port landed commit(s) {SKEW_SHA} touching '
            f'{SKEW_FILES} — do not hunt your own diff'
        )
        skew_outcome = MergeOutcome(
            'blocked',
            reason=skew_reason,
            failure_category='test_failure',
            disposition=MergeFailureDisposition.INTEGRATION_SKEW,
            failure_diagnostic={
                'disposition': 'integration_skew',
                'implicated_commits': SKEW_SHA,
                'overlap_files': SKEW_FILES,
                'failing_tests': SKEW_TESTS,
            },
        )

        mark_blocked_mock = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
        workflow._mark_blocked = mark_blocked_mock  # type: ignore[method-assign]
        workflow._write_merge_failure_review = MagicMock()

        async def _fake_enqueue(_queue, request, _event_store, **_kwargs):
            request.result.set_result(skew_outcome)

        workflow.merge_queue = asyncio.Queue()
        with patch(
            'orchestrator.merge_queue.enqueue_merge_request', _fake_enqueue,
        ):
            result = asyncio.run(
                workflow._submit_to_merge_queue('task/42', pre_rebased=False, merge_phase=True)
            )

        assert result == WorkflowOutcome.BLOCKED
        mark_blocked_mock.assert_called_once()
        call_args = mark_blocked_mock.call_args
        call_kwargs = call_args[1]
        reason_arg = call_args[0][0] if call_args[0] else call_kwargs.get('reason', '')
        detail_arg = call_kwargs.get('detail', '')

        assert call_kwargs.get('escalate_to_human') is True, (
            f'Expected escalate_to_human=True; got {call_kwargs}'
        )
        assert call_kwargs.get('category') == 'integration_skew', (
            f'Expected category=integration_skew; got {call_kwargs}'
        )
        assert call_kwargs.get('suggested_action') == 'port_landed_change', (
            f'Expected suggested_action=port_landed_change; got {call_kwargs}'
        )
        assert call_kwargs.get('merge_phase') is True, (
            f'Expected merge_phase=True (forwarded); got {call_kwargs}'
        )
        for carrier in (reason_arg, detail_arg):
            assert 'integration_skew' in carrier, carrier
            assert SKEW_SHA in carrier, carrier
            assert 'src/module/foo.py' in carrier, carrier
            assert 'src/module/bar.py' in carrier, carrier

    def test_non_skew_blocked_falls_through_to_generic_path(
        self, tmp_path: Path,
    ) -> None:
        """A blocked outcome with the default INDETERMINATE disposition must
        NOT match the INTEGRATION_SKEW branch and must fall through to the
        generic _mark_blocked path — byte-identical to pre-β behaviour."""
        config = _make_routing_config(tmp_path)
        worktree = tmp_path / 'task-wt'
        worktree.mkdir()
        workflow = _make_routing_workflow(config, worktree)

        generic_outcome = MergeOutcome(
            'blocked',
            reason='Post-merge verification failed: tsc failed',
            failure_category='compile_error',
        )

        mark_blocked_mock = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
        workflow._mark_blocked = mark_blocked_mock  # type: ignore[method-assign]
        workflow._write_merge_failure_review = MagicMock()

        async def _fake_enqueue(_queue, request, _event_store, **_kwargs):
            request.result.set_result(generic_outcome)

        workflow.merge_queue = asyncio.Queue()
        with patch(
            'orchestrator.merge_queue.enqueue_merge_request', _fake_enqueue,
        ):
            result = asyncio.run(
                workflow._submit_to_merge_queue('task/42', pre_rebased=False, merge_phase=False)
            )

        assert result == WorkflowOutcome.BLOCKED
        mark_blocked_mock.assert_called_once()
        call_kwargs = mark_blocked_mock.call_args[1]
        assert call_kwargs.get('escalate_to_human') is not True, (
            f'Generic path must not set escalate_to_human=True; got {call_kwargs}'
        )
        assert call_kwargs.get('category') != 'integration_skew', (
            f'Generic path must not set category=integration_skew; got {call_kwargs}'
        )
        assert call_kwargs.get('suggested_action') != 'port_landed_change', (
            f'Generic path must not set suggested_action=port_landed_change; got {call_kwargs}'
        )
