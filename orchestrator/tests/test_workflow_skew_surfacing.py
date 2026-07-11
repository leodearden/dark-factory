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

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from _orch_helpers import pydantic_spec

from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.workflow import TaskWorkflow, WorkflowState


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
