"""What a resumed implementer is told after ``run()``'s ESCALATED wait (task 3754).

The wait hands ``run()`` the attributed resolutions it resumes on, refuses to
resume past a record only a human can dispose of, and the resume briefing
names which escalations the resolutions answer.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from escalation.models import Escalation
from escalation.queue import EscalationQueue
from test_workflow_escalated_steward_stall import (
    _build_workflow,
    _make_advanced_worktree,
)
from test_workflow_escalated_steward_stall import config as config
from test_workflow_escalated_steward_stall import git_ops as git_ops
from test_workflow_escalated_steward_stall import git_repo as git_repo
from test_workflow_escalated_steward_stall import task_assignment as task_assignment

from orchestrator.config import OrchestratorConfig
from orchestrator.git_ops import GitOps
from orchestrator.scheduler import TaskAssignment

TASK_ID = '42'


def _submit(
    queue: EscalationQueue,
    summary: str,
    *,
    severity: str = 'blocking',
    level: int = 0,
    agent_role: str = 'implementer',
) -> Escalation:
    esc = Escalation(
        id=queue.make_id(TASK_ID),
        task_id=TASK_ID,
        agent_role=agent_role,
        severity=severity,
        category='task_failure',
        summary=summary,
        detail=f'detail for {summary}',
        level=level,
    )
    queue.submit(esc)
    return esc


@pytest.mark.asyncio
class TestWaitReturnsTheResolutionsItResumesOn:
    async def test_returns_resolved_records_in_resolution_order(
        self,
        config: OrchestratorConfig,
        git_ops: GitOps,
        task_assignment: TaskAssignment,
        tmp_path: Path,
    ):
        queue = EscalationQueue(tmp_path / 'queue')
        worktree = await _make_advanced_worktree(git_ops, TASK_ID)
        workflow, _ = _build_workflow(config, git_ops, task_assignment, queue, worktree)
        first = _submit(queue, 'first blocker')
        second = _submit(queue, 'second blocker')
        dismissed = _submit(queue, 'third blocker')
        queue.resolve(second.id, 'answer to second', resolved_by='steward')
        queue.resolve(first.id, 'answer to first', resolved_by='interactive')
        queue.resolve(
            dismissed.id, 'Auto-dismissed: superseded', dismiss=True,
            resolved_by='auto-dismissed',
        )
        _submit(queue, 'informational note', severity='info', level=1)

        carried = await asyncio.wait_for(workflow._wait_for_resolution(), 10)

        assert isinstance(carried, tuple)
        assert [(c.escalation_id, c.resolution) for c in carried] == [
            (second.id, 'answer to second'),
            (first.id, 'answer to first'),
        ]

    async def test_without_a_queue_carries_nothing(
        self,
        config: OrchestratorConfig,
        git_ops: GitOps,
        task_assignment: TaskAssignment,
        tmp_path: Path,
    ):
        workflow, _ = _build_workflow(
            config, git_ops, task_assignment, EscalationQueue(tmp_path / 'queue'), tmp_path,
        )
        workflow.escalation_queue = None

        assert await workflow._wait_for_resolution() == ()
