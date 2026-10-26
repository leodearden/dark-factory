"""A TaskWorkflow whose merge an operator cancels BLOCKS and ESCALATES (task 5248).

Leo's 2026-10-21 ruling: the cancel goes through, and the workflow turns the
cancelled merge future into BLOCKED through the existing block path, filing an
L1 that names the operator cancel and the request id. A cancel that does not
come from an operator keeps propagating as a cancellation.

The workflow's request goes onto a real (not running) lane's queue through the
production enqueue path, and the operator's cancel is the lane's own
``cancel_request``. Harness mirrors ``test_workflow_merge_blocked_surfacing.py``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _merge_lane_fakes import make_lane
from escalation.queue import EscalationQueue

from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps
from orchestrator.merge_lane import MergeLane
from orchestrator.scheduler import TaskAssignment
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome

pytestmark = pytest.mark.asyncio


def _config(tmp_path: Path) -> OrchestratorConfig:
    return OrchestratorConfig(
        project_root=tmp_path,
        max_concurrent_tasks=1,
        git=GitConfig(
            main_branch='main', branch_prefix='task/', remote='origin',
            worktree_dir='.worktrees',
        ),
    )


def _workflow(tmp_path: Path, escalations: EscalationQueue) -> TaskWorkflow:
    config = _config(tmp_path)
    worktree = tmp_path / 'task-wt'
    worktree.mkdir()
    git_ops = MagicMock(spec=GitOps)
    git_ops.get_main_sha = AsyncMock(return_value='deadbeef1234567890')
    git_ops.config = MagicMock(branch_prefix='task/', main_branch='main')
    git_ops.rebind_branch_to_head = AsyncMock(return_value=True)
    scheduler = MagicMock()
    scheduler.set_task_status = AsyncMock()
    workflow = TaskWorkflow(
        assignment=TaskAssignment(
            task_id='42',
            task={
                'id': '42', 'title': 'X', 'description': '', 'status': 'pending',
                'metadata': {'files': ['lib']}, 'dependencies': [],
            },
            modules=['lib'],
        ),
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,
        briefing=MagicMock(),
        mcp=MagicMock(),
        escalation_queue=escalations,
    )
    workflow.worktree = worktree
    artifacts = TaskArtifacts(worktree)
    artifacts.init('42', 'X', 'desc', base_commit='base-sha')
    workflow.artifacts = artifacts
    workflow.plan = {'task_id': '42', 'steps': []}
    return workflow


async def _enqueued_request_id(lane: MergeLane) -> str:
    async def _first() -> str:
        while not (entries := lane.snapshot()['entries']):
            await asyncio.sleep(0.01)
        return entries[0]['request_id']

    return await asyncio.wait_for(_first(), 10.0)


async def test_an_operator_cancel_blocks_the_task_and_files_an_l1(tmp_path: Path) -> None:
    escalations = EscalationQueue(tmp_path / 'esc')
    workflow = _workflow(tmp_path, escalations)
    queue: asyncio.Queue = asyncio.Queue()
    workflow.merge_queue = queue
    lane = make_lane(MagicMock(spec=GitOps), queue)
    submit = asyncio.create_task(
        workflow._submit_to_merge_queue('task/42', pre_rebased=False, merge_phase=False),
    )
    request_id = await _enqueued_request_id(lane)

    assert lane.cancel_request(request_id).cancelled
    outcome = await asyncio.wait_for(submit, 10.0)

    assert outcome is WorkflowOutcome.BLOCKED
    [escalation] = escalations.get_by_task('42')
    assert escalation.category == 'merge_cancelled_by_operator'
    assert escalation.level == 1
    assert request_id in escalation.summary
    assert 'cancelled by an operator' in escalation.summary


async def test_a_cancel_without_an_operator_still_propagates(tmp_path: Path) -> None:
    escalations = EscalationQueue(tmp_path / 'esc')
    workflow = _workflow(tmp_path, escalations)
    queue: asyncio.Queue = asyncio.Queue()
    workflow.merge_queue = queue
    submit = asyncio.create_task(
        workflow._submit_to_merge_queue('task/42', pre_rebased=False, merge_phase=False),
    )
    request = await asyncio.wait_for(queue.get(), 10.0)

    request.result.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(submit, 10.0)
    assert escalations.get_by_task('42') == []
