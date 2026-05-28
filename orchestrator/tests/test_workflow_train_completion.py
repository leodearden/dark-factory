"""Tests for TaskWorkflow._maybe_enqueue_group_merge() — δ₂ autonomous trigger.

_maybe_enqueue_group_merge:
  - Fires ONLY when self is the TIP (highest train.order) AND all members are
    merge-deferred (trusting self's just-written status).
  - Builds a GroupMergeRequest from self (tip), enqueues it, and maps
    MergeOutcome to WorkflowOutcome.
  - Returns None (park) when train is incomplete, self is not the tip, or
    infrastructure is unavailable.

_enter_merge_deferred calls _maybe_enqueue_group_merge after writing the
merge-deferred status; returns that outcome if non-None, else MERGE_DEFERRED.

_maybe_enqueue_group_merge does NOT exist yet → tests are RED until step-4 adds
the implementation.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import pydantic_spec
from escalation.models import Escalation  # noqa: F401 — keeps fixture parity

from orchestrator.config import OrchestratorConfig
from orchestrator.merge_queue import GroupMergeRequest, MergeOutcome
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome


# ---------------------------------------------------------------------------
# Shared fixture helper (mirrors test_workflow_train_state_escalation._make)
# ---------------------------------------------------------------------------

@dataclass
class _Fixture:
    wf: TaskWorkflow
    scheduler: MagicMock
    mark_blocked: AsyncMock
    queue: MagicMock
    merge_queue: asyncio.Queue


def _make(
    *,
    task_id: str = '103',
    metadata: dict | None = None,
    tasks_by_train_return: list[dict] | None = None,
    get_statuses_return: tuple[dict[str, str], Exception | None] | None = None,
) -> _Fixture:
    assignment = MagicMock()
    assignment.task_id = task_id
    assignment.task = {
        'id': task_id, 'title': 'T', 'description': 'd',
        'metadata': metadata or {},
    }
    assignment.modules = ['mod_a']

    _spec = pydantic_spec(OrchestratorConfig)
    config = MagicMock(spec_set=_spec)
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.lock_depth = 2
    config.steward_completion_timeout = 300.0
    config.project_root = Path('/tmp/non-existent-for-test')
    config.max_consecutive_infra_resumes = 3
    config.max_consecutive_merge_thrash = 3

    scheduler = MagicMock()
    scheduler.update_task = AsyncMock(return_value=True)
    scheduler.set_task_status = AsyncMock()
    scheduler.get_status = AsyncMock(return_value='merge-deferred')
    scheduler.get_task = AsyncMock(return_value={'id': task_id, 'metadata': metadata or {}})
    scheduler.mark_done = AsyncMock()
    scheduler.clear_requeue_count = MagicMock()

    if tasks_by_train_return is not None:
        scheduler.tasks_by_train = AsyncMock(return_value=tasks_by_train_return)
    else:
        scheduler.tasks_by_train = AsyncMock(return_value=[])

    if get_statuses_return is not None:
        scheduler.get_statuses = AsyncMock(return_value=get_statuses_return)
    else:
        scheduler.get_statuses = AsyncMock(
            return_value=({'101': 'merge-deferred', '102': 'merge-deferred', '103': 'merge-deferred'}, None)
        )

    git_ops = MagicMock()
    git_ops.config.branch_prefix = 'task/'
    git_ops.config.main_branch = 'main'

    esc_queue = MagicMock()
    esc_queue.has_open_l1 = MagicMock(return_value=False)
    esc_queue.make_id = MagicMock(return_value=f'esc-{task_id}-1')
    esc_queue.submit = MagicMock()
    esc_queue.get_by_task = MagicMock(return_value=[])

    merge_queue: asyncio.Queue = asyncio.Queue()

    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,
        briefing=MagicMock(),
        mcp=MagicMock(),
        escalation_queue=esc_queue,  # type: ignore[arg-type]
        merge_queue=merge_queue,
    )

    wf.artifacts = MagicMock()
    wf.artifacts.read_iteration_log = MagicMock(return_value=([], []))
    wf.worktree = Path(f'/tmp/wt-{task_id}')
    wf.event_store = None

    mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
    wf._mark_blocked = mark_blocked  # type: ignore[method-assign]

    return _Fixture(
        wf=wf,
        scheduler=scheduler,
        mark_blocked=mark_blocked,
        queue=esc_queue,
        merge_queue=merge_queue,
    )


# ---------------------------------------------------------------------------
# Step-3: Happy-path / tip-fires test (RED until step-4 adds the method)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tip_fires_group_merge_happy_path():
    """TIP member fires GroupMergeRequest when all three members are merge-deferred.

    Scenario: train T1, three members root→tip (101,102,103); self is 103 (tip,
    order 2); all statuses are merge-deferred.  _await_cancellable returns
    MergeOutcome('done', merge_sha='deadbeef').

    Asserts:
      (a) _maybe_enqueue_group_merge() returns WorkflowOutcome.DONE
      (b) exactly one GroupMergeRequest was placed on the real asyncio.Queue
      (c) the enqueued request has correct train_id, member_task_ids (root→tip),
          tip_branch, tip_task_id, branch, and worktree
      (d) invoking mark_member_done('101','sha9') awaits scheduler.mark_done
          with kind='merged', sha='sha9' and note containing 'T1'
      (e) invoking status_check(['101']) returns the plain {id:status} dict
    """
    members = [
        {'id': '101', 'status': 'merge-deferred', 'metadata': {'train': {'id': 'T1', 'order': 0}}},
        {'id': '102', 'status': 'merge-deferred', 'metadata': {'train': {'id': 'T1', 'order': 1}}},
        {'id': '103', 'status': 'merge-deferred', 'metadata': {'train': {'id': 'T1', 'order': 2}}},
    ]
    f = _make(
        task_id='103',
        metadata={'train': {'id': 'T1', 'order': 2}},
        tasks_by_train_return=members,
    )

    outcome_future = MergeOutcome('done', merge_sha='deadbeef')
    f.wf._await_cancellable = AsyncMock(return_value=outcome_future)  # type: ignore[method-assign]

    result = await f.wf._maybe_enqueue_group_merge()

    # (a) returns DONE
    assert result == WorkflowOutcome.DONE, (
        f'Expected WorkflowOutcome.DONE, got {result!r}'
    )

    # (b) exactly one item enqueued
    assert f.merge_queue.qsize() == 1, (
        f'Expected 1 item in merge_queue, got {f.merge_queue.qsize()}'
    )

    req = f.merge_queue.get_nowait()
    assert isinstance(req, GroupMergeRequest), (
        f'Expected GroupMergeRequest, got {type(req).__name__}'
    )

    # (c) GroupMergeRequest fields
    assert req.train_id == 'T1', f'train_id mismatch: {req.train_id!r}'
    assert req.member_task_ids == ['101', '102', '103'], (
        f'member_task_ids mismatch: {req.member_task_ids!r}'
    )
    assert req.tip_branch == '103', f'tip_branch mismatch: {req.tip_branch!r}'
    assert req.tip_task_id == '103', f'tip_task_id mismatch: {req.tip_task_id!r}'
    assert req.branch == '103', f'branch mismatch: {req.branch!r}'
    assert req.worktree == f.wf.worktree, f'worktree mismatch: {req.worktree!r}'

    # (d) mark_member_done callback
    await req.mark_member_done('101', 'sha9')
    f.scheduler.mark_done.assert_awaited_once()
    call_kwargs = f.scheduler.mark_done.call_args
    assert call_kwargs.kwargs.get('kind') == 'merged', (
        f'Expected kind=merged, got {call_kwargs.kwargs!r}'
    )
    assert call_kwargs.kwargs.get('sha') == 'sha9', (
        f'Expected sha=sha9, got {call_kwargs.kwargs!r}'
    )
    note = call_kwargs.kwargs.get('note', '')
    assert 'T1' in note, f'Expected T1 in note, got {note!r}'

    # (e) status_check callback unwraps the (dict, error) tuple
    statuses_dict = {'101': 'merge-deferred', '102': 'merge-deferred'}
    f.scheduler.get_statuses = AsyncMock(return_value=(statuses_dict, None))
    returned = await req.status_check(['101'])
    assert returned == statuses_dict, (
        f'status_check must return plain dict, got {returned!r}'
    )
