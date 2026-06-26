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
from unittest.mock import AsyncMock, MagicMock

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
    # task-1923/diff-5d: tip group-merge awaits release_lane_for_terminal_task
    # per done member and rebind_branch_to_head before enqueue.
    git_ops.release_lane_for_terminal_task = AsyncMock(return_value=None)
    git_ops.rebind_branch_to_head = AsyncMock(return_value=True)

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


# ---------------------------------------------------------------------------
# Step-5: Invariant (no-fire) tests — RED against step-4's unconditional impl
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_partial_train_does_not_fire():
    """Test A — partial train: tip present but sibling '102' is not merge-deferred.

    _maybe_enqueue_group_merge must return None and the asyncio.Queue must
    remain EMPTY (no GroupMergeRequest enqueued).
    """
    members = [
        {'id': '101', 'status': 'merge-deferred', 'metadata': {'train': {'id': 'T1', 'order': 0}}},
        {'id': '102', 'status': 'in-progress', 'metadata': {'train': {'id': 'T1', 'order': 1}}},
        {'id': '103', 'status': 'merge-deferred', 'metadata': {'train': {'id': 'T1', 'order': 2}}},
    ]
    f = _make(
        task_id='103',
        metadata={'train': {'id': 'T1', 'order': 2}},
        tasks_by_train_return=members,
    )
    # Should not be called, but stub defensively
    f.wf._await_cancellable = AsyncMock(return_value=MergeOutcome('done'))  # type: ignore[method-assign]

    result = await f.wf._maybe_enqueue_group_merge()

    assert result is None, (
        f'Expected None (partial train must not fire), got {result!r}'
    )
    assert f.merge_queue.empty(), (
        'Queue must be empty — GroupMergeRequest must not be enqueued for incomplete train'
    )


@pytest.mark.asyncio
async def test_self_status_trusted_over_lagging_get_tasks():
    """Test B — self's just-written status is trusted even if get_tasks still shows old value.

    Scenario: self is tip (103, order 2); '101' and '102' are merge-deferred;
    '103' is listed as 'in-progress' in the get_tasks result (simulating a
    lag).  The trigger must trust self's status as 'merge-deferred' (just
    written) and still FIRE (return non-None) when all others are deferred.
    """
    members = [
        {'id': '101', 'status': 'merge-deferred', 'metadata': {'train': {'id': 'T1', 'order': 0}}},
        {'id': '102', 'status': 'merge-deferred', 'metadata': {'train': {'id': 'T1', 'order': 1}}},
        # NOTE: stale status for self — trigger must override with 'merge-deferred'
        {'id': '103', 'status': 'in-progress', 'metadata': {'train': {'id': 'T1', 'order': 2}}},
    ]
    f = _make(
        task_id='103',
        metadata={'train': {'id': 'T1', 'order': 2}},
        tasks_by_train_return=members,
    )
    f.wf._await_cancellable = AsyncMock(return_value=MergeOutcome('done', merge_sha='abc'))  # type: ignore[method-assign]

    result = await f.wf._maybe_enqueue_group_merge()

    assert result == WorkflowOutcome.DONE, (
        f'Expected DONE — self status must be trusted as merge-deferred; got {result!r}'
    )
    assert not f.merge_queue.empty(), (
        'GroupMergeRequest must be enqueued when all members are effectively deferred'
    )


@pytest.mark.asyncio
async def test_non_tip_member_does_not_fire():
    """Test C — non-tip member (order 0) must not enqueue the GroupMergeRequest.

    Even when all 3 members are merge-deferred, only the tip (highest order)
    should fire.  Task '101' (order 0) must return None with empty queue.
    """
    members = [
        {'id': '101', 'status': 'merge-deferred', 'metadata': {'train': {'id': 'T1', 'order': 0}}},
        {'id': '102', 'status': 'merge-deferred', 'metadata': {'train': {'id': 'T1', 'order': 1}}},
        {'id': '103', 'status': 'merge-deferred', 'metadata': {'train': {'id': 'T1', 'order': 2}}},
    ]
    # Build workflow for the ROOT member (101, order 0) — NOT the tip
    f = _make(
        task_id='101',
        metadata={'train': {'id': 'T1', 'order': 0}},
        tasks_by_train_return=members,
    )
    f.wf._await_cancellable = AsyncMock(return_value=MergeOutcome('done'))  # type: ignore[method-assign]

    result = await f.wf._maybe_enqueue_group_merge()

    assert result is None, (
        f'Expected None — non-tip must not fire GroupMergeRequest; got {result!r}'
    )
    assert f.merge_queue.empty(), (
        'Queue must be empty — only the tip should enqueue a GroupMergeRequest'
    )


@pytest.mark.asyncio
async def test_enter_merge_deferred_returns_merge_deferred_when_trigger_parks():
    """Test D — _enter_merge_deferred returns MERGE_DEFERRED when trigger returns None.

    Patch _maybe_enqueue_group_merge to AsyncMock→None; assert that
    _enter_merge_deferred sets status=merge-deferred and returns MERGE_DEFERRED.
    """
    f = _make(
        task_id='103',
        metadata={'train': {'id': 'T1', 'order': 2}},
    )
    f.wf._maybe_enqueue_group_merge = AsyncMock(return_value=None)  # type: ignore[method-assign]

    outcome = await f.wf._enter_merge_deferred()

    # status written
    f.scheduler.set_task_status.assert_awaited_once_with('103', 'merge-deferred')
    # requeue count cleared
    f.scheduler.clear_requeue_count.assert_called_once_with('103')
    # outcome
    assert outcome == WorkflowOutcome.MERGE_DEFERRED, (
        f'Expected MERGE_DEFERRED when trigger parks, got {outcome!r}'
    )


# ---------------------------------------------------------------------------
# Step-7: Failure-path tests (blocked outcome + soft-cancel)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_blocked_outcome_escalates_to_human():
    """Tip fires but merge fails with status='blocked' → _mark_blocked(escalate_to_human=True).

    _await_cancellable returns MergeOutcome('blocked', reason='...rebase conflict...').
    Assert:
      (a) result is WorkflowOutcome.BLOCKED
      (b) _mark_blocked was awaited once with escalate_to_human=True
      (c) the reason string passed to _mark_blocked contains the train_id and
          the failed outcome status/reason.
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
    failed_outcome = MergeOutcome(
        'blocked', reason='Train merge rejected: tip branch rebase conflict on main',
    )
    f.wf._await_cancellable = AsyncMock(return_value=failed_outcome)  # type: ignore[method-assign]

    result = await f.wf._maybe_enqueue_group_merge()

    # (a)
    assert result == WorkflowOutcome.BLOCKED, (
        f'Expected WorkflowOutcome.BLOCKED on merge failure, got {result!r}'
    )

    # (b) _mark_blocked called with escalate_to_human=True
    f.mark_blocked.assert_awaited_once()
    call_args = f.mark_blocked.call_args
    kwargs = call_args.kwargs if call_args.kwargs else {}
    # escalate_to_human may come as a positional or keyword arg
    all_args = str(call_args)
    assert 'escalate_to_human=True' in all_args or kwargs.get('escalate_to_human') is True, (
        f'Expected escalate_to_human=True in _mark_blocked call: {call_args!r}'
    )

    # (c) reason contains train_id and outcome info
    reason_arg = call_args.args[0] if call_args.args else ''
    assert 'T1' in reason_arg, f'Expected train_id "T1" in reason: {reason_arg!r}'
    assert 'blocked' in reason_arg, f'Expected "blocked" in reason: {reason_arg!r}'


@pytest.mark.asyncio
async def test_soft_cancel_delegates_to_handle_soft_cancel():
    """_await_cancellable returns None → _handle_soft_cancel('group-merge') is invoked.

    Patch _handle_soft_cancel to return WorkflowOutcome.REQUEUED; assert
    _maybe_enqueue_group_merge returns that value.
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
    f.wf._await_cancellable = AsyncMock(return_value=None)  # type: ignore[method-assign]
    handle_soft_cancel = AsyncMock(return_value=WorkflowOutcome.REQUEUED)
    f.wf._handle_soft_cancel = handle_soft_cancel  # type: ignore[method-assign]

    result = await f.wf._maybe_enqueue_group_merge()

    handle_soft_cancel.assert_awaited_once_with('group-merge')
    assert result == WorkflowOutcome.REQUEUED, (
        f'Expected REQUEUED from _handle_soft_cancel, got {result!r}'
    )
