"""Tests for task 1707 δ workflow wiring — train attribution in _maybe_enqueue_group_merge.

Step-7  (RED): detection wiring — TRAIN_VERIFY_FAILED_REASON_PREFIX routes to
               _attribute_train_failure; other blocked reasons still route to
               _mark_blocked.
Step-9  (RED): all-pass → escalate the TRAIN (interaction); land nothing.
Step-11 (RED): some-fail → land passers, block offender.
Step-13 (RED): edge cases — tip-as-offender, un-stack conflict, advance failure.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from _orch_helpers import pydantic_spec

from orchestrator.config import OrchestratorConfig
from orchestrator.merge_queue import MergeOutcome
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome


# ---------------------------------------------------------------------------
# Shared fixture helper (mirrors test_workflow_train_completion._make)
# ---------------------------------------------------------------------------


@dataclass
class _Fixture:
    wf: TaskWorkflow
    scheduler: MagicMock
    git_ops: MagicMock
    mark_blocked: AsyncMock
    esc_queue: MagicMock
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
        'status': 'merge-deferred',
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
            return_value=(
                {'101': 'merge-deferred', '102': 'merge-deferred', '103': 'merge-deferred'},
                None,
            )
        )

    git_ops = MagicMock()
    git_ops.config.branch_prefix = 'task/'
    git_ops.config.main_branch = 'main'
    git_ops.advance_main = AsyncMock(return_value='advanced')
    git_ops.cleanup_merge_worktree = AsyncMock()

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
        git_ops=git_ops,
        mark_blocked=mark_blocked,
        esc_queue=esc_queue,
        merge_queue=merge_queue,
    )


def _train_members(
    train_id: str = 'T-attr', tip_id: str = '103',
) -> list[dict]:
    """Return a 3-member ordered member list suitable for tasks_by_train."""
    return [
        {'id': '101', 'status': 'merge-deferred',
         'metadata': {'train': {'id': train_id, 'order': 0, 'members': ['101', '102', tip_id]}}},
        {'id': '102', 'status': 'merge-deferred',
         'metadata': {'train': {'id': train_id, 'order': 1, 'members': ['101', '102', tip_id]}}},
        {'id': tip_id, 'status': 'merge-deferred',
         'metadata': {'train': {'id': train_id, 'order': 2, 'members': ['101', '102', tip_id]}}},
    ]


# ---------------------------------------------------------------------------
# Step-7: Detection wiring — TRAIN_VERIFY_FAILED_REASON_PREFIX branch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDetectionWiring:
    """TRAIN_VERIFY_FAILED_REASON_PREFIX routes to _attribute_train_failure."""

    async def test_tagged_outcome_calls_attribute_train_failure(self) -> None:
        """Tagged blocked outcome → _attribute_train_failure is called; _mark_blocked is NOT."""
        from orchestrator.merge_queue import TRAIN_VERIFY_FAILED_REASON_PREFIX

        members = _train_members(train_id='T-attr', tip_id='103')
        f = _make(
            task_id='103',
            metadata={'train': {'id': 'T-attr', 'order': 2, 'members': ['101', '102', '103']}},
            tasks_by_train_return=members,
        )

        tagged_outcome = MergeOutcome(
            'blocked',
            reason=f'{TRAIN_VERIFY_FAILED_REASON_PREFIX}: 3 tests failed',
            failure_category='cargo_test',
        )
        f.wf._await_cancellable = AsyncMock(return_value=tagged_outcome)  # type: ignore[method-assign]

        attr_mock = AsyncMock(return_value=WorkflowOutcome.DONE)
        f.wf._attribute_train_failure = attr_mock  # type: ignore[method-assign]

        result = await f.wf._maybe_enqueue_group_merge()

        # _attribute_train_failure called once with the result, train_id, and members
        attr_mock.assert_awaited_once()
        call_args = attr_mock.call_args
        assert call_args[0][0] is tagged_outcome, (
            f'first arg should be the result, got {call_args[0][0]!r}'
        )
        assert call_args[0][1] == 'T-attr', (
            f'second arg should be train_id="T-attr", got {call_args[0][1]!r}'
        )
        # third arg: member list (ordered root→tip)
        passed_members = call_args[0][2]
        assert len(passed_members) == 3

        # return value from _attribute_train_failure is propagated
        assert result == WorkflowOutcome.DONE

        # _mark_blocked must NOT be called
        f.mark_blocked.assert_not_awaited()

    async def test_untagged_blocked_routes_to_mark_blocked(self) -> None:
        """Non-tagged blocked outcome still routes to _mark_blocked (unaffected path)."""
        from orchestrator.merge_queue import TRAIN_VERIFY_FAILED_REASON_PREFIX

        members = _train_members(train_id='T-attr2', tip_id='103')
        f = _make(
            task_id='103',
            metadata={'train': {'id': 'T-attr2', 'order': 2, 'members': ['101', '102', '103']}},
            tasks_by_train_return=members,
        )

        other_outcome = MergeOutcome(
            'blocked',
            reason='Train merge advance failed: cas_failed',
        )
        f.wf._await_cancellable = AsyncMock(return_value=other_outcome)  # type: ignore[method-assign]

        attr_mock = AsyncMock(return_value=WorkflowOutcome.DONE)
        f.wf._attribute_train_failure = attr_mock  # type: ignore[method-assign]

        # Add a fake merge_worker with is_wip_halted=False so the orphan-halt
        # probe doesn't fire (we want to confirm _mark_blocked is the fallback)
        wip_worker = MagicMock()
        wip_worker.is_wip_halted = False
        wip_worker.halt_owner_esc_id = None
        f.wf.merge_worker = wip_worker  # type: ignore[attr-defined]

        await f.wf._maybe_enqueue_group_merge()

        # _attribute_train_failure must NOT be called
        attr_mock.assert_not_awaited()
        # _mark_blocked IS called
        f.mark_blocked.assert_awaited_once()

    async def test_train_incomplete_path_unaffected(self) -> None:
        """train_incomplete blocked outcome still returns None (park) — unaffected."""
        from orchestrator.merge_queue import (
            TRAIN_INCOMPLETE_REASON_PREFIX,
            TRAIN_VERIFY_FAILED_REASON_PREFIX,
        )

        members = _train_members(train_id='T-incomplete', tip_id='103')
        f = _make(
            task_id='103',
            metadata={'train': {'id': 'T-incomplete', 'order': 2, 'members': ['101', '102', '103']}},
            tasks_by_train_return=members,
        )

        incomplete_outcome = MergeOutcome(
            'blocked',
            reason=f'{TRAIN_INCOMPLETE_REASON_PREFIX}: member 101 is in-progress',
        )
        f.wf._await_cancellable = AsyncMock(return_value=incomplete_outcome)  # type: ignore[method-assign]

        attr_mock = AsyncMock(return_value=WorkflowOutcome.DONE)
        f.wf._attribute_train_failure = attr_mock  # type: ignore[method-assign]

        result = await f.wf._maybe_enqueue_group_merge()

        assert result is None, f'train_incomplete should return None (park), got {result!r}'
        attr_mock.assert_not_awaited()
        f.mark_blocked.assert_not_awaited()
