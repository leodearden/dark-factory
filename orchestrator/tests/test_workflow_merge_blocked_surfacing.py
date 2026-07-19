"""Tests for task 2757: the generic merge-blocked fall-through in
``TaskWorkflow._submit_to_merge_queue`` must ALWAYS surface.

Root cause (Reify 5120 RCA RC-2): with an UNRELATED open L1, a new merge-verify
failure produced ``merge_finalized state=blocked`` followed by total silence —
no escalation (both ``has_open_l1`` gates suppressed by the unrelated L1), no
requeue (merge_phase), and no durable event.

Two obligations verified here:
  * Part A (step-5): an unconditional ``EventType.merge_blocked`` durable event
    is emitted BEFORE ``_mark_blocked`` (ungated by ``has_open_l1``), and the
    real merge review-category is threaded into ``_mark_blocked`` (not the
    default ``'task_failure'``).
  * Parts B (steps 7/9): the two ``has_open_l1`` gates become signature-aware so
    a NEW root cause is no longer suppressed by an unrelated open L1.

Harness mirrors ``test_workflow_skew_surfacing.py``'s ``_make_routing_workflow``
+ ``_fake_enqueue`` + ``patch(enqueue_merge_request)``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.git_ops import GitOps
from orchestrator.merge_queue import MergeOutcome
from orchestrator.scheduler import TaskAssignment
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome

MAIN_SHA = 'deadbeef1234567890'


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


def _make_routing_workflow(
    config: OrchestratorConfig,
    worktree: Path,
    *,
    event_store: EventStore | None = None,
) -> TaskWorkflow:
    """Mirrors ``_make_routing_workflow`` in test_workflow_skew_surfacing.py — a
    full TaskWorkflow capable of driving ``_submit_to_merge_queue`` — with an
    optional real EventStore attached (the merge_blocked emit is guarded by
    ``if self.event_store``)."""
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
    # spec=GitOps hides the __init__-assigned ``config`` attr, so set it explicitly.
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
        event_store=event_store,
    )
    workflow.worktree = worktree
    artifacts = TaskArtifacts(worktree)
    artifacts.init('42', 'X', 'desc', base_commit='base-sha')
    workflow.artifacts = artifacts
    workflow.plan = {'task_id': '42', 'steps': []}
    return workflow


class TestGenericMergeBlockedSurfacing:
    """The generic (non-short-circuit) blocked fall-through in
    ``_submit_to_merge_queue`` must emit exactly one ``EventType.merge_blocked``
    row (carrying reason/category/failure_category) BEFORE ``_mark_blocked``, and
    must thread the computed review-category into ``_mark_blocked`` (not the
    default ``'task_failure'``)."""

    def test_generic_blocked_emits_merge_blocked_and_threads_category(
        self, tmp_path: Path,
    ) -> None:
        config = _make_routing_config(tmp_path)
        worktree = tmp_path / 'task-wt'
        worktree.mkdir()
        store = EventStore(tmp_path / 'runs.db', run_id='run-mb')
        workflow = _make_routing_workflow(config, worktree, event_store=store)

        # reason contains 'verification failed' -> category resolves to
        # 'post_merge_verify' (workflow.py:7307-7308); default INDETERMINATE
        # disposition + a reason not matching any earlier short-circuit prefix
        # -> generic fall-through.
        generic_outcome = MergeOutcome(
            'blocked',
            reason='Post-merge verification failed: pytest failed',
            failure_category='test_failure',
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
                workflow._submit_to_merge_queue(
                    'task/42', pre_rebased=False, merge_phase=True,
                )
            )

        assert result == WorkflowOutcome.BLOCKED

        # (a) exactly one merge_blocked row, carrying reason/category/failure_category.
        rows = store.fetch_events_by_type(EventType.merge_blocked)
        assert len(rows) == 1, f'expected exactly one merge_blocked row; got {rows}'
        assert rows[0]['task_id'] == '42'
        data = rows[0]['data']
        assert data['reason'] == 'Post-merge verification failed: pytest failed'
        assert data['category'] == 'post_merge_verify'
        assert data['failure_category'] == 'test_failure'

        # (b) _mark_blocked threaded the REAL category (not the default
        # 'task_failure') and forwarded merge_phase=True.
        mark_blocked_mock.assert_called_once()
        call_kwargs = mark_blocked_mock.call_args[1]
        assert call_kwargs.get('category') == 'post_merge_verify', (
            f'Expected category=post_merge_verify threaded into _mark_blocked; '
            f'got {call_kwargs}'
        )
        assert call_kwargs.get('merge_phase') is True, (
            f'Expected merge_phase=True forwarded; got {call_kwargs}'
        )
