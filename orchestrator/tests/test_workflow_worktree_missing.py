"""Workflow short-circuits when the merge queue reports worktree-missing
and the task is already terminal.

The bug: a human marks a task ``done`` and removes the worktree mid-merge.
The merge worker now surfaces ``MergeOutcome('blocked', reason='Worktree
missing: <path>')``.  ``TaskWorkflow._submit_to_merge_queue`` must re-check
task status; if terminal, return ``WorkflowOutcome.DONE`` cleanly without
creating an escalation or writing a merge-failure review.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.artifacts import TaskArtifacts
from orchestrator.merge_queue import (
    WORKTREE_MISSING_REASON_PREFIX,
    MergeOutcome,
    MergeRequest,
)
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome


def _make_workflow(
    *,
    tmp_path: Path,
    task_id: str = '999',
) -> TaskWorkflow:
    assignment = MagicMock()
    assignment.task_id = task_id
    assignment.task = {'id': task_id, 'title': 'Tx', 'description': 'd'}
    assignment.modules = ['mod_a']

    config = MagicMock()
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.max_review_cycles = 2
    config.max_amendment_rounds = 1
    config.lock_depth = 2
    config.steward_completion_timeout = 300.0
    config.project_root = tmp_path / 'proj'

    scheduler = MagicMock()
    git_ops = MagicMock()

    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,
        briefing=MagicMock(),
        mcp=MagicMock(),
    )
    worktree = tmp_path / 'wt'
    worktree.mkdir(parents=True, exist_ok=True)
    wf.artifacts = TaskArtifacts(worktree)
    wf.worktree = worktree
    wf.merge_queue = MagicMock()
    # _task_files is a property reading from self.plan; supply an empty plan
    # so the property returns None rather than raising.
    wf.plan = {'files': []}
    wf._module_configs = []
    return wf


def _patch_enqueue_with_outcome(monkeypatch, outcome: MergeOutcome) -> None:
    """Replace ``enqueue_merge_request`` so the request's future resolves
    immediately with ``outcome``.  Skips the real merge worker.
    """
    async def fake_enqueue(queue, req: MergeRequest, event_store):
        req.result.set_result(outcome)

    monkeypatch.setattr(
        'orchestrator.merge_queue.enqueue_merge_request', fake_enqueue,
    )


@pytest.mark.asyncio
async def test_worktree_missing_with_terminal_status_returns_done(
    tmp_path: Path, monkeypatch,
):
    """Human marked task done → merge worker surfaces worktree-missing →
    workflow short-circuits to DONE without writing a merge-failure review.
    """
    wf = _make_workflow(tmp_path=tmp_path)
    wf.scheduler.get_status = AsyncMock(return_value='done')
    write_review = MagicMock()
    wf._write_merge_failure_review = write_review  # type: ignore[method-assign]
    mark_blocked = AsyncMock()
    wf._mark_blocked = mark_blocked  # type: ignore[method-assign]

    _patch_enqueue_with_outcome(
        monkeypatch,
        MergeOutcome(
            'blocked',
            reason=f'{WORKTREE_MISSING_REASON_PREFIX}: /tmp/gone',
        ),
    )

    outcome = await wf._submit_to_merge_queue('task/999', pre_rebased=False)

    assert outcome == WorkflowOutcome.DONE
    write_review.assert_not_called()
    mark_blocked.assert_not_awaited()
    wf.scheduler.get_status.assert_awaited_once_with('999')


@pytest.mark.asyncio
async def test_worktree_missing_with_nonterminal_status_falls_through(
    tmp_path: Path, monkeypatch,
):
    """Worktree gone but task still in-progress → fall through to
    blocked + escalation (the existing path).  No silent DONE.
    """
    wf = _make_workflow(tmp_path=tmp_path)
    wf.scheduler.get_status = AsyncMock(return_value='in-progress')
    write_review = MagicMock()
    wf._write_merge_failure_review = write_review  # type: ignore[method-assign]
    mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
    wf._mark_blocked = mark_blocked  # type: ignore[method-assign]

    _patch_enqueue_with_outcome(
        monkeypatch,
        MergeOutcome(
            'blocked',
            reason=f'{WORKTREE_MISSING_REASON_PREFIX}: /tmp/gone',
        ),
    )

    outcome = await wf._submit_to_merge_queue('task/999', pre_rebased=False)

    assert outcome == WorkflowOutcome.BLOCKED
    write_review.assert_called_once()
    mark_blocked.assert_awaited_once()


@pytest.mark.asyncio
async def test_worktree_missing_with_get_status_error_falls_through(
    tmp_path: Path, monkeypatch,
):
    """If ``scheduler.get_status`` itself fails (None), don't silently
    consume the failure as DONE — fall through so a human is notified.
    """
    wf = _make_workflow(tmp_path=tmp_path)
    wf.scheduler.get_status = AsyncMock(return_value=None)
    write_review = MagicMock()
    wf._write_merge_failure_review = write_review  # type: ignore[method-assign]
    mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
    wf._mark_blocked = mark_blocked  # type: ignore[method-assign]

    _patch_enqueue_with_outcome(
        monkeypatch,
        MergeOutcome(
            'blocked',
            reason=f'{WORKTREE_MISSING_REASON_PREFIX}: /tmp/gone',
        ),
    )

    outcome = await wf._submit_to_merge_queue('task/999', pre_rebased=False)

    assert outcome == WorkflowOutcome.BLOCKED
    mark_blocked.assert_awaited_once()


# ---------------------------------------------------------------------------
# Step 4: soft-cancel primitive
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_event_during_merge_returns_done_when_terminal(
    tmp_path: Path, monkeypatch,
):
    """Set ``_cancel_event`` while merge future is unresolved → workflow
    re-checks status → returns DONE.
    """
    wf = _make_workflow(tmp_path=tmp_path)
    wf.scheduler.get_status = AsyncMock(return_value='done')

    # enqueue without resolving the future, then set cancel_event so the
    # _await_cancellable race picks the cancel.
    async def fake_enqueue(queue, req: MergeRequest, event_store):
        # Schedule the cancel after a moment so the race is real
        async def _do_cancel():
            await asyncio.sleep(0.01)
            wf._cancel_event.set()
        asyncio.create_task(_do_cancel())

    monkeypatch.setattr(
        'orchestrator.merge_queue.enqueue_merge_request', fake_enqueue,
    )

    outcome = await asyncio.wait_for(
        wf._submit_to_merge_queue('task/x', pre_rebased=False),
        timeout=2,
    )
    assert outcome == WorkflowOutcome.DONE


@pytest.mark.asyncio
async def test_cancel_event_during_merge_requeues_when_nonterminal(
    tmp_path: Path, monkeypatch,
):
    """Cancel-event set + status non-terminal → REQUEUED (slot recycles)."""
    wf = _make_workflow(tmp_path=tmp_path)
    wf.scheduler.get_status = AsyncMock(return_value='in-progress')

    async def fake_enqueue(queue, req: MergeRequest, event_store):
        async def _do_cancel():
            await asyncio.sleep(0.01)
            wf._cancel_event.set()
        asyncio.create_task(_do_cancel())

    monkeypatch.setattr(
        'orchestrator.merge_queue.enqueue_merge_request', fake_enqueue,
    )

    outcome = await asyncio.wait_for(
        wf._submit_to_merge_queue('task/x', pre_rebased=False),
        timeout=2,
    )
    assert outcome == WorkflowOutcome.REQUEUED
