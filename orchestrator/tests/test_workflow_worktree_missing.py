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
from _merge_lane_fakes import DrivenMerge, drive_merge
from _orch_helpers import pydantic_spec

from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import OrchestratorConfig
from orchestrator.merge_queue import (
    WORKTREE_MISSING_REASON_PREFIX,
    MergeOutcome,
)
from orchestrator.workflow import (
    TaskWorkflow,
    WorkflowCancelled,
    WorkflowOutcome,
    WorkflowState,
)


def _make_workflow(
    *,
    tmp_path: Path,
    task_id: str = '999',
) -> tuple[TaskWorkflow, asyncio.Event]:
    assignment = MagicMock()
    assignment.task_id = task_id
    assignment.task = {'id': task_id, 'title': 'Tx', 'description': 'd'}
    assignment.modules = ['mod_a']

    _spec = pydantic_spec(OrchestratorConfig)
    config = MagicMock(spec_set=_spec)
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.max_review_cycles = 2
    config.max_amendment_rounds = 1
    config.lock_depth = 2
    config.steward_completion_timeout = 300.0
    config.project_root = tmp_path / 'proj'
    config.git.branch_prefix = 'task/'  # task ν: real str prefix for QueuedBranch.parse

    scheduler = MagicMock()
    scheduler.set_task_status = AsyncMock()
    git_ops = MagicMock()
    # task-1923: _submit_to_merge_queue awaits rebind_branch_to_head before enqueue.
    git_ops.rebind_branch_to_head = AsyncMock(return_value=True)

    cancel_event = asyncio.Event()
    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,
        briefing=MagicMock(),
        mcp=MagicMock(),
        cancel_event=cancel_event,
    )
    worktree = tmp_path / 'wt'
    worktree.mkdir(parents=True, exist_ok=True)
    wf.artifacts = TaskArtifacts(worktree)
    # init() so write_review has a root: the merge-failure review it writes is
    # what stands in for a stub on _write_merge_failure_review.
    wf.artifacts.init(task_id, 'Tx', 'd')
    wf.worktree = worktree
    # A REAL queue, so the REAL enqueue path runs (_submit_to_merge_queue ->
    # register_and_enqueue_merge_request -> the module-global
    # enqueue_merge_request -> queue.put) and the request a test resolves is
    # the one production actually parked there.
    wf.merge_queue = asyncio.Queue()
    wf.merge_inflight_registry = None  # skip the registry attach branch
    # _task_files is a property reading from self.plan; supply an empty plan
    # so the property returns None rather than raising.
    wf.plan = {'files': []}
    wf._module_configs = []
    return wf, cancel_event


async def _drive_worktree_missing(wf: TaskWorkflow) -> DrivenMerge:
    """Run the submit to completion, playing the merger for a worktree-missing block."""
    return await drive_merge(
        wf._submit_to_merge_queue('task/999', pre_rebased=False),
        wf.merge_queue,
        MergeOutcome(
            'blocked',
            reason=f'{WORKTREE_MISSING_REASON_PREFIX}: /tmp/gone',
        ),
    )


def _merge_review(wf: TaskWorkflow) -> dict | None:
    """The merge-failure review this run wrote, or None.

    ``_write_merge_failure_review`` ends at ``artifacts.write_review('merge', ...)``,
    and these workflows hold a REAL TaskArtifacts — so the file it leaves in
    ``.task/reviews/`` is the observable the stub used to stand in for.
    """
    return wf.artifacts.read_reviews().get('merge')


@pytest.mark.asyncio
async def test_worktree_missing_with_terminal_status_returns_done(
    tmp_path: Path,
):
    """Human marked task done → merge worker surfaces worktree-missing →
    workflow short-circuits to DONE without writing a merge-failure review.
    """
    wf, _cancel_event = _make_workflow(tmp_path=tmp_path)
    wf.scheduler.get_status = AsyncMock(return_value='done')

    driven = await _drive_worktree_missing(wf)

    assert driven.result == WorkflowOutcome.DONE
    assert _merge_review(wf) is None, 'the short-circuit writes no merge review'
    # …and never marks the row blocked, which is where _mark_blocked lands.
    wf.scheduler.set_task_status.assert_not_awaited()
    wf.scheduler.get_status.assert_awaited_once_with('999')


@pytest.mark.asyncio
async def test_worktree_missing_with_cancelled_status_returns_cancelled(
    tmp_path: Path,
):
    """Human CANCELLED the task → worktree-missing → short-circuit to CANCELLED.

    Task 3538 / boundary #14b: this fallback used to collapse every
    ``TERMINAL_STATUSES`` member onto DONE, so a cancellation was reported as
    a completion.  That is a live crash as well as a lie —
    ``_OUTCOME_ALLOWED['done'] == {DONE}``, so the DONE exit fails ``run()``'s
    SM-2 consistency check against the ``cancelled`` row — and it inflated the
    completed tally, which counts ``outcome == DONE``.  The CANCELLED branch
    also enters ``WorkflowState.CANCELLED`` from MERGE (SM-1 terminal
    absorption).  Sits beside the ``done`` case above so the two terminal rows
    read as one decision table.
    """
    wf, _cancel_event = _make_workflow(tmp_path=tmp_path)
    wf.state = WorkflowState.MERGE  # where this fallback is reached from
    wf.scheduler.get_status = AsyncMock(return_value='cancelled')

    driven = await _drive_worktree_missing(wf)

    assert driven.result == WorkflowOutcome.CANCELLED
    assert wf.machine.state is WorkflowState.CANCELLED
    assert _merge_review(wf) is None, 'the short-circuit writes no merge review'
    wf.scheduler.set_task_status.assert_not_awaited()
    wf.scheduler.get_status.assert_awaited_once_with('999')


@pytest.mark.asyncio
async def test_worktree_missing_with_nonterminal_status_falls_through(
    tmp_path: Path,
):
    """Worktree gone but task still in-progress → fall through to
    blocked + escalation (the existing path).  No silent DONE.
    """
    wf, _cancel_event = _make_workflow(tmp_path=tmp_path)
    wf.scheduler.get_status = AsyncMock(return_value='in-progress')

    driven = await _drive_worktree_missing(wf)

    assert driven.result == WorkflowOutcome.BLOCKED
    review = _merge_review(wf)
    assert review is not None and review['verdict'] == 'ISSUES_FOUND', (
        f'the fall-through must leave a merge-failure review, got: {review}'
    )
    wf.scheduler.set_task_status.assert_awaited_once_with('999', 'blocked')


@pytest.mark.asyncio
async def test_worktree_missing_with_get_status_error_falls_through(
    tmp_path: Path,
):
    """If ``scheduler.get_status`` itself fails (None), don't silently
    consume the failure as DONE — fall through so a human is notified.
    """
    wf, _cancel_event = _make_workflow(tmp_path=tmp_path)
    wf.scheduler.get_status = AsyncMock(return_value=None)

    driven = await _drive_worktree_missing(wf)

    assert driven.result == WorkflowOutcome.BLOCKED
    wf.scheduler.set_task_status.assert_awaited_once_with('999', 'blocked')


# ---------------------------------------------------------------------------
# Step 4: soft-cancel primitive
# ---------------------------------------------------------------------------


async def _submit_then_cancel(
    wf: TaskWorkflow, cancel_event: asyncio.Event,
) -> WorkflowCancelled:
    """Park a real submit on its merge future, then win the race with a cancel.

    Taking the request off the REAL queue (rather than resolving it) is what
    makes the race genuine: it proves the enqueue happened and leaves the
    future unresolved, which is the state ``_await_cancellable`` arbitrates.
    """
    submit = asyncio.ensure_future(
        wf._submit_to_merge_queue('task/x', pre_rebased=False)
    )
    try:
        await asyncio.wait_for(wf.merge_queue.get(), timeout=2)
        cancel_event.set()
        with pytest.raises(WorkflowCancelled) as excinfo:
            await asyncio.wait_for(submit, timeout=2)
        return excinfo.value
    finally:
        submit.cancel()


@pytest.mark.asyncio
async def test_cancel_event_during_merge_returns_done_when_terminal(
    tmp_path: Path,
):
    """Set the cancel event while the merge future is unresolved →
    ``_await_cancellable`` raises ``WorkflowCancelled('soft')`` (W9-θ).

    The terminal-status → DONE decision this test's name refers to no longer
    happens at this layer: ``_await_cancellable`` raises unconditionally on a
    cancel-win, regardless of scheduler status, and propagates straight to
    ``run()``'s single ``WorkflowCancelled`` catch. The scheduler-status-aware
    DONE-vs-SOFT_CANCELLED decision now lives entirely in
    ``_finalise_cancellation``/``_handle_soft_cancel`` — see
    ``TestHandleSoftCancelOutcome`` in test_workflow.py, which pins the
    terminal→DONE branch directly.
    """
    wf, cancel_event = _make_workflow(tmp_path=tmp_path)
    wf.scheduler.get_status = AsyncMock(return_value='done')

    cancelled = await _submit_then_cancel(wf, cancel_event)

    assert cancelled.kind == 'soft'


@pytest.mark.asyncio
async def test_cancel_event_during_merge_soft_cancels_when_nonterminal(
    tmp_path: Path,
):
    """Cancel-event set + status non-terminal → ``_await_cancellable`` raises
    ``WorkflowCancelled('soft')`` (W9-θ) — same raise as the terminal case
    above; see that test's docstring for where the status-aware decision
    (SOFT_CANCELLED vs DONE) now lives.
    """
    wf, cancel_event = _make_workflow(tmp_path=tmp_path)
    wf.scheduler.get_status = AsyncMock(return_value='in-progress')

    cancelled = await _submit_then_cancel(wf, cancel_event)

    assert cancelled.kind == 'soft'
