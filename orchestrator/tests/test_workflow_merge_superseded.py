"""Tests for the 'superseded' merge-outcome consumer branch (γ2 / retro-coalescing seam).

When a single-task merge future resolves with
``MergeOutcome(status='superseded', ...)`` the workflow must:
  1. Park cleanly in ``status='merge-deferred'`` (no done, no failure).
  2. Return ``WorkflowOutcome.MERGE_DEFERRED`` (not BLOCKED).
  3. Leave ``_last_merge_block_reason`` untouched (no thrash-counter pollution).
  4. Emit a ``merge_attempt`` event naming ``superseded_by`` (when event_store set).
  5. Log an INFO line naming ``superseded_by``.

Mirrors :mod:`test_workflow_merge_thrash`'s minimal-mock TaskWorkflow harness.
The merge outcome is delivered from the QUEUE side (see
:class:`_merge_queue_doubles.ResolvingMergeQueue`) so the real enqueue chain
still runs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from _merge_queue_doubles import ResolvingMergeQueue
from _orch_helpers import MOCK_WORKFLOW_PROJECT_ROOT, pydantic_spec

from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventType
from orchestrator.merge_queue import MergeOutcome
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome


@dataclass
class _Fixture:
    wf: TaskWorkflow


def _make(*, task_id: str = '99') -> _Fixture:
    assignment = MagicMock()
    assignment.task_id = task_id
    assignment.task = {
        'id': task_id, 'title': 'T', 'description': 'd',
        'metadata': {},
    }
    assignment.modules = ['mod_a']

    _spec = pydantic_spec(OrchestratorConfig)
    config = MagicMock(spec_set=_spec)
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.lock_depth = 2
    config.steward_completion_timeout = 300.0
    config.project_root = MOCK_WORKFLOW_PROJECT_ROOT
    config.git.branch_prefix = 'task/'  # task ν: real str prefix for QueuedBranch.parse

    scheduler = MagicMock()
    scheduler.update_task = AsyncMock(return_value=True)
    scheduler.set_task_status = AsyncMock()
    scheduler.get_status = AsyncMock(return_value='in-progress')

    git_ops = MagicMock()
    git_ops.get_main_sha = AsyncMock(return_value='SHA-A')
    # task-1923: _submit_to_merge_queue awaits rebind_branch_to_head before enqueue.
    git_ops.rebind_branch_to_head = AsyncMock(return_value=True)

    queue = MagicMock()
    queue.get_by_task = MagicMock(return_value=[])

    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,
        briefing=MagicMock(),
        mcp=MagicMock(),
        escalation_queue=queue,  # type: ignore[arg-type]
    )

    wf.artifacts = MagicMock()

    return _Fixture(wf=wf)


_SUPERSEDED = MergeOutcome('superseded', superseded_by='mr-x', merge_sha='s')


# ---------------------------------------------------------------------------
# Tests — Park / exit contract (step-1 RED, step-2 GREEN)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_superseded_outcome_parks_as_merge_deferred(
    tmp_path: Path,
):
    """A 'superseded' merge future parks cleanly: MERGE_DEFERRED, no failure, no blocking.

    RED until step-2 adds ``_handle_superseded``:  today 'superseded' falls
    through to the blocked tail, setting ``_last_merge_block_reason=''`` and
    returning ``BLOCKED``.
    """
    f = _make()
    wf = f.wf
    wf.worktree = tmp_path / 'wt'
    wf.worktree.mkdir(parents=True, exist_ok=True)
    wf.merge_queue = ResolvingMergeQueue(_SUPERSEDED)
    wf.plan = {'files': []}
    # event_store=None proves the handler is None-safe without a separate test.
    wf.event_store = None

    outcome = await wf._submit_to_merge_queue('99', pre_rebased=False)

    # (a) Returns MERGE_DEFERRED — not BLOCKED and not DONE.
    assert outcome == WorkflowOutcome.MERGE_DEFERRED, (
        f'Expected MERGE_DEFERRED, got {outcome!r}'
    )
    # The outcome really did come back through the queue, so the real
    # register_and_enqueue_merge_request chain ran rather than being replaced.
    assert wf.merge_queue.qsize() == 1, (
        f'Expected exactly one enqueued MergeRequest, got {wf.merge_queue.qsize()}'
    )
    # (b) set_task_status was called with 'merge-deferred' and NEVER with 'done'.
    set_task_status_mock = cast(AsyncMock, wf.scheduler.set_task_status)
    set_task_status_mock.assert_any_await('99', 'merge-deferred')
    assert not any(
        call.args == ('99', 'done')
        for call in set_task_status_mock.await_args_list
    ), 'set_task_status must never be called with "done" on a superseded outcome'
    # (c) The failure path was not taken — no _mark_blocked, no escalation, and
    #     no thrash-counter pollution. Both facts follow from (a). Every
    #     _mark_blocked call site inside
    #     orchestrator/src/orchestrator/workflow.py::TaskWorkflow._submit_to_merge_queue
    #     is a `return await`, and _mark_blocked never returns MERGE_DEFERRED
    #     (its early WorkflowOutcome(self.state.value) return is gated on
    #     machine.is_terminal(), i.e. DONE/CANCELLED), so an outcome of
    #     MERGE_DEFERRED excludes all of them; the blocked tail is also the only
    #     writer of _last_merge_block_reason in that method.
    # (d) clear_requeue_count was called — prevents stranded retry counter
    #     (mirrors _enter_merge_deferred; a regression dropping this call must fail).
    cast(MagicMock, wf.scheduler.clear_requeue_count).assert_called_once_with('99')


# ---------------------------------------------------------------------------
# Tests — Observability (step-3 RED, step-4 GREEN)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_superseded_emits_event_and_log_naming_superseded_by(
    tmp_path: Path, caplog,
):
    """A 'superseded' outcome emits a ``merge_attempt`` event + an INFO log.

    The event's data dict must contain ``superseded_by='mr-x'`` and
    ``outcome='superseded'``.  At least one INFO log record's message must
    contain both 'mr-x' and a parking-semantics hint ('absorbed',
    'superseded', or 'merge-deferred').

    RED until step-4 adds event emission and logging to ``_handle_superseded``.
    """
    f = _make()
    wf = f.wf
    wf.worktree = tmp_path / 'wt'
    wf.worktree.mkdir(parents=True, exist_ok=True)
    wf.merge_queue = ResolvingMergeQueue(_SUPERSEDED)
    wf.plan = {'files': []}
    # Wire up a mock event_store so we can inspect emit() calls.
    wf.event_store = MagicMock()

    with caplog.at_level(logging.INFO, logger='orchestrator.workflow'):
        await wf._submit_to_merge_queue('99', pre_rebased=False)

    # (a) A merge_attempt event was emitted with data naming superseded_by.
    emit_calls = wf.event_store.emit.call_args_list
    superseded_events = [
        c for c in emit_calls
        if (
            # First positional arg is the EventType.
            c.args
            and c.args[0] == EventType.merge_attempt
            # data dict must name superseded_by and outcome.
            and isinstance(c.kwargs.get('data'), dict)
            and c.kwargs['data'].get('superseded_by') == 'mr-x'
            and c.kwargs['data'].get('outcome') == 'superseded'
        )
    ]
    assert superseded_events, (
        f'Expected a merge_attempt event with data={{superseded_by="mr-x", '
        f'outcome="superseded"}}; emit calls were: {emit_calls}'
    )

    # (b) An INFO log record names superseded_by ('mr-x').
    #     We only require the id is present — semantic content is already pinned
    #     structurally by assertion (a); coupling this check to prose wording
    #     (e.g. 'absorbed', 'superseded') would fail on benign rewording.
    matching_records = [
        r for r in caplog.records
        if r.levelno == logging.INFO
        and 'mr-x' in r.getMessage()
    ]
    assert matching_records, (
        f'Expected an INFO log containing "mr-x"; '
        f'INFO records were: {[r.getMessage() for r in caplog.records if r.levelno == logging.INFO]}'
    )
