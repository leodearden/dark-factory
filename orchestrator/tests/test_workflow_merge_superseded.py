"""Tests for the 'superseded' merge-outcome consumer branch (γ2 / retro-coalescing seam).

When a single-task merge future resolves with
``MergeOutcome(status='superseded', ...)`` the workflow must:
  1. Park cleanly in ``status='merge-deferred'`` (no done, no failure).
  2. Return ``WorkflowOutcome.MERGE_DEFERRED`` (not BLOCKED).
  3. Leave ``_last_merge_block_reason`` untouched (no thrash-counter pollution).
  4. Emit a ``merge_attempt`` event naming ``superseded_by`` (when event_store set).
  5. Log an INFO line naming ``superseded_by``.

Mirrors :mod:`test_workflow_merge_thrash` — same minimal-mock TaskWorkflow
harness and ``orchestrator.merge_queue.enqueue_merge_request`` monkeypatch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from _orch_helpers import pydantic_spec

from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventType
from orchestrator.merge_queue import MergeOutcome, MergeRequest
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome


@dataclass
class _Fixture:
    wf: TaskWorkflow
    mark_blocked: AsyncMock


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
    config.project_root = Path('/tmp/non-existent-for-test')

    scheduler = MagicMock()
    scheduler.update_task = AsyncMock(return_value=True)
    scheduler.set_task_status = AsyncMock()
    scheduler.get_status = AsyncMock(return_value='in-progress')

    git_ops = MagicMock()
    git_ops.get_main_sha = AsyncMock(return_value='SHA-A')

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

    mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
    wf._mark_blocked = mark_blocked  # type: ignore[method-assign]

    return _Fixture(wf=wf, mark_blocked=mark_blocked)


# ---------------------------------------------------------------------------
# Tests — Park / exit contract (step-1 RED, step-2 GREEN)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_superseded_outcome_parks_as_merge_deferred(
    tmp_path: Path, monkeypatch,
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
    wf.merge_queue = MagicMock()
    wf.plan = {'files': []}
    wf._module_configs = []
    # Pre-seed a sentinel to detect any accidental thrash-capture overwrite.
    wf._last_merge_block_reason = 'sentinel'
    # event_store=None proves the handler is None-safe without a separate test.
    wf.event_store = None

    async def fake_enqueue(queue, req: MergeRequest, event_store, **_kwargs):
        req.result.set_result(
            MergeOutcome('superseded', superseded_by='mr-x', merge_sha='s'),
        )

    monkeypatch.setattr(
        'orchestrator.merge_queue.enqueue_merge_request', fake_enqueue,
    )

    outcome = await wf._submit_to_merge_queue('99', pre_rebased=False)

    # (a) Returns MERGE_DEFERRED — not BLOCKED and not DONE.
    assert outcome == WorkflowOutcome.MERGE_DEFERRED, (
        f'Expected MERGE_DEFERRED, got {outcome!r}'
    )
    # (b) set_task_status was called with 'merge-deferred' and NEVER with 'done'.
    set_task_status_mock = cast(AsyncMock, wf.scheduler.set_task_status)
    set_task_status_mock.assert_any_await('99', 'merge-deferred')
    assert not any(
        call.args == ('99', 'done')
        for call in set_task_status_mock.await_args_list
    ), 'set_task_status must never be called with "done" on a superseded outcome'
    # (c) _mark_blocked must NOT be called — no failure path, no escalation.
    f.mark_blocked.assert_not_awaited()
    # (d) _last_merge_block_reason untouched — no thrash-counter pollution.
    assert wf._last_merge_block_reason == 'sentinel', (
        f'Thrash sentinel must be unchanged; got {wf._last_merge_block_reason!r}'
    )


# ---------------------------------------------------------------------------
# Tests — Observability (step-3 RED, step-4 GREEN)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_superseded_emits_event_and_log_naming_superseded_by(
    tmp_path: Path, monkeypatch, caplog,
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
    wf.merge_queue = MagicMock()
    wf.plan = {'files': []}
    wf._module_configs = []
    wf._last_merge_block_reason = 'sentinel'
    # Wire up a mock event_store so we can inspect emit() calls.
    wf.event_store = MagicMock()

    async def fake_enqueue(queue, req: MergeRequest, event_store, **_kwargs):
        req.result.set_result(
            MergeOutcome('superseded', superseded_by='mr-x', merge_sha='s'),
        )

    monkeypatch.setattr(
        'orchestrator.merge_queue.enqueue_merge_request', fake_enqueue,
    )

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

    # (b) An INFO log record names superseded_by and conveys parking semantics.
    parking_keywords = ('absorbed', 'superseded', 'merge-deferred')
    matching_records = [
        r for r in caplog.records
        if r.levelno == logging.INFO
        and 'mr-x' in r.getMessage()
        and any(kw in r.getMessage() for kw in parking_keywords)
    ]
    assert matching_records, (
        f'Expected an INFO log containing "mr-x" and one of {parking_keywords}; '
        f'INFO records were: {[r.getMessage() for r in caplog.records if r.levelno == logging.INFO]}'
    )
