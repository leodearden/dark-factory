"""Tests for Fix 3 — anti-thrash guard on repeated steward-resolved merge failures.

Mirrors :mod:`test_workflow_infra_thrash`: builds a minimal
:class:`TaskWorkflow` with mocks and drives ``_check_merge_outcome_thrash``
directly to assert state transitions.

Signature (a sha256-short fingerprint of the blocked-reason text) keys the
counter.  Same signature on consecutive REQUEUED outcomes → counter
increments.  Different signature → reset to 1 (steward made progress on a
different verdict).  At ``config.max_consecutive_merge_thrash`` the helper
short-circuits to ``_mark_blocked(escalate_to_human=True)`` instead of the
caller blindly resubmitting the same merge.

The ``dropped_plan_targets`` reason prefix is intentionally excluded — it
short-circuits to L1 in :meth:`_submit_to_merge_queue` (Fix 2) before
reaching the steward path, so its signature never enters the counter.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _orch_helpers import pydantic_spec

from orchestrator.config import OrchestratorConfig
from orchestrator.merge_queue import (
    DROPPED_PLAN_TARGETS_REASON_PREFIX,
    POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX,
    MergeOutcome,
    MergeRequest,
)
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome


@dataclass
class _Fixture:
    wf: TaskWorkflow
    update_task: AsyncMock
    mark_blocked: AsyncMock


def _make(
    *,
    task_id: str = '99',
    metadata: dict | None = None,
    update_task_raises: bool = False,
    max_consecutive_merge_thrash: int = 2,
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
    config.max_consecutive_merge_thrash = max_consecutive_merge_thrash

    if update_task_raises:
        update_task = AsyncMock(side_effect=RuntimeError('mcp down'))
    else:
        update_task = AsyncMock(return_value=True)

    scheduler = MagicMock()
    scheduler.update_task = update_task
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

    return _Fixture(wf=wf, update_task=update_task, mark_blocked=mark_blocked)


def _persisted_metadata(update_task: AsyncMock) -> dict:
    assert update_task.await_args is not None
    args, kwargs = update_task.await_args
    return kwargs.get('metadata') or args[1]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_signature_match_at_threshold_promotes_to_l1():
    """Two consecutive identical signatures (default threshold=2) → escalate_to_human=True."""
    f = _make(
        metadata={
            'consecutive_merge_thrash': 1,
            'last_merge_outcome_signature': 'sig-abc',
        },
        max_consecutive_merge_thrash=2,
    )

    outcome = await f.wf._check_merge_outcome_thrash(
        prev_signature='sig-abc', current_signature='sig-abc',
    )

    assert outcome == WorkflowOutcome.BLOCKED
    f.mark_blocked.assert_awaited_once()
    args, kwargs = f.mark_blocked.await_args
    assert kwargs.get('escalate_to_human') is True
    # Reason mentions thrash so an operator can grok the L1
    assert 'thrash' in args[0].lower() or 'counter=2' in args[0]


@pytest.mark.asyncio
async def test_signature_mismatch_resets_counter():
    """Different verdict on second attempt → counter resets to 1."""
    f = _make(
        metadata={
            'consecutive_merge_thrash': 1,
            'last_merge_outcome_signature': 'sig-abc',
        },
    )

    outcome = await f.wf._check_merge_outcome_thrash(
        prev_signature='sig-abc', current_signature='sig-xyz',
    )

    assert outcome is None
    md = _persisted_metadata(f.update_task)
    assert md['consecutive_merge_thrash'] == 1, (
        f'Counter must reset to 1 on signature mismatch, got {md}'
    )
    assert md['last_merge_outcome_signature'] == 'sig-xyz'
    f.mark_blocked.assert_not_awaited()


@pytest.mark.asyncio
async def test_first_signature_below_threshold_falls_through():
    """First merge thrash (counter starts at 0) → counter==1, fall through."""
    f = _make(metadata={})

    outcome = await f.wf._check_merge_outcome_thrash(
        prev_signature=None, current_signature='sig-1',
    )

    assert outcome is None
    md = _persisted_metadata(f.update_task)
    assert md['consecutive_merge_thrash'] == 1
    assert md['last_merge_outcome_signature'] == 'sig-1'


@pytest.mark.asyncio
async def test_threshold_is_configurable():
    """max_consecutive_merge_thrash=3 → promotion happens one cycle later."""
    f = _make(
        metadata={
            'consecutive_merge_thrash': 1,
            'last_merge_outcome_signature': 'sig-abc',
        },
        max_consecutive_merge_thrash=3,
    )

    # Counter goes 1 → 2 (still below threshold=3 → fall through)
    outcome = await f.wf._check_merge_outcome_thrash(
        prev_signature='sig-abc', current_signature='sig-abc',
    )
    assert outcome is None
    md = _persisted_metadata(f.update_task)
    assert md['consecutive_merge_thrash'] == 2
    f.mark_blocked.assert_not_awaited()


@pytest.mark.asyncio
async def test_corrupt_counter_metadata_treated_as_zero():
    """Non-int counter (legacy task) must not crash the helper."""
    f = _make(
        metadata={
            'consecutive_merge_thrash': 'two',  # corrupt
            'last_merge_outcome_signature': 'sig-abc',
        },
    )

    outcome = await f.wf._check_merge_outcome_thrash(
        prev_signature='sig-abc', current_signature='sig-abc',
    )

    # corrupt counter → 0; matching signature → 1; below threshold → None
    assert outcome is None
    md = _persisted_metadata(f.update_task)
    assert md['consecutive_merge_thrash'] == 1


@pytest.mark.asyncio
async def test_metadata_persistence_failure_is_non_fatal():
    """If scheduler.update_task raises, the routing decision still holds."""
    f = _make(
        metadata={
            'consecutive_merge_thrash': 1,
            'last_merge_outcome_signature': 'sig-abc',
        },
        update_task_raises=True,
    )

    outcome = await f.wf._check_merge_outcome_thrash(
        prev_signature='sig-abc', current_signature='sig-abc',
    )

    # Threshold hit → BLOCKED despite persistence failure
    assert outcome == WorkflowOutcome.BLOCKED
    f.mark_blocked.assert_awaited_once()


@pytest.mark.asyncio
async def test_dropped_plan_targets_short_circuits_to_l1_excluded_from_thrash(
    tmp_path: Path, monkeypatch,
):
    """Fix 2 short-circuits dropped_plan_targets → never enters thrash counter.

    The thrash helper is only called from the merge-phase loop after
    _submit_to_merge_queue returns REQUEUED.  The drop-guard reason
    routes through _mark_blocked(escalate_to_human=True), which returns
    BLOCKED — the loop's ``if merge_outcome != WorkflowOutcome.REQUEUED:
    return merge_outcome`` short-circuit fires before the thrash check.

    Pin this contract so a future refactor can't accidentally route
    dropped_plan_targets through the steward path and let it accumulate
    in this counter.
    """
    f = _make()
    wf = f.wf
    wf.worktree = tmp_path / 'wt'
    wf.worktree.mkdir(parents=True, exist_ok=True)
    wf.merge_queue = MagicMock()
    wf.plan = {'files': []}
    wf._module_configs = []
    # Pre-seed with a sentinel so we can distinguish "never touched" (good)
    # from "reset to None coincidentally" (would silently pass a weaker test).
    wf._last_merge_block_reason = 'sentinel'

    async def fake_enqueue(queue, req: MergeRequest, event_store, **_kwargs):
        req.result.set_result(MergeOutcome(
            'blocked',
            reason=f'{DROPPED_PLAN_TARGETS_REASON_PREFIX}: foo.py',
        ))

    monkeypatch.setattr(
        'orchestrator.merge_queue.enqueue_merge_request', fake_enqueue,
    )

    outcome = await wf._submit_to_merge_queue('99', pre_rebased=False)

    # (a) outcome is BLOCKED
    assert outcome == WorkflowOutcome.BLOCKED
    # (b) _mark_blocked awaited with escalate_to_human=True
    f.mark_blocked.assert_awaited_once()
    _, kwargs = f.mark_blocked.await_args
    assert kwargs.get('escalate_to_human') is True
    # (c) thrash counter cannot be incremented: the drop-guard short-circuit in
    # _submit_to_merge_queue returns via _mark_blocked before reaching the
    # steward-path capture that sets _last_merge_block_reason — the sentinel
    # must be unchanged.
    assert wf._last_merge_block_reason == 'sentinel'


@pytest.mark.asyncio
async def test_post_merge_pyright_broken_short_circuits_to_l1_excluded_from_thrash(
    tmp_path: Path, monkeypatch,
):
    """post_merge_pyright_broken reason routes to L1 without entering thrash counter.

    Mirrors test_dropped_plan_targets_short_circuits_to_l1_excluded_from_thrash:
    the new short-circuit branch in _submit_to_merge_queue must:
      (a) return BLOCKED
      (b) call _mark_blocked with escalate_to_human=True (L1, steward bypassed)
      (c) call _write_merge_failure_review with kind='post_merge_pyright_broken'
      (d) NOT touch _last_merge_block_reason (returns before the steward-path capture)
    """
    from unittest.mock import MagicMock

    f = _make()
    wf = f.wf
    wf.worktree = tmp_path / 'wt'
    wf.worktree.mkdir(parents=True, exist_ok=True)
    wf.merge_queue = MagicMock()
    wf.plan = {'files': []}
    wf._module_configs = []
    wf._last_merge_block_reason = 'sentinel'

    write_failure_spy = MagicMock()
    wf._write_merge_failure_review = write_failure_spy  # type: ignore[method-assign]

    broken_reason = (
        f'{POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX}: post-merge unscoped '
        'type-check failed for subpkg on abc123456789. error: src/foo.py:10: ...'
    )

    async def fake_enqueue(queue, req: MergeRequest, event_store, **_kwargs):
        req.result.set_result(MergeOutcome('blocked', reason=broken_reason))

    monkeypatch.setattr(
        'orchestrator.merge_queue.enqueue_merge_request', fake_enqueue,
    )

    outcome = await wf._submit_to_merge_queue('99', pre_rebased=False)

    # (a) outcome is BLOCKED
    assert outcome == WorkflowOutcome.BLOCKED
    # (b) _mark_blocked awaited with escalate_to_human=True
    f.mark_blocked.assert_awaited_once()
    _, kwargs = f.mark_blocked.await_args
    assert kwargs.get('escalate_to_human') is True
    # (c) _write_merge_failure_review called with kind='post_merge_pyright_broken'
    write_failure_spy.assert_called_once()
    call_args = write_failure_spy.call_args
    assert call_args[0][0] == 'post_merge_pyright_broken', (
        f'Expected kind=post_merge_pyright_broken, got {call_args[0][0]!r}'
    )
    # (d) short-circuit fires before steward-path sets _last_merge_block_reason
    assert wf._last_merge_block_reason == 'sentinel'
