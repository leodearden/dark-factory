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


# ---------------------------------------------------------------------------
# task-1688 step-5 RED: stable merge-thrash signature
# ---------------------------------------------------------------------------


def test_merge_outcome_signature_same_fingerprint_equal_despite_varied_reason():
    """_merge_outcome_signature is stable when category+normalized_cause_hint are identical.

    Two captures with the same category and cause_hint differing only by line
    number produce EQUAL signatures, even if the prose reason differs.
    A different category OR different normalised cause_hint yields a DIFFERENT
    signature.
    """
    f = _make()
    wf = f.wf

    # Set the structured fingerprint fields for 'StatusBar.tsx line 42'
    wf._last_merge_failure_category = 'gui_tsc'  # type: ignore[attr-defined]
    wf._last_merge_failure_cause_hint = 'StatusBar.tsx:42 error TS2322: Type X not assignable'  # type: ignore[attr-defined]
    wf._last_merge_block_reason = 'Post-merge verification failed: tests failed, lint issues\n\nsome output'
    sig_a = wf._merge_outcome_signature()  # type: ignore[attr-defined]

    # Same category + cause_hint differing only by line number → same normalised key
    wf._last_merge_failure_cause_hint = 'StatusBar.tsx:58 error TS2322: Type X not assignable'  # type: ignore[attr-defined]
    wf._last_merge_block_reason = 'Post-merge verification failed: unknown_test_failure\n\ndifferent output'
    sig_b = wf._merge_outcome_signature()  # type: ignore[attr-defined]

    assert sig_a == sig_b, (
        f'Same fingerprint (category, normalised cause_hint) must yield equal signature; '
        f'got {sig_a!r} vs {sig_b!r}'
    )
    assert len(sig_a) == 16 and all(c in '0123456789abcdef' for c in sig_a), (
        f'Signature must be 16 hex chars, got {sig_a!r}'
    )

    # Different category → different signature
    wf._last_merge_failure_category = 'test_failure'  # type: ignore[attr-defined]
    wf._last_merge_failure_cause_hint = 'StatusBar.tsx:42 error TS2322: Type X not assignable'  # type: ignore[attr-defined]
    sig_different_category = wf._merge_outcome_signature()  # type: ignore[attr-defined]
    assert sig_different_category != sig_a, (
        'Different category must yield a different signature'
    )

    # Different normalised cause_hint → different signature
    wf._last_merge_failure_category = 'gui_tsc'  # type: ignore[attr-defined]
    wf._last_merge_failure_cause_hint = 'OtherComponent.tsx:10 error TS9999: something else'  # type: ignore[attr-defined]
    sig_different_hint = wf._merge_outcome_signature()  # type: ignore[attr-defined]
    assert sig_different_hint != sig_a, (
        'Different cause_hint must yield a different signature'
    )


def test_merge_outcome_signature_fallback_when_structured_fields_empty():
    """_merge_outcome_signature falls back to sha256(normalised_reason) when both fields are ''.

    The fallback applies _normalize_cause_hint (lowercase + whitespace-collapse),
    so reasons that differ only by case or whitespace produce the same signature.
    """
    f = _make()
    wf = f.wf

    wf._last_merge_failure_category = ''  # type: ignore[attr-defined]
    wf._last_merge_failure_cause_hint = ''  # type: ignore[attr-defined]
    wf._last_merge_block_reason = 'Post-merge verification failed: TESTS FAILED\n\nextra output'
    sig_fallback = wf._merge_outcome_signature()  # type: ignore[attr-defined]

    assert len(sig_fallback) == 16 and all(c in '0123456789abcdef' for c in sig_fallback), (
        f'Fallback signature must be 16 hex chars, got {sig_fallback!r}'
    )

    # Reason differing only by case/whitespace → same fallback sig (normalisation applied)
    wf._last_merge_block_reason = 'post-merge verification failed:  tests  failed\n\nextra  output'
    sig_normalised = wf._merge_outcome_signature()  # type: ignore[attr-defined]
    assert sig_normalised == sig_fallback, (
        'Fallback signatures must match for reasons that differ only by case/whitespace'
    )

    # Different reason → different fallback sig
    wf._last_merge_block_reason = 'git merge failed: conflict in foo.py'
    sig_other = wf._merge_outcome_signature()  # type: ignore[attr-defined]
    assert sig_other != sig_fallback


@pytest.mark.asyncio
async def test_stable_signature_reaches_threshold_and_escalates():
    """task-1688 step-5 end-to-end: same fingerprint + varied prose → reaches threshold.

    Drive _check_merge_outcome_thrash twice using signatures from
    _merge_outcome_signature (same category+cause_hint, different reason prose).
    Counter reaches max_consecutive_merge_thrash (=2) → _mark_blocked with
    escalate_to_human=True.
    """
    f = _make(
        metadata={'consecutive_merge_thrash': 1},
        max_consecutive_merge_thrash=2,
    )
    wf = f.wf

    # Simulate capturing first failure
    wf._last_merge_failure_category = 'gui_tsc'  # type: ignore[attr-defined]
    wf._last_merge_failure_cause_hint = 'StatusBar.tsx:42 error TS2322'  # type: ignore[attr-defined]
    wf._last_merge_block_reason = 'Post-merge verification failed: tests failed\n\nfirst attempt'
    first_sig = wf._merge_outcome_signature()  # type: ignore[attr-defined]

    # Persist the first sig as prev_signature (simulates the prior iteration)
    wf.task['metadata']['last_merge_outcome_signature'] = first_sig

    # Simulate second failure — same fingerprint but prose differs (line num shifts)
    wf._last_merge_failure_cause_hint = 'StatusBar.tsx:58 error TS2322'  # type: ignore[attr-defined]
    wf._last_merge_block_reason = 'Post-merge verification failed: unknown_test_failure\n\nsecond attempt'
    second_sig = wf._merge_outcome_signature()  # type: ignore[attr-defined]

    assert first_sig == second_sig, (
        'Precondition: same fingerprint must produce equal signatures'
    )

    outcome = await wf._check_merge_outcome_thrash(
        prev_signature=first_sig, current_signature=second_sig,
    )

    assert outcome == WorkflowOutcome.BLOCKED
    f.mark_blocked.assert_awaited_once()
    _, kwargs = f.mark_blocked.await_args
    assert kwargs.get('escalate_to_human') is True


@pytest.mark.asyncio
async def test_genuinely_different_fingerprint_resets_counter_to_1():
    """task-1688 step-5: different fingerprint resets counter to 1, no escalation."""
    f = _make(
        metadata={'consecutive_merge_thrash': 1},
        max_consecutive_merge_thrash=2,
    )
    wf = f.wf

    wf._last_merge_failure_category = 'gui_tsc'  # type: ignore[attr-defined]
    wf._last_merge_failure_cause_hint = 'StatusBar.tsx:42 error TS2322'  # type: ignore[attr-defined]
    wf._last_merge_block_reason = 'Post-merge verification failed: tests failed'
    sig_first = wf._merge_outcome_signature()  # type: ignore[attr-defined]

    # Genuinely different failure: different category
    wf._last_merge_failure_category = 'test_failure'  # type: ignore[attr-defined]
    wf._last_merge_failure_cause_hint = 'some_test.py:10 AssertionError'  # type: ignore[attr-defined]
    wf._last_merge_block_reason = 'Post-merge verification failed: test exploded'
    sig_second = wf._merge_outcome_signature()  # type: ignore[attr-defined]

    assert sig_first != sig_second, 'Precondition: different fingerprints'

    outcome = await wf._check_merge_outcome_thrash(
        prev_signature=sig_first, current_signature=sig_second,
    )

    assert outcome is None
    md = _persisted_metadata(f.update_task)
    assert md['consecutive_merge_thrash'] == 1, (
        f'Counter must reset to 1 on different fingerprint, got {md}'
    )
    f.mark_blocked.assert_not_awaited()


# ---------------------------------------------------------------------------
# task-1688 step-7 RED: stable root_cause threaded to L1 escalation for L2 dedup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_thrash_threshold_passes_stable_root_cause_to_mark_blocked():
    """task-1688 step-7(a): _check_merge_outcome_thrash at threshold calls _mark_blocked
    with a non-empty root_cause kwarg derived from (category, normalised cause_hint).

    Two TaskWorkflow instances that captured the SAME fingerprint must produce
    the IDENTICAL root_cause string so find_pending_l2_by_root_cause folds them.
    """
    # Instance A
    fa = _make(
        metadata={'consecutive_merge_thrash': 1},
        max_consecutive_merge_thrash=2,
    )
    fa.wf._last_merge_failure_category = 'gui_tsc'  # type: ignore[attr-defined]
    fa.wf._last_merge_failure_cause_hint = 'StatusBar.tsx:42 error TS2322: Type X not assignable'  # type: ignore[attr-defined]
    sig_a = fa.wf._merge_outcome_signature()  # type: ignore[attr-defined]
    fa.wf.task['metadata']['last_merge_outcome_signature'] = sig_a
    outcome_a = await fa.wf._check_merge_outcome_thrash(
        prev_signature=sig_a, current_signature=sig_a,
    )
    assert outcome_a == WorkflowOutcome.BLOCKED
    fa.mark_blocked.assert_awaited_once()
    _, kwargs_a = fa.mark_blocked.await_args
    root_cause_a = kwargs_a.get('root_cause', '')
    assert root_cause_a, 'root_cause must be non-empty when fingerprint is set'
    assert 'gui_tsc' in root_cause_a, f'root_cause must mention category: {root_cause_a!r}'

    # Instance B — same fingerprint (line number differs only)
    fb = _make(
        metadata={'consecutive_merge_thrash': 1},
        max_consecutive_merge_thrash=2,
    )
    fb.wf._last_merge_failure_category = 'gui_tsc'  # type: ignore[attr-defined]
    fb.wf._last_merge_failure_cause_hint = 'StatusBar.tsx:58 error TS2322: Type X not assignable'  # type: ignore[attr-defined]
    sig_b = fb.wf._merge_outcome_signature()  # type: ignore[attr-defined]
    fb.wf.task['metadata']['last_merge_outcome_signature'] = sig_b
    outcome_b = await fb.wf._check_merge_outcome_thrash(
        prev_signature=sig_b, current_signature=sig_b,
    )
    assert outcome_b == WorkflowOutcome.BLOCKED
    _, kwargs_b = fb.mark_blocked.await_args
    root_cause_b = kwargs_b.get('root_cause', '')

    assert root_cause_a == root_cause_b, (
        f'Sibling tasks with the same fingerprint must produce identical root_cause; '
        f'got {root_cause_a!r} vs {root_cause_b!r}'
    )


@pytest.mark.asyncio
async def test_ensure_l1_escalation_stamps_root_cause():
    """task-1688 step-7(b): _ensure_l1_escalation_for_blocked stamps esc.root_cause.

    Drives _ensure_l1_escalation_for_blocked with a fake escalation_queue
    (has_open_l1 → False) and asserts the submitted Escalation has
    esc.root_cause equal to the value passed.
    """
    from unittest.mock import patch

    f = _make()
    wf = f.wf

    expected_root_cause = 'merge-outcome-thrash:gui_tsc:statusbar.tsx error ts2322: type x not assignable'

    submitted_escs: list = []
    mock_queue = MagicMock()
    mock_queue.has_open_l1 = MagicMock(return_value=False)
    mock_queue.make_id = MagicMock(return_value='esc-test-1')
    mock_queue.submit = MagicMock(side_effect=submitted_escs.append)
    wf.escalation_queue = mock_queue  # type: ignore[assignment]
    wf.event_store = None

    with patch.object(wf, '_build_train_state', new=AsyncMock(return_value=None)), patch.object(wf, '_durable_ref_suffix', return_value=''):
        await wf._ensure_l1_escalation_for_blocked(
            'Repeated merge-phase thrash',
            'detail here',
            category='task_failure',
            root_cause=expected_root_cause,
        )

    assert len(submitted_escs) == 1, f'Expected 1 submitted escalation, got {submitted_escs}'
    esc = submitted_escs[0]
    assert esc.root_cause == expected_root_cause, (
        f'Expected root_cause={expected_root_cause!r}, got {esc.root_cause!r}'
    )
