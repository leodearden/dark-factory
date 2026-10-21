"""Tests for merge-queue main-health routing wired into workflow._submit_to_merge_queue.

Step-1: compute_preexisting_main_break_fingerprint helper produces the same
        fingerprint as the inline composition in _verify_debugfix_loop, and
        returns '' on failure.
Step-13: _submit_to_merge_queue routes a MAIN_HEALTH_RED_REASON_PREFIX outcome
         to one foldable L1 carrying the shared dedupe fingerprint, then blocks
         via _mark_blocked(skip_escalation=True) (task 5471).
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from escalation.queue import EscalationQueue

from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps
from orchestrator.main_health_actions import MainHealthNextAction
from orchestrator.merge_lane.worker import MAIN_HEALTH_RED_REASON_PREFIX, MergeOutcome
from orchestrator.scheduler import TaskAssignment
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome

MAIN_SHA = 'deadbeef1234567890'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(tmp_path: Path) -> OrchestratorConfig:
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


def _make_workflow(config: OrchestratorConfig, worktree: Path) -> TaskWorkflow:
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
    # task-1923: _submit_to_merge_queue awaits rebind_branch_to_head before
    # enqueue and reads git_ops.config.branch_prefix to name the ref.  spec=GitOps
    # hides the __init__-assigned ``config`` attr, so set it explicitly.
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
    )
    workflow.worktree = worktree
    artifacts = TaskArtifacts(worktree)
    artifacts.init('42', 'X', 'desc', base_commit='base-sha')
    workflow.artifacts = artifacts
    workflow.plan = {'task_id': '42', 'steps': []}
    return workflow


# ---------------------------------------------------------------------------
# Step-1 / Step-2: compute_preexisting_main_break_fingerprint
# ---------------------------------------------------------------------------


class TestComputePreexistingMainBreakFingerprint:
    """compute_preexisting_main_break_fingerprint must produce the same
    fingerprint as the inline composition in _verify_debugfix_loop, and
    return '' (fail-safe) when composition raises."""

    def test_matches_inline_composition(self) -> None:
        from escalation.dedupe import compute_content_fingerprint

        from orchestrator.workflow import (
            _normalize_cause_hint,
            compute_preexisting_main_break_fingerprint,
        )

        category = 'compile_error'
        cause_hint = 'error TS2322: StatusBar.tsx'
        probe_sha = MAIN_SHA

        expected = compute_content_fingerprint(
            'preexisting_main_break',
            category,
            [],
            description=_normalize_cause_hint(cause_hint) + '|' + probe_sha,
        )
        actual = compute_preexisting_main_break_fingerprint(category, cause_hint, probe_sha)
        assert actual == expected, (
            f'Fingerprint mismatch: expected {expected!r}, got {actual!r}'
        )
        assert actual, 'Fingerprint must be non-empty'

    def test_returns_empty_on_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When compute_content_fingerprint raises, helper returns '' (fail-safe)."""
        import escalation.dedupe as dedupe_mod
        monkeypatch.setattr(
            dedupe_mod, 'compute_content_fingerprint',
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('injected')),
        )
        from orchestrator.workflow import compute_preexisting_main_break_fingerprint
        result = compute_preexisting_main_break_fingerprint(
            'compile_error', 'error TS2322', MAIN_SHA,
        )
        assert result == '', f'Expected empty string on failure; got {result!r}'

    def test_different_probe_sha_produces_different_fingerprint(self) -> None:
        from orchestrator.workflow import compute_preexisting_main_break_fingerprint

        fp1 = compute_preexisting_main_break_fingerprint('compile_error', 'hint', MAIN_SHA)
        fp2 = compute_preexisting_main_break_fingerprint('compile_error', 'hint', 'cafebabe')
        assert fp1 != fp2, 'Different probe_sha must produce different fingerprints'
        assert fp1 and fp2, 'Both fingerprints must be non-empty'


# ---------------------------------------------------------------------------
# Step-13: _submit_to_merge_queue routes MAIN_HEALTH_RED outcome correctly
# ---------------------------------------------------------------------------


class TestSubmitToMergeQueueMainHealthRouting:
    """_submit_to_merge_queue must route a MAIN_HEALTH_RED_REASON_PREFIX blocked
    outcome to one foldable ``preexisting_main_break`` L1 carrying the shared
    dedupe fingerprint, and must NOT touch the steward or the generic
    task-fault path.

    Pattern mirrors test_workflow_e2e.py: set workflow.merge_queue, patch
    enqueue_merge_request to deliver the MergeOutcome, call with a branch name.
    """

    def test_main_health_red_routes_to_l1_escalation(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        worktree = tmp_path / 'task-wt'
        worktree.mkdir()
        workflow = _make_workflow(config, worktree)

        fp = 'fp-abc-123'
        main_health_outcome = MergeOutcome(
            'blocked',
            reason=f'{MAIN_HEALTH_RED_REASON_PREFIX} (category=compile_error): error TS2322',
            failure_category='compile_error',
            failure_cause_hint='error TS2322',
            dedupe_fingerprint=fp,
        )

        mark_blocked_mock = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
        workflow._mark_blocked = mark_blocked_mock  # type: ignore[method-assign]
        workflow._write_merge_failure_review = MagicMock()
        workflow.escalation_queue = EscalationQueue(tmp_path / 'escalations')

        # Deliver the MergeOutcome via the enqueue hook (same pattern as e2e tests).
        async def _fake_enqueue(_queue, request, _event_store, **_kwargs):
            request.result.set_result(main_health_outcome)

        workflow.merge_queue = asyncio.Queue()
        with patch(
            'orchestrator.merge_lane.worker.enqueue_merge_request', _fake_enqueue,
        ):
            result = asyncio.run(
                workflow._submit_to_merge_queue('task/42', pre_rebased=False, merge_phase=True)
            )

        assert result == WorkflowOutcome.BLOCKED
        [esc] = workflow.escalation_queue.get_pending()
        assert esc.category == 'preexisting_main_break'
        assert esc.level == 1
        assert esc.dedupe_fingerprint == fp
        assert esc.suggested_action == MainHealthNextAction.NO_FIX_OWNER
        mark_blocked_mock.assert_called_once()
        call_kwargs = mark_blocked_mock.call_args[1]
        assert call_kwargs.get('skip_escalation') is True, (
            f'The L1 is already filed; got {call_kwargs}'
        )
        assert call_kwargs.get('merge_phase') is True, (
            f'Expected merge_phase=True (forwarded); got {call_kwargs}'
        )

    def test_non_main_health_blocked_falls_through_to_generic_path(
        self, tmp_path: Path,
    ) -> None:
        """A normal blocked outcome must NOT match the MAIN_HEALTH_RED prefix
        and must fall through to the generic _mark_blocked path."""
        config = _make_config(tmp_path)
        worktree = tmp_path / 'task-wt'
        worktree.mkdir()
        workflow = _make_workflow(config, worktree)

        generic_outcome = MergeOutcome(
            'blocked',
            reason='Post-merge verification failed: tsc failed',
            failure_category='compile_error',
        )

        mark_blocked_mock = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
        workflow._mark_blocked = mark_blocked_mock  # type: ignore[method-assign]
        workflow._write_merge_failure_review = MagicMock()

        async def _fake_enqueue(_queue, request, _event_store, **_kwargs):
            request.result.set_result(generic_outcome)

        workflow.merge_queue = asyncio.Queue()
        with patch(
            'orchestrator.merge_lane.worker.enqueue_merge_request', _fake_enqueue,
        ):
            result = asyncio.run(
                workflow._submit_to_merge_queue('task/42', pre_rebased=False, merge_phase=False)
            )

        assert result == WorkflowOutcome.BLOCKED
        mark_blocked_mock.assert_called_once()
        call_kwargs = mark_blocked_mock.call_args[1]
        # Generic path does NOT set escalate_to_human=True or category='preexisting_main_break'
        assert call_kwargs.get('escalate_to_human') is not True, (
            f'Generic path must not set escalate_to_human=True; got {call_kwargs}'
        )
        assert call_kwargs.get('category') != 'preexisting_main_break', (
            f'Generic path must not set category=preexisting_main_break; got {call_kwargs}'
        )
        assert call_kwargs.get('dedupe_fingerprint') is None, (
            f'Generic path must not set dedupe_fingerprint; got {call_kwargs}'
        )
