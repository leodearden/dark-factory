"""Tests for A1 — verify-phase infra-OSError retry/resume logic.

Step 7: Transient VerifyInfraError → bounded retry → DONE (task stays claimed,
never pending, _mark_blocked not called).

Step 9: Permanent VerifyInfraError (exhausts retries) → WorkflowOutcome.BLOCKED
with infra_hold stamped and escalate_to_human, never pending, _mark_blocked
not called from inside the loop.

Step 11: Bare infra-class OSError escaping the verify path → routed to
_mark_blocked with category='infra_issue' (not the generic 'Workflow error:').

Step 17: Boundary/repro tests end-to-end.
"""

from __future__ import annotations

import asyncio
import errno as errno_module
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import pydantic_spec

from orchestrator import verify as verify_module
from orchestrator.config import OrchestratorConfig
from orchestrator.verify import VerifyInfraError, VerifyResult
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome


# ---------------------------------------------------------------------------
# Test-fixture factory
# ---------------------------------------------------------------------------

def _make(
    *,
    task_id: str = '1883',
    metadata: dict | None = None,
    verify_infra_retry_max_attempts: int = 3,
    verify_infra_retry_backoff_secs: float = 0.01,
    verify_infra_retry_max_backoff_secs: float = 0.1,
    max_consecutive_infra_resumes: int = 3,
) -> TaskWorkflow:
    assignment = MagicMock()
    assignment.task_id = task_id
    assignment.task = {
        'id': task_id,
        'title': 'Test task',
        'description': 'Test desc',
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
    config.max_consecutive_infra_resumes = max_consecutive_infra_resumes
    config.verify_infra_retry_max_attempts = verify_infra_retry_max_attempts
    config.verify_infra_retry_backoff_secs = verify_infra_retry_backoff_secs
    config.verify_infra_retry_max_backoff_secs = verify_infra_retry_max_backoff_secs
    config.max_verify_attempts = 5
    config.max_failure_signature_repeat = 3
    config.max_opaque_timeout_attempts = 3
    config.rebase_before_verify = False
    config.escalate_preexisting_main_break = False

    scheduler = MagicMock()
    scheduler.update_task = AsyncMock(return_value=True)
    scheduler.set_task_status = AsyncMock()
    scheduler.get_status = AsyncMock(return_value='in-progress')
    scheduler.get_task = AsyncMock(return_value={'id': task_id, 'metadata': metadata or {}})

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
        escalation_queue=queue,
    )

    # Set up a fake worktree and artifacts so the loop doesn't crash on None
    wf.worktree = Path('/tmp/fake-worktree-1883')
    wf.artifacts = MagicMock()
    wf.artifacts.read_iteration_log = MagicMock(return_value=([], []))

    # Stub _mark_blocked — the loop must NOT call this on transient retry
    wf._mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)

    return wf


def _passed_result() -> VerifyResult:
    return VerifyResult(
        passed=True,
        test_output='',
        lint_output='',
        type_output='',
        summary='All checks passed',
    )


# ---------------------------------------------------------------------------
# Step 7: Transient VerifyInfraError → retry → DONE
# ---------------------------------------------------------------------------

class TestVerifyInfraTransientRetry:
    """_verify_debugfix_loop: VerifyInfraError on attempt 1, passes on attempt 2 → DONE."""

    @pytest.mark.asyncio
    async def test_retry_on_transient_infra_error_returns_done(self):
        """Loop retries on VerifyInfraError and returns DONE on clean second call."""
        wf = _make(verify_infra_retry_max_attempts=3)

        call_count = 0

        async def fake_run_scoped(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise VerifyInfraError(phase='warm_marker', errno=errno_module.ENOSPC)
            return _passed_result()

        sleep_calls = []

        async def fake_sleep(secs):
            sleep_calls.append(secs)

        with (
            patch('orchestrator.workflow.run_scoped_verification', side_effect=fake_run_scoped),
            patch('asyncio.sleep', side_effect=fake_sleep),
        ):
            outcome = await wf._verify_debugfix_loop()

        assert outcome == WorkflowOutcome.DONE
        assert call_count == 2, f'Expected 2 calls to run_scoped_verification, got {call_count}'
        assert len(sleep_calls) >= 1, 'Expected at least one backoff sleep'

    @pytest.mark.asyncio
    async def test_task_never_set_to_pending_on_transient_retry(self):
        """scheduler.set_task_status is never called with 'pending' during retry."""
        wf = _make(verify_infra_retry_max_attempts=3)

        call_count = 0

        async def fake_run_scoped(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise VerifyInfraError(phase='warm_marker', errno=errno_module.ENOSPC)
            return _passed_result()

        with (
            patch('orchestrator.workflow.run_scoped_verification', side_effect=fake_run_scoped),
            patch('asyncio.sleep', new_callable=AsyncMock),
        ):
            await wf._verify_debugfix_loop()

        # Verify set_task_status was never called with 'pending'
        for call_args in wf.scheduler.set_task_status.await_args_list:
            args = call_args.args
            # args[1] is the status argument
            if len(args) >= 2:
                assert args[1] != 'pending', (
                    f'set_task_status was called with pending: {call_args}'
                )

    @pytest.mark.asyncio
    async def test_mark_blocked_not_called_on_transient_retry(self):
        """_mark_blocked is not called when a transient VerifyInfraError is retried successfully."""
        wf = _make(verify_infra_retry_max_attempts=3)

        call_count = 0

        async def fake_run_scoped(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise VerifyInfraError(phase='warm_marker', errno=errno_module.ENOSPC)
            return _passed_result()

        with (
            patch('orchestrator.workflow.run_scoped_verification', side_effect=fake_run_scoped),
            patch('asyncio.sleep', new_callable=AsyncMock),
        ):
            await wf._verify_debugfix_loop()

        assert not wf._mark_blocked.called, (
            '_mark_blocked must NOT be called when the transient infra error is retried successfully'
        )
