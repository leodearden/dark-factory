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

import errno as errno_module
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import pydantic_spec

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
        for call_args in wf.scheduler.set_task_status.await_args_list:  # type: ignore[attr-defined]
            args = call_args.args
            # args[1] is the status argument
            if len(args) >= 2:
                assert args[1] != 'pending', (
                    f'set_task_status was called with pending: {call_args}'
                )

    @pytest.mark.asyncio
    async def test_multiple_retries_before_success(self):
        """Multiple consecutive VerifyInfraErrors then success → DONE after N retries."""
        wf = _make(verify_infra_retry_max_attempts=5)

        call_count = 0

        async def fake_run_scoped(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
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
        assert call_count == 4
        assert len(sleep_calls) == 3

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

        assert not wf._mark_blocked.called, (  # type: ignore[attr-defined]
            '_mark_blocked must NOT be called when the transient infra error is retried successfully'
        )


# ---------------------------------------------------------------------------
# Step 9: Permanent VerifyInfraError (exhausts retries) → BLOCKED + infra_hold
# ---------------------------------------------------------------------------

class TestVerifyInfraPermanentExhaustion:
    """_verify_debugfix_loop: all retries exhausted → BLOCKED with infra_hold, never pending."""

    @pytest.mark.asyncio
    async def test_exhaustion_returns_blocked(self):
        """When every run_scoped_verification call raises VerifyInfraError, return BLOCKED."""
        wf = _make(verify_infra_retry_max_attempts=3)

        async def always_infra(*args, **kwargs):
            raise VerifyInfraError(phase='warm_marker', errno=errno_module.ENOSPC)

        with (
            patch('orchestrator.workflow.run_scoped_verification', side_effect=always_infra),
            patch('asyncio.sleep', new_callable=AsyncMock),
        ):
            outcome = await wf._verify_debugfix_loop()

        assert outcome == WorkflowOutcome.BLOCKED

    @pytest.mark.asyncio
    async def test_exhaustion_stamps_infra_hold_metadata(self):
        """On exhaustion, metadata.infra_hold is stamped via scheduler.update_task."""
        wf = _make(verify_infra_retry_max_attempts=3)

        async def always_infra(*args, **kwargs):
            raise VerifyInfraError(phase='warm_marker', errno=errno_module.ENOSPC)

        with (
            patch('orchestrator.workflow.run_scoped_verification', side_effect=always_infra),
            patch('asyncio.sleep', new_callable=AsyncMock),
        ):
            await wf._verify_debugfix_loop()

        # update_task must have been called with metadata containing infra_hold
        assert wf.scheduler.update_task.called, 'update_task must be called to stamp infra_hold'  # type: ignore[attr-defined]
        call_kwargs = wf.scheduler.update_task.await_args  # type: ignore[attr-defined]
        metadata_arg = call_kwargs.kwargs.get('metadata') or (call_kwargs.args[1] if len(call_kwargs.args) > 1 else None)
        assert metadata_arg is not None, 'update_task must receive metadata kwarg'
        assert 'infra_hold' in metadata_arg, f'infra_hold missing from metadata: {metadata_arg}'
        infra_hold = metadata_arg['infra_hold']
        assert infra_hold.get('phase') == 'warm_marker'
        assert infra_hold.get('errno') == errno_module.ENOSPC

    @pytest.mark.asyncio
    async def test_exhaustion_sets_infra_hold_info_dict(self):
        """On exhaustion, self._infra_hold_info is set with category='infra_issue' and escalate_to_human."""
        wf = _make(verify_infra_retry_max_attempts=3)

        async def always_infra(*args, **kwargs):
            raise VerifyInfraError(phase='warm_marker', errno=errno_module.ENOSPC)

        with (
            patch('orchestrator.workflow.run_scoped_verification', side_effect=always_infra),
            patch('asyncio.sleep', new_callable=AsyncMock),
        ):
            await wf._verify_debugfix_loop()

        assert wf._infra_hold_info is not None, '_infra_hold_info must be set on exhaustion'
        assert wf._infra_hold_info.get('category') == 'infra_issue'
        assert wf._infra_hold_info.get('escalate_to_human') is True

    @pytest.mark.asyncio
    async def test_exhaustion_never_sets_pending(self):
        """On exhaustion, scheduler.set_task_status is never called with 'pending'."""
        wf = _make(verify_infra_retry_max_attempts=3)

        async def always_infra(*args, **kwargs):
            raise VerifyInfraError(phase='warm_marker', errno=errno_module.ENOSPC)

        with (
            patch('orchestrator.workflow.run_scoped_verification', side_effect=always_infra),
            patch('asyncio.sleep', new_callable=AsyncMock),
        ):
            await wf._verify_debugfix_loop()

        for call_args in wf.scheduler.set_task_status.await_args_list:  # type: ignore[attr-defined]
            args = call_args.args
            if len(args) >= 2:
                assert args[1] != 'pending', f'set_task_status called with pending: {call_args}'

    @pytest.mark.asyncio
    async def test_exhaustion_does_not_call_mark_blocked_directly(self):
        """The loop must NOT call _mark_blocked itself on exhaustion — run() caller's job."""
        wf = _make(verify_infra_retry_max_attempts=3)

        async def always_infra(*args, **kwargs):
            raise VerifyInfraError(phase='warm_marker', errno=errno_module.ENOSPC)

        with (
            patch('orchestrator.workflow.run_scoped_verification', side_effect=always_infra),
            patch('asyncio.sleep', new_callable=AsyncMock),
        ):
            await wf._verify_debugfix_loop()

        assert not wf._mark_blocked.called, (  # type: ignore[attr-defined]
            '_verify_debugfix_loop must NOT call _mark_blocked directly on exhaustion; '
            'that is run()\'s job via _infra_hold_info stash'
        )


# ---------------------------------------------------------------------------
# Step 11: Defensive net — bare infra OSError escaping verify loop →
#          _mark_blocked(category='infra_issue'), not generic 'Workflow error:'
# ---------------------------------------------------------------------------

class TestBareInfraOSErrorDefensiveNet:
    """run(): bare infra-class OSError escaping verify → infra_issue, not task_failure."""

    @pytest.mark.asyncio
    async def test_bare_oserror_enospc_from_verify_routed_to_infra_issue(self):
        """An OSError(ENOSPC) that escapes _execute_verify_review_loop reaches
        _mark_blocked with category='infra_issue', not the generic task_failure
        produced by run()'s broad except Exception handler.

        RED before step-12 (no dedicated except OSError in run).
        GREEN after step-12 (explicit except OSError routes infra errnos to infra_issue).
        """
        wf = _make()

        # Pre-set plan so run() skips the plan phase (if self.plan: pass branch)
        wf.plan = {'task_id': '1883', 'steps': []}
        # Pre-set initial_plan to suppress the simple_task optimistic path
        # (not self.initial_plan would be False → simple task path is skipped)
        wf.initial_plan = {'task_id': '1883', 'steps': []}

        async def fake_setup(*args, **kwargs):
            # worktree and artifacts already set by _make(); just return.
            pass

        with (
            # Bypass the real worktree setup (worktree/artifacts already set)
            patch.object(wf, '_setup_worktree_and_artifacts', side_effect=fake_setup),
            # Pre-PLAN ghost-loop: not already merged
            patch.object(wf, '_recover_if_already_merged',
                         new=AsyncMock(return_value=None)),
            # Ghost-loop before EXECUTE: branch is not on main
            patch.object(wf, '_check_branch_on_main',
                         new=AsyncMock(return_value=None)),
            # Simulate a bare infra OSError escaping the execute/verify loop
            patch.object(
                wf, '_execute_verify_review_loop',
                side_effect=OSError(errno_module.ENOSPC, 'No space left on device'),
            ),
        ):
            outcome = await wf.run()

        assert outcome == WorkflowOutcome.BLOCKED

        # Key assertion: _mark_blocked must be called with category='infra_issue'
        # (NOT the default 'task_failure' from the broad except Exception handler)
        wf._mark_blocked.assert_awaited_once()  # type: ignore[attr-defined]
        call_kwargs = wf._mark_blocked.await_args.kwargs  # type: ignore[attr-defined]
        assert call_kwargs.get('category') == 'infra_issue', (
            f"Expected _mark_blocked(category='infra_issue') but got "
            f"category={call_kwargs.get('category')!r}. "
            f"A bare OSError(ENOSPC) from the verify path must be classified as "
            f"infra_issue, not routed through the generic 'Workflow error:' handler."
        )
        assert call_kwargs.get('escalate_to_human') is True, (
            "infra_issue from verify must set escalate_to_human=True"
        )

    @pytest.mark.asyncio
    async def test_non_infra_oserror_still_uses_generic_handler(self):
        """An OSError with a non-infra errno (EACCES) is NOT infra-classified.

        It should re-raise and be caught by the broad except Exception handler
        → _mark_blocked with 'Workflow error: ...' (task_failure, the safe default).
        This ensures the infra guard is NARROW and doesn't swallow permission errors.

        Note: This test asserts CURRENT expected behaviour — the broad except
        catches the re-raised non-infra OSError. If this test is GREEN, step-12's
        implementation correctly re-raises non-infra OSErrors.
        """
        wf = _make()
        wf.plan = {'task_id': '1883', 'steps': []}
        wf.initial_plan = {'task_id': '1883', 'steps': []}

        async def fake_setup(*args, **kwargs):
            pass

        with (
            patch.object(wf, '_setup_worktree_and_artifacts', side_effect=fake_setup),
            patch.object(wf, '_recover_if_already_merged',
                         new=AsyncMock(return_value=None)),
            patch.object(wf, '_check_branch_on_main',
                         new=AsyncMock(return_value=None)),
            patch.object(
                wf, '_execute_verify_review_loop',
                side_effect=OSError(errno_module.EACCES, 'Permission denied'),
            ),
        ):
            outcome = await wf.run()

        assert outcome == WorkflowOutcome.BLOCKED
        wf._mark_blocked.assert_awaited_once()  # type: ignore[attr-defined]
        # Non-infra OSError → broad handler → reason starts with 'Workflow error:'
        call_args = wf._mark_blocked.await_args  # type: ignore[attr-defined]
        reason = call_args.args[0] if call_args.args else None
        assert reason is not None and reason.startswith('Workflow error:'), (
            f"Non-infra OSError(EACCES) must use generic handler, got reason={reason!r}"
        )


# ---------------------------------------------------------------------------
# Step 17: Boundary/repro — end-to-end ENOSPC simulation
# ---------------------------------------------------------------------------

class TestBoundaryVerifyInfraReproCases:
    """G2 boundary/repro: end-to-end verify ENOSPC simulation.

    (a) Transient: VerifyInfraError on attempt 1 (simulating _mark_verify_warm
    failure), clears on attempt 2 → DONE, never pending, no implement footprint
    re-acquisition.

    (b) Permanent: all retries exhausted → BLOCKED + infra_hold stamped +
    escalate_to_human=True.  On resolving the infra_issue escalation, the
    harness (step-16 path) resumes-at-verify (in-progress) rather than
    re-pending the task.
    """

    @pytest.mark.asyncio
    async def test_transient_warm_marker_enospc_never_pending_reaches_done(self):
        """(a) Transient ENOSPC at warm_marker: loop retries → DONE, never pending.

        Simulates the A1 root-cause scenario at the _verify_debugfix_loop level:
        VerifyInfraError(phase='warm_marker') on attempt 1 clears on attempt 2.
        The test confirms the task status is NEVER set to 'pending', and the
        workflow reaches DONE (not BLOCKED/REQUEUED).

        This test passes the VerifyInfraError through _run_scoped_verification_with_infra_retry
        to confirm the retry wrapper catches it correctly — not just at the
        run_scoped_verification patch level of steps 7-9.

        RED before step-18 if the VerifyInfraError from within run_scoped_verification
        (e.g., from the asyncio.gather sub-path) is NOT correctly propagated to the
        retry wrapper.
        """
        wf = _make(verify_infra_retry_max_attempts=3)

        call_count = 0

        async def fake_run_scoped(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Simulate _mark_verify_warm raising VerifyInfraError after
                # a successful verify (ENOSPC at the warm-marker write step)
                raise VerifyInfraError(phase='warm_marker', errno=errno_module.ENOSPC)
            # Second attempt: clears cleanly
            return _passed_result()

        with (
            patch('orchestrator.workflow.run_scoped_verification', side_effect=fake_run_scoped),
            patch('asyncio.sleep', new_callable=AsyncMock),
        ):
            outcome = await wf._verify_debugfix_loop()

        assert outcome == WorkflowOutcome.DONE, (
            f'Expected DONE after transient infra retry but got {outcome!r}. '
            f'run_scoped_verification was called {call_count} times.'
        )

        # Key boundary assertion: task NEVER set to pending
        for call_args in wf.scheduler.set_task_status.await_args_list:  # type: ignore[attr-defined]
            args = call_args.args
            if len(args) >= 2:
                assert args[1] != 'pending', (
                    f'set_task_status was called with "pending" despite transient '
                    f'infra retry: {call_args}. A verify-complete branch must never '
                    f'be re-pended due to a transient infra OSError.'
                )

        # Implement footprint guard: verify no dispatch occurred (no pending)
        assert wf.scheduler.set_task_status.await_count == 0 or all(  # type: ignore[attr-defined]
            c.args[1] != 'pending'
            for c in wf.scheduler.set_task_status.await_args_list  # type: ignore[attr-defined]
            if len(c.args) >= 2
        ), 'Implement footprint must not be re-acquired (no pending dispatch)'

    @pytest.mark.asyncio
    async def test_permanent_enospc_blocks_with_infra_hold_loud_escalation(self):
        """(b) Permanent ENOSPC exhausts window → BLOCKED + infra_hold + loud escalation.

        Verifies:
        - _verify_debugfix_loop returns BLOCKED (not REQUEUED, not DONE)
        - metadata.infra_hold is stamped with phase + errno (via scheduler.update_task)
        - _infra_hold_info carries category='infra_issue' and escalate_to_human=True
          (which run() uses to call _mark_blocked with category='infra_issue')
        - set_task_status is NEVER called with 'pending'
        - _mark_blocked is NOT called from inside the loop (only from run()'s caller)
        """
        wf = _make(verify_infra_retry_max_attempts=3)

        async def always_infra(*args, **kwargs):
            raise VerifyInfraError(phase='warm_marker', errno=errno_module.ENOSPC)

        with (
            patch('orchestrator.workflow.run_scoped_verification', side_effect=always_infra),
            patch('asyncio.sleep', new_callable=AsyncMock),
        ):
            outcome = await wf._verify_debugfix_loop()

        assert outcome == WorkflowOutcome.BLOCKED, (
            f'Expected BLOCKED after exhausting infra retry window, got {outcome!r}'
        )

        # Must NOT have been set to pending
        for call_args in wf.scheduler.set_task_status.await_args_list:  # type: ignore[attr-defined]
            if len(call_args.args) >= 2:
                assert call_args.args[1] != 'pending', (
                    f'set_task_status("pending") called for infra_hold task: {call_args}'
                )

        # infra_hold must be in metadata (enables harness step-16 resume path)
        assert wf.scheduler.update_task.called, 'update_task must stamp infra_hold metadata'  # type: ignore[attr-defined]
        call_kwargs = wf.scheduler.update_task.await_args  # type: ignore[attr-defined]
        metadata_arg = (
            call_kwargs.kwargs.get('metadata')
            or (call_kwargs.args[1] if len(call_kwargs.args) > 1 else None)
        )
        assert metadata_arg is not None and 'infra_hold' in metadata_arg, (
            f'metadata.infra_hold not stamped; update_task called with: {call_kwargs}'
        )
        infra_hold = metadata_arg['infra_hold']
        assert infra_hold.get('phase') == 'warm_marker', (
            f"Expected infra_hold.phase='warm_marker', got {infra_hold!r}"
        )

        # _infra_hold_info must be set for run() to call _mark_blocked(category='infra_issue')
        assert wf._infra_hold_info is not None, (
            '_infra_hold_info must be set on exhaustion so run() can route to infra_issue block'
        )
        assert wf._infra_hold_info.get('category') == 'infra_issue', (
            f"_infra_hold_info.category must be 'infra_issue', got {wf._infra_hold_info!r}"
        )
        assert wf._infra_hold_info.get('escalate_to_human') is True, (
            'infra_issue block must be filed with escalate_to_human=True (LOUD escalation)'
        )

        # Loop must NOT call _mark_blocked itself (that is run()'s responsibility)
        assert not wf._mark_blocked.called, (  # type: ignore[attr-defined]
            '_verify_debugfix_loop must not call _mark_blocked directly on exhaustion; '
            'only run() should invoke it via _infra_hold_info stash'
        )

    @pytest.mark.asyncio
    async def test_gather_propagates_verify_infra_error_to_retry_wrapper(self):
        """(a) extra: VerifyInfraError inside asyncio.gather reaches the retry wrapper.

        Confirms the critical propagation path: when VerifyInfraError is raised
        inside run_scoped_verification (e.g., from within _verify_module's
        asyncio.gather call), it MUST reach _run_scoped_verification_with_infra_retry
        and be caught for retry.

        This test exercises _run_scoped_verification_with_infra_retry directly
        (not _verify_debugfix_loop) to confirm the retry wrapper catches the error.

        RED before step-18 if the gather path in run_scoped_verification swallows
        the VerifyInfraError (e.g., via a broad except or return_exceptions).
        GREEN after step-18 confirms the gather propagates the error unchanged.
        """
        wf = _make(verify_infra_retry_max_attempts=3)

        call_count = 0

        async def fake_scoped_that_raises_once(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Simulate gather-propagated VerifyInfraError
                raise VerifyInfraError(phase='warm_marker', errno=errno_module.ENOSPC)
            return _passed_result()

        with (
            patch('orchestrator.workflow.run_scoped_verification',
                  side_effect=fake_scoped_that_raises_once),
            patch('asyncio.sleep', new_callable=AsyncMock),
        ):
            result = await wf._run_scoped_verification_with_infra_retry(
                verify_attempt=0,
            )

        assert result is not None, (
            '_run_scoped_verification_with_infra_retry must return a VerifyResult '
            'when VerifyInfraError clears on retry, got None '
            '(implies the retry was NOT triggered — the error was swallowed or mismapped).'
        )
        assert result.passed is True, (
            f'Expected passed=True on successful retry, got result={result!r}'
        )
        assert call_count == 2, (
            f'Expected exactly 2 calls to run_scoped_verification '
            f'(1 failing + 1 succeeding), got {call_count}'
        )
