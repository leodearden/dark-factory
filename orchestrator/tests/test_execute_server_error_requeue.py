"""First 5xx-attributed zero-output result routes to REQUEUED (task γ / 3316).

PRD `plans/server-side-api-error-handling-prd.md` task γ, contract C2
(requeue-not-block invariant). Pins boundary rows 1 (a flushed 5xx on a
zero-output result exits REQUEUED via `_repend_for_requeue`, on the FIRST
try, with no `blocked` write / no L0 / no steward / the zero-output counter
left untouched), 2 (a genuine SIGKILL wedge — no status attached — still
blocks at threshold 1) and 3 (two consecutive no-status zero-output results
still block at threshold 2, with the counter visible in the detail).

Two layers are pinned in this module, added across this task's TDD steps:

- ``TestExecuteIterations*`` (step-1 RED / step-2 GREEN, then step-5 RED /
  step-6 GREEN) — the ``_execute_iterations()`` level: rows 1-3 as the loop
  itself resolves them, plus the marker-composition defensive invariant.
- ``TestExecuteVerifyReviewLoop*`` (step-3 RED / step-4 GREEN) — the
  ``_execute_verify_review_loop()`` caller-propagation level: a REQUEUED (or
  terminal-override CANCELLED) exec outcome must leave the loop instead of
  falling through into VERIFY.

Reuses the injection harness from ``test_liveness_boundary_gate.py``
(``_make_workflow`` / ``_stub_iteration_helpers``) rather than duplicating
it — an established cross-test-module idiom under this repo's root
``--import-mode=importlib`` (see e.g. ``test_convert_to_blocked.py``,
``test_eval_boundary_suite.py``). That module owns a different PRD's
boundary gate (δ liveness); this one owns γ's, kept separate so a failure
here is never misread as a liveness regression.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from test_liveness_boundary_gate import _make_workflow, _stub_iteration_helpers

from orchestrator.agents.invoke import AgentResult
from orchestrator.scheduler import TerminalExitRejection, is_transient_api_requeue
from orchestrator.workflow import ZERO_OUTPUT_HANG_REASON, WorkflowOutcome, WorkflowState


def _zero_output_result(*, api_error_status: int | None = None, **overrides: object) -> AgentResult:
    """A zero-output ``AgentResult`` (the shape ``_zero_output_agent_result``
    uses — ``timed_out=True``, ``transcript_turns=0`` — so ``is_zero_output_timeout``
    is always True), varying only the fields each test injects.
    """
    fields: dict[str, object] = {
        'success': False,
        'output': 'Agent produced no output',
        'timed_out': True,
        'turns': 0,
        'cost_usd': 0.0,
        'duration_ms': 1_200_000,
        'transcript_turns': 0,
        'api_error_status': api_error_status,
    }
    fields.update(overrides)
    return AgentResult(**fields)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _execute_iterations — boundary row 1 (5xx-attributed zero-output → REQUEUED)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestExecuteIterationsRow1Requeue:
    """Row 1: a flushed 5xx on a zero-output result exits REQUEUED, first try."""

    async def test_row1_requeues_without_touching_breaker_or_blocking(
        self, tmp_path,
    ) -> None:
        """(a) Default threshold (2): REQUEUED on the FIRST iteration, no
        counter/breaker touched, no `blocked` write, no L0."""
        wf = _make_workflow(tmp_path=tmp_path)
        mock_invoke = _stub_iteration_helpers(wf, _zero_output_result(api_error_status=529))
        wf.scheduler.set_task_status = AsyncMock()  # type: ignore[method-assign]
        wf._mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)  # type: ignore[method-assign]

        with patch('orchestrator.workflow.submit_or_dedupe') as mock_submit:
            outcome = await wf._execute_iterations()

        assert outcome == WorkflowOutcome.REQUEUED, (
            f'Expected REQUEUED (5xx-attributed zero-output, exit-on-first); got {outcome!r}'
        )
        mock_invoke.assert_awaited_once()
        assert wf._zero_output_hang_info is None, (  # type: ignore[attr-defined]
            'zero-output counter/breaker must be untouched for this class'
        )
        assert wf._preserve_config_dir is False  # type: ignore[attr-defined]
        wf._mark_blocked.assert_not_awaited()
        mock_submit.assert_not_called()
        wf.scheduler.set_task_status.assert_awaited_once_with(wf.task_id, 'pending')

    async def test_row1_ignores_breaker_even_at_threshold_one(self, tmp_path) -> None:
        """(b) Threshold=1 still REQUEUES — proves the guard runs BEFORE the
        counter increment / threshold check, for any threshold value, not
        merely because the default threshold (2) was never reached."""
        wf = _make_workflow(tmp_path=tmp_path, max_consecutive_zero_output_timeouts=1)
        _stub_iteration_helpers(wf, _zero_output_result(api_error_status=529))
        wf.scheduler.set_task_status = AsyncMock()  # type: ignore[method-assign]
        wf._mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)  # type: ignore[method-assign]

        outcome = await wf._execute_iterations()

        assert outcome == WorkflowOutcome.REQUEUED
        assert wf._zero_output_hang_info is None  # type: ignore[attr-defined]

    async def test_row1_terminal_report_contract(self, tmp_path) -> None:
        """(c) The stashed TerminalReport carries the marker, the structured
        field, and the SM-2-safe phase pairing."""
        wf = _make_workflow(tmp_path=tmp_path)
        _stub_iteration_helpers(wf, _zero_output_result(api_error_status=529))
        wf.scheduler.set_task_status = AsyncMock()  # type: ignore[method-assign]
        wf._mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)  # type: ignore[method-assign]

        outcome = await wf._execute_iterations()

        assert outcome == WorkflowOutcome.REQUEUED
        report = wf._terminal_report
        assert report is not None
        assert report.outcome == WorkflowOutcome.REQUEUED
        assert report.api_error_status == 529
        assert 'agent API error: HTTP 529' in report.reason
        assert report.phase == wf.machine.state
        assert report.blocked_from_phase == wf.machine.state
        assert report.category is None
        assert report.counts_against_requeue_cap is True, (
            'route-1 (counts_against_cap=False) is history-only and would '
            'starve the transient cap — see design decisions'
        )

    async def test_row1_terminal_report_feeds_transient_requeue_lane(
        self, tmp_path,
    ) -> None:
        """(d) The report this task produces is recognised by the REAL
        scheduler consumer, both field-first and via the legacy prose
        fallback (INV-1: structured field over regex)."""
        wf = _make_workflow(tmp_path=tmp_path)
        _stub_iteration_helpers(wf, _zero_output_result(api_error_status=529))
        wf.scheduler.set_task_status = AsyncMock()  # type: ignore[method-assign]
        wf._mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)  # type: ignore[method-assign]

        await wf._execute_iterations()

        report = wf._terminal_report
        assert report is not None
        assert is_transient_api_requeue(
            report.reason, api_error_status=report.api_error_status,
        ) is True
        # Field-alone (INV-1): prose carrying no marker at all is still routed
        # transient when the structured field is 5xx.
        assert is_transient_api_requeue(
            'Execution failed: unrelated prose', api_error_status=report.api_error_status,
        ) is True


# ---------------------------------------------------------------------------
# _execute_iterations — the 5xx band boundary
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestExecuteIterationsServerErrorBand:
    """(e) Only the 500-599 band requeues; everything else still feeds the
    zero-output breaker. Guards against an ``api_error_status is not None``
    mis-guard, and pins the 429 policy ``is_transient_api_requeue`` documents.
    """

    @pytest.mark.parametrize('status', [500, 599])
    async def test_5xx_band_requeues(self, tmp_path, status: int) -> None:
        wf = _make_workflow(tmp_path=tmp_path, max_consecutive_zero_output_timeouts=1)
        _stub_iteration_helpers(wf, _zero_output_result(api_error_status=status))
        wf.scheduler.set_task_status = AsyncMock()  # type: ignore[method-assign]
        wf._mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)  # type: ignore[method-assign]

        outcome = await wf._execute_iterations()

        assert outcome == WorkflowOutcome.REQUEUED, (
            f'status={status} is inside the 5xx band and must requeue; got {outcome!r}'
        )

    @pytest.mark.parametrize('status', [499, 429, 404, 400, 600])
    async def test_non_5xx_band_still_trips_breaker(self, tmp_path, status: int) -> None:
        wf = _make_workflow(tmp_path=tmp_path, max_consecutive_zero_output_timeouts=1)
        _stub_iteration_helpers(wf, _zero_output_result(api_error_status=status))
        wf.scheduler.set_task_status = AsyncMock()  # type: ignore[method-assign]
        wf._mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)  # type: ignore[method-assign]

        outcome = await wf._execute_iterations()

        assert outcome == WorkflowOutcome.BLOCKED, (
            f'status={status} is OUTSIDE the 5xx band and must still trip the '
            f'wedge breaker; got {outcome!r}'
        )
        assert wf._zero_output_hang_info is not None  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# _execute_iterations — rows 2 & 3: the byte-identical SIGKILL sub-path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestExecuteIterationsRow2Row3Regression:
    """Rows 2 & 3: the genuine-wedge (no status attached) sub-path — green
    BEFORE and AFTER this task's change; deliberate characterization guards
    for "byte-identical SIGKILL path"."""

    async def test_row2_single_no_status_timeout_trips_breaker(self, tmp_path) -> None:
        """(f) Row 2: a SIGKILL kill (no JSON flushed → api_error_status=None)
        at threshold 1 blocks — no requeue write."""
        wf = _make_workflow(tmp_path=tmp_path, max_consecutive_zero_output_timeouts=1)
        _stub_iteration_helpers(wf, _zero_output_result(api_error_status=None))
        wf.scheduler.set_task_status = AsyncMock()  # type: ignore[method-assign]
        wf._mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)  # type: ignore[method-assign]

        outcome = await wf._execute_iterations()

        assert outcome == WorkflowOutcome.BLOCKED
        assert wf._zero_output_hang_info is not None  # type: ignore[attr-defined]
        assert ZERO_OUTPUT_HANG_REASON in wf._zero_output_hang_info['reason']  # type: ignore[attr-defined]
        assert wf._preserve_config_dir is True  # type: ignore[attr-defined]
        wf.scheduler.set_task_status.assert_not_awaited()

    async def test_row3_two_consecutive_no_status_timeouts_trip_breaker(
        self, tmp_path,
    ) -> None:
        """(g) Row 3: two consecutive no-status zero-output results at
        threshold 2 block after exactly 2 invocations."""
        wf = _make_workflow(tmp_path=tmp_path, max_consecutive_zero_output_timeouts=2)
        mock_invoke = _stub_iteration_helpers(wf, _zero_output_result(api_error_status=None))
        wf.scheduler.set_task_status = AsyncMock()  # type: ignore[method-assign]
        wf._mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)  # type: ignore[method-assign]

        outcome = await wf._execute_iterations()

        assert outcome == WorkflowOutcome.BLOCKED
        assert mock_invoke.await_count == 2
        assert 'consecutive_zero_output=2' in wf._zero_output_hang_info['detail']  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# _execute_verify_review_loop — caller propagation of REQUEUED / CANCELLED
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestExecuteVerifyReviewLoopPropagation:
    """A REQUEUED (or terminal-override CANCELLED) exec outcome must leave
    the EXECUTE→VERIFY→REVIEW loop instead of falling through into VERIFY.

    Today the loop's EXECUTE arm branches only on ESCALATED and BLOCKED, so
    an un-propagated REQUEUED would silently continue into VERIFY on a task
    that has already been re-pended to ``pending`` elsewhere.
    """

    async def test_row1_requeued_leaves_loop_without_running_verify(
        self, tmp_path,
    ) -> None:
        """(a) A 529 zero-output result requeues out of the loop; VERIFY and
        _mark_blocked never run, and the machine stays in EXECUTE (so
        run()'s SM-2 report.phase == machine.state check holds)."""
        wf = _make_workflow(tmp_path=tmp_path)
        wf.artifacts.get_review_cycles_total = MagicMock(return_value=0)  # type: ignore[method-assign]
        wf.artifacts.get_amendment_rounds_total = MagicMock(return_value=0)  # type: ignore[method-assign]
        wf._verify_debugfix_loop = AsyncMock(return_value=WorkflowOutcome.DONE)  # type: ignore[method-assign]
        wf._mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)  # type: ignore[method-assign]
        _stub_iteration_helpers(wf, _zero_output_result(api_error_status=529))
        wf.scheduler.set_task_status = AsyncMock()  # type: ignore[method-assign]

        outcome = await wf._execute_verify_review_loop()

        assert outcome == WorkflowOutcome.REQUEUED
        wf._verify_debugfix_loop.assert_not_awaited()
        wf._mark_blocked.assert_not_awaited()
        assert wf.machine.state is WorkflowState.EXECUTE
        assert wf._terminal_report is not None
        assert wf._terminal_report.outcome is WorkflowOutcome.REQUEUED
        wf.scheduler.set_task_status.assert_awaited_once_with(wf.task_id, 'pending')

    async def test_terminal_override_returns_cancelled_without_running_verify(
        self, tmp_path,
    ) -> None:
        """(b) A terminal-exit rejection observed during the re-pend write
        wins over the requeue intent: the loop returns CANCELLED, not
        REQUEUED, still without running VERIFY, and leaves no REQUEUED
        report stashed."""
        wf = _make_workflow(tmp_path=tmp_path)
        wf.artifacts.get_review_cycles_total = MagicMock(return_value=0)  # type: ignore[method-assign]
        wf.artifacts.get_amendment_rounds_total = MagicMock(return_value=0)  # type: ignore[method-assign]
        wf._verify_debugfix_loop = AsyncMock(return_value=WorkflowOutcome.DONE)  # type: ignore[method-assign]
        wf._mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)  # type: ignore[method-assign]
        _stub_iteration_helpers(wf, _zero_output_result(api_error_status=529))
        wf.scheduler.set_task_status = AsyncMock(  # type: ignore[method-assign]
            side_effect=TerminalExitRejection(
                task_id=wf.task_id, old_status='cancelled',
                target_status='pending', raw='terminal-exit gate',
            )
        )

        outcome = await wf._execute_verify_review_loop()

        assert outcome == WorkflowOutcome.CANCELLED
        wf._verify_debugfix_loop.assert_not_awaited()
        assert wf.machine.state is WorkflowState.CANCELLED
        assert wf._terminal_report is None

    async def test_row3_caller_regression_blocks_with_infra_issue(
        self, tmp_path,
    ) -> None:
        """(c) Row 3 at the caller layer: two consecutive no-status
        zero-output results block via ``_mark_blocked(category='infra_issue')``
        — green before and after this task (regression guard)."""
        wf = _make_workflow(tmp_path=tmp_path, max_consecutive_zero_output_timeouts=2)
        wf.artifacts.get_review_cycles_total = MagicMock(return_value=0)  # type: ignore[method-assign]
        wf.artifacts.get_amendment_rounds_total = MagicMock(return_value=0)  # type: ignore[method-assign]
        wf._verify_debugfix_loop = AsyncMock(return_value=WorkflowOutcome.DONE)  # type: ignore[method-assign]
        wf._mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)  # type: ignore[method-assign]
        _stub_iteration_helpers(wf, _zero_output_result(api_error_status=None))
        wf.scheduler.set_task_status = AsyncMock()  # type: ignore[method-assign]

        outcome = await wf._execute_verify_review_loop()

        assert outcome == WorkflowOutcome.BLOCKED
        wf._mark_blocked.assert_awaited_once()
        call_args, call_kwargs = wf._mark_blocked.await_args
        assert ZERO_OUTPUT_HANG_REASON in call_args[0]
        assert call_kwargs['category'] == 'infra_issue'
        wf.scheduler.set_task_status.assert_not_awaited()
