"""Tests for Harness main-tip integrity sweep — task 1832.

Background: merge-queue verify is SCOPED per task (diff/module), so test-suite-wide
breakage and cross-file regressions can land on main and surface only incidentally
when an unlucky later task runs a broad/__fallback__ verify.  Confirmed instances:
  - task 1829: autouse fixture silently defeated two correctness tests; only task
    1817's broad __fallback__ verify surfaced it (esc-1817-28).
  - esc-1749-16: stale tests poking retired singular fields landed on main.

Fix: a background asyncio task on the Harness wakes every
``config.main_tip_sweep_interval_secs``, and when main has advanced since the last
sweep, runs a FULL unscoped verification (all subprojects: test + lint + typecheck)
against a throwaway detached worktree pinned at the current main SHA — completely off
the serial merge lane, so per-merge latency is untouched.  On drift it files one L1
escalation per distinct bad SHA.

This file covers:
  step-1:  test_config_defaults_main_tip_sweep
  step-7:  TestRunMainTipSweepHarness — _run_main_tip_sweep single-pass tests
  step-9:  TestRunMainTipSweepHarness edge cases (pass, sha-dedup, no-queue)
  step-11: TestMainTipSweepLifecycle — start/stop wiring tests
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import _init_harness_state_for_test

from orchestrator.config import OrchestratorConfig
from orchestrator.harness import Harness
from orchestrator.verify import VerifyResult

# ---------------------------------------------------------------------------
# step-1: Config field presence and defaults
# ---------------------------------------------------------------------------


def test_config_defaults_main_tip_sweep() -> None:
    """OrchestratorConfig exposes main_tip_sweep_enabled (True) and
    main_tip_sweep_interval_secs (1800.0) with the correct defaults."""
    config = OrchestratorConfig()
    assert config.main_tip_sweep_enabled is True
    assert config.main_tip_sweep_interval_secs == 1800.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MAIN_SHA = 'b' * 40

FAILING_RESULT = VerifyResult(
    passed=False,
    test_output='FAILED test_x',
    lint_output='',
    type_output='',
    summary='test_failure',
    cause_hint='FAILED test_x',
    category='test_failure',
)

PASSING_RESULT = VerifyResult(
    passed=True,
    test_output='',
    lint_output='',
    type_output='',
    summary='all checks passed',
)


def _make_sweep_harness(*, main_sha: str = MAIN_SHA) -> Harness:
    """Build a minimal bare Harness for single-pass sweep tests."""
    h = Harness.__new__(Harness)
    _init_harness_state_for_test(h)
    h.config = OrchestratorConfig()
    h.git_ops = MagicMock()
    h.git_ops.get_main_sha = AsyncMock(return_value=main_sha)
    h._escalation_queue = MagicMock()
    h._escalation_queue.make_id = MagicMock(return_value='esc-sweep-1')
    h._escalation_queue.has_open_l1 = MagicMock(return_value=False)
    h._last_swept_main_sha = None
    return h


# ---------------------------------------------------------------------------
# step-7: _run_main_tip_sweep single-pass — drift escalation
# ---------------------------------------------------------------------------


class TestRunMainTipSweepHarness:
    """step-7: _run_main_tip_sweep files a level-1 infra_issue escalation on drift."""

    @pytest.mark.asyncio
    async def test_run_main_tip_sweep_escalates_on_drift(self) -> None:
        """When run_main_tip_sweep returns a failing VerifyResult, the harness
        calls _escalation_queue.submit with a blocking L1 infra_issue escalation
        whose summary contains the SHA prefix and failure category."""
        from orchestrator import verify as verify_module

        h = _make_sweep_harness()

        with patch.object(
            verify_module,
            'run_main_tip_sweep',
            new=AsyncMock(return_value=(MAIN_SHA, FAILING_RESULT)),
        ):
            await h._run_main_tip_sweep()

        h._escalation_queue.submit.assert_called_once()  # type: ignore[union-attr, attr-defined]
        submitted_esc = h._escalation_queue.submit.call_args[0][0]  # type: ignore[union-attr, attr-defined]
        assert submitted_esc.level == 1
        assert submitted_esc.category == 'infra_issue'
        assert submitted_esc.severity == 'blocking'
        sha_prefix = MAIN_SHA[:12]
        assert sha_prefix in submitted_esc.summary, (
            f'Expected SHA prefix {sha_prefix!r} in summary: {submitted_esc.summary!r}'
        )
        assert 'test_failure' in submitted_esc.summary, (
            f'Expected failure category in summary: {submitted_esc.summary!r}'
        )

    @pytest.mark.asyncio
    async def test_run_main_tip_sweep_pass_no_escalation(self) -> None:
        """When run_main_tip_sweep returns a passing VerifyResult, submit is NOT called
        and h._last_swept_main_sha is updated to the swept SHA."""
        from orchestrator import verify as verify_module

        h = _make_sweep_harness()

        with patch.object(
            verify_module,
            'run_main_tip_sweep',
            new=AsyncMock(return_value=(MAIN_SHA, PASSING_RESULT)),
        ):
            await h._run_main_tip_sweep()

        h._escalation_queue.submit.assert_not_called()  # type: ignore[union-attr, attr-defined]
        assert h._last_swept_main_sha == MAIN_SHA, (
            f'Expected _last_swept_main_sha={MAIN_SHA!r}, got {h._last_swept_main_sha!r}'
        )

    @pytest.mark.asyncio
    async def test_run_main_tip_sweep_sha_dedup(self) -> None:
        """When _last_swept_main_sha equals the current main SHA, verify.run_main_tip_sweep
        is NOT called (expensive full verify skipped for unchanged tip)."""
        from orchestrator import verify as verify_module

        h = _make_sweep_harness()
        h._last_swept_main_sha = MAIN_SHA  # pre-set to current SHA

        mock_sweep = AsyncMock(return_value=(MAIN_SHA, FAILING_RESULT))
        with patch.object(verify_module, 'run_main_tip_sweep', new=mock_sweep):
            await h._run_main_tip_sweep()

        mock_sweep.assert_not_called()
        h._escalation_queue.submit.assert_not_called()  # type: ignore[union-attr, attr-defined]

    @pytest.mark.asyncio
    async def test_run_main_tip_sweep_no_queue_noop(self) -> None:
        """When _escalation_queue is None and drift is detected, no exception
        is raised and the method returns cleanly.  The SHA is still marked swept
        so the loop doesn't re-trigger every tick when no queue is wired."""
        from orchestrator import verify as verify_module

        h = _make_sweep_harness()
        h._escalation_queue = None  # bare-harness: no queue attached

        with patch.object(
            verify_module,
            'run_main_tip_sweep',
            new=AsyncMock(return_value=(MAIN_SHA, FAILING_RESULT)),
        ):
            # Must not raise
            await h._run_main_tip_sweep()

        assert h._last_swept_main_sha == MAIN_SHA, (
            f'Expected _last_swept_main_sha={MAIN_SHA!r} even when queue is None, '
            f'got {h._last_swept_main_sha!r}'
        )

    @pytest.mark.asyncio
    async def test_run_main_tip_sweep_has_open_l1_dedup(self) -> None:
        """When has_open_l1 returns True for the sweep task_id, submit is NOT
        called — prevents duplicate L1 escalations across restarts for the same
        bad SHA.  In-process _last_swept_main_sha resets to None on every restart
        so has_open_l1 is the persistent dedup layer that covers restart loops.
        The SHA is still marked swept so subsequent ticks (same SHA) skip the
        expensive verify."""
        from orchestrator import verify as verify_module

        h = _make_sweep_harness()
        h._escalation_queue.has_open_l1 = MagicMock(return_value=True)  # type: ignore[union-attr]

        with patch.object(
            verify_module,
            'run_main_tip_sweep',
            new=AsyncMock(return_value=(MAIN_SHA, FAILING_RESULT)),
        ):
            await h._run_main_tip_sweep()

        h._escalation_queue.submit.assert_not_called()  # type: ignore[union-attr, attr-defined]
        assert h._last_swept_main_sha == MAIN_SHA, (
            f'SHA should be marked swept even when has_open_l1 skips filing, '
            f'got {h._last_swept_main_sha!r}'
        )

