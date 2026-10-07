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

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import _init_harness_state_for_test

from orchestrator.config import OrchestratorConfig
from orchestrator.harness import Harness
from orchestrator.main_tip_sweep_cadence import SweepSeedMode
from orchestrator.run_store import RunStore
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


def _make_sweep_harness(
    *, main_sha: str = MAIN_SHA, run_store: RunStore | None = None,
) -> Harness:
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
    h._run_store = run_store
    h._main_sweep_cold_control = None
    return h


# ---------------------------------------------------------------------------
# task 5812: warm-by-default sweep with a periodic COLD control
# ---------------------------------------------------------------------------

SHA_A = 'a' * 40
SHA_B = 'c' * 40


async def _passing_sweep(config, git_ops, *, main_sha, **kwargs):
    return (main_sha, PASSING_RESULT)


async def _tick_seed_modes(h: Harness, shas: list[str], sweep: AsyncMock) -> list:
    """Tick the sweep once per sha; return each run_main_tip_sweep call's seed_mode."""
    from orchestrator import verify as verify_module

    h.git_ops.get_main_sha = AsyncMock(side_effect=shas)
    h._close_superseded_main_sweep_escalations = AsyncMock()  # type: ignore[method-assign]
    with patch.object(verify_module, 'run_main_tip_sweep', new=sweep):
        for _ in shas:
            await h._run_main_tip_sweep()
    return [c.kwargs['seed_mode'] for c in sweep.call_args_list]


class TestMainTipSweepColdControl:
    """The harness picks each sweep's seed mode through MainSweepColdControl."""

    @pytest.mark.asyncio
    async def test_first_sweep_is_cold_and_the_next_sha_is_warm(self) -> None:
        modes = await _tick_seed_modes(
            _make_sweep_harness(), [SHA_A, SHA_B], AsyncMock(side_effect=_passing_sweep),
        )

        assert modes == [SweepSeedMode.COLD, SweepSeedMode.WARM]

    @pytest.mark.asyncio
    async def test_zero_cold_interval_sweeps_every_sha_cold(self) -> None:
        h = _make_sweep_harness()
        h.config = OrchestratorConfig(main_tip_sweep_cold_interval_secs=0)

        modes = await _tick_seed_modes(
            h, [SHA_A, SHA_B], AsyncMock(side_effect=_passing_sweep),
        )

        assert modes == [SweepSeedMode.COLD, SweepSeedMode.COLD]

    @pytest.mark.asyncio
    async def test_cold_sweep_without_a_verdict_keeps_the_next_sweep_cold(self) -> None:
        """An infra-aborted (None) cold sweep is not ground-truth evidence."""
        sweep = AsyncMock(side_effect=[None, (SHA_A, PASSING_RESULT)])

        modes = await _tick_seed_modes(_make_sweep_harness(), [SHA_A, SHA_A], sweep)

        assert modes == [SweepSeedMode.COLD, SweepSeedMode.COLD]

    @pytest.mark.asyncio
    async def test_cold_cadence_survives_a_restart(self, tmp_path) -> None:
        """Across a restart: harness #2 shares only runs.db with harness #1, and
        still sweeps warm because #1's cold verdict was persisted there."""
        store = RunStore(tmp_path / 'runs.db')

        first = await _tick_seed_modes(
            _make_sweep_harness(run_store=store), [SHA_A],
            AsyncMock(side_effect=_passing_sweep),
        )
        after_restart = await _tick_seed_modes(
            _make_sweep_harness(run_store=store), [SHA_B],
            AsyncMock(side_effect=_passing_sweep),
        )

        assert first == [SweepSeedMode.COLD]
        assert after_restart == [SweepSeedMode.WARM]

    @pytest.mark.asyncio
    async def test_sweep_logs_its_mode_before_and_with_its_verdict(self, caplog) -> None:
        from orchestrator import verify as verify_module

        h = _make_sweep_harness()
        messages_at_verify: list[str] = []

        async def _sweep(config, git_ops, *, main_sha, **kwargs):
            messages_at_verify.extend(r.getMessage() for r in caplog.records)
            return (main_sha, PASSING_RESULT)

        h.git_ops.get_main_sha = AsyncMock(return_value=SHA_A)
        h._close_superseded_main_sweep_escalations = AsyncMock()  # type: ignore[method-assign]
        with (
            caplog.at_level(logging.INFO, logger='orchestrator.harness'),
            patch.object(verify_module, 'run_main_tip_sweep', new=AsyncMock(side_effect=_sweep)),
        ):
            await h._run_main_tip_sweep()

        assert any(SHA_A[:12] in m and 'cold' in m for m in messages_at_verify), (
            f'no start line naming the sha and mode before verification: {messages_at_verify}'
        )
        after = [r.getMessage() for r in caplog.records][len(messages_at_verify):]
        assert any(
            'cold' in m and SHA_A[:12] in m and 'passed' in m for m in after
        ), f'no verdict line naming mode, sha and outcome: {after}'


# ---------------------------------------------------------------------------
# step-7: _run_main_tip_sweep single-pass — drift escalation
# ---------------------------------------------------------------------------


class TestRunMainTipSweepHarness:
    """step-7: _run_main_tip_sweep files a level-1 infra_issue escalation on drift."""

    @pytest.mark.asyncio
    async def test_run_main_tip_sweep_escalates_on_drift(self) -> None:
        """When run_main_tip_sweep returns a failing VerifyResult AND the
        confirm-before-alarm gate confirms the failure is real (task 2370),
        the harness calls _escalation_queue.submit with a blocking L1
        infra_issue escalation whose summary contains the SHA prefix and
        failure category."""
        from orchestrator import verify as verify_module

        h = _make_sweep_harness()

        with (
            patch.object(
                verify_module,
                'run_main_tip_sweep',
                new=AsyncMock(return_value=(MAIN_SHA, FAILING_RESULT)),
            ),
            patch.object(
                verify_module,
                'confirm_main_tip_failure_is_real',
                new=AsyncMock(return_value=True),
            ),
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
    async def test_run_main_tip_sweep_confirm_suppresses_no_escalation(self) -> None:
        """task 2370: when confirm_main_tip_failure_is_real returns False (the
        named failing tests passed on isolated re-run at current tip -> a
        load-induced flake, not real drift), the harness must NOT submit an
        escalation.  The SHA is still marked swept (no re-sweep next tick),
        and — deliberately — self-heal is NOT invoked: a suppressed-flake
        verdict from a scoped isolated re-run is weaker evidence than the
        genuine full-verify PASS that self-heal requires."""
        from orchestrator import verify as verify_module

        h = _make_sweep_harness()
        h._close_superseded_main_sweep_escalations = AsyncMock()  # type: ignore[method-assign]

        with (
            patch.object(
                verify_module,
                'run_main_tip_sweep',
                new=AsyncMock(return_value=(MAIN_SHA, FAILING_RESULT)),
            ),
            patch.object(
                verify_module,
                'confirm_main_tip_failure_is_real',
                new=AsyncMock(return_value=False),
            ) as mock_confirm,
        ):
            await h._run_main_tip_sweep()

        mock_confirm.assert_called_once()
        h._escalation_queue.submit.assert_not_called()  # type: ignore[union-attr, attr-defined]
        assert h._last_swept_main_sha == MAIN_SHA, (
            f'Expected _last_swept_main_sha={MAIN_SHA!r} even when suppressed '
            f'(no re-sweep of the same SHA), got {h._last_swept_main_sha!r}'
        )
        h._close_superseded_main_sweep_escalations.assert_not_called()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_run_main_tip_sweep_confirm_true_still_escalates(self) -> None:
        """task 2370: when confirm_main_tip_failure_is_real returns True (the
        failure is confirmed real, e.g. still failing on isolated re-run), the
        harness files the level-1 infra_issue escalation exactly as before —
        the confirm gate must never mask a genuine red main (regression
        guard)."""
        from orchestrator import verify as verify_module

        h = _make_sweep_harness()

        with (
            patch.object(
                verify_module,
                'run_main_tip_sweep',
                new=AsyncMock(return_value=(MAIN_SHA, FAILING_RESULT)),
            ),
            patch.object(
                verify_module,
                'confirm_main_tip_failure_is_real',
                new=AsyncMock(return_value=True),
            ) as mock_confirm,
        ):
            await h._run_main_tip_sweep()

        mock_confirm.assert_called_once()
        h._escalation_queue.submit.assert_called_once()  # type: ignore[union-attr, attr-defined]
        submitted_esc = h._escalation_queue.submit.call_args[0][0]  # type: ignore[union-attr, attr-defined]
        assert submitted_esc.level == 1
        assert submitted_esc.category == 'infra_issue'
        assert submitted_esc.severity == 'blocking'

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

    # -----------------------------------------------------------------------
    # task 2558: current-tip re-confirmation arm (composes with task 2370's
    # confirm_main_tip_failure_is_real subset re-run).  Filing now requires
    # BOTH the subset confirm AND the current main tip still being the observed
    # bad SHA — closing the "evidence since mutated" gap (main advancing past
    # the observed SHA during the minutes-long verify).
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_run_main_tip_sweep_tip_advanced_suppresses_no_escalation(self) -> None:
        """task 2558: FAIL + confirm=True but the main tip ADVANCED past swept_sha
        during the verify → NOT filed.  The observed bad SHA is stale/superseded,
        so alarming (and recommending a destructive rewind to it) would repeat the
        survey §1.7 precedent where a 'last-green' rewind named a commit that also
        failed / had since mutated.  get_main_sha is called twice (once to resolve
        the sweep SHA, once to re-confirm the current tip); confirm is still called
        once."""
        from orchestrator import verify as verify_module

        h = _make_sweep_harness()
        # 1st call = initial sweep SHA resolve (MAIN_SHA); 2nd = current-tip re-confirm
        # (advanced to a different SHA -> tip_unchanged=False).
        h.git_ops.get_main_sha = AsyncMock(side_effect=[MAIN_SHA, 'c' * 40])  # type: ignore[union-attr]

        with (
            patch.object(
                verify_module,
                'run_main_tip_sweep',
                new=AsyncMock(return_value=(MAIN_SHA, FAILING_RESULT)),
            ),
            patch.object(
                verify_module,
                'confirm_main_tip_failure_is_real',
                new=AsyncMock(return_value=True),
            ) as mock_confirm,
        ):
            await h._run_main_tip_sweep()

        mock_confirm.assert_called_once()
        assert h.git_ops.get_main_sha.call_count == 2, (  # type: ignore[union-attr]
            f'Expected get_main_sha called twice (sweep resolve + tip re-confirm), '
            f'got {h.git_ops.get_main_sha.call_count}'  # type: ignore[union-attr]
        )
        h._escalation_queue.submit.assert_not_called()  # type: ignore[union-attr, attr-defined]

    @pytest.mark.asyncio
    async def test_run_main_tip_sweep_tip_unchanged_still_escalates(self) -> None:
        """task 2558: FAIL + confirm=True + current tip UNCHANGED (default
        get_main_sha return_value=MAIN_SHA -> tip_unchanged=True) → filed exactly
        once via critical_filing_gate(rerun_confirmed=True).  The default path
        (tip has not moved) must still alarm on a genuine confirmed red main."""
        from orchestrator import verify as verify_module

        h = _make_sweep_harness()  # get_main_sha constant MAIN_SHA -> tip_unchanged=True

        with (
            patch.object(
                verify_module,
                'run_main_tip_sweep',
                new=AsyncMock(return_value=(MAIN_SHA, FAILING_RESULT)),
            ),
            patch.object(
                verify_module,
                'confirm_main_tip_failure_is_real',
                new=AsyncMock(return_value=True),
            ) as mock_confirm,
        ):
            await h._run_main_tip_sweep()

        mock_confirm.assert_called_once()
        h._escalation_queue.submit.assert_called_once()  # type: ignore[union-attr, attr-defined]
        submitted_esc = h._escalation_queue.submit.call_args[0][0]  # type: ignore[union-attr, attr-defined]
        assert submitted_esc.level == 1
        assert submitted_esc.category == 'infra_issue'

    @pytest.mark.asyncio
    async def test_run_main_tip_sweep_rerun_confirm_disabled_legacy_path(self) -> None:
        """task 2558: with main_tip_sweep_rerun_confirm_enabled=False the tip arm
        is disabled — FAIL + confirm=True files exactly as the legacy post-2370
        path did, with NO second get_main_sha re-resolution (tip_unchanged forced
        True).  The kill-switch restores byte-identical pre-2558 filing."""
        from orchestrator import verify as verify_module

        h = _make_sweep_harness()
        h.config = OrchestratorConfig(main_tip_sweep_rerun_confirm_enabled=False)
        # side_effect with a single value: a second get_main_sha call would raise
        # StopIteration, proving the disabled arm never re-resolves the tip.
        h.git_ops.get_main_sha = AsyncMock(side_effect=[MAIN_SHA])  # type: ignore[union-attr]

        with (
            patch.object(
                verify_module,
                'run_main_tip_sweep',
                new=AsyncMock(return_value=(MAIN_SHA, FAILING_RESULT)),
            ),
            patch.object(
                verify_module,
                'confirm_main_tip_failure_is_real',
                new=AsyncMock(return_value=True),
            ) as mock_confirm,
        ):
            await h._run_main_tip_sweep()

        mock_confirm.assert_called_once()
        assert h.git_ops.get_main_sha.call_count == 1, (  # type: ignore[union-attr]
            f'Disabled tip arm must not re-resolve the tip; get_main_sha called '
            f'{h.git_ops.get_main_sha.call_count} times'  # type: ignore[union-attr]
        )
        h._escalation_queue.submit.assert_called_once()  # type: ignore[union-attr, attr-defined]

