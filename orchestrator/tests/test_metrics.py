"""Tests for evals/metrics.py — composite scoring, false-green guard, null-work guard."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.evals.metrics import (
    EvalMetrics,
    _is_false_green,
    _is_null_work,
    compute_composite,
)


class TestComputeComposite:
    def test_tests_fail_scores_zero(self):
        m = EvalMetrics(tests_pass=False, plan_steps=4)
        assert compute_composite(m) == 0.0

    def test_tests_none_scores_zero(self):
        """``tests_pass=None`` (from the guards) must gate to 0."""
        m = EvalMetrics(tests_pass=None, plan_steps=4)
        assert compute_composite(m) == 0.0

    def test_clean_run_scores_one(self):
        m = EvalMetrics(
            tests_pass=True, plan_steps=8,
            review_blocking_issues=0, debug_cycles=0,
        )
        assert compute_composite(m) == 1.0

    def test_blocking_issues_reduce_score(self):
        # 2 blocking / 8 steps = 0.25 rate; 1 - 0.5 = 0.5
        m = EvalMetrics(
            tests_pass=True, plan_steps=8,
            review_blocking_issues=2, debug_cycles=0,
        )
        assert compute_composite(m) == 0.5

    def test_debug_cycles_penalty(self):
        m = EvalMetrics(
            tests_pass=True, plan_steps=8,
            review_blocking_issues=0, debug_cycles=4,
        )
        # 1 - 0 - 0.2 = 0.8
        assert compute_composite(m) == 0.8


class TestFalseGreenGuard:
    """The 404-bug signature: capped iterations, $0 cost, no diff, yet T/T/T."""

    def _sig(self, **overrides) -> EvalMetrics:
        base: dict[str, Any] = dict(
            tests_pass=True,
            lint_clean=True,
            typecheck_clean=True,
            lines_changed=0,
            files_changed=0,
            iterations=20,
            cost_usd=0.0,
        )
        base.update(overrides)
        return EvalMetrics(**base)

    def test_matches_exact_signature(self):
        assert _is_false_green(self._sig(), max_iterations=20) is True

    def test_exceeds_cap_still_matches(self):
        assert _is_false_green(self._sig(iterations=25), max_iterations=20) is True

    def test_below_cap_does_not_match(self):
        assert _is_false_green(self._sig(iterations=19), max_iterations=20) is False

    def test_any_cost_does_not_match(self):
        assert _is_false_green(self._sig(cost_usd=0.01), max_iterations=20) is False

    def test_any_lines_changed_does_not_match(self):
        assert _is_false_green(self._sig(lines_changed=1), max_iterations=20) is False

    def test_any_files_changed_does_not_match(self):
        assert _is_false_green(self._sig(files_changed=1), max_iterations=20) is False

    def test_tests_fail_does_not_match(self):
        """Baselines that already fail are score-safe and should pass through."""
        assert (
            _is_false_green(self._sig(tests_pass=False), max_iterations=20) is False
        )

    def test_tests_none_does_not_match(self):
        assert _is_false_green(self._sig(tests_pass=None), max_iterations=20) is False

    def test_legitimate_done_run_does_not_match(self):
        """A real done outcome (short, made changes, spent money) must pass through."""
        m = EvalMetrics(
            tests_pass=True,
            lint_clean=True,
            typecheck_clean=True,
            lines_changed=171,
            files_changed=10,
            iterations=4,
            cost_usd=0.0,  # vLLM runs show $0; this alone is fine
        )
        assert _is_false_green(m, max_iterations=20) is False

    def test_higher_cap_config_raises_threshold(self):
        """A task with max_iterations=30 should not match on 20 iterations."""
        assert _is_false_green(self._sig(iterations=20), max_iterations=30) is False
        assert _is_false_green(self._sig(iterations=30), max_iterations=30) is True


class TestNullWorkGuard:
    """NULL-byte implementer signature: real inference, zero code changes."""

    def _sig(self, **overrides) -> EvalMetrics:
        base: dict[str, Any] = dict(
            tests_pass=True,
            lint_clean=True,
            typecheck_clean=True,
            lines_changed=0,
            files_changed=0,
            iterations=5,
            cost_usd=0.42,
        )
        base.update(overrides)
        return EvalMetrics(**base)

    def test_matches_exact_signature(self):
        assert _is_null_work(self._sig()) is True

    def test_single_iteration_matches(self):
        assert _is_null_work(self._sig(iterations=1)) is True

    def test_high_cost_still_matches(self):
        assert _is_null_work(self._sig(cost_usd=12.50)) is True

    def test_lines_changed_does_not_match(self):
        assert _is_null_work(self._sig(lines_changed=1)) is False

    def test_files_changed_does_not_match(self):
        assert _is_null_work(self._sig(files_changed=1)) is False

    def test_zero_cost_does_not_match(self):
        """Zero cost is the 404-bug guard's domain, not ours."""
        assert _is_null_work(self._sig(cost_usd=0.0)) is False

    def test_zero_iterations_does_not_match(self):
        """No iterations means the implementer never ran — not our signature."""
        assert _is_null_work(self._sig(iterations=0)) is False

    def test_legitimate_run_does_not_match(self):
        """A real run that made changes must not trigger."""
        m = EvalMetrics(
            tests_pass=True,
            lint_clean=True,
            typecheck_clean=True,
            lines_changed=171,
            files_changed=10,
            iterations=4,
            cost_usd=1.23,
        )
        assert _is_null_work(m) is False

    def test_false_green_takes_priority_over_null_work(self):
        """The guards are mutually exclusive on the cost dimension:
        false-green requires cost==0, null-work requires cost>0."""
        # Exact false-green signature has cost_usd=0 -> null_work cannot match
        m_false_green = EvalMetrics(
            tests_pass=True, lines_changed=0, files_changed=0,
            iterations=20, cost_usd=0.0,
        )
        assert _is_false_green(m_false_green, max_iterations=20) is True
        assert _is_null_work(m_false_green) is False

        # Null-work signature has cost > 0 -> false-green cannot match
        m_null_work = EvalMetrics(
            tests_pass=True, lines_changed=0, files_changed=0,
            iterations=5, cost_usd=0.42,
        )
        assert _is_false_green(m_null_work, max_iterations=20) is False
        assert _is_null_work(m_null_work) is True


# ---------------------------------------------------------------------------
# Task 1809 step-15: _git_diff_stats returncode check + (-1,-1) sentinel
# ---------------------------------------------------------------------------

def _make_fake_proc(returncode: int, stdout: bytes, stderr: bytes) -> MagicMock:
    """Build a minimal fake asyncio Process mock."""
    proc = MagicMock()
    proc.returncode = returncode
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    return proc


@pytest.mark.asyncio
class TestGitDiffStatsReturncode:
    """Step-15 (RED): _git_diff_stats must return (-1,-1) and emit a WARNING
    when git exits non-zero, rather than returning (0,0) silently.

    Before step-16 impl: rc!=0 path is silent and returns (0,0) → RED.
    After step-16 impl: returns (-1,-1) + WARNING at 'orchestrator.evals.metrics'.
    """

    async def test_nonzero_returncode_returns_sentinel_and_warns(
        self, caplog,
    ) -> None:
        from orchestrator.evals.metrics import _git_diff_stats

        fake_proc = _make_fake_proc(
            returncode=128,
            stdout=b'',
            stderr=b'fatal: bad object deadbeef',
        )

        with (
            patch('asyncio.create_subprocess_exec', return_value=fake_proc),
            caplog.at_level(logging.WARNING, logger='orchestrator.evals.metrics'),
        ):
            result = await _git_diff_stats(Path('/fake/worktree'), 'deadbeef')

        assert result == (-1, -1), (
            f'Expected (-1,-1) sentinel on rc=128; got {result!r}'
        )
        warning_texts = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert warning_texts, (
            'Expected a WARNING at orchestrator.evals.metrics on rc!=0; '
            'got no warnings'
        )

    async def test_create_subprocess_raises_returns_sentinel_and_warns(
        self, caplog,
    ) -> None:
        from orchestrator.evals.metrics import _git_diff_stats

        with (
            patch(
                'asyncio.create_subprocess_exec',
                side_effect=OSError('no such file'),
            ),
            caplog.at_level(logging.WARNING, logger='orchestrator.evals.metrics'),
        ):
            result = await _git_diff_stats(Path('/fake/worktree'), 'deadbeef')

        assert result == (-1, -1), (
            f'Expected (-1,-1) sentinel on OSError; got {result!r}'
        )
        warning_texts = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert warning_texts, (
            'Expected a WARNING at orchestrator.evals.metrics on OSError; '
            'got no warnings'
        )

    async def test_zero_returncode_valid_stat_output_parsed(
        self, caplog,
    ) -> None:
        from orchestrator.evals.metrics import _git_diff_stats

        stat_output = (
            b' src/foo.py | 10 +++++-----\n'
            b' src/bar.py |  5 ++---\n'
            b' 2 files changed, 15 insertions(+), 0 deletions(-)\n'
        )
        fake_proc = _make_fake_proc(returncode=0, stdout=stat_output, stderr=b'')

        with (
            patch('asyncio.create_subprocess_exec', return_value=fake_proc),
            caplog.at_level(logging.WARNING, logger='orchestrator.evals.metrics'),
        ):
            lines, files = await _git_diff_stats(Path('/fake/worktree'), 'abc123')

        assert files == 2, f'Expected files=2; got {files!r}'
        assert lines == 15, f'Expected lines=15; got {lines!r}'
        parse_warnings = [
            r.message for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert not parse_warnings, (
            f'No WARNING expected for successful parse; got: {parse_warnings}'
        )

    async def test_zero_returncode_empty_stdout_returns_zero_zero(
        self, caplog,
    ) -> None:
        from orchestrator.evals.metrics import _git_diff_stats

        fake_proc = _make_fake_proc(returncode=0, stdout=b'', stderr=b'')

        with (
            patch('asyncio.create_subprocess_exec', return_value=fake_proc),
            caplog.at_level(logging.WARNING, logger='orchestrator.evals.metrics'),
        ):
            result = await _git_diff_stats(Path('/fake/worktree'), 'abc123')

        assert result == (0, 0), (
            f'Expected (0,0) for legitimate empty diff; got {result!r}'
        )
        parse_warnings = [
            r.message for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert not parse_warnings, (
            f'No WARNING expected for empty-diff (no-change); got: {parse_warnings}'
        )


# ---------------------------------------------------------------------------
# Task 2477 step-01: resolve_cost_usd — Invariant P5 cost-source resolution
# ---------------------------------------------------------------------------

class TestResolveCostUsd:
    """A PURE resolver of ``(cost_usd, cost_source)`` implementing Invariant P5.

    Cost provenance ∈ {price_table, cli, unpriced_proxy}:
      - price_table  → the run's model is listed in ``prices`` (trust the table)
      - cli          → NATIVE cloud endpoint, model unlisted (CLI cost is
                       trustworthy for the real Anthropic API)
      - unpriced_proxy → a PROXIED endpoint (is_local_model) whose model is
                       unlisted → loud WARNING + a DEFINED fallback, never the
                       raw CLI number (which is wrong for proxied endpoints).
    """

    def test_priced_model_uses_price_table(self):
        from orchestrator.evals.metrics import resolve_cost_usd

        # 1_000_000 in @ $2/1M + 500_000 out @ $8/1M = 2.0 + 4.0 = 6.0
        cost, source = resolve_cost_usd(
            1_000_000, 500_000,
            model='model-x',
            prices={'model-x': {'input_per_1m': 2.0, 'output_per_1m': 8.0}},
            cli_cost_usd=99.0,          # must be IGNORED when the model is priced
            is_local_model=False,
        )
        assert source == 'price_table'
        assert cost == pytest.approx(6.0)

    def test_priced_model_accepts_price_entry_object(self):
        """Dual handling: a PriceEntry object works exactly like a plain dict."""
        from orchestrator.config import PriceEntry
        from orchestrator.evals.metrics import resolve_cost_usd

        cost, source = resolve_cost_usd(
            1_000_000, 500_000,
            model='model-x',
            prices={'model-x': PriceEntry(input_per_1m=2.0, output_per_1m=8.0)},
            cli_cost_usd=99.0,
            is_local_model=False,
        )
        assert source == 'price_table'
        assert cost == pytest.approx(6.0)

    def test_native_cloud_unlisted_uses_cli_cost(self):
        from orchestrator.evals.metrics import resolve_cost_usd

        cost, source = resolve_cost_usd(
            1_000_000, 1_000_000,
            model='claude-sonnet-4',    # not in prices
            prices={},
            cli_cost_usd=3.5,
            is_local_model=False,       # NATIVE cloud → CLI cost is trustworthy
        )
        assert source == 'cli'
        assert cost == pytest.approx(3.5)

    def test_proxied_unlisted_warns_and_uses_defined_fallback(self, caplog):
        from orchestrator.evals.metrics import resolve_cost_usd

        with caplog.at_level(logging.WARNING, logger='orchestrator.evals.metrics'):
            cost, source = resolve_cost_usd(
                1_000_000, 1_000_000,
                model='local-llama-70b',    # not in prices
                prices={},
                cli_cost_usd=0.0,           # raw CLI cost is WRONG for proxied
                is_local_model=True,        # PROXIED endpoint (ANTHROPIC_BASE_URL)
            )

        assert source == 'unpriced_proxy'
        # Defined fallback ($2/1M in + $8/1M out): (1e6*2 + 1e6*8)/1e6 = 10.0
        assert cost == pytest.approx(10.0)
        # Crucially NOT the raw CLI cost (0.0) — a defined number, never silent.
        assert cost != 0.0
        warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert warnings, (
            'Expected a WARNING at orchestrator.evals.metrics for an unpriced '
            'proxied model; got none'
        )

    def test_proxied_but_priced_still_uses_price_table(self):
        """A proxied endpoint whose model IS listed trusts the table, not 'cli'."""
        from orchestrator.evals.metrics import resolve_cost_usd

        cost, source = resolve_cost_usd(
            1_000_000, 500_000,
            model='model-x',
            prices={'model-x': {'input_per_1m': 2.0, 'output_per_1m': 8.0}},
            cli_cost_usd=99.0,
            is_local_model=True,        # proxied — but priced → price_table wins
        )
        assert source == 'price_table'
        assert cost == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# Task 2477 step-03: blend_composite — the C4 efficiency-adjusted composite
# ---------------------------------------------------------------------------

class TestBlendComposite:
    """The C4 ``composite`` = quality blended with normalized cost + latency,
    keeping the ``tests_pass`` HARD GATE (decision 11). PURE and additive —
    ``compute_composite`` (the pure ``quality`` score) is untouched.
    """

    def test_default_weights_sum_to_one(self):
        from orchestrator.evals.metrics import DEFAULT_COMPOSITE_WEIGHTS

        assert sum(DEFAULT_COMPOSITE_WEIGHTS.values()) == pytest.approx(1.0)

    def test_tests_fail_hard_gates_to_zero(self):
        """The HARD GATE: tests_pass falsy → 0.0 regardless of the axes."""
        from orchestrator.evals.metrics import blend_composite

        assert blend_composite(
            1.0, 1.0, 1.0, tests_pass=False,
        ) == 0.0

    def test_weighted_blend_default_weights(self):
        from orchestrator.evals.metrics import blend_composite

        # 0.6*1.0 + 0.2*0.5 + 0.2*0.5 == 0.6 + 0.1 + 0.1 == 0.8
        assert blend_composite(
            1.0, 0.5, 0.5, tests_pass=True,
        ) == pytest.approx(0.8, abs=1e-4)

    def test_best_on_every_axis_is_one(self):
        from orchestrator.evals.metrics import blend_composite

        assert blend_composite(
            1.0, 1.0, 1.0, tests_pass=True,
        ) == pytest.approx(1.0)

    def test_result_clamped_to_unit_interval(self):
        from orchestrator.evals.metrics import blend_composite

        # Over-unit inputs clamp to 1.0 …
        assert blend_composite(
            2.0, 1.0, 1.0, tests_pass=True,
        ) == 1.0
        # … and a negative combination clamps to 0.0 (still, tests passed).
        assert blend_composite(
            -5.0, 0.0, 0.0, tests_pass=True,
        ) == 0.0

    # -- task 3099: the PLAN-ONLY path ------------------------------------
    # An architect eval freezes every downstream role, so the cell never runs
    # a test. The ``tests_pass`` hard gate is therefore the wrong instrument
    # there: it reads "no test signal collected" as "the answer was wrong" and
    # zeroes the number that drives survivor selection. Under ``plan_only``
    # the caller supplies plan_quality as the QUALITY axis instead.

    def test_plan_only_blends_plan_quality_without_test_signal(self):
        """plan_only=True scores the θ-rubric plan_quality, not tests_pass."""
        from orchestrator.evals.metrics import blend_composite

        # 0.6*0.9 + 0.2*1.0 + 0.2*1.0 == 0.54 + 0.2 + 0.2 == 0.94
        assert blend_composite(
            0.9, 1.0, 1.0, tests_pass=None, plan_only=True,
        ) == pytest.approx(0.94, abs=1e-4)

    def test_plan_only_bypasses_the_tests_pass_hard_gate(self):
        """Even an explicit tests_pass=False must not gate a plan-only cell.

        A plan-only cell has NO test signal at all, so ``tests_pass`` — whatever
        its value — must not decide its composite. Pinned separately from the
        ``None`` case so a future "treat False as authoritative" edit fails.
        """
        from orchestrator.evals.metrics import blend_composite

        assert blend_composite(
            0.9, 1.0, 1.0, tests_pass=False, plan_only=True,
        ) == pytest.approx(0.94, abs=1e-4)

    def test_plan_only_defaults_false_so_the_implementer_gate_is_intact(self):
        """The workflow path is byte-identical: no plan_only → hard gate."""
        from orchestrator.evals.metrics import blend_composite

        assert blend_composite(
            1.0, 1.0, 1.0, tests_pass=False,
        ) == 0.0
        assert blend_composite(
            1.0, 1.0, 1.0, tests_pass=None,
        ) == 0.0

    def test_plan_only_result_still_clamped_to_unit_interval(self):
        """The bound is a property of the blend, not of the gate."""
        from orchestrator.evals.metrics import blend_composite

        assert blend_composite(
            2.0, 1.0, 1.0, tests_pass=None, plan_only=True,
        ) == 1.0
        assert blend_composite(
            -5.0, 0.0, 0.0, tests_pass=None, plan_only=True,
        ) == 0.0


# ---------------------------------------------------------------------------
# Amendment (reviewer: code-reuse) — the cost primitives (_FALLBACK_PRICE / _rate)
# have a SINGLE home in agents.invoke; metrics re-exports rather than
# re-declares them, so the copy cannot drift. These guards fail loudly if a
# future edit reintroduces a local copy in metrics.py.
# ---------------------------------------------------------------------------

class TestCostPrimitivesSingleHome:
    """metrics._FALLBACK_PRICE / metrics._rate must BE the invoke singletons
    (identity, not mere equality) — a re-declared local copy would fail these.
    """

    def test_fallback_price_is_the_invoke_singleton(self):
        from orchestrator.agents import invoke
        from orchestrator.evals import metrics

        assert metrics._FALLBACK_PRICE is invoke._FALLBACK_PRICE

    def test_rate_accessor_is_the_invoke_singleton(self):
        from orchestrator.agents import invoke
        from orchestrator.evals import metrics

        assert metrics._rate is invoke._rate
