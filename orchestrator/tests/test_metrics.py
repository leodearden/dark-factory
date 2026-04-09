"""Tests for evals/metrics.py — composite scoring, false-green guard, null-work guard."""

from __future__ import annotations

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
        base = dict(
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
        base = dict(
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
