"""Tests for orchestrator.evals.prompt_opt.canary — the T8 prompt-variant canary.

See plans/tier1-prompt-optimization-prd.md T8 / D-7: guards the MAS
net-negative failure mode (MAS-PromptBench 2606.23664) — a role-locally-
better prompt that shifts cost downstream — by comparing four pipeline-level
metrics (cost-per-done-task, requeue-rate, mean review_cycles, mean
verify_attempts) over a post-deploy window vs a rolling pre-deploy baseline
window read from `data/orchestrator/runs.db`, and emitting a pass/regress
verdict against documented thresholds.
"""

from __future__ import annotations

import pytest

from orchestrator.evals.prompt_opt.canary import WindowMetrics, compute_window_metrics


class TestComputeWindowMetrics:
    def test_returns_window_metrics(self) -> None:
        rows = [
            {
                'outcome': 'done', 'cost_usd': 1.0, 'steward_cost_usd': 0.5,
                'review_cycles': 1, 'verify_attempts': 2,
            },
        ]
        result = compute_window_metrics(rows)
        assert isinstance(result, WindowMetrics)

    def test_cost_per_done_task_simple_example(self) -> None:
        """Plan's own illustrative numbers: 2 done rows cost 1.0+2.0, steward
        0.5+0.5 => cost_per_done_task == 2.0."""
        rows = [
            {
                'outcome': 'done', 'cost_usd': 1.0, 'steward_cost_usd': 0.5,
                'review_cycles': 0, 'verify_attempts': 0,
            },
            {
                'outcome': 'done', 'cost_usd': 2.0, 'steward_cost_usd': 0.5,
                'review_cycles': 0, 'verify_attempts': 0,
            },
        ]
        result = compute_window_metrics(rows)
        assert result.cost_per_done_task == pytest.approx(2.0)

    def test_hand_computed_metrics_with_mixed_outcomes(self) -> None:
        """A window with 2 done rows + 1 requeued row exercises every field:

        - n_rows = 3, n_done = 2
        - cost_per_done_task sums cost_usd+steward_cost_usd over ALL 3 rows
          (1.0+0.5 + 2.0+0.5 + 5.0+0.0 = 9.0) / n_done (2) = 4.5 — this is
          what makes the metric a downstream-cost-shift signal: the
          requeued row's cost still counts even though it isn't "done".
        - requeue_rate = 1/3 (one requeued row out of three)
        - mean_review_cycles / mean_verify_attempts average ONLY the two
          done rows (2+4)/2 = 3.0 and (1+3)/2 = 2.0 — the requeued row's
          review_cycles=9/verify_attempts=9 must NOT leak into these means.
        """
        rows = [
            {
                'outcome': 'done', 'cost_usd': 1.0, 'steward_cost_usd': 0.5,
                'review_cycles': 2, 'verify_attempts': 1,
            },
            {
                'outcome': 'done', 'cost_usd': 2.0, 'steward_cost_usd': 0.5,
                'review_cycles': 4, 'verify_attempts': 3,
            },
            {
                'outcome': 'requeued', 'cost_usd': 5.0, 'steward_cost_usd': 0.0,
                'review_cycles': 9, 'verify_attempts': 9,
            },
        ]
        result = compute_window_metrics(rows)
        assert result.n_rows == 3
        assert result.n_done == 2
        assert result.cost_per_done_task == pytest.approx(4.5)
        assert result.requeue_rate == pytest.approx(1 / 3)
        assert result.mean_review_cycles == pytest.approx(3.0)
        assert result.mean_verify_attempts == pytest.approx(2.0)
