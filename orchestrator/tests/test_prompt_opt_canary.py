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

import sqlite3
from pathlib import Path

import pytest

from orchestrator.evals.prompt_opt.canary import (
    Row,
    WindowMetrics,
    compute_window_metrics,
    load_window_rows,
)
from orchestrator.run_store import _SCHEMA


def _make_runs_db(path: Path, rows: list[dict]) -> None:
    """Create a synthetic runs.db — the EXACT ``runs`` + ``task_results``
    schema from :data:`orchestrator.run_store._SCHEMA` (imported, not
    hand-copied, so this fixture can never drift from the real schema) —
    and insert *rows* into ``task_results``.

    Each row dict requires ``task_id``, ``project_id``, ``outcome``, and
    ``completed_at``. Optional (default ``0``/``0.0``): ``run_id``
    (default ``'run-test'``), ``cost_usd``, ``steward_cost_usd``,
    ``review_cycles``, ``verify_attempts``.
    """
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(_SCHEMA)
        seen_runs: set[str] = set()
        for row in rows:
            run_id = row.get('run_id', 'run-test')
            if run_id not in seen_runs:
                conn.execute(
                    'INSERT INTO runs (run_id, project_id, started_at) VALUES (?, ?, ?)',
                    (run_id, row['project_id'], '2026-01-01T00:00:00+00:00'),
                )
                seen_runs.add(run_id)
            conn.execute(
                """
                INSERT INTO task_results
                    (run_id, task_id, project_id, outcome, cost_usd,
                     steward_cost_usd, review_cycles, verify_attempts, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    row['task_id'],
                    row['project_id'],
                    row['outcome'],
                    row.get('cost_usd', 0.0),
                    row.get('steward_cost_usd', 0.0),
                    row.get('review_cycles', 0),
                    row.get('verify_attempts', 0),
                    row['completed_at'],
                ),
            )
        conn.commit()
    finally:
        conn.close()


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

    def test_zero_done_rows_no_zero_division_error(self) -> None:
        """A window with rows but NO 'done' rows must not raise
        ZeroDivisionError: cost_per_done_task is incomparable (None) and the
        done-averaged metrics fall back to 0.0 rather than dividing by
        n_done == 0."""
        rows = [
            {
                'outcome': 'failed', 'cost_usd': 1.0, 'steward_cost_usd': 0.5,
                'review_cycles': 3, 'verify_attempts': 2,
            },
            {
                'outcome': 'requeued', 'cost_usd': 2.0, 'steward_cost_usd': 0.0,
                'review_cycles': 1, 'verify_attempts': 1,
            },
        ]
        result = compute_window_metrics(rows)
        assert result.n_rows == 2
        assert result.n_done == 0
        assert result.cost_per_done_task is None
        assert result.mean_review_cycles == 0.0
        assert result.mean_verify_attempts == 0.0

    def test_empty_window_returns_zeroed_metrics(self) -> None:
        """An empty window (rows == []) must not raise ZeroDivisionError on
        requeue_rate (n_rows == 0)."""
        result = compute_window_metrics([])
        assert result.n_rows == 0
        assert result.n_done == 0
        assert result.requeue_rate == 0.0
        assert result.cost_per_done_task is None

    def test_all_requeued_window_requeue_rate_is_one(self) -> None:
        rows = [
            {
                'outcome': 'requeued', 'cost_usd': 1.0, 'steward_cost_usd': 0.0,
                'review_cycles': 0, 'verify_attempts': 0,
            },
            {
                'outcome': 'requeued', 'cost_usd': 2.0, 'steward_cost_usd': 0.0,
                'review_cycles': 0, 'verify_attempts': 0,
            },
        ]
        result = compute_window_metrics(rows)
        assert result.requeue_rate == pytest.approx(1.0)


class TestLoadWindowRows:
    def test_filters_by_completed_at_range_and_project_id(self, tmp_path: Path) -> None:
        """Half-open ``[start_iso, end_iso)`` range, AND project_id match.

        Five synthetic rows: one before the window (excluded), one exactly
        AT the start boundary (included — the range is start-inclusive),
        one inside (included), one exactly AT the end boundary (excluded —
        the range is end-exclusive), and one inside the time range but for
        a different project (excluded).
        """
        db_path = tmp_path / 'runs.db'
        _make_runs_db(db_path, [
            {
                'task_id': 't-before', 'project_id': 'dark_factory', 'outcome': 'done',
                'cost_usd': 1.0, 'steward_cost_usd': 0.1, 'review_cycles': 1, 'verify_attempts': 1,
                'completed_at': '2026-06-01T00:00:00+00:00',
            },
            {
                'task_id': 't-at-start', 'project_id': 'dark_factory', 'outcome': 'done',
                'cost_usd': 2.0, 'steward_cost_usd': 0.2, 'review_cycles': 2, 'verify_attempts': 3,
                'completed_at': '2026-06-10T00:00:00+00:00',
            },
            {
                'task_id': 't-inside', 'project_id': 'dark_factory', 'outcome': 'requeued',
                'cost_usd': 3.0, 'steward_cost_usd': 0.3, 'review_cycles': 4, 'verify_attempts': 5,
                'completed_at': '2026-06-20T00:00:00+00:00',
            },
            {
                'task_id': 't-at-end', 'project_id': 'dark_factory', 'outcome': 'done',
                'cost_usd': 4.0, 'steward_cost_usd': 0.4, 'review_cycles': 6, 'verify_attempts': 7,
                'completed_at': '2026-06-25T00:00:00+00:00',
            },
            {
                'task_id': 't-wrong-project', 'project_id': 'other_project', 'outcome': 'done',
                'cost_usd': 5.0, 'steward_cost_usd': 0.5, 'review_cycles': 8, 'verify_attempts': 9,
                'completed_at': '2026-06-15T00:00:00+00:00',
            },
        ])

        rows = load_window_rows(
            db_path, '2026-06-10T00:00:00+00:00', '2026-06-25T00:00:00+00:00', 'dark_factory',
        )

        costs = {row.cost_usd for row in rows}
        assert costs == {2.0, 3.0}

    def test_returned_rows_carry_compute_window_metrics_columns(self, tmp_path: Path) -> None:
        """load_window_rows's Row must plug directly into compute_window_metrics
        (outcome, cost_usd, steward_cost_usd, review_cycles, verify_attempts)."""
        db_path = tmp_path / 'runs.db'
        _make_runs_db(db_path, [
            {
                'task_id': 't-1', 'project_id': 'dark_factory', 'outcome': 'requeued',
                'cost_usd': 1.5, 'steward_cost_usd': 0.25, 'review_cycles': 2, 'verify_attempts': 3,
                'completed_at': '2026-06-15T00:00:00+00:00',
            },
        ])

        rows = load_window_rows(
            db_path, '2026-06-10T00:00:00+00:00', '2026-06-25T00:00:00+00:00', 'dark_factory',
        )

        assert len(rows) == 1
        row = rows[0]
        assert isinstance(row, Row)
        assert row.outcome == 'requeued'
        assert row.cost_usd == pytest.approx(1.5)
        assert row.steward_cost_usd == pytest.approx(0.25)
        assert row.review_cycles == 2
        assert row.verify_attempts == 3

        # And it plugs straight into compute_window_metrics unchanged.
        metrics = compute_window_metrics(rows)
        assert metrics.n_rows == 1
        assert metrics.requeue_rate == pytest.approx(1.0)
