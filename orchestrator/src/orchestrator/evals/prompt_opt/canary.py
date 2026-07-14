"""T8 prompt-variant canary: runs.db metric-comparison + deploy verdict.

See plans/tier1-prompt-optimization-prd.md T8 / D-7: guards the MAS
net-negative failure mode (MAS-PromptBench 2606.23664) — a role-locally-
better prompt that shifts cost downstream. Reads
`data/orchestrator/runs.db` (`orchestrator.run_store`'s `task_results`
table) and compares four pipeline-level metrics over a post-deploy window
vs a rolling pre-deploy baseline window, emitting a pass/regress verdict
against documented thresholds.

Rollback is NOT re-implemented here: this module only emits the verdict an
operator's ship decision is based on. The operator unpins the artifact via
`shared.prompt_artifact.PromptArtifactStore.unpin` per
plans/tier1-prompt-optimization-runbook.md — that is the sole rollback
lever (D-4).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ['Row', 'WindowMetrics', 'compute_window_metrics', 'load_window_rows']


def _field(row: Any, name: str) -> Any:
    """Read *name* from *row* — a plain ``dict`` (unit-test rows) or a
    :class:`Row` (the real rows :func:`load_window_rows` returns).

    Keeps `compute_window_metrics` decoupled from exactly which row
    representation a caller uses, as long as it carries the five named
    fields (outcome, cost_usd, steward_cost_usd, review_cycles,
    verify_attempts).
    """
    if isinstance(row, dict):
        return row[name]
    return getattr(row, name)


@dataclass(frozen=True)
class WindowMetrics:
    """The four T8/D-7 pipeline-level metrics aggregated over one time window.

    All four are oriented "higher = regression" (see `compare_windows`):
    ``cost_per_done_task`` pairs total pipeline spend against useful output
    (a churny prompt inflates spend per done task even when it is locally
    "better" — the MAS net-negative signal D-7 guards against);
    ``requeue_rate`` is churn; ``mean_review_cycles``/``mean_verify_attempts``
    are the same loop-histogram metrics
    `dashboard.data.performance.get_loop_histograms` tracks, averaged over
    ``outcome == 'done'`` rows only (that module's done-filter convention).
    """

    n_rows: int
    n_done: int
    cost_per_done_task: float | None
    requeue_rate: float
    mean_review_cycles: float
    mean_verify_attempts: float


def compute_window_metrics(rows: Iterable[Any]) -> WindowMetrics:
    """Aggregate *rows* (task_results-shaped) into a :class:`WindowMetrics`.

    ``cost_per_done_task`` sums ``cost_usd + steward_cost_usd`` over EVERY
    row in the window (not just done rows) divided by the count of done
    rows — this is what captures the downstream cost-shift D-7 guards
    against: churn among the non-done rows still shows up as more total
    spend per unit of useful (done) output. ``requeue_rate`` is the
    fraction of ALL rows with ``outcome == 'requeued'``.
    ``mean_review_cycles``/``mean_verify_attempts`` average only over
    ``outcome == 'done'`` rows, matching
    ``dashboard.data.performance.get_loop_histograms``'s done-filter
    convention.

    Division-safe: an empty window (``n_rows == 0``) yields
    ``requeue_rate == 0.0``, and a window with no done rows
    (``n_done == 0``) yields ``cost_per_done_task is None`` and
    ``mean_review_cycles == mean_verify_attempts == 0.0`` — no
    ``ZeroDivisionError`` in either case.
    """
    rows = list(rows)
    n_rows = len(rows)
    n_requeued = sum(1 for r in rows if _field(r, 'outcome') == 'requeued')
    requeue_rate = n_requeued / n_rows if n_rows else 0.0

    done_rows = [r for r in rows if _field(r, 'outcome') == 'done']
    n_done = len(done_rows)
    if n_done:
        total_cost = sum(_field(r, 'cost_usd') + _field(r, 'steward_cost_usd') for r in rows)
        cost_per_done_task = total_cost / n_done
        mean_review_cycles = sum(_field(r, 'review_cycles') for r in done_rows) / n_done
        mean_verify_attempts = sum(_field(r, 'verify_attempts') for r in done_rows) / n_done
    else:
        cost_per_done_task = None
        mean_review_cycles = 0.0
        mean_verify_attempts = 0.0

    return WindowMetrics(
        n_rows=n_rows,
        n_done=n_done,
        cost_per_done_task=cost_per_done_task,
        requeue_rate=requeue_rate,
        mean_review_cycles=mean_review_cycles,
        mean_verify_attempts=mean_verify_attempts,
    )


@dataclass(frozen=True)
class Row:
    """One ``task_results`` row, narrowed to the five columns
    :func:`compute_window_metrics` needs."""

    outcome: str
    cost_usd: float
    steward_cost_usd: float
    review_cycles: int
    verify_attempts: int


def load_window_rows(
    db_path: Path, start_iso: str, end_iso: str, project_id: str,
) -> list[Row]:
    """Read ``task_results`` rows for *project_id* in the half-open window
    ``[start_iso, end_iso)`` from the ``runs.db`` at *db_path*.

    ISO-8601 ``completed_at`` values compare correctly as plain strings
    (lexicographic == chronological order for zero-padded ISO-8601), the
    same half-open-range convention
    ``dashboard.data.performance`` uses. Read-only sync ``sqlite3`` — no
    durability pragmas needed, mirroring
    ``reviewer_trial.__main__._fetch_titles`` /
    ``reviewer_trial.mining.mine_fn_candidates``.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.execute(
            """
            SELECT outcome, cost_usd, steward_cost_usd, review_cycles, verify_attempts
            FROM task_results
            WHERE completed_at >= ? AND completed_at < ? AND project_id = ?
            """,
            (start_iso, end_iso, project_id),
        )
        return [
            Row(
                outcome=outcome,
                cost_usd=cost_usd,
                steward_cost_usd=steward_cost_usd,
                review_cycles=review_cycles,
                verify_attempts=verify_attempts,
            )
            for outcome, cost_usd, steward_cost_usd, review_cycles, verify_attempts in cursor.fetchall()
        ]
    finally:
        conn.close()
