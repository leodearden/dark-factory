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

__all__ = [
    'CanaryThresholds',
    'CanaryVerdict',
    'MetricComparison',
    'Row',
    'WindowMetrics',
    'compare_windows',
    'compute_window_metrics',
    'load_window_rows',
]


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


@dataclass(frozen=True)
class CanaryThresholds:
    """Per-metric regression thresholds: a relative tolerance (fraction
    above baseline before flagging a regression) plus an absolute floor
    (used instead of the relative ratio when a metric's baseline == 0, to
    avoid dividing by zero on a tiny/zero baseline).

    The defaults below are PRD §9 calibration STARTING points — to be
    tuned against real ``runs.db`` baseline variance once collected. They
    are NOT asserted as numeric guarantees anywhere in this module's test
    suite, which always passes explicit thresholds instead.
    """

    cost_per_done_task_rel_tol: float = 0.2
    cost_per_done_task_abs_floor: float = 1.0
    requeue_rate_rel_tol: float = 0.2
    requeue_rate_abs_floor: float = 0.05
    mean_review_cycles_rel_tol: float = 0.2
    mean_review_cycles_abs_floor: float = 1.0
    mean_verify_attempts_rel_tol: float = 0.2
    mean_verify_attempts_abs_floor: float = 1.0


@dataclass(frozen=True)
class MetricComparison:
    """One metric's baseline-vs-post comparison.

    ``baseline``/``post`` pass through whatever :class:`WindowMetrics` held
    (``cost_per_done_task`` may be ``None`` when a window has no done
    rows). When either is ``None`` the metric is incomparable:
    ``delta_abs``/``delta_rel``/``threshold`` are ``None`` and
    ``regressed`` is ``False`` — absence of evidence is not evidence of
    regression. Otherwise ``threshold`` is the absolute cutoff ``post`` was
    compared against (``baseline * (1 + rel_tol)``, or the metric's
    ``abs_floor`` when ``baseline == 0``), so ``regressed`` is always
    exactly ``post > threshold``.
    """

    metric: str
    baseline: float | None
    post: float | None
    delta_abs: float | None
    delta_rel: float | None
    threshold: float | None
    regressed: bool


@dataclass(frozen=True)
class CanaryVerdict:
    """The overall pass/regress verdict from :func:`compare_windows`.

    ``verdict`` is ``'regress'`` when ``regressed_metrics`` is non-empty,
    else ``'pass'``. (A third value, ``'insufficient_data'``, is added by
    the insufficient-data guard.) ``baseline_n``/``post_n`` carry each
    window's row count through for display/logging.
    """

    verdict: str
    comparisons: list[MetricComparison]
    regressed_metrics: list[str]
    baseline_n: int
    post_n: int


def _compare_metric(
    metric: str, baseline: float | None, post: float | None, rel_tol: float, abs_floor: float,
) -> MetricComparison:
    """Compare one metric's baseline/post values under the "higher =
    regression" rule, applying the absolute-floor branch when
    ``baseline == 0`` and treating either side being ``None`` as
    incomparable (never regressed)."""
    if baseline is None or post is None:
        return MetricComparison(
            metric=metric, baseline=baseline, post=post,
            delta_abs=None, delta_rel=None, threshold=None, regressed=False,
        )

    delta_abs = post - baseline
    if baseline == 0:
        threshold = abs_floor
        delta_rel = None
    else:
        threshold = baseline * (1 + rel_tol)
        delta_rel = delta_abs / baseline

    return MetricComparison(
        metric=metric, baseline=baseline, post=post,
        delta_abs=delta_abs, delta_rel=delta_rel, threshold=threshold,
        regressed=post > threshold,
    )


def compare_windows(
    baseline: WindowMetrics, post: WindowMetrics, thresholds: CanaryThresholds,
) -> CanaryVerdict:
    """Compare *baseline* against *post* across all four T8/D-7 metrics and
    emit a :class:`CanaryVerdict`."""
    comparisons = [
        _compare_metric(
            'cost_per_done_task', baseline.cost_per_done_task, post.cost_per_done_task,
            thresholds.cost_per_done_task_rel_tol, thresholds.cost_per_done_task_abs_floor,
        ),
        _compare_metric(
            'requeue_rate', baseline.requeue_rate, post.requeue_rate,
            thresholds.requeue_rate_rel_tol, thresholds.requeue_rate_abs_floor,
        ),
        _compare_metric(
            'mean_review_cycles', baseline.mean_review_cycles, post.mean_review_cycles,
            thresholds.mean_review_cycles_rel_tol, thresholds.mean_review_cycles_abs_floor,
        ),
        _compare_metric(
            'mean_verify_attempts', baseline.mean_verify_attempts, post.mean_verify_attempts,
            thresholds.mean_verify_attempts_rel_tol, thresholds.mean_verify_attempts_abs_floor,
        ),
    ]
    regressed_metrics = [c.metric for c in comparisons if c.regressed]

    return CanaryVerdict(
        verdict='regress' if regressed_metrics else 'pass',
        comparisons=comparisons,
        regressed_metrics=regressed_metrics,
        baseline_n=baseline.n_rows,
        post_n=post.n_rows,
    )
