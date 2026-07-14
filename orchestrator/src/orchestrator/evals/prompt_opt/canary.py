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

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

__all__ = ['WindowMetrics', 'compute_window_metrics']


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
    """
    rows = list(rows)
    n_rows = len(rows)
    n_requeued = sum(1 for r in rows if _field(r, 'outcome') == 'requeued')
    requeue_rate = n_requeued / n_rows

    done_rows = [r for r in rows if _field(r, 'outcome') == 'done']
    n_done = len(done_rows)
    total_cost = sum(_field(r, 'cost_usd') + _field(r, 'steward_cost_usd') for r in rows)
    cost_per_done_task = total_cost / n_done
    mean_review_cycles = sum(_field(r, 'review_cycles') for r in done_rows) / n_done
    mean_verify_attempts = sum(_field(r, 'verify_attempts') for r in done_rows) / n_done

    return WindowMetrics(
        n_rows=n_rows,
        n_done=n_done,
        cost_per_done_task=cost_per_done_task,
        requeue_rate=requeue_rate,
        mean_review_cycles=mean_review_cycles,
        mean_verify_attempts=mean_verify_attempts,
    )
