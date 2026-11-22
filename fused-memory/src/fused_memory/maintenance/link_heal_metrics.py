"""One link-heal sweep run's metric series, and the rolling 7-day summary gate θ reads.

Each sweep run becomes one M1 metric series (``shared.memory_eval_metrics``)
under ``<root>/link-heal/``. ``summary.json`` beside the series is rebuilt from
the series stamped in the last seven days, never from the ledger, so it cannot
disagree with what it summarises. The summary's contract with gate θ is
:data:`THETA_SUMMARY_KEYS` (θ requires ``runs_7d``, ``would_escape_7d`` and
``adjudication_failure_rate_7d``), and no summary that lacks one, or holds a
non-number under one, is returned. The summary counts the series and the UTC
days they were stamped on, gives each per-run metric's 7-day sum under its id
with the ``_7d`` suffix, and gives the failure rate as the failed sum over the
asked sum, 0.0 when nothing was asked. It also carries ``expires_at``,
:data:`SUMMARY_TTL` after its window ends, and
``scripts/check_write_triage_readiness_gate.py`` fails closed on a report past
its ``expires_at``, so a summary no later run replaced (a refused or crashed
run, or a stopped timer) stops being evidence for gate θ.

A per-run metric added later needs no reset of the window: a series that
predates it adds nothing to its sum. Only the metrics in
:data:`CONTRACT_METRICS` must be carried by every series in the window.

Every metric is a reported scalar: the sweep's escapes are filed by the sweep
itself, and the gate on the week is θ's, so no M2 rule alarms on the series.
Each run still judges its series into ``limits-current.json`` beside it, as the
memory-eval runner does for its own evals, because every eval directory with
runs on the shared root is read as one whose evaluator completes.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from shared.memory_eval_limits import (
    LimitsConfig,
    evaluate_series,
    limits_artifact_path,
    load_limits_artifact,
    write_limits_artifact,
)
from shared.memory_eval_metrics import (
    Corpus,
    Metric,
    MetricSeries,
    canonical_json_text,
    eval_dir,
    format_stamp,
    load_metric_series,
    load_series_window,
    metrics_artifact_path,
    stamp_datetime,
    write_metric_series,
)
from shared.safe_io import atomic_write_text

from fused_memory.maintenance.link_heal_report import SweepRunView

logger = logging.getLogger(__name__)

EVAL_ID = 'link-heal'
SUMMARY_FILENAME = 'summary.json'
SUMMARY_FORMAT = 'link-heal-summary/1'
SUMMARY_WINDOW = timedelta(days=7)
#: One nightly interval (scripts/fused-memory-link-heal.timer) plus jitter and run time; a missed night expires it.
SUMMARY_TTL = timedelta(hours=26)
SUMMARY_SUFFIX = '_7d'
#: plans/write-triage-link-healing-prd.capability-manifest.md, "Summary key contract (ε → θ)".
THETA_SUMMARY_KEYS: tuple[str, ...] = (
    'runs_7d', 'writing_runs_7d', 'would_escape_7d', 'adjudication_failure_rate_7d',
    'actions_planned_7d', 'skipped_cap_7d',
)


#: One nightly run a day; with only scalars in the series nothing spends the budget.
LIMITS = LimitsConfig(runs_per_quarter=90, min_samples=1, baseline_window=7)


@dataclass(frozen=True)
class _RunMetric:
    """A per-run metric: its id and how one run measures it."""

    metric_id: str
    measure: Callable[[SweepRunView], int]


_FAILED, _ASKED = 'adjudication_failed', 'adjudications_asked'

_RUN_METRICS: tuple[_RunMetric, ...] = (
    _RunMetric('writing_runs', lambda view: int(view.apply is not None)),
    _RunMetric(
        'would_escape', lambda view: sum(len(run.counts.would_escape) for run in view.runs),
    ),
    _RunMetric('actions_planned', lambda view: view.plan.counts.planned),
    _RunMetric('skipped_cap', lambda view: sum(run.counts.skipped_cap for run in view.runs)),
    _RunMetric(_ASKED, lambda view: view.plan.counts.adjudications_asked),
    _RunMetric(_FAILED, lambda view: view.plan.counts.adjudication_failed),
    _RunMetric('adjudication_capped', lambda view: view.plan.counts.adjudication_capped),
)

#: The per-run metric ids; each one's 7-day sum is a summary key.
SUMMED_METRICS: tuple[str, ...] = tuple(metric.metric_id for metric in _RUN_METRICS)

#: The per-run metrics :data:`THETA_SUMMARY_KEYS` rest on: those whose sum is a
#: contract key, and the failure rate's two terms.
CONTRACT_METRICS: frozenset[str] = frozenset({
    *(
        metric_id for metric_id in SUMMED_METRICS
        if f'{metric_id}{SUMMARY_SUFFIX}' in THETA_SUMMARY_KEYS
    ),
    _FAILED,
    _ASKED,
})


def sweep_metric_series(view: SweepRunView, *, project_id: str, stamp: str) -> MetricSeries:
    """*view*'s run as one series of the home project *project_id*, stamped *stamp*.

    The run is the unit, so every metric has ``n=1``; the corpus counts the
    examined links per project, since one sweep spans several.
    """
    return MetricSeries(
        schema_version=1,
        eval_id=EVAL_ID,
        run_stamp=stamp,
        corpus=Corpus(
            project_id=project_id,
            counts=dict(Counter(record.project_id for record in view.records)),
        ),
        metrics=[
            Metric(metric_id=metric.metric_id, kind='scalar', value=metric.measure(view), n=1)
            for metric in _RUN_METRICS
        ],
    )


def judge_series(root: Path, series: MetricSeries) -> Path:
    """Judge *series* against the earlier series under *root* and rewrite the
    eval's ``limits-current.json``, resuming from the one an earlier run left, as
    the memory-eval runner's ``evaluate_eval`` does; return its path. An earlier
    artifact that does not parse raises."""
    path = limits_artifact_path(root, EVAL_ID)
    prior = load_limits_artifact(path) if path.is_file() else None
    window = load_series_window(
        root, EVAL_ID, limit=LIMITS.baseline_window, before_stamp=series.run_stamp,
    )
    result = evaluate_series(
        series,
        window,
        LIMITS,
        frozenset(prior.grandfather_set) if prior is not None else None,
        prior.snapshotted_metric_ids if prior is not None else None,
    )
    return write_limits_artifact(result, path)


def summary_path(root: Path) -> Path:
    """Where the rolling summary over *root*'s link-heal series lives."""
    return eval_dir(root, EVAL_ID) / SUMMARY_FILENAME


def summarize_series(root: Path, *, now: datetime) -> dict[str, Any]:
    """The summary of the series under *root* stamped in ``[now - 7 days, now]``.

    Every key is present whatever the window holds. A series that does not parse
    raises, as :func:`load_metric_series` does, and so do a series that lacks one
    of :data:`CONTRACT_METRICS` and a summary that breaks
    :func:`check_summary_contract`.
    """
    since, until = format_stamp(now - SUMMARY_WINDOW), format_stamp(now)
    series = [load_metric_series(path) for path in _window_paths(root, since, until)]
    sums = {metric_id: _sum(series, metric_id) for metric_id in SUMMED_METRICS}
    asked = sums[_ASKED]
    summary = {
        'format': SUMMARY_FORMAT,
        'eval_id': EVAL_ID,
        'window': {'since': since, 'until': until},
        'expires_at': (now + SUMMARY_TTL).astimezone(UTC).isoformat(timespec='seconds'),
        f'runs{SUMMARY_SUFFIX}': len(series),
        f'run_days{SUMMARY_SUFFIX}': len({stamp_datetime(run.run_stamp).date() for run in series}),
        **{f'{metric_id}{SUMMARY_SUFFIX}': total for metric_id, total in sums.items()},
        f'adjudication_failure_rate{SUMMARY_SUFFIX}': sums[_FAILED] / asked if asked else 0.0,
    }
    check_summary_contract(summary)
    return summary


def check_summary_contract(summary: Mapping[str, object]) -> None:
    """Raise :class:`ValueError` naming the first gate θ key *summary* lacks, or
    holds a non-number under. A bool is not a number, as θ's ``--require`` holds."""
    for key in THETA_SUMMARY_KEYS:
        if key not in summary:
            raise ValueError(f'{EVAL_ID} summary is missing gate θ key {key!r}')
        value = summary[key]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f'{EVAL_ID} summary gate θ key {key!r} is not a number: {value!r}')


def _window_paths(root: Path, since: str, until: str) -> list[Path]:
    """The metrics artifacts stamped in ``[since, until]``, ordered by filename as
    ``shared.memory_eval_metrics.load_series_window`` orders them."""
    directory = eval_dir(root, EVAL_ID)
    if not directory.is_dir():
        return []
    low = metrics_artifact_path(root, EVAL_ID, since).name
    high = metrics_artifact_path(root, EVAL_ID, until).name
    return [path for path in sorted(directory.glob('metrics-*.json')) if low <= path.name <= high]


def _sum(series: Sequence[MetricSeries], metric_id: str) -> int:
    """*metric_id*'s total over *series*. A series without it raises when the
    metric is one of :data:`CONTRACT_METRICS`, and otherwise predates it and adds
    nothing."""
    total = 0
    for run in series:
        found = [metric for metric in run.metrics if metric.metric_id == metric_id]
        if found:
            total += int(found[0].value)
        elif metric_id in CONTRACT_METRICS:
            raise ValueError(f'{EVAL_ID} series {run.run_stamp} has no metric {metric_id!r}')
    return total


@dataclass(frozen=True)
class SweepMetrics:
    """Where one run's series and the summary went, and why they were not
    written, when they were not."""

    metrics_path: Path
    summary_path: Path
    unwritten: str | None = None

    def as_detail(self) -> dict[str, str | None]:
        return {
            'metrics_json': str(self.metrics_path),
            'summary_json': str(self.summary_path),
            'metrics_unwritten': self.unwritten,
        }


@dataclass(frozen=True)
class SweepMetricsTarget:
    """The metrics root a sweep writes under, and the home project its series names."""

    root: Path
    project_id: str

    def write(self, view: SweepRunView, now: datetime) -> SweepMetrics:
        """Write *view*'s series stamped *now*, then rebuild the summary.

        A series that cannot be built raises. A failure to write, or to re-read an
        earlier artifact, is logged and returned as ``unwritten``, and the summary an
        earlier run left is removed, because it does not count tonight. A summary no
        run replaces at all is not removed; it stops being evidence at its own
        ``expires_at``.
        """
        stamp = format_stamp(now)
        series = sweep_metric_series(view, project_id=self.project_id, stamp=stamp)
        emitted = SweepMetrics(
            metrics_path=metrics_artifact_path(self.root, EVAL_ID, stamp),
            summary_path=summary_path(self.root),
        )
        try:
            write_metric_series(series, self.root)
            judge_series(self.root, series)
            summary = summarize_series(self.root, now=now)
            atomic_write_text(emitted.summary_path, canonical_json_text(summary), mkdir=True)
        except (OSError, ValueError) as failure:
            logger.error(
                'link-heal sweep %s: cannot write its metrics: %s', view.plan.run_id[:8], failure,
            )
            unwritten = f'{type(failure).__name__}: {failure}'
            return replace(emitted, unwritten=unwritten + _retire(emitted.summary_path))
        return emitted


def _retire(stale: Path) -> str:
    """Remove the summary *stale*; what to add to the run's ``unwritten`` when it
    could not be removed, else nothing."""
    try:
        stale.unlink()
    except (FileNotFoundError, NotADirectoryError):
        return ''
    except OSError as failure:
        logger.error('link-heal sweep: cannot remove the stale summary %s: %s', stale, failure)
        return f'; the stale summary stays ({type(failure).__name__}: {failure})'
    return ''
