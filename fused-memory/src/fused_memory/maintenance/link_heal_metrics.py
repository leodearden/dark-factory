"""One link-heal sweep run's metric series, and the rolling 7-day summary gate θ reads.

Each sweep run becomes one M1 metric series (``shared.memory_eval_metrics``)
under ``<root>/link-heal/``. ``summary.json`` beside the series is rebuilt from
the series stamped in the last seven days, never from the ledger, so it cannot
disagree with what it summarises. The keys gate θ reads are
:data:`THETA_SUMMARY_KEYS`, and no summary that lacks one, or holds a
non-number under one, is returned. The summary counts the series, gives each
per-run metric's 7-day sum under its id with the ``_7d`` suffix, and gives the
failure rate as the failed sum over the asked sum, 0.0 when nothing was asked.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from shared.memory_eval_metrics import (
    Corpus,
    Metric,
    MetricSeries,
    canonical_json_text,
    eval_dir,
    format_stamp,
    load_metric_series,
    metrics_artifact_path,
    write_metric_series,
)
from shared.safe_io import atomic_write_text

from fused_memory.maintenance.link_heal_report import SweepRunView

logger = logging.getLogger(__name__)

EVAL_ID = 'link-heal'
SUMMARY_FILENAME = 'summary.json'
SUMMARY_FORMAT = 'link-heal-summary/1'
SUMMARY_WINDOW = timedelta(days=7)
SUMMARY_SUFFIX = '_7d'
#: plans/write-triage-link-healing-prd.capability-manifest.md, "Summary key contract (ε → θ)".
THETA_SUMMARY_KEYS: tuple[str, ...] = (
    'runs_7d', 'writing_runs_7d', 'would_escape_7d', 'adjudication_failure_rate_7d',
    'actions_planned_7d', 'skipped_cap_7d',
)


@dataclass(frozen=True)
class _RunMetric:
    """A per-run metric: its id, how one run measures it, and whether a rise in it
    is a regression (an alarmed count) or only reported (a scalar)."""

    metric_id: str
    measure: Callable[[SweepRunView], int]
    alarmed: bool


_RUN_METRICS: tuple[_RunMetric, ...] = (
    _RunMetric('writing_runs', lambda view: int(view.apply is not None), alarmed=False),
    _RunMetric(
        'would_escape', lambda view: sum(len(run.counts.would_escape) for run in view.runs),
        alarmed=True,
    ),
    _RunMetric('actions_planned', lambda view: view.plan.counts.planned, alarmed=False),
    _RunMetric(
        'skipped_cap', lambda view: sum(run.counts.skipped_cap for run in view.runs),
        alarmed=True,
    ),
    _RunMetric(
        'adjudications_asked', lambda view: view.plan.counts.adjudications_asked, alarmed=False,
    ),
    _RunMetric(
        'adjudication_failed', lambda view: view.plan.counts.adjudication_failed, alarmed=True,
    ),
    _RunMetric(
        'adjudication_capped', lambda view: view.plan.counts.adjudication_capped, alarmed=True,
    ),
)

#: The per-run metric ids; each one's 7-day sum is a summary key.
SUMMED_METRICS: tuple[str, ...] = tuple(metric.metric_id for metric in _RUN_METRICS)


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
            Metric(
                metric_id=metric.metric_id,
                kind='count' if metric.alarmed else 'scalar',
                value=metric.measure(view),
                n=1,
                direction='higher_is_worse' if metric.alarmed else None,
            )
            for metric in _RUN_METRICS
        ],
    )


def summary_path(root: Path) -> Path:
    """Where the rolling summary over *root*'s link-heal series lives."""
    return eval_dir(root, EVAL_ID) / SUMMARY_FILENAME


def summarize_series(root: Path, *, now: datetime) -> dict[str, Any]:
    """The summary of the series under *root* stamped in ``[now - 7 days, now]``.

    Every key is present whatever the window holds. A series that does not parse
    raises, as :func:`load_metric_series` does, and so does a summary that breaks
    :func:`check_summary_contract`.
    """
    since, until = format_stamp(now - SUMMARY_WINDOW), format_stamp(now)
    series = [load_metric_series(path) for path in _window_paths(root, since, until)]
    sums = {metric_id: _sum(series, metric_id) for metric_id in SUMMED_METRICS}
    asked = sums['adjudications_asked']
    summary = {
        'format': SUMMARY_FORMAT,
        'eval_id': EVAL_ID,
        'window': {'since': since, 'until': until},
        f'runs{SUMMARY_SUFFIX}': len(series),
        **{f'{metric_id}{SUMMARY_SUFFIX}': total for metric_id, total in sums.items()},
        f'adjudication_failure_rate{SUMMARY_SUFFIX}': (
            sums['adjudication_failed'] / asked if asked else 0.0
        ),
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
    total = 0
    for run in series:
        found = [metric for metric in run.metrics if metric.metric_id == metric_id]
        if not found:
            raise ValueError(f'{EVAL_ID} series {run.run_stamp} has no metric {metric_id!r}')
        total += int(found[0].value)
    return total


@dataclass(frozen=True)
class SweepMetrics:
    """Where one run's series and the summary went, and why they were not
    written, when they were not."""

    metrics_path: Path
    summary_path: Path
    unwritten: str | None = None

    def as_detail(self) -> dict[str, str]:
        detail = {'metrics_json': str(self.metrics_path), 'summary_json': str(self.summary_path)}
        if self.unwritten is not None:
            detail['metrics_unwritten'] = self.unwritten
        return detail


@dataclass(frozen=True)
class SweepMetricsTarget:
    """The metrics root a sweep writes under, and the home project its series names."""

    root: Path
    project_id: str

    def write(self, view: SweepRunView, now: datetime) -> SweepMetrics:
        """Write *view*'s series stamped *now*, then rebuild the summary.

        A series that cannot be built raises. A failure to write, or to re-read an
        earlier artifact, is logged and returned as ``unwritten``.
        """
        stamp = format_stamp(now)
        series = sweep_metric_series(view, project_id=self.project_id, stamp=stamp)
        emitted = SweepMetrics(
            metrics_path=metrics_artifact_path(self.root, EVAL_ID, stamp),
            summary_path=summary_path(self.root),
        )
        try:
            write_metric_series(series, self.root)
            summary = summarize_series(self.root, now=now)
            atomic_write_text(emitted.summary_path, canonical_json_text(summary), mkdir=True)
        except (OSError, ValueError) as failure:
            logger.error(
                'link-heal sweep %s: cannot write its metrics: %s', view.plan.run_id[:8], failure,
            )
            return replace(emitted, unwritten=f'{type(failure).__name__}: {failure}')
        return emitted
