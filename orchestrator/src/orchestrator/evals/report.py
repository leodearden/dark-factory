"""Structured reporting for Elo judge results."""

from __future__ import annotations

import json
import logging
import statistics
from datetime import UTC, datetime
from itertools import combinations
from math import sqrt
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .elo import (
    INDISTINGUISHABLE_THRESHOLD,
    JudgeState,
    TaskPool,
    _pair_key,
)

if TYPE_CHECKING:
    from .runner import EvalResult

logger = logging.getLogger(__name__)

REPORT_FILE = Path(__file__).parent / 'judge_report.json'


# ---------------------------------------------------------------------------
# Composite-report statistics substrate (task 2477 λ) — stdlib only.
#
# Neither scipy nor numpy is a declared dependency, so small-sample confidence
# intervals use stdlib ``statistics`` plus this df→t-critical lookup. Trials are
# few (decision 10: ≥3/cell), so the Student-t interval — not a normal z — is
# the correct estimator; the ``sufficient`` flag surfaces when n<3.
# ---------------------------------------------------------------------------

# Two-sided 95% (0.975 one-tail) Student-t critical values, indexed by degrees
# of freedom (n-1). df>30 (or df<1) falls back to the normal approximation 1.96.
_T_CRITICAL_0975: dict[int, float] = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}
_T_CRITICAL_NORMAL = 1.96


def _t_critical(df: int) -> float:
    """Two-sided 95% Student-t critical value for *df* degrees of freedom,
    falling back to the normal approximation (1.96) for df>30 or df<1.
    """
    if df < 1:
        return _T_CRITICAL_NORMAL
    return _T_CRITICAL_0975.get(df, _T_CRITICAL_NORMAL)


def mean_ci95(values: list[float]) -> dict[str, Any]:
    """Mean and two-sided 95% Student-t confidence interval of *values*.

    Returns ``{mean, lo, hi, n, sufficient}``. A CI is computed for n>=2 (via
    the df→t-critical table); for n<2 the interval collapses to the point
    estimate (``lo == hi == mean``). ``sufficient`` is ``n>=3`` (decision 10:
    ≥3 trials/cell), surfacing when the sample is too small to trust. An empty
    input yields ``mean==0.0, n==0``. Pure, stdlib-only (statistics.stdev is the
    ddof=1 sample stdev).
    """
    n = len(values)
    if n == 0:
        return {'mean': 0.0, 'lo': 0.0, 'hi': 0.0, 'n': 0, 'sufficient': False}
    mean = statistics.mean(values)
    if n < 2:
        return {'mean': mean, 'lo': mean, 'hi': mean, 'n': n, 'sufficient': False}
    stdev = statistics.stdev(values)  # ddof=1 (sample stdev)
    half_width = _t_critical(n - 1) * stdev / sqrt(n)
    return {
        'mean': mean,
        'lo': mean - half_width,
        'hi': mean + half_width,
        'n': n,
        'sufficient': n >= 3,
    }


def _ratio_score(value: float, best: float) -> float:
    """Normalize *value* against the *best* (min) observed, as ``best/value``
    clamped to ``[0, 1]``.

    ``1.0`` == this run IS the best (cheapest / fastest). A larger *value* (more
    cost / higher latency) scores lower. When either operand is non-positive —
    a single-config fixture, a zero denominator, or an otherwise undefined
    normalization — returns the neutral ``1.0`` so such fixtures never
    spuriously penalize.
    """
    if best <= 0 or value <= 0:
        return 1.0
    return min(max(best / value, 0.0), 1.0)


def compute_aggregate_ratings(state: JudgeState) -> dict[str, float]:
    """Mean Elo across tasks, only for configs present in ALL tasks."""
    if not state.per_task:
        return {}

    config_sets = [set(pool.ratings.keys()) for pool in state.per_task.values()]
    common = set.intersection(*config_sets) if config_sets else set()

    agg: dict[str, float] = {}
    for cfg in common:
        ratings = [state.per_task[t].ratings[cfg] for t in state.per_task]
        agg[cfg] = round(sum(ratings) / len(ratings), 1)
    return agg


def compute_tiers(ratings: dict[str, float]) -> list[list[str]]:
    """Group configs within ``INDISTINGUISHABLE_THRESHOLD`` Elo into tiers.

    Sorted descending by rating.  Consecutive configs whose rating is within
    the threshold of the tier's highest-rated config are grouped together.
    """
    if not ratings:
        return []

    sorted_cfgs = sorted(ratings.keys(), key=lambda c: -ratings[c])
    tiers: list[list[str]] = [[sorted_cfgs[0]]]

    for cfg in sorted_cfgs[1:]:
        tier_top_rating = ratings[tiers[-1][0]]
        if tier_top_rating - ratings[cfg] < INDISTINGUISHABLE_THRESHOLD:
            tiers[-1].append(cfg)
        else:
            tiers.append([cfg])

    return tiers


def _confidence_label(pool: TaskPool, config: str) -> str:
    """Label rating confidence based on number of matches played."""
    matches_played = sum(
        1 for m in pool.matches
        if m['config_a'] == config or m['config_b'] == config
    )
    if matches_played >= 4:
        return 'solid'
    elif matches_played >= 2:
        return 'tentative'
    return 'preliminary'


def build_report(state: JudgeState) -> dict[str, Any]:
    """Build the full structured report."""
    report: dict[str, Any] = {
        'generated_at': datetime.now(UTC).isoformat(),
        'tasks': {},
        'aggregate': {},
    }

    for task_id, pool in state.per_task.items():
        sorted_ratings = sorted(pool.ratings.items(), key=lambda x: -x[1])
        tiers = compute_tiers(pool.ratings)

        # Find indistinguishable pairs (maxed out and still close)
        indistinguishable: list[str] = []
        for a, b in combinations(pool.ratings.keys(), 2):
            key = _pair_key(a, b)
            count = pool.pair_counts.get(key, 0)
            gap = abs(pool.ratings[a] - pool.ratings[b])
            if count >= 3 and gap < INDISTINGUISHABLE_THRESHOLD:
                indistinguishable.append(f'{a} \u2248 {b}')

        report['tasks'][task_id] = {
            'leaderboard': [
                {
                    'config': cfg,
                    'elo': rating,
                    'confidence': _confidence_label(pool, cfg),
                }
                for cfg, rating in sorted_ratings
            ],
            'tiers': [
                {
                    'rank': i + 1,
                    'configs': tier,
                    'elo_range': f'{min(pool.ratings[c] for c in tier):.0f}'
                                 f'-{max(pool.ratings[c] for c in tier):.0f}',
                }
                for i, tier in enumerate(tiers)
            ],
            'indistinguishable_pairs': indistinguishable,
            'total_matches': len(pool.matches),
            'matches': pool.matches,
        }

    # Aggregate across tasks
    agg_ratings = compute_aggregate_ratings(state)
    agg_tiers = compute_tiers(agg_ratings)
    report['aggregate'] = {
        'leaderboard': [
            {'config': c, 'mean_elo': r}
            for c, r in sorted(agg_ratings.items(), key=lambda x: -x[1])
        ],
        'tiers': [
            {'rank': i + 1, 'configs': tier}
            for i, tier in enumerate(agg_tiers)
        ],
        'tasks_included': list(state.per_task.keys()),
    }

    return report


def save_report(report: dict[str, Any], path: Path = REPORT_FILE) -> Path:
    """Write JSON report to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    return path


def format_markdown(report: dict[str, Any]) -> str:
    """Format report as markdown for console output."""
    lines: list[str] = []
    lines.append('# Eval Judge Report')
    lines.append(f'Generated: {report["generated_at"]}')
    lines.append('')

    # Aggregate leaderboard
    agg = report.get('aggregate', {})
    if agg.get('leaderboard'):
        lines.append('## Aggregate Leaderboard')
        lines.append(f'(across {len(agg["tasks_included"])} tasks)')
        lines.append('')
        for i, entry in enumerate(agg['leaderboard'], 1):
            lines.append(f'  {i}. {entry["config"]:30s}  {entry["mean_elo"]:.0f}')
        lines.append('')

        if agg.get('tiers'):
            lines.append('### Tiers (within 50 Elo = statistically indistinguishable)')
            for tier in agg['tiers']:
                lines.append(f'  Tier {tier["rank"]}: {", ".join(tier["configs"])}')
            lines.append('')

    # Per-task leaderboards
    for task_id, task_data in report.get('tasks', {}).items():
        lines.append(f'## {task_id}')
        lines.append(f'({task_data["total_matches"]} matches)')
        lines.append('')
        for entry in task_data['leaderboard']:
            marker = '*' if entry['confidence'] == 'preliminary' else ' '
            lines.append(
                f'  {marker} {entry["config"]:30s}  '
                f'{entry["elo"]:.0f}  ({entry["confidence"]})'
            )
        if task_data.get('indistinguishable_pairs'):
            lines.append(
                f'  Too close to call: '
                f'{"; ".join(task_data["indistinguishable_pairs"])}'
            )
        lines.append('')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# recovery_score surface (eval-revival η)
#
# An ADDITIVE, interim per-(task_id, config_name) column distinct from the Elo
# leaderboard above. It surfaces the ``recovery_score`` that
# ``metrics.collect_metrics`` writes into ``EvalResult.metrics`` — a populated
# float for adversarial fixtures, the ``None`` null sentinel otherwise. It does
# NOT touch the Elo ``build_report`` / ``format_markdown`` schema: the full C4
# per-config composite+cost+latency+recovery report row is owned by Phase-3
# task λ, which η does not depend on; this is the distinct-column surface μ/λ
# consume in the interim.
# ---------------------------------------------------------------------------

_RECOVERY_COLUMNS = ('task_id', 'config_name', 'adversarial', 'recovery_score')


def build_recovery_report(results: list[EvalResult]) -> dict[str, Any]:
    """Build the recovery-score column from a list of :class:`EvalResult`.

    Each row carries ``task_id`` / ``config_name``, the ``recovery_score``
    pulled from ``EvalResult.metrics`` (a populated float for adversarial
    fixtures, ``None`` — the C4 ``recovery_score | null`` sentinel — otherwise),
    and an ``adversarial`` bool.

    The ``adversarial`` flag is taken from the explicit ``metrics['adversarial']``
    boolean that ``collect_metrics`` stamps from the task record — NOT inferred
    from ``recovery_score is not None``. That distinction matters: recovery
    scoring can fail on a genuinely adversarial fixture, in which case
    ``collect_metrics``' guard leaves ``recovery_score=None``; inferring the flag
    from the score would then silently mislabel that run ``adversarial: false``
    and hide the scoring failure. Keying on the explicit flag renders such a run
    as ``adversarial: true`` with a null score instead. For results persisted
    BEFORE the flag existed (no ``adversarial`` key in ``metrics``), we fall
    back to the old ``recovery_score is not None`` inference so old reports
    render unchanged. Rows are sorted by ``(task_id, config_name)`` so the
    surface is deterministic.
    """
    rows: list[dict[str, Any]] = []
    for result in results:
        recovery_score = result.metrics.get('recovery_score')
        explicit_adversarial = result.metrics.get('adversarial')
        adversarial = (
            bool(explicit_adversarial)
            if explicit_adversarial is not None
            else recovery_score is not None
        )
        rows.append({
            'task_id': result.task_id,
            'config_name': result.config_name,
            'adversarial': adversarial,
            'recovery_score': recovery_score,
        })
    rows.sort(key=lambda r: (r['task_id'], r['config_name']))
    return {'rows': rows}


def format_recovery_table(report: dict[str, Any]) -> str:
    """Render :func:`build_recovery_report` output as a deterministic table.

    A ``recovery_score`` column shows the populated float (4 dp) for adversarial
    rows and ``-`` (the null sentinel) for ordinary rows. The same report always
    renders byte-identically (no wall-clock or dict-order dependence).
    """
    rows = report.get('rows', [])

    def _score_cell(value: float | None) -> str:
        return '-' if value is None else f'{value:.4f}'

    rendered = [
        {
            'task_id': str(r['task_id']),
            'config_name': str(r['config_name']),
            'adversarial': 'yes' if r['adversarial'] else 'no',
            'recovery_score': _score_cell(r['recovery_score']),
        }
        for r in rows
    ]
    widths = {
        col: (
            max(len(col), *(len(rr[col]) for rr in rendered))
            if rendered else len(col)
        )
        for col in _RECOVERY_COLUMNS
    }

    def _fmt(cells: dict[str, str]) -> str:
        return '  '.join(cells[col].ljust(widths[col]) for col in _RECOVERY_COLUMNS)

    lines = ['recovery_score report:']
    lines.append(_fmt({col: col for col in _RECOVERY_COLUMNS}))
    lines.append('  '.join('-' * widths[col] for col in _RECOVERY_COLUMNS))
    lines.extend(_fmt(rr) for rr in rendered)
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# plan_quality surface (eval-revival θ)
#
# An ADDITIVE, interim per-(task_id, config_name, role_under_test) column
# distinct from the Elo leaderboard above — the architect-eval analogue of the
# recovery_score surface. It surfaces the ``plan_quality`` that
# ``run_architect_eval`` writes into ``EvalResult.metrics`` — a populated float
# for architect runs (role_under_test=='architect'), the ``None`` null sentinel
# otherwise (ordinary implementer runs never invoke the plan judge). It does NOT
# touch the Elo ``build_report`` / ``format_markdown`` schema: the full C4
# per-config composite+cost+latency+plan_quality report row is owned by Phase-3
# task λ, which θ does not depend on; this is the distinct-column surface μ/λ
# consume in the interim, mirroring η's recovery surface.
# ---------------------------------------------------------------------------

_PLAN_QUALITY_COLUMNS = ('task_id', 'config_name', 'role_under_test', 'plan_quality')


def build_plan_quality_report(results: list[EvalResult]) -> dict[str, Any]:
    """Build the plan-quality column from a list of :class:`EvalResult` (θ).

    Each row carries ``task_id`` / ``config_name`` / ``role_under_test`` and the
    ``plan_quality`` pulled from ``EvalResult.metrics`` — a populated float for
    architect runs (``role_under_test=='architect'``), ``None`` (the C4
    ``plan_quality | null`` sentinel) otherwise. Both are read directly from the
    persisted metrics dict; a result predating the θ fields (no ``plan_quality``
    / ``role_under_test`` key) reads back as ``None`` for each, so old reports
    render unchanged. Rows are sorted by ``(task_id, config_name)`` so the
    surface is deterministic.
    """
    rows: list[dict[str, Any]] = []
    for result in results:
        rows.append({
            'task_id': result.task_id,
            'config_name': result.config_name,
            'role_under_test': result.metrics.get('role_under_test'),
            'plan_quality': result.metrics.get('plan_quality'),
        })
    rows.sort(key=lambda r: (r['task_id'], r['config_name']))
    return {'rows': rows}


def format_plan_quality_table(report: dict[str, Any]) -> str:
    """Render :func:`build_plan_quality_report` output as a deterministic table.

    A ``plan_quality`` column shows the populated float (4 dp) for architect rows
    and ``-`` (the null sentinel) for non-architect rows; ``role_under_test``
    renders ``-`` when ``None``. The same report always renders byte-identically
    (no wall-clock or dict-order dependence).
    """
    rows = report.get('rows', [])

    def _score_cell(value: float | None) -> str:
        return '-' if value is None else f'{value:.4f}'

    def _role_cell(value: str | None) -> str:
        return '-' if value is None else str(value)

    rendered = [
        {
            'task_id': str(r['task_id']),
            'config_name': str(r['config_name']),
            'role_under_test': _role_cell(r['role_under_test']),
            'plan_quality': _score_cell(r['plan_quality']),
        }
        for r in rows
    ]
    widths = {
        col: (
            max(len(col), *(len(rr[col]) for rr in rendered))
            if rendered else len(col)
        )
        for col in _PLAN_QUALITY_COLUMNS
    }

    def _fmt(cells: dict[str, str]) -> str:
        return '  '.join(cells[col].ljust(widths[col]) for col in _PLAN_QUALITY_COLUMNS)

    lines = ['plan_quality report:']
    lines.append(_fmt({col: col for col in _PLAN_QUALITY_COLUMNS}))
    lines.append('  '.join('-' * widths[col] for col in _PLAN_QUALITY_COLUMNS))
    lines.extend(_fmt(rr) for rr in rendered)
    return '\n'.join(lines)
