"""Structured reporting for Elo judge results."""

from __future__ import annotations

import json
import logging
import statistics
from collections import defaultdict
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
from .metrics import _rate, blend_composite

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


def _price_role_entry(model: str, prices: dict[str, Any] | None) -> dict[str, Any]:
    """One role's price cell for *model*: its ``input/output_per_1m`` rates, or
    the EXPLICIT ``{'source': 'unpriced'}`` marker for an unlisted model — never a
    fabricated default (loud-over-silent). Shared by :func:`build_price_table`
    (individual-config keys) and :func:`build_pairwise_price_table` (the μ
    end-to-end combined ``arch+impl`` keys) so both build cells identically.
    """
    entry = prices.get(model) if prices else None
    if entry is None:
        return {'source': 'unpriced'}
    return {
        'input_per_1m': _rate(entry, 'input_per_1m'),
        'output_per_1m': _rate(entry, 'output_per_1m'),
    }


def build_price_table(
    configs: list[Any], prices: dict[str, Any] | None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Build the C4 per-config price table ``{config_name: {role: entry}}``.

    Each config's ``model`` is looked up in *prices* (a ``config.prices``-shaped
    map; ``PriceEntry`` objects and plain ``{'input_per_1m','output_per_1m'}``
    dicts are both accepted via :func:`_rate`). A listed model yields
    ``{role: {'input_per_1m', 'output_per_1m'}}``; an UNLISTED model yields the
    explicit ``{role: {'source': 'unpriced'}}`` marker — never a fabricated
    default (loud-over-silent). Pure over ``(configs, prices)``; the caller
    (μ / CLI) seeds *prices* from ``default_price_table()`` / ``config.prices``.
    Config keys are inserted in sorted order so the table is byte-deterministic.
    """
    table: dict[str, dict[str, dict[str, Any]]] = {}
    for config in sorted(configs, key=lambda c: c.name):
        table[config.name] = {config.role: _price_role_entry(config.model, prices)}
    return table


def build_pairwise_price_table(
    pairs: list[Any], prices: dict[str, Any] | None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Combined-name price table for the μ end-to-end stages (matrix / confirm).

    Each ``(architect_cfg, implementer_cfg)`` pair keys ONE entry under the
    combined ``f'{arch.name}+{impl.name}'`` config name — byte-identical to the
    ``config_name`` :func:`run_end_to_end` stamps on that pair's ``EvalResult`` —
    carrying BOTH roles' per-model rates. Keying by the combined name keeps the
    rendered price-table section's ``config`` column ALIGNED with an end-to-end
    composite report's combined rows; individual-config :func:`build_price_table`
    keys never lined up with those rows (reviewer: correctness). Pure over
    ``(pairs, prices)``; keys inserted in sorted order for byte-determinism.
    """
    table: dict[str, dict[str, dict[str, Any]]] = {}
    for arch_cfg, impl_cfg in sorted(pairs, key=lambda p: f'{p[0].name}+{p[1].name}'):
        table[f'{arch_cfg.name}+{impl_cfg.name}'] = {
            arch_cfg.role: _price_role_entry(arch_cfg.model, prices),
            impl_cfg.role: _price_role_entry(impl_cfg.model, prices),
        }
    return table


def _summarize_cost_source(sources: set[str]) -> str:
    """Collapse a config's per-trial cost sources to ONE row-level label.

    A config's ``cost_usd`` is a mean across all its trials; when those trials
    carry more than one distinct ``cost_source`` (e.g. ``price_table`` on one
    fixture, ``unpriced_proxy`` on another) reporting just the first trial's
    source would mislabel the cross-source mean, so surface ``'mixed'`` instead
    (reviewer: robustness). Exactly one distinct source → that source; none (a
    config with no trials, unreachable here) → ``'cli'`` for back-compat.
    """
    if not sources:
        return 'cli'
    if len(sources) == 1:
        return next(iter(sources))
    return 'mixed'


def build_composite_report(
    results: list[EvalResult],
    *,
    price_table: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the C4 per-config composite report over the UNION of configs.

    This retires the April all-tasks-INTERSECTION collapse (see
    :func:`compute_aggregate_ratings`): a config that ran in *any* fixture gets a
    row, so a config present in only one fixture is no longer silently dropped.

    Aggregation (Open-Q3 / decision 10) is per-fixture NORMALIZED composite →
    mean with a small-sample CI:

    1. For each fixture, ``best_cost`` / ``best_latency`` = the minimum POSITIVE
       cost / latency across the PASSING (``tests_pass`` truthy) trials of all
       configs on that fixture. A cheap-but-WRONG run (which itself hard-gates
       to 0) must not set the efficiency floor and deflate the correct configs'
       cost/latency scores (reviewer: correctness); a fixture with NO passing
       trial falls back to all trials so its baseline stays defined.
    2. For each trial, the efficiency-adjusted composite is
       ``blend_composite(quality=composite_score,
       cost_score=_ratio_score(cost, best_cost),
       latency_score=_ratio_score(latency, best_latency),
       tests_pass=tests_pass)`` — so the cheapest+fastest+top-quality run of a
       fixture scores highest, and a failing trial hard-gates to 0.
    3. For each config, its per-trial composite / cost / latency / quality are
       pooled across ALL its trials (every fixture) and reduced to a mean +
       95% CI via :func:`mean_ci95`.

    ``latency_secs`` is ``workflow_duration_ms / 1000`` (the workflow's own
    working time, not wall-clock, so eval/scheduler overhead is not charged to
    the config). ``recovery_score`` / ``plan_quality`` / ``role_under_test`` are
    passthroughs of any trial's metrics (η/θ surfaces). ``cost_source`` (P5) is
    the config's single distinct per-trial source, or ``'mixed'`` when its
    trials span more than one — since ``cost_usd`` is a cross-trial mean,
    labelling it with just the first trial's source would mislabel a blended
    figure (reviewer: robustness).
    Rows are emitted sorted by ``config`` so the surface is byte-deterministic;
    *price_table* (μ/CLI seeds it from :func:`build_price_table`) is echoed
    verbatim, defaulting to ``{}`` when omitted.
    """
    # 1. Per-fixture best (min positive) cost and latency, taken from the
    #    PASSING trials only so a cheap-but-WRONG run cannot set the efficiency
    #    floor and deflate the correct configs (reviewer: correctness). The
    #    ``*_all`` maps mirror the same minima over ALL trials and seed the
    #    fallback for a fixture where nothing passed.
    best_cost: dict[str, float] = {}
    best_latency: dict[str, float] = {}
    best_cost_all: dict[str, float] = {}
    best_latency_all: dict[str, float] = {}
    for r in results:
        fixture = r.task_id
        cost = float(r.metrics.get('cost_usd', 0.0) or 0.0)
        latency = float(r.metrics.get('workflow_duration_ms', 0) or 0) / 1000.0
        passed = bool(r.metrics.get('tests_pass'))
        if cost > 0:
            best_cost_all[fixture] = min(best_cost_all.get(fixture, cost), cost)
            if passed:
                best_cost[fixture] = min(best_cost.get(fixture, cost), cost)
        if latency > 0:
            best_latency_all[fixture] = min(best_latency_all.get(fixture, latency), latency)
            if passed:
                best_latency[fixture] = min(best_latency.get(fixture, latency), latency)
    # Fixtures with no passing trial fall back to the all-trials baseline so the
    # normalization denominator stays defined (every such trial hard-gates to 0
    # regardless, so this only keeps the baseline non-empty).
    for fixture, v in best_cost_all.items():
        best_cost.setdefault(fixture, v)
    for fixture, v in best_latency_all.items():
        best_latency.setdefault(fixture, v)

    # 2/3. Accumulate per-trial normalized composites, pooled per config.
    def _acc() -> dict[str, Any]:
        return {
            'composite': [], 'cost': [], 'latency': [], 'quality': [],
            'passes': 0, 'trials': 0, 'fixtures': set(),
            'judge_invocations': 0, 'judge_cost_usd': 0.0,
            'cost_sources': set(),
            'first_metrics': None,
        }

    by_config: dict[str, dict[str, Any]] = defaultdict(_acc)
    for r in results:
        fixture = r.task_id
        m = r.metrics
        quality = float(m.get('composite_score', 0.0) or 0.0)
        cost = float(m.get('cost_usd', 0.0) or 0.0)
        latency = float(m.get('workflow_duration_ms', 0) or 0) / 1000.0
        tests_pass = m.get('tests_pass')
        composite_trial = blend_composite(
            quality,
            _ratio_score(cost, best_cost.get(fixture, 0.0)),
            _ratio_score(latency, best_latency.get(fixture, 0.0)),
            tests_pass=tests_pass,
        )
        acc = by_config[r.config_name]
        acc['composite'].append(composite_trial)
        acc['cost'].append(cost)
        acc['latency'].append(latency)
        acc['quality'].append(quality)
        acc['passes'] += 1 if tests_pass else 0
        acc['trials'] += 1
        acc['fixtures'].add(fixture)
        acc['judge_invocations'] += int(m.get('judge_invocations', 0) or 0)
        acc['judge_cost_usd'] += float(m.get('judge_cost_usd', 0.0) or 0.0)
        acc['cost_sources'].add(str(m.get('cost_source', 'cli')))
        if acc['first_metrics'] is None:
            acc['first_metrics'] = m

    rows: list[dict[str, Any]] = []
    for cfg in sorted(by_config):
        acc = by_config[cfg]
        fm = acc['first_metrics'] or {}
        composite_ci = mean_ci95(acc['composite'])
        cost_ci = mean_ci95(acc['cost'])
        latency_ci = mean_ci95(acc['latency'])
        quality_ci = mean_ci95(acc['quality'])
        trials = acc['trials']
        rows.append({
            'config': cfg,
            'role_under_test': fm.get('role_under_test'),
            'composite': composite_ci['mean'],
            'quality': quality_ci['mean'],
            'tests_pass_rate': (acc['passes'] / trials) if trials else 0.0,
            'cost_usd': cost_ci['mean'],
            'cost_source': _summarize_cost_source(acc['cost_sources']),
            'latency_secs': latency_ci['mean'],
            'trials': trials,
            'fixtures': len(acc['fixtures']),
            'ci95': {
                'composite': composite_ci,
                'cost': cost_ci,
                'latency': latency_ci,
            },
            'judge': {
                'invocations': acc['judge_invocations'],
                'cost_usd': acc['judge_cost_usd'],
            },
            'recovery_score': fm.get('recovery_score'),
            'plan_quality': fm.get('plan_quality'),
        })

    return {
        'generated_at': datetime.now(UTC).isoformat(),
        'aggregation': 'per_fixture_normalized_mean_ci',
        'price_table': price_table if price_table is not None else {},
        'configs': rows,
    }


# ===== μ methodology-driver substrate (PRD plans/eval-framework-revival-prd.md
# task μ) =====
#
# select_survivors / build_methodology_report / format_methodology_report model
# the AUTOMATIC ofat→select_survivors→matrix→confirm→methodology-report flow.
# That single-command auto-driver is a planned follow-up and is NOT yet wired:
# the shipped CLI surface (eval-ofat / eval-matrix / eval-confirm) instead runs
# each stage independently, with the operator picking survivors manually via
# --arch/--impl between stages. These three PURE functions are the tested
# substrate that follow-up will consume; today only test_eval_driver_report.py
# exercises them. Kept here (not deferred) so the auto-driver lands as a thin
# wiring change over an already-verified core.
def select_survivors(
    composite_report: dict[str, Any],
    *,
    top_k: int,
    roles: list[str],
) -> dict[str, list[str]]:
    """The top-K config names per ``role_under_test`` — the μ OFAT survivor gate.

    Groups the λ :func:`build_composite_report` ``'configs'`` rows by
    ``role_under_test``, ranks each requested role's group by DESCENDING
    ``composite`` mean (ties broken deterministically by ascending ``config``
    name, so the surface is byte-stable), and returns the top-``top_k`` config
    names per role. A role with fewer than ``top_k`` rows returns all it has; a
    requested role with no rows returns ``[]``. Pure over the report dict — the
    OFAT screen feeds these survivors into :func:`run_matrix_stage`.
    """
    by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in composite_report.get('configs', []):
        by_role[row.get('role_under_test')].append(row)

    survivors: dict[str, list[str]] = {}
    for role in roles:
        ranked = sorted(
            by_role.get(role, []),
            key=lambda r: (-float(r.get('composite', 0.0) or 0.0), str(r.get('config', ''))),
        )
        survivors[role] = [str(r['config']) for r in ranked[:top_k]]
    return survivors


_METHODOLOGY_STAGES = ('ofat', 'matrix', 'confirm')


def build_methodology_report(
    ofat_results: list[EvalResult],
    matrix_results: list[EvalResult],
    confirm_results: list[EvalResult],
    *,
    price_table: dict[str, Any] | None = None,
    survivors: dict[str, list[str]] | None = None,
    winner: str | None = None,
) -> dict[str, Any]:
    """The μ methodology report: three λ composite sub-reports + survivors + winner.

    Nests a full :func:`build_composite_report` under each stage key
    (``'ofat'`` / ``'matrix'`` / ``'confirm'``), echoing *price_table* into each
    (so every stage is a self-contained C4 composite report), and carries the
    OFAT *survivors*, the confirmed *winner*, and the echoed *price_table* at the
    top level. A thin, deterministic aggregator — it invents NO new per-config
    schema, reusing λ's substrate wholesale (decision 7). ``price_table`` /
    ``survivors`` default to ``{}`` and ``winner`` to ``None`` when omitted.
    """
    pt = price_table if price_table is not None else {}
    return {
        'generated_at': datetime.now(UTC).isoformat(),
        'price_table': pt,
        'survivors': survivors if survivors is not None else {},
        'winner': winner,
        'stages': {
            'ofat': build_composite_report(ofat_results, price_table=pt),
            'matrix': build_composite_report(matrix_results, price_table=pt),
            'confirm': build_composite_report(confirm_results, price_table=pt),
        },
    }


def format_methodology_report(report: dict[str, Any]) -> str:
    """Render :func:`build_methodology_report` output byte-stably.

    A deterministic survivors/winner header, then each stage
    (``ofat`` → ``matrix`` → ``confirm``) rendered via
    :func:`format_composite_table` (which carries that stage's per-config
    composite / cost / latency / CI95 columns and its price-table section). No
    wall-clock is rendered and every row/section is sorted with fixed float
    precision, so the same report always renders byte-identically.
    """
    survivors = report.get('survivors') or {}
    lines = [
        'methodology report:',
        f'winner: {report.get("winner") if report.get("winner") is not None else "-"}',
        'survivors:',
    ]
    for role in sorted(survivors):
        names = ', '.join(str(n) for n in (survivors[role] or []))
        lines.append(f'  {role}: {names}')

    stages = report.get('stages') or {}
    for stage in _METHODOLOGY_STAGES:
        sub = stages.get(stage)
        lines.append('')
        lines.append(f'== {stage} stage ==')
        if sub is None:
            lines.append('(no results)')
            continue
        lines.append(format_composite_table(sub))
    return '\n'.join(lines)


def compute_aggregate_ratings(state: JudgeState) -> dict[str, float]:
    """Mean Elo across tasks over the UNION of configs.

    Retires the April all-tasks-INTERSECTION collapse (task 2477 λ / decision
    10): a config that ran in ANY task gets an aggregate rating, averaged over
    only the tasks in which it actually appears — rather than being dropped
    unless it appeared in every task (which collapsed the leaderboard to the
    handful of ever-present configs). ``build_report`` / ``compute_tiers`` /
    ``format_markdown`` consume the returned dict unchanged. Configs are inserted
    in sorted order so the mapping is deterministic.
    """
    if not state.per_task:
        return {}

    union = {cfg for pool in state.per_task.values() for cfg in pool.ratings}

    agg: dict[str, float] = {}
    for cfg in sorted(union):
        ratings = [
            pool.ratings[cfg]
            for pool in state.per_task.values()
            if cfg in pool.ratings
        ]
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


# ---------------------------------------------------------------------------
# Composite report renderer (eval-revival λ / C4)
#
# The deterministic table surface for :func:`build_composite_report`, mirroring
# the width-computed ljust idiom of format_recovery_table / format_plan_quality_
# table above. Emits a per-config table (composite / quality / cost / cost_source
# / latency / CI95 / trials / fixtures) followed by a DISTINCT price-table
# section (config → role → input/output per-1M). Byte-stable: rows and sections
# are sorted, floats fixed-precision, and no wall-clock is rendered.
# ---------------------------------------------------------------------------

_COMPOSITE_COLUMNS = (
    'config', 'composite', 'quality', 'cost_usd', 'cost_source',
    'latency_secs', 'ci95_composite', 'trials', 'fixtures',
)
_PRICE_TABLE_COLUMNS = ('config', 'role', 'input_per_1m', 'output_per_1m')


def _ci_cell(ci: dict[str, Any] | None) -> str:
    """Render a ``mean_ci95`` sub-dict as a ``[lo, hi]`` interval (4 dp)."""
    if not ci:
        return '-'
    return f'[{float(ci.get("lo", 0.0)):.4f}, {float(ci.get("hi", 0.0)):.4f}]'


def _format_price_table_section(price_table: dict[str, Any]) -> list[str]:
    """Render the ``{config: {role: entry}}`` price table as a deterministic
    block. A listed entry shows its ``input_per_1m`` / ``output_per_1m`` (4 dp);
    an UNPRICED entry (``{'source': 'unpriced'}``) shows the explicit marker in
    both cells — never a blank or fabricated price. Configs and roles are sorted.
    """
    rendered: list[dict[str, str]] = []
    for cfg in sorted(price_table):
        roles = price_table[cfg] or {}
        for role in sorted(roles):
            entry = roles[role] or {}
            if 'input_per_1m' in entry:
                inp = f'{float(entry["input_per_1m"]):.4f}'
                outp = f'{float(entry["output_per_1m"]):.4f}'
            else:
                inp = outp = str(entry.get('source', 'unpriced'))
            rendered.append({
                'config': str(cfg),
                'role': str(role),
                'input_per_1m': inp,
                'output_per_1m': outp,
            })
    widths = {
        col: (
            max(len(col), *(len(rr[col]) for rr in rendered))
            if rendered else len(col)
        )
        for col in _PRICE_TABLE_COLUMNS
    }

    def _fmt(cells: dict[str, str]) -> str:
        return '  '.join(cells[col].ljust(widths[col]) for col in _PRICE_TABLE_COLUMNS)

    lines = ['price table:']
    lines.append(_fmt({col: col for col in _PRICE_TABLE_COLUMNS}))
    lines.append('  '.join('-' * widths[col] for col in _PRICE_TABLE_COLUMNS))
    lines.extend(_fmt(rr) for rr in rendered)
    return lines


def format_composite_table(report: dict[str, Any]) -> str:
    """Render :func:`build_composite_report` output as a deterministic table.

    A per-config table carries composite / quality / cost_usd / cost_source /
    latency_secs, a ``[lo, hi]`` CI95 rendering of the composite, and trial /
    fixture counts, followed by a distinct ``price table`` section. The same
    report always renders byte-identically (rows sorted by config, price-table
    sections sorted, fixed float precision, no wall-clock/dict-order dependence).
    An unpriced config's ``cost_source`` shows the explicit marker (e.g.
    ``unpriced_proxy``), never a blank.
    """
    configs = sorted(report.get('configs', []), key=lambda r: str(r.get('config', '')))

    rendered = [
        {
            'config': str(r.get('config', '')),
            'composite': f'{float(r.get("composite", 0.0)):.4f}',
            'quality': f'{float(r.get("quality", 0.0)):.4f}',
            'cost_usd': f'{float(r.get("cost_usd", 0.0)):.4f}',
            'cost_source': str(r.get('cost_source', '')),
            'latency_secs': f'{float(r.get("latency_secs", 0.0)):.2f}',
            'ci95_composite': _ci_cell((r.get('ci95') or {}).get('composite')),
            'trials': str(r.get('trials', 0)),
            'fixtures': str(r.get('fixtures', 0)),
        }
        for r in configs
    ]
    widths = {
        col: (
            max(len(col), *(len(rr[col]) for rr in rendered))
            if rendered else len(col)
        )
        for col in _COMPOSITE_COLUMNS
    }

    def _fmt(cells: dict[str, str]) -> str:
        return '  '.join(cells[col].ljust(widths[col]) for col in _COMPOSITE_COLUMNS)

    lines = ['composite report:']
    lines.append(_fmt({col: col for col in _COMPOSITE_COLUMNS}))
    lines.append('  '.join('-' * widths[col] for col in _COMPOSITE_COLUMNS))
    lines.extend(_fmt(rr) for rr in rendered)

    # Distinct price-table section.
    lines.append('')
    lines.extend(_format_price_table_section(report.get('price_table') or {}))
    return '\n'.join(lines)
