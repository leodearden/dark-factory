"""The per-config composite report over persisted eval cells, as rows, a ranked
survivor set, a per-stage methodology nest and a deterministic table.

Rationale for the composite report: docs/plan-scoring-and-judge.md, "The composite
report (task 3099)". Rationale for the ranking: docs/plan-scoring-and-judge.md,
"Survivor ranking (`select_survivors`, tasks 3099 and 3379)".
"""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, TypeAlias

from .cell_admission import (
    cost_per_plan,
    has_plan_quality_score,
    is_plan_only,
    is_unmeasurable,
    judged_blind_count,
    judged_without_reference,
    mean_plan_quality,
    plan_quality_score,
    plan_rate,
)
from .metrics import blend_composite
from .plan_predicates import produced_a_plan
from .prices import format_price_table_section
from .stats import mean_ci95
from .table_render import ci_cell, optional_float_cell, render_fixed_table

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .results import EvalResult


def _ratio_score(value: float, best: float) -> float:
    """``best / value`` clamped to ``[0, 1]``: ``1.0`` is the cheapest or fastest run.

    A non-positive operand (a single-config fixture, a zero denominator) gives the
    neutral ``1.0``, so an undefined normalization never penalizes.
    """
    if best <= 0 or value <= 0:
        return 1.0
    return min(max(best / value, 0.0), 1.0)


def _summarize_cost_source(sources: set[str]) -> str:
    """One row-level label for the cost sources of a config's measured trials.

    Exactly one distinct source is that source, more than one is ``'mixed'``, and
    none (every trial unmeasurable, so ``cost_usd`` is ``None``) is ``'cli'``.
    """
    if not sources:
        return 'cli'
    if len(sources) == 1:
        return next(iter(sources))
    return 'mixed'


# ``(fixture, role_group)``: the key of every efficiency baseline map. Module-level
# because a name bound inside a function body is not usable in a type expression.
_BaselineKey: TypeAlias = tuple[str, str]


def build_composite_report(
    results: list[EvalResult],
    *,
    price_table: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The per-config composite report over the UNION of configs in *results*.

    Each trial is blended (:func:`blend_composite`) against the minimum positive
    cost and latency of its ``(fixture, role_group)`` group's passing trials, then
    pooled per config into a mean and 95% CI (:func:`mean_ci95`). A plan-only
    (architect) trial's quality axis is its :func:`plan_quality_score`, a workflow
    trial's is its ``composite_score``. An unmeasurable trial
    (:func:`is_unmeasurable`) leaves every measured pool, but is still counted in
    ``trials``, ``fixtures`` and ``judge``.

    Returns ``generated_at``, ``aggregation``, ``price_table`` (*price_table*
    echoed, ``{}`` when omitted) and ``configs``: one row per config, sorted by
    ``config``. A row's keys are ``config``, ``role_under_test``, ``composite``,
    ``quality``, ``tests_pass_rate``, ``cost_usd``, ``cost_source``,
    ``latency_secs`` (``workflow_duration_ms / 1000``), ``trials``, ``fixtures``,
    ``ci95`` (``composite`` / ``cost`` / ``latency``), ``judge`` (``invocations`` /
    ``cost_usd``), ``recovery_score``, ``plan_quality``,
    ``plan_quality_cap_excluded``, ``plan_quality_no_plan``,
    ``plan_quality_judged_without_reference`` and its ``_unmeasured`` count,
    ``plan_quality_n``, ``plan_rate`` and ``cost_per_plan``.

    A row with no measured trial has ``None`` for ``composite``, ``quality``,
    ``cost_usd``, ``latency_secs`` and every ``ci95`` entry. An all-plan-only
    row's ``tests_pass_rate`` is ``None``. ``plan_quality`` and ``plan_rate`` are
    ``None`` for an empty θ pool, ``cost_per_plan`` when no admitted cell planned,
    and ``plan_quality_judged_without_reference`` when an admitted cell predates
    its marker.
    """
    # 1. Per-group minimum positive cost/latency over the PASSING trials; ``*_all``,
    #    over every measurable trial, is the fallback for a group where none passed.
    best_cost: dict[_BaselineKey, float] = {}
    best_latency: dict[_BaselineKey, float] = {}
    best_cost_all: dict[_BaselineKey, float] = {}
    best_latency_all: dict[_BaselineKey, float] = {}

    def _baseline_key(result: EvalResult) -> _BaselineKey:
        return (
            result.task_id,
            'plan_only' if is_plan_only(result.metrics) else 'workflow',
        )

    for r in results:
        # A cell that measured nothing seeds neither map, not even the fallback.
        if is_unmeasurable(r.metrics):
            continue
        key = _baseline_key(r)
        cost = float(r.metrics.get('cost_usd', 0.0) or 0.0)
        latency = float(r.metrics.get('workflow_duration_ms', 0) or 0) / 1000.0
        # A plan-only trial passes when it was scored AND planned: a no-plan cell is
        # cheap because it failed, so it must not set the floor (task 3302).
        passed = (
            (
                has_plan_quality_score(r.metrics)
                and produced_a_plan(r.metrics)
            ) if is_plan_only(r.metrics)
            else bool(r.metrics.get('tests_pass'))
        )
        if cost > 0:
            best_cost_all[key] = min(best_cost_all.get(key, cost), cost)
            if passed:
                best_cost[key] = min(best_cost.get(key, cost), cost)
        if latency > 0:
            best_latency_all[key] = min(best_latency_all.get(key, latency), latency)
            if passed:
                best_latency[key] = min(best_latency.get(key, latency), latency)
    # A sole no-plan cell in its group earns 1.0 ratios from this fallback, which is
    # why blend_composite(no_plan=True) hard-gates it below (task 3302).
    for key, v in best_cost_all.items():
        best_cost.setdefault(key, v)
    for key, v in best_latency_all.items():
        best_latency.setdefault(key, v)

    # 2/3. Accumulate per-trial normalized composites, pooled per config.
    def _acc() -> dict[str, Any]:
        return {
            'composite': [], 'cost': [], 'latency': [], 'quality': [],
            'passes': 0, 'trials': 0, 'plan_only_trials': 0, 'fixtures': set(),
            'judge_invocations': 0, 'judge_cost_usd': 0.0,
            'cost_sources': set(),
            'first_metrics': None,
            'plan_quality_scores': [],
            'plan_quality_cap_excluded': 0,
            'plan_quality_no_plan': 0,
            'plan_quality_judged_without_reference': 0,
            # Admitted cells with NO marker: they make the count above unknown.
            'plan_quality_judged_without_reference_unmeasured': 0,
            # The cost_per_plan numerator, summed when plan_quality_scores is.
            'plan_cost_usd': 0.0,
        }

    by_config: dict[str, dict[str, Any]] = defaultdict(_acc)
    for r in results:
        fixture = r.task_id
        m = r.metrics
        cost = float(m.get('cost_usd', 0.0) or 0.0)
        latency = float(m.get('workflow_duration_ms', 0) or 0) / 1000.0
        tests_pass = m.get('tests_pass')
        plan_only = is_plan_only(m)
        acc = by_config[r.config_name]
        # The axis that fed the composite: the θ score for a plan-only cell (floored
        # when it produced no plan), compute_composite's score for a workflow one.
        scored_pq = plan_quality_score(m, where=f'{fixture} x {r.config_name}')
        quality = (
            scored_pq if plan_only and scored_pq is not None
            else float(m.get('composite_score', 0.0) or 0.0)
        )
        # Shared by the hard gate and the no-plan count. Scoped by
        # has_plan_quality_score, so a cap-tainted cell is never a content failure.
        no_plan = has_plan_quality_score(m) and not produced_a_plan(m)
        # ONE admission decision for every measured pool. The trial is still counted
        # in ``trials`` and ``plan_quality_cap_excluded``.
        if not is_unmeasurable(m):
            # Each trial normalizes against its OWN (fixture, role_group) floor.
            bkey = _baseline_key(r)
            acc['composite'].append(blend_composite(
                quality,
                _ratio_score(cost, best_cost.get(bkey, 0.0)),
                _ratio_score(latency, best_latency.get(bkey, 0.0)),
                tests_pass=tests_pass,
                plan_only=plan_only,
                no_plan=no_plan,
            ))
            acc['quality'].append(quality)
            acc['cost'].append(cost)
            acc['latency'].append(latency)
            acc['cost_sources'].add(str(m.get('cost_source', 'cli')))
        acc['passes'] += 1 if tests_pass else 0
        acc['plan_only_trials'] += 1 if plan_only else 0
        acc['trials'] += 1
        acc['fixtures'].add(fixture)
        acc['judge_invocations'] += int(m.get('judge_invocations', 0) or 0)
        acc['judge_cost_usd'] += float(m.get('judge_cost_usd', 0.0) or 0.0)
        if acc['first_metrics'] is None:
            acc['first_metrics'] = m
        # The θ pool, the cost_per_plan numerator and the judged-blind count share
        # this admission condition, the one build_plan_quality_report applies.
        if scored_pq is not None:
            acc['plan_quality_scores'].append(scored_pq)
            # A no-plan cell's real spend lands here too: $ per usable plan.
            acc['plan_cost_usd'] += cost
            marker = judged_without_reference(m)
            if marker is None:
                acc['plan_quality_judged_without_reference_unmeasured'] += 1
            elif marker:
                acc['plan_quality_judged_without_reference'] += 1
        # Architect trials only, to agree with build_plan_quality_report's count.
        if m.get('cap_tainted') and plan_only:
            acc['plan_quality_cap_excluded'] += 1
        if no_plan:
            acc['plan_quality_no_plan'] += 1

    rows: list[dict[str, Any]] = []
    for cfg in sorted(by_config):
        acc = by_config[cfg]
        fm = acc['first_metrics'] or {}
        composite_ci = mean_ci95(acc['composite'])
        cost_ci = mean_ci95(acc['cost'])
        latency_ci = mean_ci95(acc['latency'])
        quality_ci = mean_ci95(acc['quality'])
        trials = acc['trials']
        # The pools share one admission decision, so they are empty together, and
        # an empty pool is None: never mean_ci95([])'s confident 0.0.
        measured = bool(acc['composite'])
        # A plan-only config ran no test, so it has no pass rate, not a 0% one.
        all_plan_only = acc['plan_only_trials'] == trials and trials > 0
        # Bound once, so plan_quality_n and plan_rate describe the same pool.
        n_admitted = len(acc['plan_quality_scores'])
        rows.append({
            'config': cfg,
            'role_under_test': fm.get('role_under_test'),
            'composite': composite_ci['mean'] if measured else None,
            'quality': quality_ci['mean'] if measured else None,
            'tests_pass_rate': (
                None if all_plan_only
                else (acc['passes'] / trials) if trials else 0.0
            ),
            'cost_usd': cost_ci['mean'] if measured else None,
            'cost_source': _summarize_cost_source(acc['cost_sources']),
            'latency_secs': latency_ci['mean'] if measured else None,
            'trials': trials,
            'fixtures': len(acc['fixtures']),
            'ci95': {
                'composite': composite_ci if measured else None,
                'cost': cost_ci if measured else None,
                'latency': latency_ci if measured else None,
            },
            'judge': {
                'invocations': acc['judge_invocations'],
                'cost_usd': acc['judge_cost_usd'],
            },
            'recovery_score': fm.get('recovery_score'),
            'plan_quality': mean_plan_quality(acc['plan_quality_scores']),
            'plan_quality_cap_excluded': acc['plan_quality_cap_excluded'],
            'plan_quality_no_plan': acc['plan_quality_no_plan'],
            'plan_quality_judged_without_reference': judged_blind_count(
                acc['plan_quality_judged_without_reference'],
                acc['plan_quality_judged_without_reference_unmeasured'],
            ),
            'plan_quality_judged_without_reference_unmeasured': acc[
                'plan_quality_judged_without_reference_unmeasured'
            ],
            'plan_quality_n': n_admitted,
            'plan_rate': plan_rate(n_admitted, acc['plan_quality_no_plan']),
            'cost_per_plan': cost_per_plan(
                acc['plan_cost_usd'], n_admitted - acc['plan_quality_no_plan'],
            ),
        })

    return {
        'generated_at': datetime.now(UTC).isoformat(),
        'aggregation': 'per_fixture_normalized_mean_ci',
        'price_table': price_table if price_table is not None else {},
        'configs': rows,
    }


# The μ methodology-driver substrate: docs/plan-scoring-and-judge.md, "The composite
# report (task 3099)".
def select_survivors(
    composite_report: dict[str, Any],
    *,
    top_k: int,
    roles: Sequence[str],
) -> dict[str, list[str]]:
    """The top-*top_k* config names per requested ``role_under_test``.

    Groups :func:`build_composite_report`'s ``configs`` rows by role and ranks each
    group by, in order:

    1. a measured ``composite`` before ``None``;
    2. descending ``composite``;
    3. a ``plan_quality`` before ``None``;
    4. descending ``plan_quality``;
    5. a ``plan_rate`` before ``None``;
    6. descending ``plan_rate``;
    7. ascending ``config`` name.

    A role with fewer than *top_k* rows returns all it has, and a requested role
    with no rows returns ``[]``. A workflow row has no ``plan_quality`` or
    ``plan_rate``, so keys 3-6 are inert for it.
    """
    by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in composite_report.get('configs', []):
        by_role[row.get('role_under_test')].append(row)

    def _rank_key(
        r: dict[str, Any],
    ) -> tuple[bool, float, bool, float, bool, float, str]:
        composite = r.get('composite')
        plan_quality = r.get('plan_quality')
        # ``.get`` like its siblings: a partially-populated row dict (several
        # consumers build one) degrades to None-last rather than raising.
        rate = r.get('plan_rate')
        return (
            composite is None,
            -float(composite or 0.0),
            plan_quality is None,
            -float(plan_quality or 0.0),
            rate is None,
            -float(rate or 0.0),
            str(r.get('config', '')),
        )

    survivors: dict[str, list[str]] = {}
    for role in roles:
        ranked = sorted(by_role.get(role, []), key=_rank_key)
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
    """The μ methodology report: a composite sub-report per stage, survivors, winner.

    ``stages`` nests a :func:`build_composite_report` for each of ``ofat``,
    ``matrix`` and ``confirm``, each with *price_table* echoed into it. The top
    level carries ``generated_at``, the echoed ``price_table`` and the OFAT
    ``survivors`` (both ``{}`` when omitted), and the confirmed ``winner``
    (``None`` when omitted).
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


_COMPOSITE_COLUMNS = (
    'config', 'composite', 'quality', 'plan_quality', 'pq_excluded',
    'pq_no_plan', 'pq_no_ref', 'plan_rate', 'cost_usd', 'cost_per_plan',
    'cost_source', 'latency_secs', 'ci95_composite', 'trials', 'fixtures',
)


def format_composite_table(report: dict[str, Any]) -> str:
    """Render :func:`build_composite_report` output as a deterministic table.

    One row per config, sorted by ``config``, in :data:`_COMPOSITE_COLUMNS` order,
    then a distinct ``price table`` section. Floats are fixed-precision and no
    wall-clock is rendered, so a report always renders byte-identically.
    ``composite``, ``quality``, ``plan_quality``, ``plan_rate``, ``cost_usd``,
    ``cost_per_plan`` and ``latency_secs`` render ``-`` for ``None``, and so does
    ``pq_no_ref`` for an unmeasured bound. ``pq_excluded`` and ``pq_no_plan``
    render a missing count as ``0``. An unpriced config's ``cost_source`` shows its
    explicit marker (e.g. ``unpriced_proxy``), never a blank.
    """
    configs = sorted(report.get('configs', []), key=lambda r: str(r.get('config', '')))

    rendered = [
        {
            'config': str(r.get('config', '')),
            'composite': optional_float_cell(r.get('composite')),
            'quality': optional_float_cell(r.get('quality')),
            'plan_quality': optional_float_cell(r.get('plan_quality')),
            'pq_excluded': str(int(r.get('plan_quality_cap_excluded', 0) or 0)),
            'pq_no_plan': str(int(r.get('plan_quality_no_plan', 0) or 0)),
            # '-' for an unmeasured bound, not the 0 default its siblings use.
            'pq_no_ref': (
                '-' if r.get('plan_quality_judged_without_reference') is None
                else str(int(r['plan_quality_judged_without_reference']))
            ),
            'plan_rate': optional_float_cell(r.get('plan_rate')),
            'cost_usd': optional_float_cell(r.get('cost_usd', 0.0)),
            'cost_per_plan': optional_float_cell(r.get('cost_per_plan')),
            'cost_source': str(r.get('cost_source', '')),
            'latency_secs': optional_float_cell(
                r.get('latency_secs', 0.0), precision=2,
            ),
            'ci95_composite': ci_cell((r.get('ci95') or {}).get('composite')),
            'trials': str(r.get('trials', 0)),
            'fixtures': str(r.get('fixtures', 0)),
        }
        for r in configs
    ]
    lines = render_fixed_table(
        _COMPOSITE_COLUMNS, rendered, header='composite report:',
    )

    # Distinct price-table section.
    lines.append('')
    lines.extend(format_price_table_section(report.get('price_table') or {}))
    return '\n'.join(lines)
