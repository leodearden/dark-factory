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
from typing import TYPE_CHECKING, Any, TypeAlias

from .elo import (
    INDISTINGUISHABLE_THRESHOLD,
    JudgeState,
    TaskPool,
    _pair_key,
)
from .metrics import _rate, blend_composite, produced_a_plan

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

    A config's ``cost_usd`` is a mean across its MEASURED trials; when those
    trials carry more than one distinct ``cost_source`` (e.g. ``price_table`` on
    one fixture, ``unpriced_proxy`` on another) reporting just the first trial's
    source would mislabel the cross-source mean, so surface ``'mixed'`` instead
    (reviewer: robustness). A trial dropped from that mean
    (:func:`_is_unmeasurable`) drops its source with it, for the same reason:
    the label must describe the figure actually taken. Exactly one distinct
    source → that source; none (a config whose every trial was unmeasurable, so
    ``cost_usd`` is ``None``) → ``'cli'`` for back-compat.
    """
    if not sources:
        return 'cli'
    if len(sources) == 1:
        return next(iter(sources))
    return 'mixed'


def _is_plan_only(metrics: dict[str, Any]) -> bool:
    """Is this trial a PLAN-ONLY cell (task 3099)?

    ``run_architect_eval`` is the only producer of ``role_under_test ==
    'architect'`` and is plan-only BY CONSTRUCTION — it freezes
    implementer/debugger/reviewer/verify, so the cell never runs a test and
    carries its quality in ``plan_quality`` rather than ``composite_score``.

    Keyed on the EXISTING ``role_under_test`` stamp rather than a new persisted
    ``plan_only`` field for two reasons: ``build_composite_report`` already keys
    ``plan_quality_cap_excluded`` on exactly this test (a second field could
    drift from the first), and a new field would read back as its default on
    every result JSON written before this task — silently mis-classifying the
    whole existing corpus, including the campaign this fix exists to make
    readable.
    """
    return metrics.get('role_under_test') == 'architect'


def _has_plan_quality_score(metrics: dict[str, Any]) -> bool:
    """THE plan-quality admission cascade: architect / not tainted / scored.

    One function, so every surface that admits or refuses a plan-only cell asks
    the SAME question (task 3099, reviewer amendment).
    :func:`build_plan_quality_report` and :func:`build_composite_report` both
    call it, and the two clauses used to coincide only by COUPLING —
    ``run_architect_eval`` happens to taint exactly when it has no score, so a
    ``cap_tainted``-only test and a ``plan_quality is None``-only test agreed by
    accident. A hand-edited result, a legacy JSON, or a future taint cause that
    preserves the structural floor would break that coincidence and feed a score
    the θ pool REFUSES into the composite pool ``select_survivors`` ranks on.
    """
    return (
        _is_plan_only(metrics)
        and not metrics.get('cap_tainted')
        and metrics.get('plan_quality') is not None
    )


def _plan_quality_score(
    metrics: dict[str, Any], *, where: str = '?',
) -> float | None:
    """THE θ score of one plan-only cell — ``None`` when it must be EXCLUDED.

    ONE accessor, called by both :func:`build_composite_report` and
    :func:`build_plan_quality_report`, so ``mean_plan_quality`` and the composite
    row's ``plan_quality`` cannot drift (task 3302, extending the
    :func:`_mean_plan_quality` discipline from the reduction to the value being
    reduced). It COMPOSES with :func:`_has_plan_quality_score` rather than
    re-deriving its cascade.

    Three outcomes, two of them deliberately distinct:

    - ``None`` — UNMEASURABLE. Not an admissible plan-only cell at all
      (:func:`_has_plan_quality_score`: not an architect run, cap-tainted, or no
      persisted score). The transport layer refused us, so every number would be
      fabricated: the cell leaves the pool entirely (task 3118, unchanged here).
    - ``0.0`` — NO PLAN WAS PRODUCED (``not produced_a_plan``). A healthy
      architect that emitted a stepless artifact is the OPPOSITE of unmeasurable:
      it is a genuine CONTENT measurement worth exactly 0.0, which is what
      :func:`judge.score_plan_structure` already returns for that artifact. This
      reproduces the 2026-07-27 campaign's own hand-computed ``meanPQ_all``
      ("scores a no-plan cell as 0") on the automated table.
    - the persisted float — a plan with steps, scored on its content.

    The persisted score is DISCARDED on the ``0.0`` path, and loudly: a nonzero
    ``plan_quality`` beside ``plan_steps=0`` IS the two-scorer disagreement
    (Graphiti e2066ec6 — ``score_plan_structure`` floors a stepless plan to 0.0
    as an anti-fabrication guard while the LLM judge has no such guard), and
    correcting it silently would hide the very cells an operator needs to see.

    *where* names the cell (``"fixture x config"``) for that warning; the metrics
    dict carries neither, so the caller — which holds the ``EvalResult`` — passes
    it. Purely diagnostic: it never changes the returned score.
    """
    if not _has_plan_quality_score(metrics):
        return None
    if not produced_a_plan(metrics):
        persisted = metrics.get('plan_quality')
        if persisted:
            logger.warning(
                'Plan-quality floor applied to %s: persisted plan_quality=%s '
                'with plan_steps=0 — a stepless artifact is not a plan, so this '
                "cell scores 0.0 (score_plan_structure's anti-fabrication "
                "answer), NOT the LLM judge's number. Two scorers disagreed "
                'here; the deterministic one wins.',
                where, persisted,
            )
        return 0.0
    return float(metrics['plan_quality'])


def _is_unmeasurable(metrics: dict[str, Any]) -> bool:
    """Did this trial measure NOTHING — so it must leave EVERY pool?

    True only for a PLAN-ONLY cell with no admissible θ score
    (:func:`_has_plan_quality_score`): the architect was refused, so the cell has
    no quality, and its recorded ``cost_usd`` / ``workflow_duration_ms`` are the
    $0.00 / 0 ms of a run that never happened rather than a measurement.

    Averaging those zeros into the row's cost/latency means was the MIRROR IMAGE
    of the defect task 3118 removed from ``mean_plan_quality``: it handed a
    schedule-attributable "2x cheaper, 2x faster" bonus to whichever candidate
    happened to be scheduled inside a cap window — on the very table task 3099
    makes the operator's ranking surface. A WORKFLOW trial is never unmeasurable
    here: a failing run really did burn its cost and latency.
    """
    return _is_plan_only(metrics) and not _has_plan_quality_score(metrics)


def _mean_plan_quality(scores: list[float]) -> float | None:
    """Reduce a pool of θ-rubric plan-quality scores to its mean (4 dp).

    THE ONE reduction, called by BOTH :func:`build_composite_report`'s row and
    :func:`build_plan_quality_report`'s per-config aggregate (task 3099). The
    composite row's ``plan_quality`` used to be a first-untainted-TRIAL
    passthrough — a harmless diagnostic echo until this task promoted the field
    to a DECISION surface (:func:`select_survivors` ranks on it and
    :func:`format_composite_table` renders it beside the composite). A trial-1
    score sitting next to a mean would let the two tables the CLI now prints
    ADJACENTLY — and the ``quality`` / ``plan_quality`` cells of one row — give
    contradictory answers about the same quantity, reintroducing the "two
    exclusion surfaces that disagree are worse than one" hazard on the very
    surface this task exists to make trustworthy. Sharing the reduction (rather
    than duplicating the arithmetic and asserting the two agree) makes that drift
    structurally impossible.

    An EMPTY pool is ``None``, never ``0.0``: "we measured nothing" must not read
    as "it scored nothing".
    """
    if not scores:
        return None
    return round(sum(scores) / len(scores), 4)


# ``(fixture, role_group)`` — the key of every efficiency baseline map in
# :func:`build_composite_report` (task 3099). Module-level, not function-local:
# a name bound inside a function body is not usable in a type expression.
_BaselineKey: TypeAlias = tuple[str, str]


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

    1. For each ``(fixture, role_group)`` — ``role_group`` being ``'plan_only'``
       for an architect trial and ``'workflow'`` otherwise (task 3099) —
       ``best_cost`` / ``best_latency`` = the minimum POSITIVE cost / latency
       across that group's MEASURED trials: ``tests_pass`` truthy for a workflow
       trial, :func:`_has_plan_quality_score` for a plan-only one. A
       cheap-but-WRONG
       (or cheap-but-UNMEASURABLE) run must not set the efficiency floor and
       deflate the runs that were actually measured (reviewer: correctness); a
       group with NO such trial falls back to all its trials so its baseline
       stays defined. Scoping by role is what keeps a ~$0.30/60s plan-only cell
       from becoming the floor for the ~$5/900s full-workflow cells that
       ``ofat_candidates()`` puts over the same fixtures.
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
    the config). ``recovery_score`` / ``role_under_test`` are passthroughs of any
    trial's metrics (η surfaces). ``plan_quality`` (θ) is NOT a passthrough: it is
    the config's POOLED MEAN over its scored plan-only cells, reduced by
    :func:`_mean_plan_quality` — the SAME function
    :func:`build_plan_quality_report` reduces ``mean_plan_quality`` with, so the
    two surfaces the CLI prints adjacently are structurally incapable of
    disagreeing (task 3099; it was a first-untainted-TRIAL passthrough while it
    was only a diagnostic echo, which is not defensible once
    :func:`select_survivors` RANKS on it). Its pool admits a cell on the
    byte-identical cascade that surface applies — architect-only, skip
    CAP-TAINTED (task 3118), score not ``None`` — so an infra refusal is never
    averaged in as a zero; ``plan_quality_cap_excluded`` reports how many were
    skipped, counted over ARCHITECT trials only so it agrees exactly with that
    surface's architect-scoped ``cap_excluded`` (reviewer: docs-accuracy — two
    exclusion surfaces that disagree are worse than one).

    THE PLAN-PRODUCTION INVARIANT (task 3302): the θ score of every admitted cell
    comes from :func:`_plan_quality_score`, so a cell whose architect ran fine
    but produced NO PLAN (``not metrics.produced_a_plan`` — ``plan_steps == 0``)
    is FLOORED to ``0.0`` rather than reported at whatever the LLM plan judge
    said, and is counted as ``plan_quality_no_plan``. A no-plan cell is also
    barred from SETTING its ``(fixture, 'plan_only')`` cost/latency baseline
    (step 1 below), though its real spend still enters the row's pools. The two
    treatments are deliberately different because the causes are: a cap-tainted
    cell is EXCLUDED and counted (we never asked the model, so any number would
    be fabricated); a no-plan cell is FLOORED and counted (we asked, and the
    answer was nothing — which is worth exactly 0.0, the same answer
    :func:`judge.score_plan_structure` gives that artifact). Neither is silent.

    SCOPE of the taint exclusion (REVISED by task 3099): an UNMEASURABLE
    plan-only trial (:func:`_is_unmeasurable`) leaves EVERY measured pool —
    ``plan_quality``, ``composite``, ``quality``, ``cost``, ``latency``, the
    efficiency baselines and the ``cost_source`` label — on ONE admission
    decision, so no two surfaces can disagree about the same cell.

    The earlier scope kept tainted trials in the composite pool on the grounds
    that it measured the implementer-path gates, which a tainted architect cell
    reports as an honest failure; once the composite is DERIVED from
    ``plan_quality`` that justification no longer holds, and a fabricated ``0.0``
    there would penalise whichever candidate was scheduled inside a cap window —
    the defect task 3118 removed from ``mean_plan_quality``, reintroduced one
    layer up in the number ``select_survivors`` actually ranks on. Keeping such a
    trial in the ``cost`` / ``latency`` pools was that defect's MIRROR IMAGE: the
    cell reports the $0.00 / 0 ms of a run that never happened, so averaging it
    in REWARDED the cap window instead — reporting a config as "2x cheaper and 2x
    faster" than an identically-priced sibling purely because one of its trials
    was refused (reviewer: correctness).

    The ``trials`` denominator, ``fixtures``, and the ``judge`` spend totals do
    still COUNT every trial, and ``plan_quality_cap_excluded`` reports how many
    were skipped — so the sample is reported honestly by the columns that exist
    to report it, never by silently averaging a zero into a price. A config with
    NO measured trial reports ``composite`` / ``quality`` / ``cost_usd`` /
    ``latency_secs`` and all three ``ci95`` intervals as ``None`` rather than
    ``0.0`` — "we measured nothing" must never read as "it scored nothing, for
    free, instantly, and we are confident".

    PLAN-ONLY rows (task 3099, :func:`_is_plan_only`): a trial whose
    ``role_under_test`` is ``'architect'`` ran no test at all, so it is blended
    with ``quality=plan_quality, plan_only=True`` instead of being hard-gated to
    ``0.0`` by its absent ``tests_pass``, and ``quality`` reports the axis that
    actually fed the composite. Such a config's ``tests_pass_rate`` is ``None``:
    no test ran, so there is no pass rate — not a 0% one. ``cost_source`` (P5) is
    the config's single distinct per-trial source over its MEASURED trials, or
    ``'mixed'`` when they span more than one — since ``cost_usd`` is a mean,
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
    #    Keyed on ``(fixture, role_group)`` — NOT on fixture alone (task 3099).
    #    A plan-only cell's cost is ONE architect invocation (~$0.30/60s); a
    #    workflow cell's is a full run (~$5/900s), and ``ofat_candidates()`` puts
    #    both over the SAME fixtures in one result set. A shared floor would
    #    therefore deflate every workflow row by ~16x AND clamp every plan-only
    #    row's efficiency axes to 1.0, so the same architect campaign would rank
    #    differently depending on whether unrelated implementer rows happened to
    #    be present. Scoping makes the plan-only composite a well-defined
    #    quantity rather than an artifact of who else was in the run.
    best_cost: dict[_BaselineKey, float] = {}
    best_latency: dict[_BaselineKey, float] = {}
    best_cost_all: dict[_BaselineKey, float] = {}
    best_latency_all: dict[_BaselineKey, float] = {}

    def _baseline_key(result: EvalResult) -> _BaselineKey:
        return (
            result.task_id,
            'plan_only' if _is_plan_only(result.metrics) else 'workflow',
        )

    for r in results:
        # A cell that measured nothing is not a data point about cost or
        # latency, so it seeds NEITHER baseline map — not even the ``*_all``
        # fallback, whose whole job is to keep the denominator defined for
        # trials that ARE scored. (Every trial it would have contributed is
        # itself excluded from the composite pool below, so a group left with an
        # empty fallback is never divided by.)
        if _is_unmeasurable(r.metrics):
            continue
        key = _baseline_key(r)
        cost = float(r.metrics.get('cost_usd', 0.0) or 0.0)
        latency = float(r.metrics.get('workflow_duration_ms', 0) or 0) / 1000.0
        # A plan-only trial has no tests_pass to consult, so its admission test
        # is the shared θ cascade — preserving the same rule the workflow group
        # applies: a cheap-but-UNMEASURABLE run cannot set the floor and deflate
        # the runs that were actually measured.
        passed = (
            _has_plan_quality_score(r.metrics) if _is_plan_only(r.metrics)
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
    # Groups with no passing trial fall back to the all-trials baseline so the
    # normalization denominator stays defined (every such workflow trial
    # hard-gates to 0 regardless, so this only keeps the baseline non-empty).
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
        }

    by_config: dict[str, dict[str, Any]] = defaultdict(_acc)
    for r in results:
        fixture = r.task_id
        m = r.metrics
        cost = float(m.get('cost_usd', 0.0) or 0.0)
        latency = float(m.get('workflow_duration_ms', 0) or 0) / 1000.0
        tests_pass = m.get('tests_pass')
        plan_only = _is_plan_only(m)
        acc = by_config[r.config_name]
        # A plan-only cell's quality axis is its θ-rubric plan_quality; a
        # workflow cell's is the pure compute_composite score. Accumulating the
        # axis that ACTUALLY fed the composite is what keeps the row internally
        # consistent — otherwise every architect row renders quality=0.0000
        # beside a non-zero composite (task 3099).
        #
        # The θ score comes from THE shared accessor (task 3302), never a raw
        # ``m['plan_quality']`` read: a cell whose architect produced NO plan is
        # floored to 0.0 there, so the judge's number for a stepless artifact
        # cannot reach the quality axis, the composite, or select_survivors.
        scored_pq = _plan_quality_score(m, where=f'{fixture} x {r.config_name}')
        quality = (
            scored_pq if plan_only and scored_pq is not None
            else float(m.get('composite_score', 0.0) or 0.0)
        )
        # ONE admission decision (:func:`_is_unmeasurable`), applied to EVERY
        # measured pool. An unmeasurable plan-only trial is excluded from the
        # composite/quality pools rather than scored 0.0 — task 3118's invariant,
        # now applied to the number that drives survivor selection — AND from the
        # cost/latency pools, whose $0.00 / 0 ms it never actually spent. Its
        # cost_source is dropped with it, so the row-level label describes the
        # mean that was actually taken. It is still counted in `trials` and in
        # `plan_quality_cap_excluded` below, so nothing is dropped silently.
        if not _is_unmeasurable(m):
            # Each trial normalizes against its OWN (fixture, role_group) floor.
            bkey = _baseline_key(r)
            acc['composite'].append(blend_composite(
                quality,
                _ratio_score(cost, best_cost.get(bkey, 0.0)),
                _ratio_score(latency, best_latency.get(bkey, 0.0)),
                tests_pass=tests_pass,
                plan_only=plan_only,
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
        # The plan-quality POOL admits a cell on THE shared accessor
        # (:func:`_plan_quality_score`, which composes _has_plan_quality_score:
        # architect-only / skip cap_tainted / score is not None, then floors a
        # no-plan cell to 0.0), so a cap-tainted cell is counted-and-excluded
        # here exactly as build_plan_quality_report excludes it (task 3118) and
        # is never averaged in as a zero, while a no-plan cell is floored on both
        # surfaces identically (task 3302). Reusing the accessor's return rather
        # than re-reading the metrics dict keeps the two sourced from one place.
        if scored_pq is not None:
            acc['plan_quality_scores'].append(scored_pq)
        # Counted over ARCHITECT trials only, matching
        # build_plan_quality_report's architect-scoped cap_excluded — the two
        # exclusion surfaces describe the same cells and must not disagree. (A
        # tainted non-architect trial has no plan_quality to exclude in the first
        # place, so it is counted by neither surface.)
        if m.get('cap_tainted') and plan_only:
            acc['plan_quality_cap_excluded'] += 1
        # …and the FLOOR gets its own count, parallel to the exclusion but
        # describing a DISJOINT cause (task 3302): this cell was admissible and
        # measured — a healthy architect that produced nothing — so it stays in
        # every pool at 0.0 rather than leaving them. Counting it is what keeps a
        # mean that absorbed zeros from reading identically to a mean over cells
        # that all planned badly (loud-over-silent-degradation).
        if _has_plan_quality_score(m) and not produced_a_plan(m):
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
        # An EMPTY pool means every trial was unmeasurable. mean_ci95([]) returns
        # mean 0.0 with a zero-width [0.0, 0.0] interval, which would read as
        # "it scored nothing, for free, instantly, and we are confident" — so
        # every derived cell reports None (rendered '-'), following the
        # mean_plan_quality=None precedent. ONE flag, because the composite,
        # quality, cost and latency pools now share ONE admission decision
        # (:func:`_is_unmeasurable`) and so are empty together or not at all.
        measured = bool(acc['composite'])
        # A plan-only config ran no test at all, so it has no pass RATE — not a
        # 0% one. Fabricating 0% there would report a failure that never happened.
        all_plan_only = acc['plan_only_trials'] == trials and trials > 0
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
            # The config's POOLED MEAN over its scored plan-only cells, through
            # the SAME reduction build_plan_quality_report uses. An empty pool
            # (every cell unmeasurable, or a workflow config with no plan-only
            # cell at all) is None → rendered '-', with the accompanying
            # plan_quality_cap_excluded count saying why.
            'plan_quality': _mean_plan_quality(acc['plan_quality_scores']),
            'plan_quality_cap_excluded': acc['plan_quality_cap_excluded'],
            # How many of the cells that pool DID admit produced no plan at all
            # and were scored 0.0 — the content-failure counterpart of the
            # transport-failure count above.
            'plan_quality_no_plan': acc['plan_quality_no_plan'],
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
    ``role_under_test``, ranks each requested role's group, and returns the
    top-``top_k`` config names per role. A role with fewer than ``top_k`` rows
    returns all it has; a requested role with no rows returns ``[]``. Pure over
    the report dict — the OFAT screen feeds these survivors into
    :func:`run_matrix_stage`.

    RANKING (task 3099), in order:

    1. rows with a measured ``composite`` before rows with ``composite is None``
       — ``None`` means "we measured nothing", which must never outrank a config
       that genuinely scored zero. A bare ``or 0.0`` coercion would silently tie
       the two and then hand the win to whichever sorted first by name,
       promoting the candidate we know LEAST about;
    2. DESCENDING ``composite`` mean;
    3. rows with a ``plan_quality`` before rows without (same ``None``-last
       rule on the secondary axis);
    4. DESCENDING ``plan_quality`` — the meaningful secondary signal for a
       PLAN-ONLY architect row;
    5. ASCENDING ``config`` name, surviving ONLY as the final byte-stability
       guarantee.

    Why 3-4 exist at all: with every architect composite hard-gated to 0.0 (the
    bug this task fixes), step 5 had silently become the ENTIRE selection
    mechanism for the architect role — ``architect-fable-high`` was "selected"
    for sorting first, not for planning best (plans/eval-architect-effort-
    verdict-2026-07-27.md, defect 2). Fixing the composite alone would leave the
    same trap armed for the next pair of genuinely-tied real composites, so the
    alphabet is demoted below every axis that carries signal. A workflow row
    carries no ``plan_quality``, so steps 3-4 are inert for it and the existing
    implementer ordering is unchanged.
    """
    by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in composite_report.get('configs', []):
        by_role[row.get('role_under_test')].append(row)

    def _rank_key(r: dict[str, Any]) -> tuple[bool, float, bool, float, str]:
        composite = r.get('composite')
        plan_quality = r.get('plan_quality')
        return (
            composite is None,
            -float(composite or 0.0),
            plan_quality is None,
            -float(plan_quality or 0.0),
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
# Shared fixed-width table renderer
#
# The single home for the width-computed ljust idiom every deterministic table
# below shares (recovery / plan-quality / per-config mean / price / composite).
# It was copied five times before task 3118, which is exactly the drift this
# invites: the '-'-not-'0.0000' fix had to be applied to two copies by hand and
# the other three silently kept their trailing padding.
# ---------------------------------------------------------------------------

def _render_fixed_table(
    columns: tuple[str, ...],
    rendered: list[dict[str, str]],
    *,
    header: str | None = None,
) -> list[str]:
    """Render PRE-STRINGIFIED rows as a fixed-width, byte-deterministic block.

    *rendered* is a list of ``{column: cell}`` dicts whose values are already
    strings — every value/None/precision decision stays with the caller, which
    is what keeps this generic. Column widths are the max of the header and all
    cells, so the block is a function of its input alone: no wall-clock, no
    dict-order dependence, and the same rows always render byte-identically.

    Returns the block as LINES (optional *header* line, column headers, a dashes
    rule, then the rows) rather than a joined string, so callers can append
    further sections — several emit two or three blocks separated by blanks.

    Lines are ``rstrip``ed: trailing padding on the last column is invisible
    whitespace that makes an otherwise-identical table differ byte-for-byte,
    and it lets a caller assert on a line's real ending (e.g. the per-config
    mean block's ``-`` for "nothing scored", which must never read as
    ``0.0000``).
    """
    widths = {
        col: (
            max(len(col), *(len(rr[col]) for rr in rendered))
            if rendered else len(col)
        )
        for col in columns
    }

    def _fmt(cells: dict[str, str]) -> str:
        return '  '.join(cells[col].ljust(widths[col]) for col in columns).rstrip()

    lines = [header] if header is not None else []
    lines.append(_fmt({col: col for col in columns}))
    lines.append('  '.join('-' * widths[col] for col in columns))
    lines.extend(_fmt(rr) for rr in rendered)
    return lines


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
    return '\n'.join(_render_fixed_table(
        _RECOVERY_COLUMNS, rendered, header='recovery_score report:',
    ))


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

_PLAN_QUALITY_COLUMNS = (
    'task_id', 'config_name', 'role_under_test', 'plan_quality',
    'invocation_error',
)
_PLAN_QUALITY_MEAN_COLUMNS = (
    'config_name', 'n', 'cap_excluded', 'no_plan', 'mean_plan_quality',
)
# The plan_quality cell of a cap-tainted row. Deliberately NOT a number and NOT
# the bare '-' null sentinel (which already means "not an architect run"): a
# reader must be able to tell "this candidate scored nothing" from "we could not
# measure this candidate" at a glance.
#
# Spelled 'excluded', not 'cap-excluded' (reviewer: design-coherence): the
# ``cap_tainted`` flag covers every unmeasurable cause — cap hit, auth failure,
# model-not-found, zero-output wedge, harness error — and a PERMANENT config
# error ("this candidate can never run") must not render as a TRANSIENT one
# ("rerun after the cap window"). The cause is always carried in the adjacent
# ``invocation_error`` column and broken out by cause in the summary line, so
# the marker itself stays cause-neutral rather than asserting a cap.
_CAP_EXCLUDED_CELL = 'excluded'
_PLAN_QUALITY_MEAN_HEADER = 'plan_quality by config:'


def _taint_cause(marker: str | None) -> str:
    """Reduce a stage-prefixed ``invocation_error`` to its bare CAUSE key.

    ``"architect:cap_hit: You've hit your session limit · resets 8pm"`` →
    ``"cap_hit"``; ``"architect:model_not_found: ..."`` → ``"model_not_found"``.
    Pure string surgery on the marker ``run_architect_eval`` builds — the eval
    layer deliberately does not re-derive the cause from the AgentResult, which
    is long gone by report time.

    Falls back to ``'unknown'`` for an absent or unparseable marker rather than
    guessing, so a mis-shaped marker shows up as its own bucket instead of being
    silently folded into a real cause.
    """
    if not marker:
        return 'unknown'
    # Only the FIRST stage marker is used: the join order is architect-then-
    # judge, and a judge-only refusal never taints, so the leading marker is
    # always the one that drove the exclusion.
    first = marker.split(';')[0].strip()
    _stage, sep, rest = first.partition(':')
    if not sep:
        return first or 'unknown'
    return rest.strip().partition(':')[0].strip() or 'unknown'


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

    Rows additionally carry the task-3118 infra markers ``cap_tainted`` /
    ``invocation_error`` (likewise default-safe: ``False`` / ``None`` for
    metrics predating them), and the report gains a per-config aggregate over
    ARCHITECT rows plus a report-level ``cap_excluded`` total.

    THE INVARIANT: a cap-tainted cell is an infra failure — we never got to ask
    the model — so it is EXCLUDED from ``mean_plan_quality`` and COUNTED as
    excluded, never averaged in as a zero. Averaging it in would penalise
    whichever candidate happened to be scheduled inside a cap window, which is a
    property of the schedule, not of the candidate. A config with no scored cells
    reports ``mean_plan_quality=None`` rather than ``0.0``, so "we measured
    nothing" can never read as "it scored nothing".

    ITS COUNTERPART (task 3302): a cell whose architect ran fine but produced NO
    PLAN (``not metrics.produced_a_plan``) is the opposite case — a genuine
    content measurement — so it is FLOORED to ``0.0`` by
    :func:`_plan_quality_score`, KEPT in the mean, and counted separately as
    ``no_plan``. Two causes, two treatments, two counts, neither silent:
    transport refusal → excluded + ``cap_excluded``; content failure → floored +
    ``no_plan``. Conflating them would either let the candidate that failed to
    plan escape the pool entirely or penalise the one that merely hit a cap.

    ``cap_excluded_by_cause`` breaks that total out by CAUSE
    (``{'cap_hit': 2, 'model_not_found': 1}``, key-sorted for determinism)
    because the causes are not interchangeable: a cap hit is transient and
    schedule-attributable ("rerun after the window"), whereas a model-not-found
    or auth failure is a PERMANENT, candidate-specific configuration error. A
    single "cap-excluded" total would let the latter masquerade as the former
    and hide a config that can never run behind ``n=0, mean=None``.
    """
    rows: list[dict[str, Any]] = []
    for result in results:
        rows.append({
            'task_id': result.task_id,
            'config_name': result.config_name,
            'role_under_test': result.metrics.get('role_under_test'),
            'plan_quality': result.metrics.get('plan_quality'),
            # The PLAN-PRODUCTION predicate's input (task 3302), carried
            # verbatim so the row is self-describing: a reader can see the step
            # count the adjacent plan_quality was — or was not — scored over.
            'plan_steps': result.metrics.get('plan_steps'),
            'cap_tainted': bool(result.metrics.get('cap_tainted')),
            'invocation_error': result.metrics.get('invocation_error'),
            # The θ score this cell contributes to the aggregate, through THE
            # shared accessor (task 3302): the persisted float for a real plan,
            # a floored 0.0 for a cell that produced none, None when the cell is
            # excluded outright. Kept beside the RAW persisted ``plan_quality``
            # rather than replacing it — the per-row table reports what was
            # persisted, and an operator must still be able to see the judge's
            # number that the floor overrode.
            'scored_plan_quality': _plan_quality_score(
                result.metrics,
                where=f'{result.task_id} x {result.config_name}',
            ),
        })
    rows.sort(key=lambda r: (r['task_id'], r['config_name']))

    # Per-config aggregate over ARCHITECT rows only — an implementer run never
    # invokes the plan judge, so its null score is a different thing entirely
    # and must not dilute the architect mean.
    scored: dict[str, list[float]] = defaultdict(list)
    excluded: dict[str, int] = defaultdict(int)
    no_plan: dict[str, int] = defaultdict(int)
    totals: dict[str, int] = defaultdict(int)
    by_cause: dict[str, int] = defaultdict(int)
    for row in rows:
        if row['role_under_test'] != 'architect':
            continue
        cfg = row['config_name']
        totals[cfg] += 1
        if row['cap_tainted']:
            excluded[cfg] += 1
            by_cause[_taint_cause(row['invocation_error'])] += 1
        elif row['scored_plan_quality'] is not None:
            # The row carries ``plan_steps`` verbatim from the metrics dict, so
            # the predicate reads it here exactly as it does one layer up.
            if not produced_a_plan(row):
                # DISJOINT from the exclusion above: this cell was measured and
                # kept, at the 0.0 the accessor floored it to (task 3302).
                no_plan[cfg] += 1
            # THE shared accessor's answer (task 3302), not the raw persisted
            # score: a cell whose architect produced no plan lands here as a
            # floored 0.0, identically to build_composite_report's pool, so
            # mean_plan_quality and the composite row cannot drift.
            scored[cfg].append(row['scored_plan_quality'])

    configs = [
        {
            'config_name': cfg,
            'n': len(scored[cfg]),
            'cap_excluded': excluded[cfg],
            # How many of those ``n`` scored cells produced NO plan and were
            # floored to 0.0 (task 3302). Disjoint from ``cap_excluded``: those
            # cells are not in ``n`` at all.
            'no_plan': no_plan[cfg],
            'total': totals[cfg],
            # THE ONE reduction, shared with build_composite_report's row so the
            # two surfaces cannot drift (task 3099, :func:`_mean_plan_quality`).
            'mean_plan_quality': _mean_plan_quality(scored[cfg]),
        }
        for cfg in sorted(totals)
    ]
    return {
        'rows': rows,
        'configs': configs,
        'cap_excluded': sum(excluded.values()),
        'cap_excluded_by_cause': {c: by_cause[c] for c in sorted(by_cause)},
    }


def _format_plan_quality_mean_section(report: dict[str, Any]) -> list[str]:
    """Render the per-config mean block: how many cells actually scored.

    The point of the block is that ``mean_plan_quality`` is reported ALONGSIDE
    the ``n`` it was computed over, the ``cap_excluded`` count it left out and
    the ``no_plan`` count it scored as zeros, so a mean over 19 of 22 cells —
    four of which produced no plan at all — reads as exactly that instead of
    looking like a mean over 22 comparable plans. A config with nothing scored
    renders ``-``, never ``0.0000``.
    """
    configs = report.get('configs', [])
    rendered = [
        {
            'config_name': str(c['config_name']),
            'n': str(c['n']),
            'cap_excluded': str(c['cap_excluded']),
            'no_plan': str(c['no_plan']),
            'mean_plan_quality': (
                '-' if c['mean_plan_quality'] is None
                else f'{float(c["mean_plan_quality"]):.4f}'
            ),
        }
        for c in configs
    ]
    return _render_fixed_table(
        _PLAN_QUALITY_MEAN_COLUMNS, rendered, header=_PLAN_QUALITY_MEAN_HEADER,
    )


def format_plan_quality_table(report: dict[str, Any]) -> str:
    """Render :func:`build_plan_quality_report` output as a deterministic table.

    A ``plan_quality`` column shows the populated float (4 dp) for architect rows
    and ``-`` (the null sentinel) for non-architect rows; ``role_under_test``
    and ``invocation_error`` render ``-`` when ``None``. A CAP-TAINTED row shows
    the explicit ``excluded`` marker in place of a score, alongside the
    ``invocation_error`` that caused it — an infra failure must never render as
    ``0.0000``, and must stay distinguishable from the ``-`` a non-architect row
    uses.

    Two sections follow the rows: the exclusion count — how many architect cells
    were not measurable, BROKEN OUT BY CAUSE so a permanent config error is not
    read as a transient cap window — and the per-config mean block. Both are
    computed from the report, so the CLI caller picks the exclusion up with no
    change of its own. The same report always renders byte-identically (no
    wall-clock or dict-order dependence).
    """
    rows = report.get('rows', [])

    def _score_cell(row: dict[str, Any]) -> str:
        if row.get('cap_tainted'):
            return _CAP_EXCLUDED_CELL
        value = row['plan_quality']
        return '-' if value is None else f'{value:.4f}'

    def _text_cell(value: str | None) -> str:
        return '-' if value is None else str(value)

    rendered = [
        {
            'task_id': str(r['task_id']),
            'config_name': str(r['config_name']),
            'role_under_test': _text_cell(r['role_under_test']),
            'plan_quality': _score_cell(r),
            'invocation_error': _text_cell(r.get('invocation_error')),
        }
        for r in rows
    ]
    lines = _render_fixed_table(
        _PLAN_QUALITY_COLUMNS, rendered, header='plan_quality report:',
    )

    architect_cells = sum(c['total'] for c in report.get('configs', []))
    # Broken out by CAUSE so a permanent config error (model_not_found /
    # auth_failed — "this candidate can never run") cannot masquerade as a
    # transient schedule artifact (cap_hit — "rerun after the window").
    by_cause = report.get('cap_excluded_by_cause') or {}
    causes = ', '.join(f'{cause}: {n}' for cause, n in sorted(by_cause.items()))
    lines.append('')
    lines.append(
        f'excluded: {report.get("cap_excluded", 0)} unmeasurable cell(s) of '
        f'{architect_cells} architect cell(s)'
        + (f' ({causes})' if causes else '')
    )
    lines.append('')
    lines.extend(_format_plan_quality_mean_section(report))
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Composite report renderer (eval-revival λ / C4)
#
# The deterministic table surface for :func:`build_composite_report`, mirroring
# the width-computed ljust idiom of format_recovery_table / format_plan_quality_
# table above. Emits a per-config table (composite / quality / plan_quality /
# pq_excluded / cost / cost_source / latency / CI95 / trials / fixtures) followed
# by a DISTINCT price-table section (config → role → input/output per-1M).
# Byte-stable: rows and sections are sorted, floats fixed-precision, and no
# wall-clock is rendered.
#
# ``plan_quality`` / ``pq_excluded`` (task 3099) make an architect campaign
# rankable from the table alone. ``pq_excluded`` is a COUNT of that config's
# unmeasurable (cap-tainted) architect cells; the per-CAUSE breakdown — which is
# what separates a transient cap window from a permanent model-not-found — lives
# only in :func:`format_plan_quality_table`, so the CLI emits both for an
# architect result set.
# ---------------------------------------------------------------------------

_COMPOSITE_COLUMNS = (
    'config', 'composite', 'quality', 'plan_quality', 'pq_excluded',
    'cost_usd', 'cost_source', 'latency_secs', 'ci95_composite',
    'trials', 'fixtures',
)
_PRICE_TABLE_COLUMNS = ('config', 'role', 'input_per_1m', 'output_per_1m')


def _ci_cell(ci: dict[str, Any] | None) -> str:
    """Render a ``mean_ci95`` sub-dict as a ``[lo, hi]`` interval (4 dp)."""
    if not ci:
        return '-'
    return f'[{float(ci.get("lo", 0.0)):.4f}, {float(ci.get("hi", 0.0)):.4f}]'


def _optional_float_cell(value: Any, *, precision: int = 4) -> str:
    """Render an OPTIONAL float cell: fixed precision, or ``-`` when unmeasured.

    The same `-`-not-``0.0000`` idiom as ``_ci_cell`` above and
    :func:`format_plan_quality_table`'s ``_score_cell``: a cell that measured
    NOTHING (``None``) must never render as one that scored zero. Task 3099
    extends the idiom to ``composite`` / ``quality``, which now report ``None``
    for a config whose every trial was unmeasurable.
    """
    return '-' if value is None else f'{float(value):.{precision}f}'


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
    return _render_fixed_table(
        _PRICE_TABLE_COLUMNS, rendered, header='price table:',
    )


def format_composite_table(report: dict[str, Any]) -> str:
    """Render :func:`build_composite_report` output as a deterministic table.

    A per-config table carries composite / quality / plan_quality / pq_excluded /
    cost_usd / cost_source / latency_secs, a ``[lo, hi]`` CI95 rendering of the
    composite, and trial / fixture counts, followed by a distinct ``price table``
    section. The same report always renders byte-identically (rows sorted by
    config, price-table sections sorted, fixed float precision, no
    wall-clock/dict-order dependence). An unpriced config's ``cost_source`` shows
    the explicit marker (e.g. ``unpriced_proxy``), never a blank.

    ``plan_quality`` (task 3099) is a PLAN-ONLY architect config's MEAN θ-rubric
    score over its scored cells — not one trial's — and ``-`` for a workflow row
    that never invoked the plan judge. Being the config-level mean is exactly what
    makes an architect campaign rankable from this table alone instead of
    recomputing from the per-cell result JSONs, and is why it renders the same
    number as the ``plan_quality by config:`` block of the
    :func:`format_plan_quality_table` the CLI prints directly beneath it
    (:func:`_mean_plan_quality` is shared). ``pq_excluded`` is that config's
    COUNT of unmeasurable (cap-tainted) architect cells; the per-CAUSE breakdown
    that distinguishes a transient cap window from a permanent model-not-found is
    deliberately NOT duplicated here — it lives in
    :func:`format_plan_quality_table`, which the CLI emits alongside this table
    for an architect result set.

    ``composite`` / ``quality`` / ``plan_quality`` all render ``-`` when the row
    measured nothing (:func:`_optional_float_cell`): "we measured nothing" must
    never read as "it scored nothing".
    """
    configs = sorted(report.get('configs', []), key=lambda r: str(r.get('config', '')))

    rendered = [
        {
            'config': str(r.get('config', '')),
            'composite': _optional_float_cell(r.get('composite')),
            'quality': _optional_float_cell(r.get('quality')),
            'plan_quality': _optional_float_cell(r.get('plan_quality')),
            'pq_excluded': str(int(r.get('plan_quality_cap_excluded', 0) or 0)),
            'cost_usd': _optional_float_cell(r.get('cost_usd', 0.0)),
            'cost_source': str(r.get('cost_source', '')),
            'latency_secs': _optional_float_cell(
                r.get('latency_secs', 0.0), precision=2,
            ),
            'ci95_composite': _ci_cell((r.get('ci95') or {}).get('composite')),
            'trials': str(r.get('trials', 0)),
            'fixtures': str(r.get('fixtures', 0)),
        }
        for r in configs
    ]
    lines = _render_fixed_table(
        _COMPOSITE_COLUMNS, rendered, header='composite report:',
    )

    # Distinct price-table section.
    lines.append('')
    lines.extend(_format_price_table_section(report.get('price_table') or {}))
    return '\n'.join(lines)
