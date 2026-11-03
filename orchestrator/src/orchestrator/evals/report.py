"""The composite, methodology, recovery and plan-quality report surfaces over
persisted eval cells.
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
from .measurement import UnmeasuredCause
from .metrics import (
    DECLINE_KINDS,
    RoleUnderTest,
    blend_composite,
    terminal_kind_of,
)
from .plan_predicates import produced_a_plan
from .prices import format_price_table_section
from .stats import mean_ci95
from .table_render import ci_cell, optional_float_cell, render_fixed_table

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from .measurement import UnmeasuredRecord
    from .results import EvalResult


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


def _summarize_cost_source(sources: set[str]) -> str:
    """Collapse a config's per-trial cost sources to ONE row-level label.

    A config's ``cost_usd`` is a mean across its MEASURED trials; when those
    trials carry more than one distinct ``cost_source`` (e.g. ``price_table`` on
    one fixture, ``unpriced_proxy`` on another) reporting just the first trial's
    source would mislabel the cross-source mean, so surface ``'mixed'`` instead
    (reviewer: robustness). A trial dropped from that mean
    (:func:`is_unmeasurable`) drops its source with it, for the same reason:
    the label must describe the figure actually taken. Exactly one distinct
    source → that source; none (a config whose every trial was unmeasurable, so
    ``cost_usd`` is ``None``) → ``'cli'`` for back-compat.
    """
    if not sources:
        return 'cli'
    if len(sources) == 1:
        return next(iter(sources))
    return 'mixed'


def _merge_counts(per_key: Mapping[str, Mapping[str, int]]) -> dict[str, int]:
    """Fold per-config ``{kind: count}`` tallies into ONE campaign-wide tally.

    THE reduction behind ``build_plan_quality_report``'s report-level
    ``declined_by_kind`` (task 4760), so that total is summed from the very
    counters the per-config entries are emitted from rather than re-derived from
    the rendered ``configs`` list — the :func:`mean_plan_quality` discipline
    applied to a histogram: one arithmetic, so the two surfaces cannot disagree
    about the same quantity.

    Pure: reads the mapping, mutates nothing, and returns insertion-ordered
    counts the caller key-sorts for determinism.
    """
    merged: dict[str, int] = defaultdict(int)
    for counts in per_key.values():
        for key, n in counts.items():
            merged[key] += n
    return dict(merged)


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
    :func:`elo_report.compute_aggregate_ratings`): a config that ran in *any*
    fixture gets a row, so a config present in only one fixture is no longer
    silently dropped.

    Aggregation (Open-Q3 / decision 10) is per-fixture NORMALIZED composite →
    mean with a small-sample CI:

    1. For each ``(fixture, role_group)`` — ``role_group`` being ``'plan_only'``
       for an architect trial and ``'workflow'`` otherwise (task 3099) —
       ``best_cost`` / ``best_latency`` = the minimum POSITIVE cost / latency
       across that group's MEASURED trials: ``tests_pass`` truthy for a workflow
       trial, :func:`has_plan_quality_score` for a plan-only one. A
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
    :func:`mean_plan_quality` — the SAME function
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
    comes from :func:`plan_quality_score`, so a cell whose architect ran fine
    but produced NO PLAN (``not plan_predicates.produced_a_plan`` — ``plan_steps == 0``)
    is FLOORED to ``0.0`` rather than reported at whatever the LLM plan judge
    said, and is counted as ``plan_quality_no_plan``. A no-plan cell is also
    barred from SETTING its ``(fixture, 'plan_only')`` cost/latency baseline
    (step 1 below), though its real spend still enters the row's pools. The two
    treatments are deliberately different because the causes are: a cap-tainted
    cell is EXCLUDED and counted (we never asked the model, so any number would
    be fabricated); a no-plan cell is FLOORED and counted (we asked, and the
    answer was nothing — which is worth exactly 0.0, the same answer
    :func:`judge.score_plan_structure` gives that artifact). Neither is silent.

    THE RELIABILITY AXIS (task 3379): ``plan_rate`` is the fraction of that
    config's ADMITTED plan-only cells that emitted a plan at all —
    ``(plan_quality_n - plan_quality_no_plan) / plan_quality_n`` through THE
    shared :func:`plan_rate` reducer, with ``plan_quality_n`` (the admitted
    pool's size, identical to the θ table's ``n``) carried beside it so the
    ratio is verifiable from the row alone. It is DERIVED entirely from counts
    task 3302 already collects: no new collection, no new predicate. It exists
    because the 2026-07-27 architect campaign
    (``plans/eval-architect-effort-verdict-2026-07-27.md``) found that what
    separates the candidates is not how WELL they plan but how OFTEN they plan
    at all, and its operator had to hand-compute that figure from the per-cell
    result JSONs because no report surface exposed it. The denominator is the
    ADMITTED pool and deliberately NOT ``trials``: a cap-tainted cell leaves
    both numerator and denominator, so a candidate is never reported as less
    reliable for a question a session-cap window stopped it from answering (see
    :func:`plan_rate`). An empty pool is ``None``, never ``0.0`` or ``1.0``.

    ``cost_per_plan`` is that axis priced: the spend over the config's ADMITTED
    plan-only cells — INCLUDING the ones that produced nothing — over only the
    cells that DID produce a plan (:func:`cost_per_plan`). The asymmetry is
    deliberate and is the whole figure: a no-plan cell burns real budget and
    returns nothing, so a cheap-but-unreliable candidate's per-fixture
    ``cost_usd`` advantage is partly illusory, and netting the failed attempts
    out of the numerator would report that illusion rather than remove it. Its
    numerator accumulates under the SAME admission decision that fills the θ
    pool, so an unmeasurable cell's $0.00 dilutes neither side. It lives on
    THIS row only: :func:`build_plan_quality_report` carries no cost data, and
    threading some there would create a SECOND cost surface free to drift from
    ``cost_usd``. ``None`` when no admitted cell planned.

    SCOPE of the taint exclusion (REVISED by task 3099): an UNMEASURABLE
    plan-only trial (:func:`is_unmeasurable`) leaves EVERY measured pool —
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

    PLAN-ONLY rows (task 3099, :func:`is_plan_only`): a trial whose
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
    *price_table* (μ/CLI seeds it from :func:`prices.build_price_table`) is echoed
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
            'plan_only' if is_plan_only(result.metrics) else 'workflow',
        )

    for r in results:
        # A cell that measured nothing is not a data point about cost or
        # latency, so it seeds NEITHER baseline map — not even the ``*_all``
        # fallback, whose whole job is to keep the denominator defined for
        # trials that ARE scored. (Every trial it would have contributed is
        # itself excluded from the composite pool below, so a group left with an
        # empty fallback is never divided by.)
        if is_unmeasurable(r.metrics):
            continue
        key = _baseline_key(r)
        cost = float(r.metrics.get('cost_usd', 0.0) or 0.0)
        latency = float(r.metrics.get('workflow_duration_ms', 0) or 0) / 1000.0
        # A plan-only trial has no tests_pass to consult, so its admission test
        # is the shared θ cascade — preserving the same rule the workflow group
        # applies: a cheap-but-UNMEASURABLE run cannot set the floor and deflate
        # the runs that were actually measured.
        #
        # …AND it must actually have produced a plan (task 3302). A no-plan cell
        # is measurable — it stays in every pool at 0.0 — but it is cheap and
        # fast precisely BECAUSE it returned nothing:
        # plans/eval-architect-effort-verdict-2026-07-27.md measured one at
        # $0.5-$3 against a real plan's ~$3.7. Letting it seed the floor hands a
        # schedule-independent "2x cheaper, 2x faster" bonus to the candidate
        # that FAILED to plan and deflates every candidate that succeeded. This
        # is the plan-only spelling of the rule the workflow group already
        # applies to a failing trial, not a new one.
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
    # Groups with no passing trial fall back to the all-trials baseline so the
    # normalization denominator stays defined (every such workflow trial
    # hard-gates to 0 regardless, so this only keeps the baseline non-empty).
    # The same holds for a plan-only group in which NOTHING produced a plan: the
    # fallback still admits those cells, so the denominator exists — which means
    # a no-plan cell that is the SOLE member of its (fixture, 'plan_only') group
    # DOES earn ratios of 1.0 on both efficiency axes here. That is exactly why
    # its composite is hard-gated to 0.0 by blend_composite(no_plan=True) below:
    # flooring its quality axis bounds only the 0.6 quality weight, leaving 0.4
    # of pure efficiency credit earned by FAILING to plan (task 3302 review —
    # measured: a no-plan cell at 0.40 outranked a real 6-step plan at 0.26 and
    # survived top_k=2). BOTH closures are needed and they close different
    # routes: barring the cell from SEEDING the floor (above) closes the
    # INTRA-group route, where it would deflate its plan-producing siblings; the
    # gate closes the CROSS-group route, where it would outrank a plan-producing
    # config normalized against a different fixture's floor.
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
            # …and how many admitted cells carry NO marker at all, which is a
            # different fact from "marker False" and makes the count above
            # unknowable rather than zero (task 3628 amendment).
            'plan_quality_judged_without_reference_unmeasured': 0,
            # The $/usable-plan NUMERATOR (task 3379): spend over admitted
            # plan-only cells, summed under the SAME condition that fills
            # plan_quality_scores — see the accumulation site.
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
        scored_pq = plan_quality_score(m, where=f'{fixture} x {r.config_name}')
        quality = (
            scored_pq if plan_only and scored_pq is not None
            else float(m.get('composite_score', 0.0) or 0.0)
        )
        # THE no-plan predicate, evaluated ONCE and reused by both the
        # blend_composite hard gate below and the plan_quality_no_plan counter
        # further down — so the gate and the count can never disagree about
        # which cells produced no plan (task 3302).
        #
        # Scoped by has_plan_quality_score, not by plan_only alone: a
        # cap-tainted cell is UNMEASURABLE (we never got to ask), so it is
        # counted by plan_quality_cap_excluded and must not also be counted as
        # a content failure — the two causes are disjoint by construction. At
        # the gate site the two spellings coincide exactly, because
        # is_unmeasurable is precisely "plan-only and not
        # has_plan_quality_score", so an inadmissible cell never reaches
        # blend_composite at all.
        no_plan = has_plan_quality_score(m) and not produced_a_plan(m)
        # ONE admission decision (:func:`is_unmeasurable`), applied to EVERY
        # measured pool. An unmeasurable plan-only trial is excluded from the
        # composite/quality pools rather than scored 0.0 — task 3118's invariant,
        # now applied to the number that drives survivor selection — AND from the
        # cost/latency pools, whose $0.00 / 0 ms it never actually spent. Its
        # cost_source is dropped with it, so the row-level label describes the
        # mean that was actually taken. It is still counted in `trials` and in
        # `plan_quality_cap_excluded` below, so nothing is dropped silently.
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
        # The plan-quality POOL admits a cell on THE shared accessor
        # (:func:`plan_quality_score`, which composes has_plan_quality_score:
        # architect-only / skip cap_tainted / score is not None, then floors a
        # no-plan cell to 0.0), so a cap-tainted cell is counted-and-excluded
        # here exactly as build_plan_quality_report excludes it (task 3118) and
        # is never averaged in as a zero, while a no-plan cell is floored on both
        # surfaces identically (task 3302). Reusing the accessor's return rather
        # than re-reading the metrics dict keeps the two sourced from one place.
        if scored_pq is not None:
            acc['plan_quality_scores'].append(scored_pq)
            # …and the cell's REAL spend joins the $/usable-plan numerator under
            # exactly that one condition (task 3379), so numerator and
            # denominator are sourced from ONE admission decision and cannot
            # describe different cells. An unmeasurable/cap-tainted cell is
            # excluded here for the same reason is_unmeasurable keeps its
            # $0.00 out of the `cost` pool: it is the price of a run that never
            # happened, and averaging it in would report the cap window as a
            # discount. A no-plan cell's spend DOES land here — that is what
            # makes the figure $/usable plan rather than $/cell.
            acc['plan_cost_usd'] += cost
            # …and the σ VALIDITY bound (task 3628) is counted under that SAME
            # one admission decision, which is what makes it a strict subset of
            # plan_quality_n and keeps it in lockstep with the θ surface's pool.
            # Alone among the three counts this one changes NO number: the cell
            # contributes its real score to the quality axis, the composite and
            # the cost pools exactly as a healthy cell does. It neither excludes
            # (cap_excluded) nor floors (no_plan) — it reports that the score,
            # while real, was produced without ground truth to grade against, so
            # only the CONFIDENCE attached to it is reduced.
            # Read through THE shared accessor, so a cell written before σ
            # landed lands in the UNMEASURED tally rather than being silently
            # counted as validity-bounded=False.
            marker = judged_without_reference(m)
            if marker is None:
                acc['plan_quality_judged_without_reference_unmeasured'] += 1
            elif marker:
                acc['plan_quality_judged_without_reference'] += 1
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
        # every pool (at 0.0 on quality, at its real spend on cost/latency, and
        # hard-gated to 0.0 on the composite) rather than leaving them. Counting
        # it is what keeps a mean that absorbed zeros from reading identically to
        # a mean over cells that all planned badly
        # (loud-over-silent-degradation). Reuses the SAME local the gate used.
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
        # An EMPTY pool means every trial was unmeasurable. mean_ci95([]) returns
        # mean 0.0 with a zero-width [0.0, 0.0] interval, which would read as
        # "it scored nothing, for free, instantly, and we are confident" — so
        # every derived cell reports None (rendered '-'), following the
        # mean_plan_quality=None precedent. ONE flag, because the composite,
        # quality, cost and latency pools now share ONE admission decision
        # (:func:`is_unmeasurable`) and so are empty together or not at all.
        measured = bool(acc['composite'])
        # A plan-only config ran no test at all, so it has no pass RATE — not a
        # 0% one. Fabricating 0% there would report a failure that never happened.
        all_plan_only = acc['plan_only_trials'] == trials and trials > 0
        # The size of the ADMITTED θ pool, bound ONCE and reused by both
        # plan_quality_n and plan_rate below (task 3379) — so the rate and the
        # denominator it is reported beside can never describe different pools.
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
            # The config's POOLED MEAN over its scored plan-only cells, through
            # the SAME reduction build_plan_quality_report uses. An empty pool
            # (every cell unmeasurable, or a workflow config with no plan-only
            # cell at all) is None → rendered '-', with the accompanying
            # plan_quality_cap_excluded count saying why.
            'plan_quality': mean_plan_quality(acc['plan_quality_scores']),
            'plan_quality_cap_excluded': acc['plan_quality_cap_excluded'],
            # How many of the cells that pool DID admit produced no plan at all
            # and were scored 0.0 — the content-failure counterpart of the
            # transport-failure count above.
            'plan_quality_no_plan': acc['plan_quality_no_plan'],
            # …and how many of that same admitted pool were judged WITHOUT a
            # reference diff (task 3628) — a validity bound on the
            # plan_quality above, not a third way of removing cells from it.
            # ``None`` (rendered '-') when ANY admitted cell predates the
            # marker: the bound is then UNKNOWN, and a 0 there would read as
            # "every score was graded against a landed diff" when in fact
            # nothing measured whether any of them was.
            'plan_quality_judged_without_reference': judged_blind_count(
                acc['plan_quality_judged_without_reference'],
                acc['plan_quality_judged_without_reference_unmeasured'],
            ),
            # How many admitted cells drove that None — 1-of-50 unmeasured and
            # 50-of-50 are very different situations to an operator, and the
            # None alone cannot tell them apart.
            'plan_quality_judged_without_reference_unmeasured': acc[
                'plan_quality_judged_without_reference_unmeasured'
            ],
            # The ADMITTED θ pool's size — the denominator both derived figures
            # below are taken over, carried on the row so each is verifiable
            # from the row ALONE rather than by re-deriving it from the θ
            # table (the module's report-a-rate-beside-its-n norm). It is the
            # same number build_plan_quality_report reports as ``n``.
            'plan_quality_n': n_admitted,
            # RELIABILITY (task 3379): how often this config emitted a plan at
            # all, over what was actually measured. Through THE shared reducer,
            # which build_plan_quality_report's aggregate also calls.
            'plan_rate': plan_rate(n_admitted, acc['plan_quality_no_plan']),
            # $ per USABLE plan (task 3379): the admitted cells' full spend —
            # including what the planless ones burned — over only the cells
            # that produced a plan. Same n_admitted local, so it describes the
            # same pool plan_rate does.
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
    roles: Sequence[str],
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
    5. rows with a ``plan_rate`` before rows without (the same ``None``-last
       rule again, on the tertiary axis);
    6. DESCENDING ``plan_rate`` (task 3379) — HOW OFTEN the config emitted a
       plan at all, breaking a tie of both means above;
    7. ASCENDING ``config`` name, surviving ONLY as the final byte-stability
       guarantee.

    Why 3-4 exist at all: with every architect composite hard-gated to 0.0 (the
    bug task 3099 fixed), the name tiebreak had silently become the ENTIRE
    selection mechanism for the architect role — ``architect-fable-high`` was
    "selected" for sorting first, not for planning best (plans/eval-architect-
    effort-verdict-2026-07-27.md, defect 2). Fixing the composite alone would
    leave the same trap armed for the next pair of genuinely-tied real
    composites, so the alphabet is demoted below every axis that carries signal.
    A workflow row carries no ``plan_quality``, so steps 3-4 are inert for it and
    the existing implementer ordering is unchanged.

    Why 5-6 sit BELOW ``plan_quality`` rather than above it: task 3302 already
    folds every no-plan cell's zero into BOTH ``plan_quality`` (floored to 0.0
    by :func:`plan_quality_score` and KEPT in the mean) and ``composite``
    (hard-gated by ``blend_composite(no_plan=True)``), so an unreliable
    candidate is penalised on both primary axes already; ranking on reliability
    PRIMARILY would charge it a third time for the same cells. Its residual —
    and real — value is breaking a genuine tie of both means: one cell at 0.0
    beside one at 1.0 yields the SAME mean as two cells at 0.5 while the two
    configs differ sharply in how often they plan at all, and without this axis
    that tie falls through to the alphabet — the very defect step 7 exists to
    be a last resort against. A workflow row carries no ``plan_rate`` either, so
    steps 5-6 are likewise inert for it.
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


# ---------------------------------------------------------------------------
# recovery_score surface (eval-revival η)
#
# An ADDITIVE, interim per-(task_id, config_name) column distinct from the Elo
# leaderboard above. It surfaces the ``recovery_score`` that
# ``runner.collect_metrics`` writes into ``EvalResult.metrics`` — a populated
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
    return '\n'.join(render_fixed_table(
        _RECOVERY_COLUMNS, rendered, header='recovery_score report:',
    ))


# ---------------------------------------------------------------------------
# plan_quality surface (eval-revival θ)
#
# An ADDITIVE, interim per-(task_id, config_name, role_under_test) column
# distinct from the Elo leaderboard above — the architect-eval analogue of the
# recovery_score surface. It surfaces the ``plan_quality`` that
# ``run_architect_eval`` writes into ``EvalResult.metrics`` — a populated float
# for architect runs (role_under_test ARCHITECT), the ``None`` null sentinel
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
    # ``declined`` sits IMMEDIATELY AFTER ``no_plan`` (task 4760) because it is
    # the explanation OF that number — how many of the cells that scored zero
    # got there by taking an explicit plan-tools decline exit rather than by
    # failing to plan. Appending it at the far end would separate the count from
    # the one it qualifies, which is the whole defect this column removes.
    'config_name', 'n', 'cap_excluded', 'no_plan', 'declined',
    'judged_without_reference', 'plan_rate', 'mean_plan_quality',
)
# The plan_quality cell of a cap-tainted row. Deliberately NOT a number and NOT
# the bare '-' null sentinel (which already means "not an architect run"): a
# reader must be able to tell "this candidate scored nothing" from "we could not
# measure this candidate" at a glance.
#
# Spelled 'excluded', not 'cap-excluded' (reviewer: design-coherence): the
# ``cap_tainted`` flag covers every unmeasurable cause — cap hit, auth failure,
# model-not-found, server error (5xx), zero-output wedge, harness error — and a
# PERMANENT config error ("this candidate can never run") must not render as a
# TRANSIENT one ("rerun after the cap window"). The cause is always carried in
# the adjacent ``invocation_error`` column (the rendered ``invocation_errors``
# records) and broken out by cause in the summary line, so the marker itself
# stays cause-neutral rather than asserting a cap.
_CAP_EXCLUDED_CELL = 'excluded'
_PLAN_QUALITY_MEAN_HEADER = 'plan_quality by config:'


def format_invocation_errors(
    records: Sequence[UnmeasuredRecord] | None,
) -> str | None:
    """*records* on one line as ``stage:cause: reason``, ``'; '``-joined; ``None`` if none."""
    if not records:
        return None
    return '; '.join(f"{r['stage']}:{r['cause']}: {r['reason']}" for r in records)


def _exclusion_cause(row: dict[str, Any]) -> UnmeasuredCause:
    """Why a cap-tainted *row* was excluded: its first record's cause.

    Architect records precede judge ones and a judge-only failure never
    taints, so the first record is the one that drove the exclusion.
    """
    cell = f'{row["task_id"]} x {row["config_name"]}'
    records = row['invocation_errors']
    if not records:
        raise ValueError(
            f'{cell}: a cap_tainted cell must carry at least one '
            f'invocation_errors record naming why it was not measured, and '
            f'this one carries none. Load cells through ResultStore.load_all, '
            f'which supplies the record for pre-6734 cells.'
        )
    cause = records[0]['cause']
    try:
        return UnmeasuredCause(cause)
    except ValueError:
        raise ValueError(
            f'{cell}: invocation_errors[0] names cause {cause!r}, which is not '
            f'an UnmeasuredCause ({", ".join(c.value for c in UnmeasuredCause)}).'
        ) from None


def build_plan_quality_report(results: list[EvalResult]) -> dict[str, Any]:
    """Build the plan-quality column from a list of :class:`EvalResult` (θ).

    Each row carries ``task_id`` / ``config_name`` / ``role_under_test`` and the
    ``plan_quality`` pulled from ``EvalResult.metrics`` — a populated float for
    architect runs (``role_under_test`` ARCHITECT), ``None`` (the C4
    ``plan_quality | null`` sentinel) otherwise. Both are read directly from the
    persisted metrics dict; a result predating the θ fields (no ``plan_quality``
    / ``role_under_test`` key) reads back as ``None`` for each, so old reports
    render unchanged. Rows are sorted by ``(task_id, config_name)`` so the
    surface is deterministic.

    Rows additionally carry the task-3118 infra markers ``cap_tainted`` /
    ``invocation_errors`` (likewise default-safe: ``False`` / ``None`` for
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
    PLAN (``not plan_predicates.produced_a_plan``) is the opposite case — a genuine
    content measurement — so it is FLOORED to ``0.0`` by
    :func:`plan_quality_score`, KEPT in the mean, and counted separately as
    ``no_plan``. Two causes, two treatments, two counts, neither silent:
    transport refusal → excluded + ``cap_excluded``; content failure → floored +
    ``no_plan``. Conflating them would either let the candidate that failed to
    plan escape the pool entirely or penalise the one that merely hit a cap.

    That ``no_plan`` count now also drives a per-config ``plan_rate`` (task
    3379) — the fraction of admitted cells that emitted a plan at all, reduced
    by THE shared :func:`plan_rate` that :func:`build_composite_report`'s row
    calls, over the SAME admitted ``n`` ``mean_plan_quality`` averages. So the
    reliability figure and the quality figure describe the same pool, and the
    two tables the CLI prints adjacently cannot disagree about either. No new
    per-ROW field: the aggregate reads counts it already holds.

    ``cap_excluded_by_cause`` breaks that total out by CAUSE
    (``{'cap_hit': 2, 'model_not_found': 1}``, key-sorted for determinism)
    because the causes are not interchangeable: a cap hit is transient and
    schedule-attributable ("rerun after the window"), whereas a model-not-found
    or auth failure is a PERMANENT, candidate-specific configuration error. A
    single "cap-excluded" total would let the latter masquerade as the former
    and hide a config that can never run behind ``n=0, mean=None``.

    A THIRD treatment, disjoint from both (eval-revival σ, task 3628):
    ``judged_without_reference`` counts the admitted cells whose score is the
    LLM judge's own number produced from an EMPTY reference diff — graded on
    plausibility, never against the landed change. Unlike the other two this
    changes NO number: the cell is KEPT AND SCORED, stays in ``n``, and its
    score stays in ``mean_plan_quality``. It is a VALIDITY bound, not an
    exclusion. Averaging these cells out would discard real measurements, while
    reporting them silently is the defect σ exists to remove — the v1 campaign
    judged half its hard subset this way and it was findable only by
    archaeology. So: transport refusal → EXCLUDED + ``cap_excluded``; content
    failure → FLOORED + ``no_plan``; judged blind → KEPT, SCORED, and FLAGGED.

    That third count is ``None``, never ``0``, whenever any admitted cell
    predates the marker (:func:`judged_without_reference`), with
    ``judged_without_reference_unmeasured`` carrying how many did — an
    unmeasured bound and a measured-clean one are opposite findings and must
    not render alike.

    A FOURTH treatment (task 4760), which — like the third — subtracts nothing
    and floors nothing, and unlike the third does not even bound a number:
    ``declined`` / ``declined_by_kind`` / ``no_plan_declined`` are EXPLANATORY.
    They move no cell between pools and change no existing figure; they say WHY
    the ``no_plan`` cells are there. ``no_plan`` has always answered HOW MANY
    architect cells emitted no plan and never WHY, and the two causes it
    conflates are opposite verdicts on the candidate: a genuine "could not
    plan" is a content failure, whereas ``report_false_premise`` and its four
    siblings are the architect CORRECTLY refusing an unplannable task. THE
    tranche-1 misreading is the case in point: 47 of 47 ``plan_steps = 0``
    architect cells were explicit, adversarially-verified-true declines, and the
    campaign readout could not say so — it reported an 89% planning failure that
    the transcripts flatly contradicted, and separating the two took manual
    forensics over destroyed-on-cleanup artifacts.

    So: transport refusal → EXCLUDED + ``cap_excluded``; content failure →
    FLOORED + ``no_plan``; judged blind → KEPT, SCORED, FLAGGED; terminal
    decline → KEPT, SCORED, UNCHANGED, and EXPLAINED. ``declined`` counts
    admitted cells whose ``terminal_kind`` is one of the five explicit decline
    exits — including a PLAN-THEN-DECLINE cell, which is in ``declined`` but not
    in ``no_plan``, so ``declined - no_plan_declined`` recovers that population
    exactly. ``declined_by_kind`` is key-sorted for the same reason
    ``cap_excluded_by_cause`` is: the kinds are not interchangeable.
    ``terminal_kind_unmeasured`` counts the admitted cells whose terminal
    statement was never resolved — read through ``terminal_kind_of``, so an
    ABSENT key (the cell predates the field, the only cause the live instrument
    can produce) and an explicit ``None`` count ALIKE — so a PARTIAL split never
    wears the appearance of a complete one.

    ``plan_rate`` and ``mean_plan_quality`` are DELIBERATELY untouched by all of
    this: planRate keeps its exact historical derivation (``plan_steps > 0``
    over the admitted pool) so every campaign artifact already committed stays
    comparable, and the split is reported ALONGSIDE it.
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
            'invocation_errors': result.metrics.get('invocation_errors'),
            # The σ validity marker (task 3628) — read through THE shared
            # accessor, NOT like cap_tainted above. cap_tainted's absence
            # genuinely means "no infra failure was recorded"; this field's
            # absence means the cell's validity was never measured at all, and
            # collapsing that to False is the one mistake that would make the
            # count reassuring instead of true. ``None`` is therefore a third
            # value here and travels all the way to the rendered '-'.
            'judged_without_reference': judged_without_reference(
                result.metrics
            ),
            # The TERMINAL STATEMENT this cell ended on (task 4760), read
            # through THE shared accessor rather than a bare ``.get`` for the
            # same reason ``judged_without_reference`` above is: one expression,
            # one contract, so the report layer and the campaign driver can
            # never disagree about what the field means. ``None`` — an
            # implementer cell (the field does not apply) or a cell persisted
            # before the field existed — travels through to the aggregate's
            # ``terminal_kind_unmeasured``, never collapsed to "did not
            # decline".
            'terminal_kind': terminal_kind_of(result.metrics),
            # The θ score this cell contributes to the aggregate, through THE
            # shared accessor (task 3302): the persisted float for a real plan,
            # a floored 0.0 for a cell that produced none, None when the cell is
            # excluded outright. Kept beside the RAW persisted ``plan_quality``
            # rather than replacing it — the per-row table reports what was
            # persisted, and an operator must still be able to see the judge's
            # number that the floor overrode.
            'scored_plan_quality': plan_quality_score(
                result.metrics,
                where=f'{result.task_id} x {result.config_name}',
            ),
            # The predicate's ANSWER, computed here from the metrics dict its
            # contract is written against — never re-derived downstream from the
            # row (reviewer: robustness). ``plan_steps`` above is carried for the
            # READER; asking ``produced_a_plan(row)`` instead would work only
            # while the row happens to spell that key identically to the metrics
            # dict, and a rename would silently answer False for EVERY row —
            # flooring every scored cell to 0.0 with no exception, no log and no
            # failing test. Sourced beside ``scored_plan_quality`` so the count
            # below and the score above can never describe different cells.
            'produced_a_plan': produced_a_plan(result.metrics),
        })
    rows.sort(key=lambda r: (r['task_id'], r['config_name']))

    # Per-config aggregate over ARCHITECT rows only — an implementer run never
    # invokes the plan judge, so its null score is a different thing entirely
    # and must not dilute the architect mean.
    scored: dict[str, list[float]] = defaultdict(list)
    excluded: dict[str, int] = defaultdict(int)
    no_plan: dict[str, int] = defaultdict(int)
    judged_blind: dict[str, int] = defaultdict(int)
    judged_blind_unmeasured: dict[str, int] = defaultdict(int)
    totals: dict[str, int] = defaultdict(int)
    by_cause: dict[UnmeasuredCause, int] = defaultdict(int)
    # The task-4760 decline split. Four counters, all over the SAME admitted
    # pool ``scored`` / ``no_plan`` / ``judged_blind`` are over.
    declined: dict[str, int] = defaultdict(int)
    declined_by_kind: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    no_plan_declined: dict[str, int] = defaultdict(int)
    terminal_unmeasured: dict[str, int] = defaultdict(int)
    for row in rows:
        if row['role_under_test'] != RoleUnderTest.ARCHITECT:
            continue
        cfg = row['config_name']
        totals[cfg] += 1
        if row['cap_tainted']:
            excluded[cfg] += 1
            by_cause[_exclusion_cause(row)] += 1
        elif row['scored_plan_quality'] is not None:
            # Subscript, not ``.get``: the answer was computed from the metrics
            # dict when the row was built, so a dropped key must raise here
            # rather than read back as "no config produced a plan".
            if not row['produced_a_plan']:
                # DISJOINT from the exclusion above: this cell was measured and
                # kept, at the 0.0 the accessor floored it to (task 3302).
                no_plan[cfg] += 1
            # Counted inside THIS branch on purpose (task 3628): the count is a
            # strict subset of the admitted pool ``n`` reports, never of the
            # excluded cells — a cap-tainted cell has no plan_quality to bound,
            # so it belongs to cap_excluded alone. The keyless case is tallied
            # separately rather than falling into the False arm: it is the pool
            # membership that is measured here, not the validity.
            if row['judged_without_reference'] is None:
                judged_blind_unmeasured[cfg] += 1
            elif row['judged_without_reference']:
                judged_blind[cfg] += 1
            # The decline split (task 4760), counted INSIDE this same branch and
            # for the same reason the two counts above are: it describes the
            # ADMITTED pool ``n`` reports. A cap-tainted cell may well carry a
            # stale terminal kind, but it belongs to ``cap_excluded`` alone —
            # counting it here would describe a cell the mean beside it never
            # saw. The UNMEASURED case is tallied separately rather than falling
            # into the "not a decline" arm: a cell whose terminal kind reads
            # ``None`` — key absent (it predates the field) or written null,
            # which ``terminal_kind_of`` deliberately does not distinguish —
            # never had its terminal statement resolved, and reading that as
            # "did not decline" is precisely the
            # partial-count-wearing-a-complete-count's-clothes defect this
            # split exists to remove.
            kind = row['terminal_kind']
            if kind is None:
                terminal_unmeasured[cfg] += 1
            elif kind in DECLINE_KINDS:
                declined[cfg] += 1
                declined_by_kind[cfg][kind] += 1
                # The INTERSECTION with ``no_plan`` — THE tranche-1 number.
                # Gated on the same predicate ``no_plan`` above is gated on, so
                # the two counts cannot describe different cells; a
                # plan-then-decline cell increments ``declined`` and NOT this.
                if not row['produced_a_plan']:
                    no_plan_declined[cfg] += 1
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
            # WHY those cells are there (task 4760). ``declined`` counts the
            # admitted cells whose terminal statement was one of the five
            # explicit plan-tools decline exits — a CORRECT refusal, not a
            # planning failure — and subtracts nothing: every one of them is
            # still in ``n``, in ``plan_rate`` and in ``mean_plan_quality`` at
            # the score it earned.
            'declined': declined[cfg],
            # …broken out by KIND, key-sorted for determinism, exactly as
            # ``cap_excluded_by_cause`` is: 'already_done' and 'false_premise'
            # are opposite findings about the fixture and must not merge.
            'declined_by_kind': {
                kind: declined_by_kind[cfg][kind]
                for kind in sorted(declined_by_kind[cfg])
            },
            # The intersection of ``declined`` and ``no_plan`` — THE tranche-1
            # sentence ("47 of 47 no-plan cells were explicit declines"). A
            # plan-then-decline cell is in ``declined`` but NOT here, so
            # ``declined - no_plan_declined`` recovers that population exactly.
            'no_plan_declined': no_plan_declined[cfg],
            # How many of those same ``n`` cells carry NO resolved terminal
            # kind — in practice, cells that predate the field — so a PARTIAL
            # split is never read as a complete one. Its siblings above are
            # counts over the MEASURED sub-population only.
            'terminal_kind_unmeasured': terminal_unmeasured[cfg],
            # How many of those same ``n`` scored cells were judged WITHOUT a
            # reference diff (task 3628). Unlike the two counts above this one
            # subtracts nothing and floors nothing — the cells are in ``n`` and
            # in the mean at their real scores; only the confidence is bounded.
            # ``None`` when ANY of those cells predates the marker: the bound
            # is then unknown, and 0 would assert a validity nobody measured.
            'judged_without_reference': judged_blind_count(
                judged_blind[cfg], judged_blind_unmeasured[cfg],
            ),
            # …with the size of the unmeasured population beside it, so the
            # None above is diagnosable from the row alone.
            'judged_without_reference_unmeasured': judged_blind_unmeasured[cfg],
            'total': totals[cfg],
            # The RELIABILITY figure over the very same admitted pool ``n``
            # counts and ``mean_plan_quality`` averages (task 3379), through THE
            # shared :func:`plan_rate` build_composite_report's row also calls —
            # no re-derived arithmetic, so the two adjacent tables cannot drift.
            'plan_rate': plan_rate(len(scored[cfg]), no_plan[cfg]),
            # THE ONE reduction, shared with build_composite_report's row so the
            # two surfaces cannot drift (task 3099, :func:`mean_plan_quality`).
            'mean_plan_quality': mean_plan_quality(scored[cfg]),
        }
        for cfg in sorted(totals)
    ]
    return {
        'rows': rows,
        'configs': configs,
        'cap_excluded': sum(excluded.values()),
        'cap_excluded_by_cause': {c.value: by_cause[c] for c in sorted(by_cause)},
        # Report-level decline totals (task 4760), summed from the very same
        # per-config counters above so the two surfaces cannot drift. No
        # absence rule of their own: unlike ``judged_without_reference``, an
        # unmeasured cell here does not make the campaign-wide count UNKNOWN —
        # it makes it PARTIAL, and ``terminal_kind_unmeasured`` beside it says
        # by how much, which is the honest shape for a count whose
        # sub-population is nameable.
        'declined': sum(declined.values()),
        'declined_by_kind': {
            kind: total
            for kind, total in sorted(_merge_counts(declined_by_kind).items())
        },
        'no_plan_declined': sum(no_plan_declined.values()),
        'terminal_kind_unmeasured': sum(terminal_unmeasured.values()),
        # Report-level total, with the same absence rule as the per-config
        # counts (task 3628): one keyless admitted cell ANYWHERE makes the
        # campaign-wide bound unknown, so the total is None rather than a
        # partial sum wearing the appearance of a complete one.
        'judged_without_reference': judged_blind_count(
            sum(judged_blind.values()), sum(judged_blind_unmeasured.values()),
        ),
        'judged_without_reference_unmeasured': sum(
            judged_blind_unmeasured.values()
        ),
    }


def _format_plan_quality_mean_section(report: dict[str, Any]) -> list[str]:
    """Render the per-config mean block: how many cells actually scored.

    The point of the block is that ``mean_plan_quality`` is reported ALONGSIDE
    the ``n`` it was computed over, the ``cap_excluded`` count it left out and
    the ``no_plan`` count it scored as zeros, so a mean over 19 of 22 cells —
    four of which produced no plan at all — reads as exactly that instead of
    looking like a mean over 22 comparable plans. A config with nothing scored
    renders ``-``, never ``0.0000``.

    ``judged_without_reference`` (task 3628) is the same discipline applied to
    VALIDITY rather than to membership: a mean reported without saying how many
    of its cells were graded with no landed diff to grade against is the same
    silent degradation one layer up. Those cells ARE in ``n`` and in the mean.

    ``declined`` (task 4760) is the same discipline applied to CAUSE: it sits
    immediately after the ``no_plan`` it explains and says how many of those
    zero-scoring cells got there by an explicit plan-tools refusal rather than
    by failing to plan. It adjusts nothing — those cells are in ``n``, in
    ``plan_rate`` and in the mean exactly as before. A measured zero renders
    ``0``, not the ``-`` its neighbours use for "unmeasurable".
    """
    configs = report.get('configs', [])
    rendered = [
        {
            'config_name': str(c['config_name']),
            'n': str(c['n']),
            'cap_excluded': str(c['cap_excluded']),
            'no_plan': str(c['no_plan']),
            # How many of those n cells ended on an explicit decline exit (task
            # 4760). A measured ZERO renders '0', never the '-' sentinel its
            # neighbours use: on this surface '-' already means "not an
            # architect run" / "we could not measure this", and a config that
            # simply never declined is a MEASURED fact about it.
            'declined': str(c['declined']),
            # How many of those n cells the judge scored with NO reference diff
            # (task 3628). Rendered here rather than only in JSON because a
            # mean reported without saying how many of its cells were judged
            # blind is the same silent degradation one layer up. ``-`` — the
            # same sentinel the mean beside it uses — when the count is None
            # because some cell predates the marker; printing 0 there would be
            # the reassuring answer to a question nobody asked the instrument.
            'judged_without_reference': (
                '-' if c.get('judged_without_reference') is None
                else str(c['judged_without_reference'])
            ),
            # The same '-'-when-None rule the mean beside it uses: a config
            # whose every cell was refused has no rate, and a fabricated 0.0
            # would assert a reliability failure we never observed (task 3379).
            'plan_rate': (
                '-' if c.get('plan_rate') is None
                else f'{float(c["plan_rate"]):.4f}'
            ),
            'mean_plan_quality': (
                '-' if c['mean_plan_quality'] is None
                else f'{float(c["mean_plan_quality"]):.4f}'
            ),
        }
        for c in configs
    ]
    return render_fixed_table(
        _PLAN_QUALITY_MEAN_COLUMNS, rendered, header=_PLAN_QUALITY_MEAN_HEADER,
    )


def format_plan_quality_table(report: dict[str, Any]) -> str:
    """Render :func:`build_plan_quality_report` output as a deterministic table.

    A ``plan_quality`` column shows the populated float (4 dp) for architect rows
    and ``-`` (the null sentinel) for non-architect rows; ``role_under_test``
    and ``invocation_error`` (the :func:`format_invocation_errors` rendering of
    the row's records) render ``-`` when ``None``. A CAP-TAINTED row shows the
    explicit ``excluded`` marker in place of a score, alongside the records
    that caused it — an infra failure must never render as
    ``0.0000``, and must stay distinguishable from the ``-`` a non-architect row
    uses.

    Sections follow the rows: the exclusion count — how many architect cells
    were not measurable, BROKEN OUT BY CAUSE so a permanent config error is not
    read as a transient cap window — the task-3628 validity bound, the task-4760
    DECLINE SPLIT (how many of the scored cells ended on an explicit plan-tools
    refusal, broken out by kind, and how much of the ``no_plan`` band that
    accounts for), and the per-config mean block. All are computed from the
    report, so the CLI caller picks them up with no change of its own. The same
    report always renders byte-identically (no wall-clock or dict-order
    dependence).
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
            'invocation_error': _text_cell(
                format_invocation_errors(r.get('invocation_errors'))
            ),
        }
        for r in rows
    ]
    lines = render_fixed_table(
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
    # The σ validity bound beside the exclusion count (task 3628), stated
    # against ITS OWN pool: the SCORED cells, not every architect cell, because
    # that is the population whose plan_quality it bounds. Suppressed only when
    # the bound is MEASURED and zero — a clean campaign must not grow a scary
    # line — which is also what keeps this byte-deterministic.
    #
    # An UNMEASURED bound is emphatically not that case and must print (task
    # 3628 amendment): a rebuild over the pre-σ corpus has no measured cell at
    # all, and suppressing the line there would make "nothing bounded these
    # scores" look exactly like "everything was graded against a landed diff".
    judged_blind = report.get('judged_without_reference', 0)
    unmeasured = report.get('judged_without_reference_unmeasured', 0) or 0
    scored_cells = sum(c['n'] for c in report.get('configs', []))
    if judged_blind is None or unmeasured:
        lines.append(
            f'judged without reference: UNMEASURED — {unmeasured} of '
            f'{scored_cells} scored cell(s) predate the marker, so their '
            f'plan_quality validity is UNKNOWN (not bounded, not clean)'
        )
    elif judged_blind:
        lines.append(
            f'judged without reference: {judged_blind} of {scored_cells} '
            f'scored cell(s) (plan_quality bounded)'
        )
    # The decline split (task 4760), modelled on the ``cap_excluded_by_cause``
    # section above and stated against ITS OWN pool — the SCORED cells, which is
    # the population ``terminal_kind`` was resolved over. It explains the
    # ``no_plan`` count printed below rather than adjusting it: a decline is the
    # architect CORRECTLY refusing an unplannable task, and the tranche-1
    # readout could not say so.
    #
    # Suppressed only when the split is MEASURED and empty (no declines, nothing
    # unmeasured) — the judged-without-reference discipline: a campaign where
    # every candidate planned must not grow a line about refusals it never made,
    # and that is also what keeps this byte-deterministic.
    declined_total = report.get('declined', 0) or 0
    declined_by_kind = report.get('declined_by_kind') or {}
    no_plan_cells = sum(c['no_plan'] for c in report.get('configs', []))
    no_plan_declined = report.get('no_plan_declined', 0) or 0
    terminal_unmeasured = report.get('terminal_kind_unmeasured', 0) or 0
    if declined_total:
        kinds = ', '.join(
            f'{kind}: {n}' for kind, n in sorted(declined_by_kind.items())
        )
        lines.append(
            f'declined: {declined_total} of {scored_cells} scored cell(s) '
            f'ended on an explicit decline exit'
            + (f' ({kinds})' if kinds else '')
            + f' — {no_plan_declined} of {no_plan_cells} no-plan cell(s) were '
            f'a refusal, not a planning failure'
        )
    # An UNMEASURED population is emphatically NOT the suppressible case: a
    # rebuild over the pre-4760 corpus resolved no terminal kind at all, and a
    # silent '0 declines' there would read as "every candidate tried and
    # failed to plan" — the exact reading this split exists to refute.
    if terminal_unmeasured:
        lines.append(
            # "carry no resolved terminal kind", not "predate the field": the
            # count is over ``terminal_kind_of``'s answer, which reads an
            # absent key and an explicit null alike, so naming only the first
            # cause would state more than the number knows. Predating the
            # field IS the only cause the live instrument can produce — the
            # write is unconditional — but the sentence must stay true of the
            # count, not of today's producer.
            f'terminal kind: UNMEASURED — {terminal_unmeasured} of '
            f'{scored_cells} scored cell(s) carry no resolved terminal kind, '
            f'so the decline split is PARTIAL (not measured-clean)'
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
#
# ``pq_no_plan`` (task 3302) is the COUNT beside it, and both belong on THIS
# table because this is the surface ``select_survivors`` ranks on: the two
# treatments of a zero-scoring cell — transport refusal EXCLUDED, no-plan
# FLOORED — are otherwise indistinguishable here, and a column of
# ``plan_quality 0.0000`` would read as "these candidates all planned badly"
# when it means "these candidates produced no plan at all".
#
# ``pq_no_ref`` (task 3628) is the THIRD count, and differs from both above in
# that it changes no number on this row: the flagged cells are in
# ``plan_quality``, in ``composite`` and in ``cost_usd`` at their real values.
# It bounds the CONFIDENCE in ``plan_quality`` — those cells were graded by the
# judge on plausibility, with no landed diff to grade against — so a reader
# ranking on this table can see that a high plan_quality rests partly on scores
# with no ground truth behind them. Beside its two siblings deliberately: the
# three describe one pool and are only interpretable together. It is also the
# one of the three that renders ``-``: unlike an exclusion or a floor, whose
# absence from a cell's metrics really is zero of them, a MISSING validity
# marker means the question was never asked of that cell, and rendering the
# answer 0 there would be a fabrication (task 3628 amendment).
#
# ``plan_rate`` / ``cost_per_plan`` (task 3379) are the DERIVED pair over those
# counts, and belong here for the same reason. ``plan_rate`` is the RELIABILITY
# discriminator: the 2026-07-27 verdict found that what actually separates the
# candidates is how often they emit a plan AT ALL, not how well they plan when
# they do — a comparison an operator could not make from a table of counts
# without dividing by the right denominator by hand. ``cost_per_plan`` is that
# axis priced, and unmasks a cheap candidate's ILLUSORY cost advantage: fable
# read $3.456/fixture beside the opus-max incumbent but $4.731 per usable plan,
# i.e. no cheaper at all while failing to plan 5x as often. Both are on THIS
# table because it is the surface ``select_survivors`` ranks on and the one an
# operator reads to choose between candidates.
# ---------------------------------------------------------------------------

_COMPOSITE_COLUMNS = (
    'config', 'composite', 'quality', 'plan_quality', 'pq_excluded',
    'pq_no_plan', 'pq_no_ref', 'plan_rate', 'cost_usd', 'cost_per_plan',
    'cost_source', 'latency_secs', 'ci95_composite', 'trials', 'fixtures',
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
    (:func:`mean_plan_quality` is shared). ``pq_excluded`` is that config's
    COUNT of unmeasurable (cap-tainted) architect cells; the per-CAUSE breakdown
    that distinguishes a transient cap window from a permanent model-not-found is
    deliberately NOT duplicated here — it lives in
    :func:`format_plan_quality_table`, which the CLI emits alongside this table
    for an architect result set.

    ``pq_no_plan`` (task 3302) is its counterpart: how many of that config's
    architect cells were FLOORED to 0.0 because they produced no plan. Rendered
    here and not only on the θ table because ranking happens on THIS surface —
    without it a row reads ``plan_quality 0.0000`` beside ``pq_excluded 0`` and
    an operator cannot tell a config that planned badly from one that did not
    plan at all. Two causes, two treatments, two counts, neither silent.

    ``plan_rate`` (task 3379) is the RELIABILITY column beside those two counts:
    the fraction of that config's ADMITTED architect cells that emitted a plan
    at all. The 2026-07-27 verdict found that what separates the candidates is
    how OFTEN they plan rather than how well, so this is the discriminator an
    operator most needs — and it renders the same number as the ``plan_rate`` in
    the ``plan_quality by config:`` block beneath (:func:`plan_rate` is
    shared). ``cost_per_plan`` is that axis PRICED: the admitted cells' full
    spend, including what the planless ones burned, over only the cells that
    produced a plan. It sits immediately after ``cost_usd`` because it exists to
    correct it — a candidate reading cheap per fixture can be no cheaper per
    USABLE plan (the doc's fable row: $3.456/fixture, $4.731/plan), and the two
    numbers only make that point side by side.

    ``composite`` / ``quality`` / ``plan_quality`` / ``plan_rate`` /
    ``cost_per_plan`` all render ``-`` when the row measured nothing
    (:func:`optional_float_cell`): "we measured nothing" must never read as "it
    scored nothing" — nor, for a rate, as the perfect 1.0000 a fabricated
    "planned every time" would show.
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
            # NOT the .get-with-0-default its siblings use (task 3628
            # amendment): a row whose count is None — because some admitted
            # cell predates the marker — renders the '-' this table already
            # spells "unmeasured" for every optional float, and a row from a
            # report dict built before the field existed takes the same path.
            # 0 is reserved for a bound that was actually measured at zero.
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
