"""Tests for evals/report.py — the C4 composite report surface (task 2477 λ).

Covers the statistics substrate (mean_ci95, _ratio_score), the per-config
price table, the composite report over the UNION of configs (retiring the
all-tasks-intersection collapse), its deterministic renderer, and the
union-aggregation of Elo ratings.
"""

from __future__ import annotations

from math import sqrt

import pytest

# ---------------------------------------------------------------------------
# Task 2477 step-05: mean_ci95 (Student-t small-sample CI) + _ratio_score
# ---------------------------------------------------------------------------

class TestMeanCI95:
    """Small-sample Student-t two-sided 95% CI, stdlib-only (no scipy/numpy)."""

    def test_three_values_student_t_interval(self):
        from orchestrator.evals.report import mean_ci95

        out = mean_ci95([0.4, 0.5, 0.6])
        assert out['mean'] == pytest.approx(0.5)
        assert out['n'] == 3
        assert out['sufficient'] is True

        # Hand-computed: stdev(ddof=1) == 0.1; t(df=2, 0.975) == 4.303;
        # half-width == 4.303 * 0.1 / sqrt(3) == 0.24843…
        half = 4.303 * 0.1 / sqrt(3)
        assert out['lo'] == pytest.approx(0.5 - half, abs=1e-3)
        assert out['hi'] == pytest.approx(0.5 + half, abs=1e-3)
        # Cross-check the plan's stated endpoints.
        assert out['lo'] == pytest.approx(0.2515, abs=1e-3)
        assert out['hi'] == pytest.approx(0.7485, abs=1e-3)

    def test_single_value_has_no_interval(self):
        from orchestrator.evals.report import mean_ci95

        out = mean_ci95([0.5])
        assert out['mean'] == pytest.approx(0.5)
        assert out['n'] == 1
        assert out['sufficient'] is False
        # No CI for n<2 → the interval collapses to the point estimate.
        assert out['lo'] == pytest.approx(0.5)
        assert out['hi'] == pytest.approx(0.5)

    def test_empty_is_zero_and_insufficient(self):
        from orchestrator.evals.report import mean_ci95

        out = mean_ci95([])
        assert out['mean'] == 0.0
        assert out['n'] == 0
        assert out['sufficient'] is False

    def test_two_values_has_interval_but_insufficient(self):
        """n==2 → a CI IS computed, but 'sufficient' stays False (decision 10)."""
        from orchestrator.evals.report import mean_ci95

        out = mean_ci95([0.4, 0.6])
        assert out['n'] == 2
        assert out['sufficient'] is False
        assert out['lo'] < out['mean'] < out['hi']


class TestRatioScore:
    """best/value normalization, single-config-safe, clamped to [0, 1]."""

    def test_worse_value_scores_below_one(self):
        from orchestrator.evals.report import _ratio_score

        # value twice the best → 0.5
        assert _ratio_score(2.0, 1.0) == pytest.approx(0.5)

    def test_best_equals_value_scores_one(self):
        from orchestrator.evals.report import _ratio_score

        assert _ratio_score(1.0, 1.0) == pytest.approx(1.0)

    def test_nonpositive_best_or_value_is_neutral_one(self):
        """Undefined normalization (single-config / zero denominator) → 1.0."""
        from orchestrator.evals.report import _ratio_score

        assert _ratio_score(1.0, 0.0) == pytest.approx(1.0)
        assert _ratio_score(1.0, -3.0) == pytest.approx(1.0)
        assert _ratio_score(0.0, 1.0) == pytest.approx(1.0)
        assert _ratio_score(-2.0, 1.0) == pytest.approx(1.0)

    def test_result_clamped_to_unit_interval(self):
        from orchestrator.evals.report import _ratio_score

        # A value BETTER than 'best' (best/value > 1) clamps down to 1.0.
        assert _ratio_score(0.5, 1.0) == 1.0


# ---------------------------------------------------------------------------
# Task 2477 step-07: build_price_table — per-config C4 price table
# ---------------------------------------------------------------------------

def _cfg(name, model, role='implementer'):
    from orchestrator.evals.configs import EvalConfig

    return EvalConfig(name=name, backend='claude', model=model, effort='high', role=role)


class TestBuildPriceTable:
    """{config_name: {role: {input_per_1m, output_per_1m}}} pulled from prices
    by model — an unlisted model yields an EXPLICIT unpriced marker, never a
    silently fabricated price.
    """

    def test_priced_configs_map_role_to_price_entry(self):
        from orchestrator.evals.report import build_price_table

        prices = {
            'model-x': {'input_per_1m': 2.0, 'output_per_1m': 8.0},
            'model-y': {'input_per_1m': 1.0, 'output_per_1m': 4.0},
        }
        configs = [
            _cfg('cfg-a', 'model-x', role='implementer'),
            _cfg('cfg-b', 'model-y', role='architect'),
        ]
        table = build_price_table(configs, prices)
        assert table['cfg-a'] == {
            'implementer': {'input_per_1m': 2.0, 'output_per_1m': 8.0},
        }
        assert table['cfg-b'] == {
            'architect': {'input_per_1m': 1.0, 'output_per_1m': 4.0},
        }

    def test_unlisted_model_gets_explicit_unpriced_marker(self):
        from orchestrator.evals.report import build_price_table

        prices = {'model-x': {'input_per_1m': 2.0, 'output_per_1m': 8.0}}
        configs = [_cfg('cfg-z', 'unlisted-model', role='implementer')]
        table = build_price_table(configs, prices)
        entry = table['cfg-z']['implementer']
        # An EXPLICIT marker, NOT a fabricated price.
        assert 'input_per_1m' not in entry
        assert entry.get('source') == 'unpriced'

    def test_accepts_price_entry_objects(self):
        from orchestrator.config import PriceEntry
        from orchestrator.evals.report import build_price_table

        prices = {'model-x': PriceEntry(input_per_1m=2.0, output_per_1m=8.0)}
        table = build_price_table([_cfg('cfg-a', 'model-x')], prices)
        assert table['cfg-a']['implementer'] == {
            'input_per_1m': 2.0, 'output_per_1m': 8.0,
        }

    def test_deterministic_sorted_config_keys(self):
        from orchestrator.evals.report import build_price_table

        prices = {'m': {'input_per_1m': 1.0, 'output_per_1m': 2.0}}
        configs = [_cfg('z', 'm'), _cfg('a', 'm'), _cfg('m', 'm')]
        table = build_price_table(configs, prices)
        assert list(table.keys()) == ['a', 'm', 'z']


# ---------------------------------------------------------------------------
# Task 2478 μ (amend): build_pairwise_price_table — the combined arch+impl
# price table for the end-to-end (matrix/confirm) stages. Keyed to match
# run_end_to_end's ``f'{arch.name}+{impl.name}'`` config_name so the rendered
# price section aligns with the end-to-end composite rows (was individual-keyed).
# ---------------------------------------------------------------------------

class TestBuildPairwisePriceTable:
    """{arch+impl: {architect: entry, implementer: entry}} — one combined-name
    entry per (architect, implementer) pair, carrying BOTH roles' rates.
    """

    def test_pair_keys_combined_name_with_both_roles(self):
        from orchestrator.evals.report import build_pairwise_price_table

        prices = {
            'arch-model': {'input_per_1m': 3.0, 'output_per_1m': 12.0},
            'impl-model': {'input_per_1m': 1.0, 'output_per_1m': 4.0},
        }
        arch = _cfg('architect-x', 'arch-model', role='architect')
        impl = _cfg('impl-y', 'impl-model', role='implementer')
        table = build_pairwise_price_table([(arch, impl)], prices)
        # Key is the combined name run_end_to_end stamps; both roles present.
        assert table == {
            'architect-x+impl-y': {
                'architect': {'input_per_1m': 3.0, 'output_per_1m': 12.0},
                'implementer': {'input_per_1m': 1.0, 'output_per_1m': 4.0},
            }
        }

    def test_unlisted_model_gets_explicit_unpriced_marker(self):
        from orchestrator.evals.report import build_pairwise_price_table

        arch = _cfg('a', 'unlisted', role='architect')
        impl = _cfg('b', 'unlisted', role='implementer')
        table = build_pairwise_price_table([(arch, impl)], {})
        # EXPLICIT marker for each role, never a fabricated price.
        assert table['a+b']['architect'] == {'source': 'unpriced'}
        assert table['a+b']['implementer'] == {'source': 'unpriced'}

    def test_deterministic_sorted_combined_keys(self):
        from orchestrator.evals.report import build_pairwise_price_table

        prices = {'m': {'input_per_1m': 1.0, 'output_per_1m': 2.0}}
        i1 = _cfg('impl-b', 'm', role='implementer')
        pairs = [
            (_cfg('arch-z', 'm', role='architect'), i1),
            (_cfg('arch-a', 'm', role='architect'), i1),
        ]
        table = build_pairwise_price_table(pairs, prices)
        assert list(table.keys()) == ['arch-a+impl-b', 'arch-z+impl-b']


# ---------------------------------------------------------------------------
# Task 2477 step-09: build_composite_report — the C4 per-config composite report
# over the UNION of configs (retiring the all-tasks-intersection collapse).
# ---------------------------------------------------------------------------

def _mresult(
    task_id, config_name, trial, *, quality, cost_usd, duration_ms,
    tests_pass: bool | None = True, cost_source='price_table', recovery_score=None,
    plan_quality=None, role_under_test='implementer', plan_steps=0,
    judge_invocations=0, judge_cost_usd=0.0,
    cap_tainted=False, invocation_error=None,
):
    """Build a synthetic EvalResult with a production-shaped metrics dict.

    ``plan_steps`` is threaded explicitly (task 3302) because it is the
    PLAN-PRODUCTION predicate the report layer reads: a fixture that declares a
    ``plan_quality`` without the step count it came from is itself the
    self-contradictory shape this task removes.
    """
    from orchestrator.evals.metrics import EvalMetrics
    from orchestrator.evals.runner import EvalResult

    m = EvalMetrics(
        tests_pass=tests_pass,
        cost_usd=cost_usd,
        cost_source=cost_source,
        workflow_duration_ms=duration_ms,
        composite_score=quality,          # the PURE quality (compute_composite)
        recovery_score=recovery_score,
        plan_quality=plan_quality,
        role_under_test=role_under_test,
        plan_steps=plan_steps,
        judge_invocations=judge_invocations,
        judge_cost_usd=judge_cost_usd,
        cap_tainted=cap_tainted,
        invocation_error=invocation_error,
    )
    return EvalResult(
        task_id=task_id,
        config_name=config_name,
        outcome='completed',
        metrics=m.to_dict(),
        worktree_path='/tmp/eval',
        trial=trial,
    )


def _union_dataset():
    """Configs A,B across fixtures f1,f2 (3 trials each); C in f1 ONLY (3 trials).

    C is an ARCHITECT (plan-only) row, so the efficiency baselines are keyed on
    ``(fixture, role_group)`` and C normalizes against its OWN group — a
    plan-only cell's cost is one architect invocation, a workflow cell's is a
    full run, so a shared floor would crush the workflow rows (task 3099).

    Per-(fixture, role_group) best cost/latency drives the normalization:
      f1 workflow:  cost A=2 B=4 → best 2 (A); latency A=2 B=4 → best 2 (A)
      f2 workflow:  cost A=2 B=4 → best 2 (A); latency A=2 B=4 → best 2 (A)
      f1 plan_only: cost C=1     → best 1 (C); latency C=1     → best 1 (C)

    Hand-computed per-trial blends (weights .6/.2/.2):
      f1 A: blend(1.0, 2→1.0, 2→1.0)  = 1.0   f2 A: blend(1.0, 1.0, 1.0) = 1.0
      f1 B: blend(1.0, 4→0.5, 4→0.5)  = 0.8   f2 B: blend(1.0, 0.5, 0.5) = 0.8
      f1 C: blend(0.9, 1.0, 1.0, plan_only) = 0.94   ← quality is plan_quality
    Per-config composite means: A=1.0, B=0.8, C=0.94.

    Results are inserted in scrambled config order (C, A, B) to prove the impl
    sorts rows by config name.
    """
    results = []
    for tr in (1, 2, 3):
        results.append(_mresult(
            'f1', 'C', tr, quality=1.0, cost_usd=1.0, duration_ms=1000,
            role_under_test='architect', plan_quality=0.9, plan_steps=6,
        ))
    for tr in (1, 2, 3):
        results.append(_mresult(
            'f1', 'A', tr, quality=1.0, cost_usd=2.0, duration_ms=2000,
            recovery_score=0.42,
        ))
        results.append(_mresult(
            'f2', 'A', tr, quality=1.0, cost_usd=2.0, duration_ms=2000,
            recovery_score=0.42,
        ))
    for tr in (1, 2, 3):
        results.append(_mresult(
            'f1', 'B', tr, quality=1.0, cost_usd=4.0, duration_ms=4000,
        ))
        results.append(_mresult(
            'f2', 'B', tr, quality=1.0, cost_usd=4.0, duration_ms=4000,
        ))
    return results


class TestBuildCompositeReport:
    """C4 per-config composite report over the UNION of configs."""

    def test_union_of_configs_no_intersection_collapse(self):
        from orchestrator.evals.report import build_composite_report

        report = build_composite_report(_union_dataset())
        names = [row['config'] for row in report['configs']]
        # C is present in f1 ONLY, yet MUST survive: this is the union, not the
        # all-tasks-intersection collapse (which would drop C).
        assert names == ['A', 'B', 'C']

    def test_row_schema_and_hand_computed_aggregates(self):
        from orchestrator.evals.report import build_composite_report

        report = build_composite_report(_union_dataset())
        rows = {row['config']: row for row in report['configs']}

        a = rows['A']
        # composite = mean per-(fixture, role_group)-normalized blend. A is the
        # cheapest+fastest WORKFLOW row on both fixtures → mean([1.0]*6). The
        # architect row C no longer sets the workflow floor (task 3099).
        assert a['composite'] == pytest.approx(1.0)
        assert a['quality'] == pytest.approx(1.0)     # mean pure composite_score
        assert a['cost_usd'] == pytest.approx(2.0)    # mean cost
        assert a['latency_secs'] == pytest.approx(2.0)  # mean duration_ms/1000
        assert a['cost_source'] == 'price_table'
        assert a['trials'] == 6
        assert a['fixtures'] == 2
        assert a['tests_pass_rate'] == pytest.approx(1.0)
        # ci95 carries composite/cost/latency sub-dicts, sufficient at >=3 trials
        for axis in ('composite', 'cost', 'latency'):
            assert a['ci95'][axis]['sufficient'] is True
        assert a['ci95']['composite']['mean'] == pytest.approx(1.0)

        b = rows['B']
        assert b['composite'] == pytest.approx(0.8)  # mean([0.8]*6)
        assert b['cost_usd'] == pytest.approx(4.0)
        assert b['latency_secs'] == pytest.approx(4.0)
        assert b['trials'] == 6
        assert b['fixtures'] == 2

        c = rows['C']
        # C is an ARCHITECT row, so it scores through the plan-only path
        # (task 3099): quality is its plan_quality 0.9, not its composite_score.
        # Cheapest+fastest in f1 → both efficiency axes 1.0 →
        # 0.6*0.9 + 0.2*1.0 + 0.2*1.0 == 0.94.
        assert c['composite'] == pytest.approx(0.94)
        assert c['quality'] == pytest.approx(0.9)
        assert c['cost_usd'] == pytest.approx(1.0)
        assert c['trials'] == 3
        assert c['fixtures'] == 1
        assert c['ci95']['composite']['sufficient'] is True

    def test_passthrough_recovery_plan_quality_role(self):
        from orchestrator.evals.report import build_composite_report

        report = build_composite_report(_union_dataset())
        rows = {row['config']: row for row in report['configs']}
        # η/θ passthroughs taken straight from the metrics dict.
        assert rows['A']['recovery_score'] == pytest.approx(0.42)
        assert rows['A']['role_under_test'] == 'implementer'
        assert rows['A']['plan_quality'] is None
        assert rows['C']['role_under_test'] == 'architect'
        assert rows['C']['plan_quality'] == pytest.approx(0.9)

    def test_cap_tainted_first_trial_does_not_blank_the_plan_quality_passthrough(self):
        # The passthrough takes the FIRST trial's metrics, so a config whose
        # first trial was cap-refused would report plan_quality=None despite
        # having healthy trials — the infra failure blanking a real measurement
        # (task 3118). It must skip tainted trials and count the exclusions.
        from orchestrator.evals.report import build_composite_report

        results = [
            _mresult('f1', 'capped-first', 1, quality=1.0, cost_usd=1.0,
                     duration_ms=1000, role_under_test='architect',
                     plan_quality=None, cap_tainted=True,
                     invocation_error='architect:cap_hit: session limit'),
            _mresult('f1', 'capped-first', 2, quality=1.0, cost_usd=1.0,
                     duration_ms=1000, role_under_test='architect',
                     plan_quality=0.82, plan_steps=6),
            _mresult('f1', 'healthy', 1, quality=1.0, cost_usd=1.0,
                     duration_ms=1000, role_under_test='architect',
                     plan_quality=0.55, plan_steps=6),
        ]
        rows = {r['config']: r for r in build_composite_report(results)['configs']}

        assert rows['capped-first']['plan_quality'] == pytest.approx(0.82)
        assert rows['capped-first']['plan_quality_cap_excluded'] == 1
        # A config with no tainted trials keeps today's passthrough exactly.
        assert rows['healthy']['plan_quality'] == pytest.approx(0.55)
        assert rows['healthy']['plan_quality_cap_excluded'] == 0

    def test_cap_exclusion_counter_agrees_with_the_plan_quality_surface(self):
        # The two exclusion counters describe the SAME cells, so they must not
        # disagree: build_plan_quality_report counts ARCHITECT rows only, so a
        # tainted non-architect trial (which has no plan_quality to exclude in
        # the first place) must not inflate the composite counter either
        # (reviewer: docs-accuracy).
        from orchestrator.evals.report import (
            build_composite_report,
            build_plan_quality_report,
        )

        results = [
            _mresult('f1', 'mixed', 1, quality=1.0, cost_usd=1.0,
                     duration_ms=1000, role_under_test='architect',
                     plan_quality=None, cap_tainted=True,
                     invocation_error='architect:cap_hit: session limit'),
            _mresult('f1', 'mixed', 2, quality=1.0, cost_usd=1.0,
                     duration_ms=1000, role_under_test='implementer',
                     plan_quality=None, cap_tainted=True,
                     invocation_error='architect:cap_hit: session limit'),
            _mresult('f1', 'mixed', 3, quality=1.0, cost_usd=1.0,
                     duration_ms=1000, role_under_test='architect',
                     plan_quality=0.6, plan_steps=6),
        ]
        rows = {r['config']: r for r in build_composite_report(results)['configs']}
        plan_report = build_plan_quality_report(results)

        assert rows['mixed']['plan_quality_cap_excluded'] == 1
        assert plan_report['cap_excluded'] == 1
        assert (
            rows['mixed']['plan_quality_cap_excluded']
            == plan_report['cap_excluded']
        )
        # The tainted trial is still skipped as a passthrough SOURCE regardless
        # of role, so a healthy trial's measurement survives.
        assert rows['mixed']['plan_quality'] == pytest.approx(0.6)

    def test_price_table_echo_and_sorted_rows(self):
        from orchestrator.evals.report import build_composite_report

        price_table = {
            'A': {'implementer': {'input_per_1m': 2.0, 'output_per_1m': 8.0}},
        }
        report = build_composite_report(_union_dataset(), price_table=price_table)
        # (5) top-level price_table echoes the passed table; rows are sorted.
        assert report['price_table'] == price_table
        assert report['aggregation'] == 'per_fixture_normalized_mean_ci'
        assert [row['config'] for row in report['configs']] == ['A', 'B', 'C']

    def test_price_table_defaults_to_empty_dict_when_omitted(self):
        from orchestrator.evals.report import build_composite_report

        report = build_composite_report(_union_dataset())
        assert report['price_table'] == {}

    def test_tests_pass_false_trial_zeroes_that_trials_composite(self):
        from orchestrator.evals.report import build_composite_report

        # Config P in g1: 3 trials, identical cost/latency (best == self → each
        # efficiency score 1.0). Trials 1,2 pass → blend 1.0; trial 3 FAILS →
        # hard gate → 0.0. Without the gate all three would be 1.0 (mean 1.0);
        # with it the pool is [1.0, 1.0, 0.0] → mean 2/3.
        results = [
            _mresult('g1', 'P', 1, quality=1.0, cost_usd=1.0, duration_ms=1000,
                     tests_pass=True),
            _mresult('g1', 'P', 2, quality=1.0, cost_usd=1.0, duration_ms=1000,
                     tests_pass=True),
            _mresult('g1', 'P', 3, quality=1.0, cost_usd=1.0, duration_ms=1000,
                     tests_pass=False),
        ]
        report = build_composite_report(results)
        row = report['configs'][0]
        assert row['config'] == 'P'
        assert row['composite'] == pytest.approx(2 / 3)
        assert row['tests_pass_rate'] == pytest.approx(2 / 3)

    def test_judge_invocations_and_cost_are_summed(self):
        from orchestrator.evals.report import build_composite_report

        results = [
            _mresult('f1', 'A', tr, quality=1.0, cost_usd=2.0, duration_ms=2000,
                     judge_invocations=1, judge_cost_usd=0.01)
            for tr in (1, 2, 3)
        ]
        report = build_composite_report(results)
        row = report['configs'][0]
        assert row['judge']['invocations'] == 3
        assert row['judge']['cost_usd'] == pytest.approx(0.03)


# ---------------------------------------------------------------------------
# Task 3099: the PLAN-ONLY composite path.
#
# An architect eval freezes every downstream role, so its cells carry no test
# signal. Scored through the workflow path they ALL collapsed to composite
# 0.0000 — which then made select_survivors' alphabetical tie-break the entire
# architect selection mechanism (plans/eval-architect-effort-verdict-2026-07-27.md,
# defects 1-2). A plan-only row is scored on its θ-rubric plan_quality instead.
# ---------------------------------------------------------------------------

def _arch(task_id, config_name, trial, *, plan_quality, cost_usd, duration_ms,
          cap_tainted=False, plan_steps=6):
    """A plan-only architect cell, shaped as run_architect_eval writes it:
    tests_pass=None (no test signal) and quality carried by plan_quality.

    ``plan_steps`` defaults NONZERO (task 3302): the ordinary architect cell
    these tests describe DID produce a plan, and its ``plan_quality`` is a score
    over that plan's steps. A ``plan_steps=0`` cell is the distinct no-plan
    shape, requested explicitly by the tests that exercise it.
    """
    return _mresult(
        task_id, config_name, trial,
        quality=0.0,                 # composite_score is never set on this path
        cost_usd=cost_usd, duration_ms=duration_ms,
        tests_pass=None, role_under_test='architect',
        plan_quality=plan_quality, plan_steps=plan_steps,
        cap_tainted=cap_tainted,
        invocation_error='architect:cap_hit: session limit' if cap_tainted else None,
    )


def _table_row_cells(text, name, *, section=''):
    """The whitespace-split cells of the rendered row for *name*.

    *section* optionally scopes the search to everything after a section header
    (e.g. ``'plan_quality by config:'``), because one rendering can carry
    several tables whose rows each start with the same config name.
    """
    scoped = text.partition(section)[2] if section else text
    return next(ln.split() for ln in scoped.splitlines()
                if ln.split() and ln.split()[0] == name)


class TestPlanOnlyComposite:
    """A plan-only (architect) row scores on plan_quality, not on tests_pass."""

    def test_plan_only_row_scores_its_plan_quality(self):
        """The reported defect: this row used to report composite 0.0000."""
        from orchestrator.evals.report import build_composite_report

        # Sole config on the fixture → it IS its own cost/latency best → both
        # efficiency axes 1.0. 0.6*0.9 + 0.2*1.0 + 0.2*1.0 == 0.94.
        results = [
            _arch('p1', 'arch-a', tr, plan_quality=0.9, cost_usd=0.3,
                  duration_ms=60000)
            for tr in (1, 2, 3)
        ]
        row = build_composite_report(results)['configs'][0]
        assert row['composite'] == pytest.approx(0.94, abs=1e-4)
        # …and `quality` is the axis that ACTUALLY fed the composite, so the
        # table can never show quality=0.0000 beside a non-zero composite.
        assert row['quality'] == pytest.approx(0.9, abs=1e-4)

    def test_plan_only_rows_rank_by_plan_quality(self):
        """Identical but for plan_quality → DIFFERENT, correctly-ordered
        composites. Every architect row collapsing to 0.0000 is the defect."""
        from orchestrator.evals.report import build_composite_report

        results = [
            _arch('p1', 'arch-good', tr, plan_quality=0.9, cost_usd=0.3,
                  duration_ms=60000)
            for tr in (1, 2, 3)
        ] + [
            _arch('p1', 'arch-weak', tr, plan_quality=0.4, cost_usd=0.3,
                  duration_ms=60000)
            for tr in (1, 2, 3)
        ]
        rows = {r['config']: r
                for r in build_composite_report(results)['configs']}
        # 0.6*0.9 + 0.4 == 0.94   vs   0.6*0.4 + 0.4 == 0.64
        assert rows['arch-good']['composite'] == pytest.approx(0.94, abs=1e-4)
        assert rows['arch-weak']['composite'] == pytest.approx(0.64, abs=1e-4)
        assert rows['arch-good']['composite'] > rows['arch-weak']['composite']

    def test_cap_tainted_trial_is_excluded_not_scored_zero(self):
        """Task 3118's invariant, applied to the number that DRIVES selection.

        A tainted cell measured NOTHING, so averaging it in as 0.0 would
        penalise whichever candidate happened to be scheduled inside a cap
        window. It is excluded from the composite/quality pools — but still
        COUNTED, in `trials` and in `plan_quality_cap_excluded`, so nothing is
        dropped silently.
        """
        from orchestrator.evals.report import build_composite_report

        results = [
            # One tainted (no plan_quality, zero cost — a 429 refusal) + one healthy.
            _arch('p1', 'arch-mixed', 1, plan_quality=None, cost_usd=0.0,
                  duration_ms=0, cap_tainted=True),
            _arch('p1', 'arch-mixed', 2, plan_quality=0.8, cost_usd=0.3,
                  duration_ms=60000),
            # The control: the SAME healthy trial, with no tainted sibling.
            _arch('p1', 'arch-clean', 1, plan_quality=0.8, cost_usd=0.3,
                  duration_ms=60000),
        ]
        rows = {r['config']: r
                for r in build_composite_report(results)['configs']}

        assert rows['arch-mixed']['composite'] == pytest.approx(
            rows['arch-clean']['composite'], abs=1e-9,
        )
        assert rows['arch-mixed']['composite'] == pytest.approx(0.88, abs=1e-4)
        # Counted, not dropped.
        assert rows['arch-mixed']['trials'] == 2
        assert rows['arch-mixed']['plan_quality_cap_excluded'] == 1
        assert rows['arch-clean']['plan_quality_cap_excluded'] == 0

    def test_wholly_unmeasured_config_reports_none_not_zero(self):
        """"We measured nothing" must never read as "it scored nothing"."""
        from orchestrator.evals.report import build_composite_report

        results = [
            _arch('p1', 'arch-dark', tr, plan_quality=None, cost_usd=0.0,
                  duration_ms=0, cap_tainted=True)
            for tr in (1, 2, 3)
        ]
        row = build_composite_report(results)['configs'][0]
        assert row['composite'] is None
        assert row['quality'] is None
        assert row['trials'] == 3
        assert row['plan_quality_cap_excluded'] == 3

    def test_plan_only_row_has_no_fabricated_tests_pass_rate(self):
        """No test ran, so there is no pass RATE — not a 0% one."""
        from orchestrator.evals.report import build_composite_report

        results = [
            _arch('p1', 'arch-a', 1, plan_quality=0.9, cost_usd=0.3,
                  duration_ms=60000),
            _mresult('p2', 'impl-a', 1, quality=1.0, cost_usd=5.0,
                     duration_ms=900000, tests_pass=True),
        ]
        rows = {r['config']: r
                for r in build_composite_report(results)['configs']}
        assert rows['arch-a']['tests_pass_rate'] is None
        # …while a workflow row keeps a real float.
        assert rows['impl-a']['tests_pass_rate'] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Task 3099 (reviewer amendment): an UNMEASURABLE plan-only cell leaves EVERY
# pool, and the decision to drop it has ONE home.
#
# Excluding it from the composite pool alone left the mirror image of the defect
# task 3118 removed: the tainted cell reports $0.00 / 0 ms because it never ran,
# so averaging those into cost/latency handed a schedule-attributable BONUS to
# whichever candidate happened to be scheduled inside a cap window — on the very
# table this task makes the operator's ranking surface.
# ---------------------------------------------------------------------------

class TestUnmeasurableCellLeavesEveryPool:
    """One admission decision, applied to composite / quality / cost / latency."""

    def _mixed_and_clean(self):
        """Two configs with IDENTICAL measured cells; one also has a dead trial."""
        return [
            _arch('p1', 'arch-mixed', 1, plan_quality=None, cost_usd=0.0,
                  duration_ms=0, cap_tainted=True),
            _arch('p1', 'arch-mixed', 2, plan_quality=0.8, cost_usd=0.3,
                  duration_ms=60000),
            _arch('p1', 'arch-clean', 1, plan_quality=0.8, cost_usd=0.3,
                  duration_ms=60000),
        ]

    def test_cost_and_latency_are_not_deflated_by_an_unmeasurable_cell(self):
        """Identical measured cells ⇒ identical cost/latency, cap window or not.

        Averaging the tainted cell's fabricated $0.00 / 0 ms in reported
        arch-mixed as 2x cheaper and 2x faster than arch-clean purely because it
        was scheduled inside a cap window (reviewer: correctness).
        """
        from orchestrator.evals.report import build_composite_report

        rows = {r['config']: r for r in
                build_composite_report(self._mixed_and_clean())['configs']}

        assert rows['arch-mixed']['cost_usd'] == pytest.approx(0.3)
        assert rows['arch-mixed']['latency_secs'] == pytest.approx(60.0)
        assert rows['arch-mixed']['cost_usd'] == rows['arch-clean']['cost_usd']
        assert (rows['arch-mixed']['latency_secs']
                == rows['arch-clean']['latency_secs'])
        # The sample size is still reported honestly — by the columns that exist
        # to report it, not by silently averaging a zero into the cost figure.
        assert rows['arch-mixed']['trials'] == 2
        assert rows['arch-mixed']['plan_quality_cap_excluded'] == 1

    def test_wholly_unmeasured_config_is_not_the_cheapest_row_in_the_table(self):
        """All-tainted ⇒ cost/latency are None ('-'), not 0.0 ("free and instant")."""
        from orchestrator.evals.report import build_composite_report

        results = [
            _arch('p1', 'arch-dark', tr, plan_quality=None, cost_usd=0.0,
                  duration_ms=0, cap_tainted=True)
            for tr in (1, 2, 3)
        ]
        row = build_composite_report(results)['configs'][0]
        assert row['cost_usd'] is None
        assert row['latency_secs'] is None
        assert row['trials'] == 3

    def test_unmeasured_row_has_no_zero_width_interval_at_zero(self):
        """One row must not say both "we measured nothing" and "we measured a
        zero-width interval at zero" (reviewer: correctness)."""
        from orchestrator.evals.report import build_composite_report

        results = [
            _arch('p1', 'arch-dark', tr, plan_quality=None, cost_usd=0.0,
                  duration_ms=0, cap_tainted=True)
            for tr in (1, 2, 3)
        ]
        row = build_composite_report(results)['configs'][0]
        assert row['composite'] is None
        assert row['ci95']['composite'] is None
        assert row['ci95']['cost'] is None
        assert row['ci95']['latency'] is None

    def test_a_tainted_cell_carrying_a_score_is_refused_by_every_surface(self):
        """ONE predicate, so the surfaces cannot disagree (reviewer: robustness).

        ``run_architect_eval`` currently taints exactly when it has no score, so
        the two predicates coincided by COUPLING rather than by construction. A
        hand-edited result, a legacy JSON, or a future taint cause that keeps the
        structural floor would feed a score the plan_quality pool refuses into
        the composite/quality pools — making one row's ``quality`` and
        ``plan_quality`` cells disagree and putting a fabricated number back into
        the figure ``select_survivors`` ranks on.
        """
        from orchestrator.evals.report import (
            build_composite_report,
            build_plan_quality_report,
        )

        results = [
            # Tainted, yet carrying a score AND an outlier cost/latency.
            _arch('p1', 'arch-odd', 1, plan_quality=0.2, cost_usd=9.9,
                  duration_ms=999000, cap_tainted=True),
            _arch('p1', 'arch-odd', 2, plan_quality=0.8, cost_usd=0.3,
                  duration_ms=60000),
        ]
        row = build_composite_report(results)['configs'][0]

        # Only the healthy cell survives: it IS its group's floor, so both
        # efficiency axes are 1.0 → 0.6*0.8 + 0.2 + 0.2 == 0.88.
        assert row['plan_quality'] == pytest.approx(0.8)
        assert row['quality'] == pytest.approx(row['plan_quality'])
        assert row['composite'] == pytest.approx(0.88, abs=1e-4)
        assert row['cost_usd'] == pytest.approx(0.3)
        assert row['latency_secs'] == pytest.approx(60.0)
        assert row['plan_quality_cap_excluded'] == 1
        # …and the θ surface reduces the identical pool, bit for bit.
        assert row['plan_quality'] == (
            build_plan_quality_report(results)['configs'][0]['mean_plan_quality']
        )


# ---------------------------------------------------------------------------
# Task 3099 (reviewer: correctness): the composite row's `plan_quality` is the
# config's POOLED MEAN, not its first untainted trial.
#
# The field was a harmless diagnostic ECHO of one trial until this task promoted
# it to a DECISION surface — select_survivors' steps 3-4 rank on it and
# format_composite_table renders it beside the composite. Every plan_quality
# case that predates this block uses IDENTICAL per-trial scores, which is
# exactly why a trial-1 passthrough survived review, so every case below carries
# per-trial VARIANCE.
# ---------------------------------------------------------------------------

class TestPlanQualityIsTheConfigMean:
    """The composite row's ``plan_quality`` is the config's pooled mean."""

    def _varied(self):
        """One architect config whose trials VARY: tainted, then 0.9, then 0.3.

        Scored pool ``[0.9, 0.3]`` → mean 0.6. A first-scored-trial passthrough
        reports 0.9 instead; averaging the tainted cell in as a zero reports 0.4.
        """
        return [
            _arch('p1', 'arch-varied', 1, plan_quality=None, cost_usd=0.0,
                  duration_ms=0, cap_tainted=True),
            _arch('p1', 'arch-varied', 2, plan_quality=0.9, cost_usd=0.3,
                  duration_ms=60000),
            _arch('p1', 'arch-varied', 3, plan_quality=0.3, cost_usd=0.3,
                  duration_ms=60000),
        ]

    def test_row_reports_the_pooled_mean_not_the_first_scored_trial(self):
        from orchestrator.evals.report import build_composite_report

        row = build_composite_report(self._varied())['configs'][0]
        assert row['plan_quality'] == pytest.approx(0.6)
        # Neither the first SCORED trial (0.9) nor the tainted-as-zero mean (0.4).
        assert row['plan_quality'] != pytest.approx(0.9)
        assert row['plan_quality'] != pytest.approx(0.4)
        # Counted-and-excluded: the unmeasurable cell is still reported, so
        # nothing is silently dropped from the sample it was averaged over.
        assert row['plan_quality_cap_excluded'] == 1
        assert row['trials'] == 3

    def test_agrees_bit_identically_with_the_plan_quality_surface(self):
        """ONE shared reduction, so ``==`` and not ``approx``: the two surfaces
        must be structurally incapable of drifting, not merely close today."""
        from orchestrator.evals.report import (
            build_composite_report,
            build_plan_quality_report,
        )

        results = self._varied()
        row = build_composite_report(results)['configs'][0]
        cfg = build_plan_quality_report(results)['configs'][0]
        assert cfg['config_name'] == row['config']
        assert row['plan_quality'] == cfg['mean_plan_quality']

    def test_the_two_adjacent_tables_render_the_same_cell(self):
        """``_emit_composite_report`` now prints both tables on one screen, so an
        operator must not be able to read contradictory answers to one question.
        """
        from orchestrator.evals.report import (
            _PLAN_QUALITY_MEAN_HEADER,
            build_composite_report,
            build_plan_quality_report,
            format_composite_table,
            format_plan_quality_table,
        )

        results = self._varied()
        composite = _table_row_cells(
            format_composite_table(build_composite_report(results)),
            'arch-varied',
        )
        means = _table_row_cells(
            format_plan_quality_table(build_plan_quality_report(results)),
            'arch-varied', section=_PLAN_QUALITY_MEAN_HEADER,
        )
        # _COMPOSITE_COLUMNS[3] is plan_quality; the mean section's last column
        # is mean_plan_quality.
        assert composite[3] == '0.6000'
        assert means[-1] == '0.6000'
        # …and the single-trial score must not appear in the composite row at all.
        assert '0.9000' not in composite

    def test_quality_and_plan_quality_cells_of_one_row_agree(self):
        """Two cells of the SAME row must not contradict either: for a plan-only
        row ``quality`` is the mean of the axis that fed the composite, which is
        the same pool ``plan_quality`` now reduces."""
        from orchestrator.evals.report import build_composite_report

        row = build_composite_report(self._varied())['configs'][0]
        assert row['quality'] == pytest.approx(row['plan_quality'])

    def test_identical_per_trial_scores_are_unchanged(self):
        """Regression: with no per-trial variance the mean IS the passthrough
        score, and a workflow row still reports ``None``."""
        from orchestrator.evals.report import build_composite_report

        rows = {r['config']: r
                for r in build_composite_report(_union_dataset())['configs']}
        assert rows['C']['plan_quality'] == pytest.approx(0.9)  # three 0.9 cells
        assert rows['A']['plan_quality'] is None                # a workflow row


class TestSurvivorRankingUsesTheConfigMean:
    """A REAL composite tie is broken by the better MEAN plan quality."""

    def _tied(self):
        """Two architect configs on ``p1`` constructed to tie EXACTLY at 0.76.

        ``arch-zeta``  cost 0.3 / 60s — it sets its group's floor, so both
          efficiency axes are 1.0; plan_quality ``[0.9, 0.3]`` → per-trial blends
          ``0.6*0.9 + 0.4 = 0.94`` and ``0.6*0.3 + 0.4 = 0.58`` → composite mean
          0.76, mean plan_quality 0.6.
        ``arch-alpha`` cost 0.5 / 75s — cost score ``0.3/0.5 = 0.6``, latency
          score ``60/75 = 0.8``; plan_quality ``[0.8, 0.8]`` →
          ``0.6*0.8 + 0.2*0.6 + 0.2*0.8 = 0.76`` on both trials → composite mean
          0.76, mean plan_quality 0.8.
        """
        return [
            _arch('p1', 'arch-zeta', 1, plan_quality=0.9, cost_usd=0.3,
                  duration_ms=60000),
            _arch('p1', 'arch-zeta', 2, plan_quality=0.3, cost_usd=0.3,
                  duration_ms=60000),
            _arch('p1', 'arch-alpha', 1, plan_quality=0.8, cost_usd=0.5,
                  duration_ms=75000),
            _arch('p1', 'arch-alpha', 2, plan_quality=0.8, cost_usd=0.5,
                  duration_ms=75000),
        ]

    def test_the_tie_is_exact(self):
        """Pinned separately and EXACTLY so the ranking test below can never pass
        for the wrong reason — a tie that quietly stopped being one."""
        from orchestrator.evals.report import build_composite_report

        rows = {r['config']: r
                for r in build_composite_report(self._tied())['configs']}
        assert rows['arch-zeta']['composite'] == pytest.approx(0.76)
        assert rows['arch-zeta']['composite'] == rows['arch-alpha']['composite']

    def test_better_mean_plan_quality_wins_the_tie(self):
        from orchestrator.evals.report import (
            build_composite_report,
            select_survivors,
        )

        report = build_composite_report(self._tied())
        rows = {r['config']: r for r in report['configs']}
        assert rows['arch-zeta']['composite'] == rows['arch-alpha']['composite']
        # THE DEFECT: on a first-scored-trial passthrough zeta reports 0.9
        # against alpha's 0.8, so the WORSE-mean config wins the tie.
        assert rows['arch-zeta']['plan_quality'] == pytest.approx(0.6)
        assert rows['arch-alpha']['plan_quality'] == pytest.approx(0.8)
        assert select_survivors(report, top_k=1, roles=['architect']) == {
            'architect': ['arch-alpha'],
        }


class TestRoleScopedEfficiencyBaseline:
    """The cost/latency floor is keyed on ``(fixture, role_group)``.

    ``ofat_candidates()`` returns architect + implementer + judge candidates and
    ``run_ofat_stage`` runs them over the SAME fixtures into ONE result list. A
    plan-only cell's cost is a single architect invocation (~$0.30/60s); a
    workflow cell's is a full run (~$5/900s). Sharing one floor per fixture
    therefore both crushes the workflow rows and clamps every plan-only row's
    efficiency axes to 1.0 — making an architect campaign's ranking depend on
    whether unrelated implementer rows happened to be in the same result set.
    """

    def _mixed(self):
        """The shape ofat_candidates() actually produces: one fixture carrying a
        cheap plan-only architect cell alongside expensive workflow cells."""
        return [
            _mresult('m1', 'impl-a', tr, quality=1.0, cost_usd=5.0,
                     duration_ms=900000, tests_pass=True)
            for tr in (1, 2, 3)
        ] + [
            _mresult('m1', 'impl-b', tr, quality=1.0, cost_usd=10.0,
                     duration_ms=1800000, tests_pass=True)
            for tr in (1, 2, 3)
        ]

    def test_plan_only_cell_never_sets_the_workflow_floor(self):
        """Adding an architect cell must not move ANY implementer composite."""
        from orchestrator.evals.report import build_composite_report

        without = {r['config']: r['composite']
                   for r in build_composite_report(self._mixed())['configs']}
        with_arch = {
            r['config']: r['composite']
            for r in build_composite_report(self._mixed() + [
                _arch('m1', 'arch-a', tr, plan_quality=0.9, cost_usd=0.3,
                      duration_ms=60000)
                for tr in (1, 2, 3)
            ])['configs']
        }

        assert without['impl-a'] == with_arch['impl-a']
        assert without['impl-b'] == with_arch['impl-b']
        # …and the workflow rows are still normalized against each other:
        # impl-a is the cheapest+fastest workflow run → 1.0; impl-b is 2x on
        # both axes → 0.6 + 0.2*0.5 + 0.2*0.5 == 0.8.
        assert with_arch['impl-a'] == pytest.approx(1.0)
        assert with_arch['impl-b'] == pytest.approx(0.8)

    def test_plan_only_rows_normalize_against_each_other(self):
        """Two architect configs differing only in cost must get DIFFERENT
        cost-driven composites, not both clamp to 1.0 against implementer cost.
        """
        from orchestrator.evals.report import build_composite_report

        rows = {
            r['config']: r['composite']
            for r in build_composite_report(self._mixed() + [
                _arch('m1', 'arch-cheap', tr, plan_quality=0.8, cost_usd=0.3,
                      duration_ms=60000)
                for tr in (1, 2, 3)
            ] + [
                _arch('m1', 'arch-dear', tr, plan_quality=0.8, cost_usd=0.6,
                      duration_ms=120000)
                for tr in (1, 2, 3)
            ])['configs']
        }

        # 0.6*0.8 + 0.2*1.0 + 0.2*1.0 == 0.88  vs  0.6*0.8 + 0.2*0.5 + 0.2*0.5 == 0.68
        assert rows['arch-cheap'] == pytest.approx(0.88, abs=1e-4)
        assert rows['arch-dear'] == pytest.approx(0.68, abs=1e-4)
        assert rows['arch-cheap'] > rows['arch-dear']


class TestEfficiencyBaselineIgnoresFailingTrials:
    """Amendment (reviewer: correctness): the per-fixture cost/latency floor is
    taken from PASSING trials only, so a cheap-but-WRONG run cannot deflate the
    efficiency scores of the correct configs on that fixture.
    """

    def test_failing_cheap_run_does_not_deflate_passing_config(self):
        from orchestrator.evals.report import build_composite_report

        # Fixture h1: GOOD passes at cost/latency 2.0/2000; BADCHEAP FAILS but is
        # cheaper+faster (1.0/1000). If the failing run set the baseline, GOOD's
        # cost_score/latency_score would each drop to 0.5 → composite 0.8.
        # Restricting the baseline to PASSING trials makes GOOD the best passing
        # run → each efficiency score 1.0 → composite 1.0.
        results = [
            _mresult('h1', 'GOOD', tr, quality=1.0, cost_usd=2.0,
                     duration_ms=2000, tests_pass=True)
            for tr in (1, 2, 3)
        ] + [
            _mresult('h1', 'BADCHEAP', tr, quality=1.0, cost_usd=1.0,
                     duration_ms=1000, tests_pass=False)
            for tr in (1, 2, 3)
        ]
        rows = {r['config']: r
                for r in build_composite_report(results)['configs']}
        assert rows['GOOD']['composite'] == pytest.approx(1.0)
        # The failing config still hard-gates to 0 regardless of its cheapness.
        assert rows['BADCHEAP']['composite'] == pytest.approx(0.0)

    def test_all_failing_fixture_falls_back_to_all_trials_baseline(self):
        """No passing trial on a fixture → baseline falls back to all trials, so
        the report still builds (every such trial hard-gates to 0 anyway)."""
        from orchestrator.evals.report import build_composite_report

        results = [
            _mresult('h2', 'X', tr, quality=1.0, cost_usd=3.0, duration_ms=3000,
                     tests_pass=False)
            for tr in (1, 2, 3)
        ]
        row = build_composite_report(results)['configs'][0]
        assert row['config'] == 'X'
        assert row['composite'] == pytest.approx(0.0)
        assert row['tests_pass_rate'] == pytest.approx(0.0)


class TestCostSourceMixedLabel:
    """Amendment (reviewer: robustness): when a config's trials span more than
    one distinct cost_source, the row-level label is 'mixed' rather than
    silently the first trial's source — since cost_usd is a cross-trial mean.
    """

    def test_mixed_sources_render_as_mixed(self):
        from orchestrator.evals.report import build_composite_report

        results = [
            _mresult('f1', 'M', 1, quality=1.0, cost_usd=2.0, duration_ms=2000,
                     cost_source='price_table'),
            _mresult('f2', 'M', 1, quality=1.0, cost_usd=2.0, duration_ms=2000,
                     cost_source='unpriced_proxy'),
            _mresult('f3', 'M', 1, quality=1.0, cost_usd=2.0, duration_ms=2000,
                     cost_source='cli'),
        ]
        row = build_composite_report(results)['configs'][0]
        assert row['config'] == 'M'
        assert row['cost_source'] == 'mixed'

    def test_single_source_reports_that_source(self):
        from orchestrator.evals.report import build_composite_report

        results = [
            _mresult('f1', 'S', tr, quality=1.0, cost_usd=2.0, duration_ms=2000,
                     cost_source='unpriced_proxy')
            for tr in (1, 2, 3)
        ]
        row = build_composite_report(results)['configs'][0]
        assert row['cost_source'] == 'unpriced_proxy'


# ---------------------------------------------------------------------------
# Task 2477 step-11: format_composite_table — deterministic renderer of the
# C4 composite report (per-config table + a distinct price-table section).
# ---------------------------------------------------------------------------

def _priced_unpriced_report():
    """A report with a PRICED config A and an UNPRICED-proxy config U.

    A's cost_source is 'price_table' and it is listed in the price table; U's
    cost_source is 'unpriced_proxy' and it carries the explicit unpriced marker.
    """
    from orchestrator.evals.report import build_composite_report

    results = [
        _mresult('f1', 'A', tr, quality=1.0, cost_usd=2.0, duration_ms=2000,
                 cost_source='price_table')
        for tr in (1, 2, 3)
    ] + [
        _mresult('f1', 'U', tr, quality=1.0, cost_usd=5.0, duration_ms=5000,
                 cost_source='unpriced_proxy')
        for tr in (1, 2, 3)
    ]
    price_table = {
        'A': {'implementer': {'input_per_1m': 2.0, 'output_per_1m': 8.0}},
        'U': {'implementer': {'source': 'unpriced'}},
    }
    return build_composite_report(results, price_table=price_table)


class TestFormatCompositeTable:
    """Deterministic ljust-width renderer mirroring format_recovery_table."""

    def test_per_config_rows_and_columns(self):
        from orchestrator.evals.report import format_composite_table

        out = format_composite_table(_priced_unpriced_report())
        # A row line per config.
        assert 'A' in out
        assert 'U' in out
        # The required columns are present as headers.
        for header in ('composite', 'quality', 'cost_usd', 'cost_source', 'latency'):
            assert header in out

    def test_ci95_bracket_rendering(self):
        import re

        from orchestrator.evals.report import format_composite_table

        out = format_composite_table(_priced_unpriced_report())
        # A CI95 interval renders as ``[lo, hi]`` (e.g. [1.0000, 1.0000]).
        assert re.search(r'\[\s*-?\d+\.\d+,\s*-?\d+\.\d+\]', out) is not None

    def test_distinct_price_table_section(self):
        from orchestrator.evals.report import format_composite_table

        out = format_composite_table(_priced_unpriced_report())
        lower = out.lower()
        # A distinct 'price table' section.
        assert 'price table' in lower
        # …listing each config→role→input/output per-1M.
        assert 'input_per_1m' in out
        assert 'output_per_1m' in out
        assert 'implementer' in out

    def test_unpriced_config_shows_explicit_marker_not_blank(self):
        from orchestrator.evals.report import format_composite_table

        out = format_composite_table(_priced_unpriced_report())
        # The per-config cost_source column shows the explicit proxy marker…
        assert 'unpriced_proxy' in out
        assert 'price_table' in out
        # …and the price-table section shows the explicit 'unpriced' marker for U
        # rather than a fabricated/blank price.
        assert 'unpriced' in out

    def test_renders_byte_identically_across_two_calls(self):
        from orchestrator.evals.report import format_composite_table

        report = _priced_unpriced_report()
        # No wall-clock / dict-order dependence: same report → identical bytes.
        assert format_composite_table(report) == format_composite_table(report)

    # -- task 3099: the plan-only columns ---------------------------------

    def _arch_report(self):
        from orchestrator.evals.report import build_composite_report

        return build_composite_report(
            [
                _arch('p1', 'arch-good', tr, plan_quality=0.9, cost_usd=0.3,
                      duration_ms=60000)
                for tr in (1, 2, 3)
            ] + [
                _arch('p1', 'arch-weak', tr, plan_quality=0.4, cost_usd=0.3,
                      duration_ms=60000)
                for tr in (1, 2, 3)
            ] + [
                # Every trial tainted → nothing measured at all.
                _arch('p1', 'arch-dark', tr, plan_quality=None, cost_usd=0.0,
                      duration_ms=0, cap_tainted=True)
                for tr in (1, 2, 3)
            ] + [
                # One tainted trial among healthy ones → an exclusion COUNT.
                _arch('p1', 'arch-mixed', 1, plan_quality=None, cost_usd=0.0,
                      duration_ms=0, cap_tainted=True),
                _arch('p1', 'arch-mixed', 2, plan_quality=0.7, cost_usd=0.3,
                      duration_ms=60000),
            ] + [
                _mresult('p2', 'impl-a', 1, quality=1.0, cost_usd=5.0,
                         duration_ms=900000, tests_pass=True),
            ],
        )

    def _row(self, out, config):
        """The rendered line for *config*, split into whitespace-separated cells."""
        return _table_row_cells(out, config)

    def test_header_carries_plan_quality_and_exclusion_columns(self):
        from orchestrator.evals.report import format_composite_table

        out = format_composite_table(self._arch_report())
        header = out.splitlines()[1]
        assert 'plan_quality' in header
        assert 'pq_excluded' in header

    def test_architect_row_renders_plan_quality_and_exclusion_count(self):
        """Asserted by COLUMN INDEX, not membership (reviewer: test-quality).

        A bare ``'1' in row`` is satisfied by the trailing trials/fixtures cells,
        so it would still pass if ``pq_excluded`` regressed to 0 — it pins
        nothing. The index is asserted against ``_COMPOSITE_COLUMNS`` first, so a
        column reorder fails loudly here instead of silently re-aiming every
        positional assertion below at the wrong cell.
        """
        from orchestrator.evals.report import (
            _COMPOSITE_COLUMNS,
            format_composite_table,
        )

        assert _COMPOSITE_COLUMNS[3] == 'plan_quality'
        assert _COMPOSITE_COLUMNS[4] == 'pq_excluded'
        out = format_composite_table(self._arch_report())
        assert self._row(out, 'arch-good')[3] == '0.9000'
        assert self._row(out, 'arch-good')[4] == '0'
        assert self._row(out, 'arch-mixed')[4] == '1'   # the exclusion COUNT
        assert self._row(out, 'arch-dark')[4] == '3'

    def test_non_architect_row_renders_dash_not_a_fabricated_zero(self):
        from orchestrator.evals.report import format_composite_table

        out = format_composite_table(self._arch_report())
        cells = self._row(out, 'impl-a')
        # A workflow row has no plan_quality: '-' , never '0.0000'. Positional:
        # a bare `'-' in cells` would also pass if `composite` wrongly rendered
        # '-' (reviewer: test-quality).
        assert cells[3] == '-'
        assert cells[1] != '-'   # …while its composite IS measured.

    def test_wholly_unmeasured_row_renders_dash_for_every_measured_cell(self):
        """Not just composite/quality: an all-tainted row must not render as the
        cheapest and fastest config in the table, nor carry a zero-width CI."""
        from orchestrator.evals.report import (
            _COMPOSITE_COLUMNS,
            format_composite_table,
        )

        assert _COMPOSITE_COLUMNS[5] == 'cost_usd'
        assert _COMPOSITE_COLUMNS[7] == 'latency_secs'
        assert _COMPOSITE_COLUMNS[8] == 'ci95_composite'
        out = format_composite_table(self._arch_report())
        cells = self._row(out, 'arch-dark')
        # composite and quality are both None → '-'. "We measured nothing" must
        # never render as the 0.0000 that "it scored nothing" would.
        assert cells[1] == '-'
        assert cells[2] == '-'
        assert '0.0000' not in cells[1:3]
        # …and neither may the cost/latency cells (0.0000 / 0.00 would make the
        # config that never ran the cheapest and fastest row on the screen), nor
        # the CI95 cell, which used to render '[0.0000, 0.0000]' right beside the
        # composite '-' (reviewer: correctness).
        assert cells[5] == '-'
        assert cells[7] == '-'
        assert cells[8] == '-'

    def test_cap_window_is_not_rendered_as_a_cost_advantage(self):
        """arch-mixed's measured cell is arch-good's price; only its plan quality
        differs. The table must not present it as half the cost/latency."""
        from orchestrator.evals.report import format_composite_table

        out = format_composite_table(self._arch_report())
        mixed = self._row(out, 'arch-mixed')
        good = self._row(out, 'arch-good')
        assert mixed[5] == good[5] == '0.3000'
        assert mixed[7] == good[7] == '60.00'

    def test_operator_can_rank_architects_from_the_table_alone(self):
        """The acceptance assertion for the reported defect.

        In the 2026-07-27 OFAT every architect row rendered composite 0.0000, so
        the table could not be ranked and the scores had to be recomputed by
        hand from the per-cell result JSONs (defect 1).
        """
        from orchestrator.evals.report import format_composite_table

        out = format_composite_table(self._arch_report())
        good = self._row(out, 'arch-good')[1]
        weak = self._row(out, 'arch-weak')[1]
        assert good == '0.9400'
        assert weak == '0.6400'
        assert float(good) > float(weak)

    def test_plan_only_table_renders_byte_identically(self):
        from orchestrator.evals.report import format_composite_table

        report = self._arch_report()
        assert format_composite_table(report) == format_composite_table(report)


# ---------------------------------------------------------------------------
# _render_fixed_table — the ONE home for the width-computed ljust idiom every
# deterministic table above shares (task 3118 amendment). It was copied five
# times, and that duplication is what let the '-'-not-'0.0000' rstrip fix land
# in only two of the copies. Pinned here so the shared contract can't drift.
# ---------------------------------------------------------------------------

class TestRenderFixedTable:
    _COLUMNS = ('name', 'score')

    def test_widths_are_max_of_header_and_cells(self):
        from orchestrator.evals.report import _render_fixed_table

        lines = _render_fixed_table(
            self._COLUMNS,
            [{'name': 'a-very-long-config', 'score': '1.0000'}],
            header='demo:',
        )
        assert lines[0] == 'demo:'
        # The header cell is padded out to the widest row cell, so the columns
        # line up; the dashes rule matches those same widths.
        assert lines[1].startswith('name'.ljust(len('a-very-long-config')))
        assert len(lines[2].split('  ')[0]) == len('a-very-long-config')
        assert lines[3].startswith('a-very-long-config')

    def test_trailing_padding_is_stripped(self):
        # Invisible trailing whitespace makes an otherwise-identical table
        # differ byte-for-byte, and it hides a line's real ending — the
        # per-config mean block asserts on a trailing '-' meaning "nothing
        # scored", which must never read as 0.0000.
        from orchestrator.evals.report import _render_fixed_table

        lines = _render_fixed_table(
            self._COLUMNS,
            [{'name': 'a', 'score': 'a-long-score-cell'}, {'name': 'b', 'score': '-'}],
            header='demo:',
        )
        assert all(ln == ln.rstrip() for ln in lines)
        assert lines[-1].endswith('-')

    def test_empty_rows_still_render_headers_and_rule(self):
        # No rows must not mean no table: the header/rule still render at the
        # header widths rather than raising on the max() of an empty sequence.
        from orchestrator.evals.report import _render_fixed_table

        lines = _render_fixed_table(self._COLUMNS, [], header='demo:')
        assert lines[0] == 'demo:'
        assert lines[1] == 'name  score'
        assert lines[2] == '----  -----'
        assert len(lines) == 3

    def test_header_is_optional(self):
        from orchestrator.evals.report import _render_fixed_table

        lines = _render_fixed_table(self._COLUMNS, [])
        assert lines[0] == 'name  score'


# ---------------------------------------------------------------------------
# Task 2477 step-13: compute_aggregate_ratings — UNION, not intersection.
# ---------------------------------------------------------------------------

class TestAggregateRatingsUnion:
    """The April all-tasks-INTERSECTION collapse is retired: the aggregate
    leaderboard spans the UNION of configs, each averaged over only the tasks in
    which it actually appears.
    """

    def _state(self):
        from orchestrator.evals.elo import JudgeState, TaskPool

        return JudgeState(per_task={
            't1': TaskPool(ratings={'A': 1100.0, 'B': 1000.0, 'C': 900.0}),
            't2': TaskPool(ratings={'A': 1200.0, 'B': 1050.0}),
        })

    def test_union_includes_config_present_in_only_one_task(self):
        from orchestrator.evals.report import compute_aggregate_ratings

        agg = compute_aggregate_ratings(self._state())
        # C ran ONLY in t1 but MUST survive — no collapse-to-2-entries.
        assert set(agg.keys()) == {'A', 'B', 'C'}

    def test_aggregate_is_mean_over_tasks_where_config_appears(self):
        from orchestrator.evals.report import compute_aggregate_ratings

        agg = compute_aggregate_ratings(self._state())
        assert agg['A'] == pytest.approx(1150.0)   # mean(1100, 1200)
        assert agg['B'] == pytest.approx(1025.0)   # mean(1000, 1050)
        assert agg['C'] == pytest.approx(900.0)    # only t1 → its t1 rating

    def test_empty_state_is_empty(self):
        from orchestrator.evals.elo import JudgeState
        from orchestrator.evals.report import compute_aggregate_ratings

        assert compute_aggregate_ratings(JudgeState(per_task={})) == {}
