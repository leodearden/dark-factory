"""Tests for evals/report.py — the C4 composite report surface (task 2477 λ).

Covers the statistics substrate (mean_ci95, _ratio_score), the per-config
price table, the composite report over the UNION of configs (retiring the
all-tasks-intersection collapse), its deterministic renderer, and the
union-aggregation of Elo ratings.
"""

from __future__ import annotations

import logging
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
    judged_without_reference=False,
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
        judged_without_reference=judged_without_reference,
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

    # -- judged_without_reference (eval-revival σ, task 3628) ----------------
    #
    # The composite twin of the θ-surface count. Same accumulator idiom as
    # plan_quality_cap_excluded, one decisive difference: it excludes nothing.

    def test_judged_without_reference_is_counted_per_config(self):
        from orchestrator.evals.report import build_composite_report

        results = [
            _mresult('f1', 'blind-first', 1, quality=1.0, cost_usd=1.0,
                     duration_ms=1000, role_under_test='architect',
                     plan_quality=0.8, plan_steps=6,
                     judged_without_reference=True),
            _mresult('f2', 'blind-first', 2, quality=1.0, cost_usd=1.0,
                     duration_ms=1000, role_under_test='architect',
                     plan_quality=0.6, plan_steps=6),
            _mresult('f1', 'healthy', 1, quality=1.0, cost_usd=1.0,
                     duration_ms=1000, role_under_test='architect',
                     plan_quality=0.55, plan_steps=6),
        ]
        rows = {r['config']: r for r in build_composite_report(results)['configs']}

        assert rows['blind-first']['plan_quality_judged_without_reference'] == 1
        assert rows['healthy']['plan_quality_judged_without_reference'] == 0

    def test_judged_blind_cell_stays_in_the_plan_quality_pool(self):
        """POOL MEMBERSHIP: bounds validity, does NOT exclude.

        The flagged cell's score is in the mean and the count never exceeds the
        pool it is drawn from — the composite twin of the θ-surface's
        "bounds, does not exclude".
        """
        from orchestrator.evals.report import build_composite_report

        results = [
            _mresult('f1', 'mixed', 1, quality=1.0, cost_usd=1.0,
                     duration_ms=1000, role_under_test='architect',
                     plan_quality=0.9, plan_steps=6,
                     judged_without_reference=True),
            _mresult('f2', 'mixed', 1, quality=1.0, cost_usd=1.0,
                     duration_ms=1000, role_under_test='architect',
                     plan_quality=0.5, plan_steps=6),
        ]
        rows = {r['config']: r for r in build_composite_report(results)['configs']}
        row = rows['mixed']

        # Over BOTH cells (0.9 + 0.5) / 2 — not just the healthy one.
        assert row['plan_quality'] == pytest.approx(0.7)
        assert row['plan_quality_judged_without_reference'] == 1
        assert (
            row['plan_quality_judged_without_reference'] <= row['plan_quality_n']
        )

    def test_judged_blind_counter_agrees_with_the_plan_quality_surface(self):
        """The two accumulators describe the SAME cells and must not drift.

        Follows the cap_excluded agreement assertion's idiom exactly; this is
        the test that stops the composite and θ surfaces diverging.
        """
        from orchestrator.evals.report import (
            build_composite_report,
            build_plan_quality_report,
        )

        results = [
            _mresult('f1', 'mixed', 1, quality=1.0, cost_usd=1.0,
                     duration_ms=1000, role_under_test='architect',
                     plan_quality=0.8, plan_steps=6,
                     judged_without_reference=True),
            _mresult('f2', 'mixed', 1, quality=1.0, cost_usd=1.0,
                     duration_ms=1000, role_under_test='architect',
                     plan_quality=0.6, plan_steps=6),
        ]
        rows = {r['config']: r for r in build_composite_report(results)['configs']}
        plan_report = build_plan_quality_report(results)

        assert rows['mixed']['plan_quality_judged_without_reference'] == 1
        assert plan_report['judged_without_reference'] == 1
        assert (
            rows['mixed']['plan_quality_judged_without_reference']
            == plan_report['judged_without_reference']
        )

    def test_cap_tainted_cell_carrying_the_flag_is_only_cap_excluded(self):
        """DISJOINT: a tainted cell has no plan_quality to bound."""
        from orchestrator.evals.report import build_composite_report

        results = [
            _mresult('f1', 'capped', 1, quality=1.0, cost_usd=1.0,
                     duration_ms=1000, role_under_test='architect',
                     plan_quality=None, cap_tainted=True,
                     invocation_error='architect:cap_hit: session limit',
                     judged_without_reference=True),
            _mresult('f2', 'capped', 1, quality=1.0, cost_usd=1.0,
                     duration_ms=1000, role_under_test='architect',
                     plan_quality=0.7, plan_steps=6),
        ]
        rows = {r['config']: r for r in build_composite_report(results)['configs']}

        assert rows['capped']['plan_quality_cap_excluded'] == 1
        assert rows['capped']['plan_quality_judged_without_reference'] == 0

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
# Task 3302: the PLAN-PRODUCTION predicate applied to the composite pipeline.
#
# run_architect_eval used to call the LLM plan judge with no scorability gate,
# so a HEALTHY architect that produced a stepless artifact persisted the
# self-contradictory cell `cap_tainted=False, plan_steps=0, plan_quality=0.95`.
# The report layer admitted that cell on `plan_quality is not None` and averaged
# the judge's number into every surface that ranks candidates.
#
# The rule, from plans/eval-architect-effort-verdict-2026-07-27.md's own hand
# analysis: a no-plan cell scores 0 (`meanPQ_all`), it is FLOORED and counted —
# never excluded the way a cap-tainted (transport-refused) cell is.
# ---------------------------------------------------------------------------

class TestNoPlanCellScoresZeroNotTheJudgesNumber:
    """A healthy architect that emitted NO plan scores 0.0, not the judge's."""

    def test_no_plan_cell_scores_zero_across_every_surface(self):
        from orchestrator.evals.report import build_composite_report

        # The exact shape the ungated LLM judge writes for a stepless artifact:
        # healthy (not tainted), zero steps, a confident 0.95.
        results = [
            _arch('p1', 'arch-noplan', tr, plan_steps=0, plan_quality=0.95,
                  cost_usd=0.3, duration_ms=60000)
            for tr in (1, 2, 3)
        ]
        row = build_composite_report(results)['configs'][0]

        # meanPQ_all: the cell produced nothing, so it is worth 0 — never 0.95.
        assert row['plan_quality'] == pytest.approx(0.0, abs=1e-9)
        assert row['quality'] == pytest.approx(0.0, abs=1e-9)
        # A no-plan cell scores 0.0 on the quality axis AND on the composite —
        # not 0.0 on quality with full marks on efficiency. The 0.4 this once
        # asserted WAS the 0.2 cost + 0.2 latency credit a cell earned for
        # failing to plan (it is the sole config on its fixture, so the *_all
        # fallback baseline hands it ratios of 1.0 on both axes); the
        # blend_composite(no_plan=True) hard gate removes it.
        assert row['composite'] == 0.0

    def test_no_plan_cell_is_floored_not_excluded(self):
        """A cap-tainted cell is EXCLUDED (we never asked the model); a no-plan
        cell is a genuine CONTENT measurement worth 0.0 and stays in every pool,
        including the cost and latency it really did burn."""
        from orchestrator.evals.report import build_composite_report

        results = [
            _arch('p1', 'arch-noplan', tr, plan_steps=0, plan_quality=0.95,
                  cost_usd=0.3, duration_ms=60000)
            for tr in (1, 2, 3)
        ]
        row = build_composite_report(results)['configs'][0]

        assert row['trials'] == 3
        assert row['plan_quality_cap_excluded'] == 0
        assert row['cost_usd'] == pytest.approx(0.3)
        assert row['latency_secs'] == pytest.approx(60.0)
        assert row['composite'] is not None

    def test_both_surfaces_agree_bit_identically(self):
        """The composite row and the θ table are ONE reduction, so the floor
        must be applied where they share it — not on one side only."""
        from orchestrator.evals.report import (
            build_composite_report,
            build_plan_quality_report,
        )

        results = [
            _arch('p1', 'arch-noplan', tr, plan_steps=0, plan_quality=0.95,
                  cost_usd=0.3, duration_ms=60000)
            for tr in (1, 2, 3)
        ]
        row = build_composite_report(results)['configs'][0]
        theta = build_plan_quality_report(results)['configs'][0]

        assert row['plan_quality'] == theta['mean_plan_quality']
        assert theta['mean_plan_quality'] == pytest.approx(0.0, abs=1e-9)
        # The floored cells were SCORED (as zeros), not excluded, so they count
        # toward the n the mean was taken over.
        assert theta['n'] == 3
        assert theta['cap_excluded'] == 0
        assert theta['total'] == 3

    def test_mixed_pool_averages_the_floor_in(self):
        """One no-plan trial among two real ones: (0.0 + 0.6 + 0.6) / 3."""
        from orchestrator.evals.report import (
            build_composite_report,
            build_plan_quality_report,
        )

        results = [
            _arch('p1', 'arch-mixed', 1, plan_steps=0, plan_quality=0.9,
                  cost_usd=0.3, duration_ms=60000),
            _arch('p1', 'arch-mixed', 2, plan_steps=6, plan_quality=0.6,
                  cost_usd=0.3, duration_ms=60000),
            _arch('p1', 'arch-mixed', 3, plan_steps=6, plan_quality=0.6,
                  cost_usd=0.3, duration_ms=60000),
        ]
        row = build_composite_report(results)['configs'][0]
        theta = build_plan_quality_report(results)['configs'][0]

        assert row['plan_quality'] == pytest.approx(0.4, abs=1e-9)
        assert theta['mean_plan_quality'] == pytest.approx(0.4, abs=1e-9)
        assert row['plan_quality'] == theta['mean_plan_quality']

    def test_a_real_plan_still_reports_its_judged_score(self):
        """The control: the floor fires ONLY on a stepless artifact."""
        from orchestrator.evals.report import build_composite_report

        results = [
            _arch('p1', 'arch-real', tr, plan_steps=6, plan_quality=0.95,
                  cost_usd=0.3, duration_ms=60000)
            for tr in (1, 2, 3)
        ]
        row = build_composite_report(results)['configs'][0]
        assert row['plan_quality'] == pytest.approx(0.95, abs=1e-9)
        assert row['quality'] == pytest.approx(0.95, abs=1e-9)


class TestNoPlanCellsAreCounted:
    """The floor is COUNTED, never silent (loud-over-silent-degradation).

    A mean that silently absorbs zeros for cells that produced nothing reads
    identically to a mean over cells that all planned badly. The two causes get
    two disjoint counts: ``no_plan`` (content failure, floored to 0.0 and kept)
    and ``cap_excluded`` (transport refusal, dropped from the pool entirely).
    """

    @staticmethod
    def _one_no_plan_among_two():
        return [
            _arch('p1', 'arch-mixed', 1, plan_steps=0, plan_quality=0.9,
                  cost_usd=0.3, duration_ms=60000),
            _arch('p1', 'arch-mixed', 2, plan_steps=6, plan_quality=0.6,
                  cost_usd=0.3, duration_ms=60000),
            _arch('p1', 'arch-mixed', 3, plan_steps=6, plan_quality=0.6,
                  cost_usd=0.3, duration_ms=60000),
        ]

    def test_both_surfaces_count_the_floored_cell_and_agree(self):
        from orchestrator.evals.report import (
            build_composite_report,
            build_plan_quality_report,
        )

        results = self._one_no_plan_among_two()
        row = build_composite_report(results)['configs'][0]
        theta = build_plan_quality_report(results)['configs'][0]

        assert row['plan_quality_no_plan'] == 1
        assert theta['no_plan'] == 1
        assert row['plan_quality_no_plan'] == theta['no_plan']

    def test_an_all_planning_config_reports_zero_never_absent(self):
        """A count that vanishes when it is zero cannot be read as 'none'."""
        from orchestrator.evals.report import (
            build_composite_report,
            build_plan_quality_report,
        )

        results = [
            _arch('p1', 'arch-real', tr, plan_steps=6, plan_quality=0.6,
                  cost_usd=0.3, duration_ms=60000)
            for tr in (1, 2, 3)
        ]
        row = build_composite_report(results)['configs'][0]
        theta = build_plan_quality_report(results)['configs'][0]

        assert row['plan_quality_no_plan'] == 0
        assert theta['no_plan'] == 0

    def test_cap_tainted_is_counted_as_excluded_not_as_no_plan(self):
        """Disjoint causes, disjoint counts: a transport refusal never got to
        ask the model, so it is not a cell that 'produced no plan' — it is a
        cell we could not measure at all (task 3118)."""
        from orchestrator.evals.report import (
            build_composite_report,
            build_plan_quality_report,
        )

        results = [
            # Refused at the transport layer: no score, no plan, EXCLUDED.
            _arch('p1', 'arch-both', 1, plan_steps=0, plan_quality=None,
                  cost_usd=0.0, duration_ms=0, cap_tainted=True),
            # Healthy, but produced nothing: FLOORED to 0.0 and kept.
            _arch('p1', 'arch-both', 2, plan_steps=0, plan_quality=0.9,
                  cost_usd=0.3, duration_ms=60000),
            _arch('p1', 'arch-both', 3, plan_steps=6, plan_quality=0.6,
                  cost_usd=0.3, duration_ms=60000),
        ]
        row = build_composite_report(results)['configs'][0]
        theta = build_plan_quality_report(results)['configs'][0]

        assert row['plan_quality_cap_excluded'] == 1
        assert row['plan_quality_no_plan'] == 1
        assert theta['cap_excluded'] == 1
        assert theta['no_plan'] == 1
        # Neither treatment shrinks the sample silently.
        assert row['trials'] == 3
        assert theta['total'] == 3
        # …and the mean is over the two admitted cells: (0.0 + 0.6) / 2.
        assert theta['n'] == 2
        assert theta['mean_plan_quality'] == pytest.approx(0.3, abs=1e-9)

    def test_the_theta_mean_table_renders_the_count(self):
        """mean_plan_quality must be readable beside the no-plan cells it
        scored as zeros, not just the n it averaged."""
        from orchestrator.evals.report import (
            _PLAN_QUALITY_MEAN_HEADER,
            build_plan_quality_report,
            format_plan_quality_table,
        )

        out = format_plan_quality_table(
            build_plan_quality_report(self._one_no_plan_among_two())
        )
        header = _table_row_cells(
            out, 'config_name', section=_PLAN_QUALITY_MEAN_HEADER,
        )
        cells = _table_row_cells(
            out, 'arch-mixed', section=_PLAN_QUALITY_MEAN_HEADER,
        )
        assert 'no_plan' in header
        assert cells[header.index('no_plan')] == '1'

    def test_the_composite_table_renders_the_count(self):
        """The count must reach the RANKING surface, not only the θ table.

        format_composite_table is what select_survivors ranks on and what an
        operator reads to compare candidates; there, a floored 0.0 is otherwise
        indistinguishable from a badly-planned one. ``pq_no_plan`` beside
        ``pq_excluded`` is the second of the two counts the invariant promises
        (reviewer: design-coherence).
        """
        from orchestrator.evals.report import (
            _COMPOSITE_COLUMNS,
            build_composite_report,
            format_composite_table,
        )

        # Pinned by index against the column tuple first, so a reorder fails
        # here rather than silently re-aiming the cell assertion below.
        assert _COMPOSITE_COLUMNS[4] == 'pq_excluded'
        assert _COMPOSITE_COLUMNS[5] == 'pq_no_plan'
        out = format_composite_table(
            build_composite_report(self._one_no_plan_among_two())
        )
        header = out.splitlines()[1].split()
        cells = _table_row_cells(out, 'arch-mixed')

        assert 'pq_no_plan' in header
        assert cells[header.index('pq_no_plan')] == '1'
        # …and the exclusion count stays its own, disjoint cell.
        assert cells[header.index('pq_excluded')] == '0'

    def test_an_all_planning_config_renders_a_zero_count(self):
        """The control: the column is a COUNT, not a marker that only appears
        when it fires — a blank would read as 'not applicable'."""
        from orchestrator.evals.report import (
            build_composite_report,
            format_composite_table,
        )

        results = [
            _arch('p1', 'arch-real', tr, plan_steps=6, plan_quality=0.6,
                  cost_usd=0.3, duration_ms=60000)
            for tr in (1, 2, 3)
        ]
        out = format_composite_table(build_composite_report(results))
        header = out.splitlines()[1].split()
        cells = _table_row_cells(out, 'arch-real')
        assert cells[header.index('pq_no_plan')] == '0'


class TestPlanRateIsTheReliabilityColumn:
    """``plan_rate`` — how OFTEN a config emitted a plan at all (task 3379).

    The 2026-07-27 architect campaign's own verdict
    (``plans/eval-architect-effort-verdict-2026-07-27.md``) found that what
    actually separates the candidates is not how WELL they plan but how often
    they plan at all — and the operator had to hand-compute that figure from the
    per-cell result JSONs because no report surface exposed it. Task 3302
    collected the counts (``plan_quality_no_plan`` beside the admitted θ pool);
    this is the derived ratio over them.
    """

    @staticmethod
    def _one_no_plan_among_three():
        """3 admitted architect cells, ONE of which produced no plan.

        Deliberately the shape of ``TestNoPlanCellsAreCounted.
        _one_no_plan_among_two`` — same config, same costs — so the two
        reliability figures this task adds are read over a dataset whose
        no-plan/θ accounting is already pinned by that class.
        """
        return [
            _arch('p1', 'arch-mixed', 1, plan_steps=0, plan_quality=0.9,
                  cost_usd=0.3, duration_ms=60000),
            _arch('p1', 'arch-mixed', 2, plan_steps=6, plan_quality=0.6,
                  cost_usd=0.3, duration_ms=60000),
            _arch('p1', 'arch-mixed', 3, plan_steps=6, plan_quality=0.6,
                  cost_usd=0.3, duration_ms=60000),
        ]

    @staticmethod
    def _one_tainted_one_no_plan_one_planned():
        """One cell of each kind: REFUSED, admitted-but-planless, planned."""
        return [
            # Refused at the transport layer: never asked, so never answered.
            _arch('p1', 'arch-both', 1, plan_steps=0, plan_quality=None,
                  cost_usd=0.0, duration_ms=0, cap_tainted=True),
            # Healthy architect, stepless artifact: asked, answered nothing.
            _arch('p1', 'arch-both', 2, plan_steps=0, plan_quality=0.9,
                  cost_usd=0.3, duration_ms=60000),
            # Healthy architect, real plan.
            _arch('p1', 'arch-both', 3, plan_steps=6, plan_quality=0.6,
                  cost_usd=0.3, duration_ms=60000),
        ]

    def test_the_rate_is_planned_over_admitted(self):
        from orchestrator.evals.report import build_composite_report

        row = build_composite_report(self._one_no_plan_among_three())['configs'][0]

        # The denominator travels WITH the rate on the row, so the figure is
        # verifiable from the row alone rather than by re-deriving it from a
        # different surface (the module's report-a-rate-beside-its-n norm).
        assert row['plan_quality_n'] == 3
        # Hand-computed: (3 admitted - 1 no-plan) / 3 admitted.
        assert row['plan_rate'] == pytest.approx(0.6667, abs=1e-4)

    def test_a_cap_tainted_cell_leaves_BOTH_numerator_and_denominator(self):
        """THE DENOMINATOR PIN: the rate is over what was actually MEASURED.

        A cap-tainted cell is a transport refusal — we never got to ask the
        model — so it is neither a cell that planned nor a cell that failed to
        plan. Counting it in the denominator (i.e. ranging over ``trials``)
        would report a candidate as LESS RELIABLE for a question it never got
        to answer, purely because it happened to be scheduled inside a session-
        cap window: precisely the schedule-attributable penalty tasks 3118 and
        3099 spent two rounds removing, reintroduced on the surface
        ``select_survivors`` ranks on. It also matches the campaign this
        automates — ``plans/eval-architect-effort-verdict-2026-07-27.md``
        computes its planRate over 19 fixtures, dropping the 3 cap-contaminated
        ones from the denominator, not over all 22.
        """
        from orchestrator.evals.report import build_composite_report

        row = build_composite_report(
            self._one_tainted_one_no_plan_one_planned()
        )['configs'][0]

        assert row['plan_quality_n'] == 2
        assert row['plan_rate'] == pytest.approx(0.5, abs=1e-9)
        # NOT 1/3 — the refused cell is not a failure to plan.
        assert row['plan_rate'] != pytest.approx(1 / 3, abs=1e-4)
        # …and the narrowed denominator is never silent: the sample is still
        # reported honestly by the columns that exist to report it.
        assert row['plan_quality_cap_excluded'] == 1
        assert row['trials'] == 3

    def test_an_all_planning_config_reports_one_never_absent(self):
        """A rate that only appears when it FIRES cannot be read.

        The control for the column being a always-present ratio rather than a
        marker: a blank here would read as "not applicable".
        """
        from orchestrator.evals.report import build_composite_report

        results = [
            _arch('p1', 'arch-real', tr, plan_steps=6, plan_quality=0.6,
                  cost_usd=0.3, duration_ms=60000)
            for tr in (1, 2, 3)
        ]
        row = build_composite_report(results)['configs'][0]

        assert row['plan_quality_n'] == 3
        assert row['plan_rate'] == pytest.approx(1.0, abs=1e-9)

    def test_a_workflow_config_reports_None_never_zero_and_never_one(self):
        """An EMPTY admitted pool is ``None`` — the failure mode is two-sided.

        A fabricated ``0.0`` would slander a config that never ran an architect
        cell ("it never planned"); a fabricated ``1.0`` would flatter it ("it
        always planned"). Only ``None`` says the true thing: we measured
        nothing (``_mean_plan_quality`` / ``_optional_float_cell`` precedent).
        """
        from orchestrator.evals.report import build_composite_report

        results = [
            _mresult('w1', 'impl-only', tr, quality=0.8, cost_usd=5.0,
                     duration_ms=900000)
            for tr in (1, 2, 3)
        ]
        row = build_composite_report(results)['configs'][0]

        assert row['plan_quality_n'] == 0
        assert row['plan_rate'] is None


class TestBothSurfacesReportTheSamePlanRate:
    """The θ aggregate carries ``plan_rate`` too — through THE SAME reducer.

    The CLI prints ``format_composite_table`` and ``format_plan_quality_table``
    adjacently, and ``plan_rate`` is derivable from counts BOTH surfaces already
    hold (``n`` / ``no_plan``). Deriving it twice would leave two tables free to
    answer the reliability question differently — the "two exclusion surfaces
    that disagree are worse than one" hazard this module repeatedly closes, most
    recently by :func:`_mean_plan_quality` (task 3099) and
    :func:`_plan_quality_score` (task 3302).
    """

    def test_the_theta_aggregate_carries_the_same_hand_computed_rate(self):
        from orchestrator.evals.report import build_plan_quality_report

        theta = build_plan_quality_report(
            TestPlanRateIsTheReliabilityColumn._one_no_plan_among_three()
        )['configs'][0]

        # The SAME hand-computed (3 - 1) / 3 the composite row reports.
        assert theta['plan_rate'] == pytest.approx(0.6667, abs=1e-4)

    def test_agrees_bit_identically_with_the_composite_row(self):
        """``==``, not ``approx``: the point is not that the two surfaces round
        to the same place but that they are the same computation."""
        from orchestrator.evals.report import (
            build_composite_report,
            build_plan_quality_report,
        )

        results = TestPlanRateIsTheReliabilityColumn._one_no_plan_among_three()
        row = build_composite_report(results)['configs'][0]
        theta = build_plan_quality_report(results)['configs'][0]

        assert row['plan_rate'] == theta['plan_rate']

    def test_agrees_bit_identically_on_the_cap_tainted_dataset(self):
        """…and agrees where the DENOMINATOR is the interesting part.

        The composite row's exposed denominator IS the θ table's ``n``: a
        reader who checks the rate against either table's sample gets the same
        answer, which is the whole reason ``plan_quality_n`` is on the row.
        """
        from orchestrator.evals.report import (
            build_composite_report,
            build_plan_quality_report,
        )

        results = (
            TestPlanRateIsTheReliabilityColumn
            ._one_tainted_one_no_plan_one_planned()
        )
        row = build_composite_report(results)['configs'][0]
        theta = build_plan_quality_report(results)['configs'][0]

        assert row['plan_rate'] == theta['plan_rate']
        assert theta['n'] == row['plan_quality_n'] == 2

    def test_an_empty_admitted_pool_is_None_on_the_theta_surface_too(self):
        """Every cell refused → nothing measured → no rate, on both surfaces.

        ``mean_plan_quality`` is already ``None`` here; a ``plan_rate`` of 0.0
        beside it would assert a reliability failure the transport layer never
        let us observe.
        """
        from orchestrator.evals.report import (
            build_composite_report,
            build_plan_quality_report,
        )

        results = [
            _arch('p1', 'arch-refused', tr, plan_steps=0, plan_quality=None,
                  cost_usd=0.0, duration_ms=0, cap_tainted=True)
            for tr in (1, 2, 3)
        ]
        theta = build_plan_quality_report(results)['configs'][0]
        row = build_composite_report(results)['configs'][0]

        assert theta['n'] == 0
        assert theta['mean_plan_quality'] is None
        assert theta['plan_rate'] is None
        assert row['plan_rate'] is None


class TestCostPerUsablePlan:
    """``cost_per_plan`` — what a config charges per plan you can actually USE.

    The asymmetry IS the column: the numerator keeps the spend of cells that
    produced NOTHING, the denominator counts only the cells that produced
    something. A no-plan cell burns real budget and returns nothing
    (``plans/eval-architect-effort-verdict-2026-07-27.md`` measured $0.5–$3
    against a real plan's ~$3.7), so a cheap-but-unreliable candidate's
    per-fixture cost advantage is partly ILLUSORY — the doc's own arithmetic put
    fable at $3.456/fixture but $4.731 per usable plan, i.e. no cheaper than the
    opus-max incumbent while failing to plan 5x as often. Netting the failed
    attempts out of the numerator would report exactly the illusion this column
    exists to remove.
    """

    @staticmethod
    def _three_at_30c_one_planless():
        return [
            _arch('p1', 'arch-mixed', 1, plan_steps=0, plan_quality=0.9,
                  cost_usd=0.30, duration_ms=60000),
            _arch('p1', 'arch-mixed', 2, plan_steps=6, plan_quality=0.6,
                  cost_usd=0.30, duration_ms=60000),
            _arch('p1', 'arch-mixed', 3, plan_steps=6, plan_quality=0.6,
                  cost_usd=0.30, duration_ms=60000),
        ]

    def test_the_failed_attempts_spend_stays_in_the_numerator(self):
        from orchestrator.evals.report import build_composite_report

        row = build_composite_report(self._three_at_30c_one_planless())['configs'][0]

        # Hand-computed: (0.30 * 3 admitted) / 2 that planned. You paid for the
        # failed attempt; that is the entire content of "$ per USABLE plan".
        assert row['cost_per_plan'] == pytest.approx(0.45, abs=1e-9)

    def test_it_exceeds_the_plain_cost_mean_whenever_a_cell_failed_to_plan(self):
        """The illusory-cheapness signal, stated as a relation.

        Reading ``cost_usd`` alone, this config looks like a $0.30 candidate.
        Two thirds of the time that is what a plan costs; the other third bought
        nothing, and the gap between the two cells is what an operator comparing
        candidates needs to see.
        """
        from orchestrator.evals.report import build_composite_report

        row = build_composite_report(self._three_at_30c_one_planless())['configs'][0]

        assert row['cost_usd'] == pytest.approx(0.30, abs=1e-9)
        assert row['cost_per_plan'] > row['cost_usd']

    def test_a_cap_tainted_cell_contributes_to_NEITHER_side(self):
        """One admission decision sources both numerator and denominator.

        A refused cell's recorded $0.00 is the price of a run that never
        happened, not a measurement — the same reason ``_is_unmeasurable`` keeps
        it out of the ``cost`` pool. Letting it into the numerator would DILUTE
        $/plan with a free run, reporting the cap window as a discount.
        """
        from orchestrator.evals.report import build_composite_report

        results = [
            _arch('p1', 'arch-both', 1, plan_steps=0, plan_quality=None,
                  cost_usd=0.0, duration_ms=0, cap_tainted=True),
            _arch('p1', 'arch-both', 2, plan_steps=0, plan_quality=0.9,
                  cost_usd=0.30, duration_ms=60000),
            _arch('p1', 'arch-both', 3, plan_steps=6, plan_quality=0.6,
                  cost_usd=0.30, duration_ms=60000),
        ]
        row = build_composite_report(results)['configs'][0]

        # (0.30 + 0.30) / 1 planned — NOT (0.00 + 0.30 + 0.30) / 1.
        assert row['cost_per_plan'] == pytest.approx(0.60, abs=1e-9)

    def test_a_config_that_never_planned_reports_None_not_zero(self):
        """"We got no plan at any price" must not render as "plans were free".

        Nor may it raise: a config that failed to plan on every cell is a real
        campaign outcome, and it is the one whose row an operator most needs.
        """
        from orchestrator.evals.report import build_composite_report

        results = [
            _arch('p1', 'arch-planless', tr, plan_steps=0, plan_quality=0.9,
                  cost_usd=0.30, duration_ms=60000)
            for tr in (1, 2, 3)
        ]
        row = build_composite_report(results)['configs'][0]

        assert row['plan_quality_n'] == 3
        assert row['plan_rate'] == pytest.approx(0.0, abs=1e-9)
        assert row['cost_per_plan'] is None

    def test_a_workflow_config_reports_None(self):
        """No plan-only cell at all → no $/plan, rather than a fabricated 0.0."""
        from orchestrator.evals.report import build_composite_report

        results = [
            _mresult('w1', 'impl-only', tr, quality=0.8, cost_usd=5.0,
                     duration_ms=900000)
            for tr in (1, 2, 3)
        ]
        row = build_composite_report(results)['configs'][0]

        assert row['cost_per_plan'] is None


class TestTheReliabilityColumnsAreRendered:
    """Both figures must reach the surfaces an OPERATOR reads.

    ``format_composite_table`` is what ``select_survivors`` ranks on and what a
    human compares candidates with; a figure that exists only in the row JSON
    leaves the 2026-07-27 operator exactly where they started — recomputing it
    by hand from the per-cell result JSONs.
    """

    def test_the_composite_table_renders_both_new_columns(self):
        from orchestrator.evals.report import (
            _COMPOSITE_COLUMNS,
            build_composite_report,
            format_composite_table,
        )

        # Pin the column tuple FIRST, so a reorder fails here rather than
        # silently re-aiming the cell assertions below (the idiom
        # TestNoPlanCellsAreCounted established). The pre-existing pins must
        # still hold: plan_rate is INSERTED after them, not in place of them.
        assert _COMPOSITE_COLUMNS[4] == 'pq_excluded'
        assert _COMPOSITE_COLUMNS[5] == 'pq_no_plan'
        assert _COMPOSITE_COLUMNS[7] == 'plan_rate'
        # $/plan reads beside the plain cost it corrects, not across the table.
        assert (
            _COMPOSITE_COLUMNS[_COMPOSITE_COLUMNS.index('cost_usd') + 1]
            == 'cost_per_plan'
        )

        out = format_composite_table(build_composite_report(
            TestCostPerUsablePlan._three_at_30c_one_planless()
        ))
        header = out.splitlines()[1].split()
        cells = _table_row_cells(out, 'arch-mixed')

        assert cells[header.index('plan_rate')] == '0.6667'
        assert cells[header.index('cost_per_plan')] == '0.4500'
        # …and it reads as dearer than the $0.30/cell it would look like from
        # the cost column alone.
        assert cells[header.index('cost_usd')] == '0.3000'

    def test_a_workflow_row_renders_dashes_in_both(self):
        """Never ``0.0000`` and never ``1.0000``: a cell that measured nothing
        must not read as one that scored zero — nor, here, as a perfect one."""
        from orchestrator.evals.report import (
            build_composite_report,
            format_composite_table,
        )

        results = [
            _mresult('w1', 'impl-only', tr, quality=0.8, cost_usd=5.0,
                     duration_ms=900000)
            for tr in (1, 2, 3)
        ]
        out = format_composite_table(build_composite_report(results))
        header = out.splitlines()[1].split()
        cells = _table_row_cells(out, 'impl-only')

        assert cells[header.index('plan_rate')] == '-'
        assert cells[header.index('cost_per_plan')] == '-'

    def test_an_all_planning_config_renders_a_full_rate(self):
        """The control: a rate that only appears when it FIRES cannot be read,
        so the column is present-and-1.0000, not blank."""
        from orchestrator.evals.report import (
            build_composite_report,
            format_composite_table,
        )

        results = [
            _arch('p1', 'arch-real', tr, plan_steps=6, plan_quality=0.6,
                  cost_usd=0.30, duration_ms=60000)
            for tr in (1, 2, 3)
        ]
        out = format_composite_table(build_composite_report(results))
        header = out.splitlines()[1].split()
        cells = _table_row_cells(out, 'arch-real')

        assert cells[header.index('plan_rate')] == '1.0000'
        # Every cell planned, so $/plan IS the per-cell cost.
        assert cells[header.index('cost_per_plan')] == '0.3000'

    def test_the_theta_mean_table_renders_the_rate(self):
        """On the θ surface too, beside the n / cap_excluded / no_plan it was
        computed over — the module's report-a-figure-with-its-sample norm."""
        from orchestrator.evals.report import (
            _PLAN_QUALITY_MEAN_COLUMNS,
            _PLAN_QUALITY_MEAN_HEADER,
            build_plan_quality_report,
            format_plan_quality_table,
        )

        assert 'plan_rate' in _PLAN_QUALITY_MEAN_COLUMNS
        out = format_plan_quality_table(build_plan_quality_report(
            TestPlanRateIsTheReliabilityColumn._one_no_plan_among_three()
        ))
        header = _table_row_cells(
            out, 'config_name', section=_PLAN_QUALITY_MEAN_HEADER,
        )
        cells = _table_row_cells(
            out, 'arch-mixed', section=_PLAN_QUALITY_MEAN_HEADER,
        )

        assert cells[header.index('plan_rate')] == '0.6667'
        # The sample it was computed over stays legible beside it.
        assert cells[header.index('n')] == '3'
        assert cells[header.index('no_plan')] == '1'

    def test_rendering_is_byte_deterministic(self):
        """The widened tuples must not cost the surface its byte-stability."""
        from orchestrator.evals.report import (
            build_composite_report,
            build_plan_quality_report,
            format_composite_table,
            format_plan_quality_table,
        )

        results = TestCostPerUsablePlan._three_at_30c_one_planless()
        composite = build_composite_report(results)
        theta = build_plan_quality_report(results)

        assert format_composite_table(composite) == format_composite_table(composite)
        assert (
            format_plan_quality_table(theta) == format_plan_quality_table(theta)
        )


class TestTheDiscardedJudgeScoreIsLoggedNotSwallowed:
    """The floor DISCARDS a persisted LLM-judge score — loudly, or not at all.

    `_plan_quality_score`'s warning is the ONLY runtime signal that a number
    written to disk was overridden at read time, and the two-scorer disagreement
    it reports (Graphiti e2066ec6) is exactly what an operator needs to see. An
    untested log line is a log line a refactor deletes (reviewer: test-coverage);
    these also exercise the `where=` plumbing that exists solely to serve it.
    """

    _LOGGER = 'orchestrator.evals.report'

    @staticmethod
    def _floor_records(caplog):
        return [r for r in caplog.records if 'Plan-quality floor' in r.getMessage()]

    def test_a_discarded_nonzero_score_warns_once_and_names_the_cell(self, caplog):
        from orchestrator.evals.report import build_composite_report

        results = [
            _arch('p1', 'arch-noplan', 1, plan_steps=0, plan_quality=0.9,
                  cost_usd=0.3, duration_ms=60000),
            _arch('p1', 'arch-noplan', 2, plan_steps=6, plan_quality=0.6,
                  cost_usd=0.3, duration_ms=60000),
        ]
        with caplog.at_level(logging.WARNING, logger=self._LOGGER):
            build_composite_report(results)

        records = self._floor_records(caplog)
        # ONE build, ONE floored cell, ONE warning — the second (planning) trial
        # must not warn, and the floored one must not warn twice per build.
        assert len(records) == 1
        message = records[0].getMessage()
        # The `where=` plumbing: fixture x config, so the line identifies the
        # cell an operator has to go and look at. Rendered through getMessage(),
        # so a mis-shaped lazy %s arg fails here rather than shipping green.
        assert 'p1 x arch-noplan' in message
        # …and it reports the number that was DISCARDED, not just that one was.
        assert '0.9' in message
        assert records[0].levelno == logging.WARNING

    def test_a_cell_that_scored_zero_anyway_does_not_warn(self, caplog):
        """Nothing was discarded: the judge and the floor AGREE at 0.0, so there
        is no two-scorer disagreement to report and the line would be noise."""
        from orchestrator.evals.report import build_composite_report

        results = [
            _arch('p1', 'arch-noplan', tr, plan_steps=0, plan_quality=0.0,
                  cost_usd=0.3, duration_ms=60000)
            for tr in (1, 2)
        ]
        with caplog.at_level(logging.WARNING, logger=self._LOGGER):
            build_composite_report(results)

        assert self._floor_records(caplog) == []

    def test_a_real_plan_never_warns(self, caplog):
        """The control: the floor — and its warning — fire ONLY on a stepless
        artifact, so a healthy campaign's log stays readable."""
        from orchestrator.evals.report import (
            build_composite_report,
            build_plan_quality_report,
        )

        results = [
            _arch('p1', 'arch-real', tr, plan_steps=6, plan_quality=0.9,
                  cost_usd=0.3, duration_ms=60000)
            for tr in (1, 2, 3)
        ]
        with caplog.at_level(logging.WARNING, logger=self._LOGGER):
            build_composite_report(results)
            build_plan_quality_report(results)

        assert self._floor_records(caplog) == []

    def test_the_theta_surface_warns_on_its_own_read(self, caplog):
        """Each surface reports the floor decisions IT made: the accessor is
        pure, so the warning belongs to the build that read the cell. The CLI
        builds both, so a floored cell warns once per surface — documented on
        :func:`_plan_quality_score`, and pinned here so the count is a decision
        rather than an accident.
        """
        from orchestrator.evals.report import build_plan_quality_report

        results = [
            _arch('p1', 'arch-noplan', 1, plan_steps=0, plan_quality=0.9,
                  cost_usd=0.3, duration_ms=60000),
        ]
        with caplog.at_level(logging.WARNING, logger=self._LOGGER):
            build_plan_quality_report(results)

        records = self._floor_records(caplog)
        assert len(records) == 1
        assert 'p1 x arch-noplan' in records[0].getMessage()


class TestNoPlanCellCannotSetTheEfficiencyFloor:
    """A cell that returned NOTHING must not become the cost/latency baseline.

    plans/eval-architect-effort-verdict-2026-07-27.md measured a no-plan cell at
    $0.5-$3 against a real plan's ~$3.7: it is cheap and fast precisely BECAUSE
    it failed. Letting it seed its group's floor hands the failing candidate a
    "2x cheaper, 2x faster" bonus and deflates every candidate that succeeded —
    the same rule already applied to a failing workflow trial, applied to the
    plan-only group.
    """

    @staticmethod
    def _two_configs():
        return [
            # Cheap and fast because it returned nothing.
            _arch('p1', 'arch-noplan', tr, plan_steps=0, plan_quality=0.95,
                  cost_usd=0.3, duration_ms=60000)
            for tr in (1, 2, 3)
        ] + [
            _arch('p1', 'arch-real', tr, plan_steps=6, plan_quality=0.7,
                  cost_usd=1.2, duration_ms=240000)
            for tr in (1, 2, 3)
        ]

    def test_the_plan_producing_config_is_its_own_floor(self):
        from orchestrator.evals.report import build_composite_report

        rows = {r['config']: r
                for r in build_composite_report(self._two_configs())['configs']}

        # arch-real is the only plan-producing cell, so it IS its
        # (fixture, plan_only) floor: 0.6*0.7 + 0.2*1.0 + 0.2*1.0.
        assert rows['arch-real']['composite'] == pytest.approx(0.82, abs=1e-4)
        # …and explicitly NOT the deflated number a no-plan floor produces:
        # 0.6*0.7 + 0.2*(0.3/1.2) + 0.2*(60/240).
        assert rows['arch-real']['composite'] != pytest.approx(0.52, abs=1e-4)

    def test_the_no_plan_cells_spend_is_still_reported(self):
        """Barred from the BASELINE, not from the pools: it really did burn
        budget, and dropping that would hide what the failure cost."""
        from orchestrator.evals.report import build_composite_report

        rows = {r['config']: r
                for r in build_composite_report(self._two_configs())['configs']}

        assert rows['arch-noplan']['cost_usd'] == pytest.approx(0.3)
        assert rows['arch-noplan']['latency_secs'] == pytest.approx(60.0)
        assert rows['arch-noplan']['trials'] == 3

    def test_survivor_selection_promotes_the_config_that_planned(self):
        """The end-to-end consequence, through the real pipeline path."""
        from orchestrator.evals.report import (
            build_composite_report,
            select_survivors,
        )

        report = build_composite_report(self._two_configs())
        assert select_survivors(report, top_k=1, roles=['architect']) == {
            'architect': ['arch-real'],
        }


class TestNoPlanCellCannotOutrankAPlanProducingConfig:
    """The CROSS-FIXTURE route to the same defect (task 3302 review).

    Barring a no-plan cell from SEEDING its group's efficiency floor closes
    only the INTRA-group route. Flooring its quality axis bounds 0.6 of the
    composite weight; the remaining 0.2 cost + 0.2 latency is still collected,
    and a cell that is the sole member of its ``(fixture, 'plan_only')`` group
    takes the ``*_all`` fallback baseline — earning ratios of 1.0 on both axes
    and banking the full 0.40 for having produced NOTHING.

    Measured on the pre-fix HEAD: A = 0.40 outranked B = 0.26 (a real 6-step
    plan) and ``select_survivors(top_k=2)`` returned ``['C', 'A']`` — the
    config that produced NO PLAN AT ALL survived and the one that produced a
    real (if mediocre) plan was cut. That is precisely the "schedule-independent
    2x cheaper, 2x faster bonus to the candidate that FAILED to plan" this task
    exists to remove.
    """

    @staticmethod
    def _counterexample():
        return [
            # Fixture p1: the sole config there, so its group's floor is its
            # OWN cost/latency via the *_all fallback → ratios 1.0 / 1.0.
            *(_arch('p1', 'A', tr, plan_steps=0, plan_quality=0.95,
                    cost_usd=0.3, duration_ms=60000)
              for tr in (1, 2, 3)),
            # Fixture p2: a real but mediocre plan, expensive and slow …
            *(_arch('p2', 'B', tr, plan_steps=6, plan_quality=0.35,
                    cost_usd=4.0, duration_ms=400000)
              for tr in (1, 2, 3)),
            # … against a real, good, cheap and fast plan (its group's floor).
            *(_arch('p2', 'C', tr, plan_steps=6, plan_quality=0.90,
                    cost_usd=0.5, duration_ms=50000)
              for tr in (1, 2, 3)),
        ]

    def _rows(self):
        from orchestrator.evals.report import build_composite_report

        report = build_composite_report(self._counterexample())
        return report, {r['config']: r for r in report['configs']}

    def test_the_no_plan_config_banks_no_efficiency_credit(self):
        """Hard-gated: a cell that produced nothing scores 0.0, not 0.40."""
        _, rows = self._rows()

        assert rows['A']['composite'] == 0.0

    def test_the_fix_touches_only_the_no_plan_cell(self):
        """Both plan-producing composites are exact DEFAULT_COMPOSITE_WEIGHTS
        identities, unchanged before and after the gate."""
        _, rows = self._rows()

        # 0.6*0.35 + 0.2*(0.5/4.0) + 0.2*(50/400) == 0.21 + 0.025 + 0.025
        assert rows['B']['composite'] == pytest.approx(0.26, abs=1e-4)
        # 0.6*0.90 + 0.2*1.0 + 0.2*1.0 — C is its own group's floor.
        assert rows['C']['composite'] == pytest.approx(0.94, abs=1e-4)

    def test_survivor_selection_cuts_the_config_that_produced_nothing(self):
        """The end-to-end consequence, through the real ranking path.

        Today this returns ``['C', 'A']``: the no-plan config survives and B —
        which produced a real plan — is cut.
        """
        from orchestrator.evals.report import select_survivors

        report, _ = self._rows()
        assert select_survivors(report, top_k=2, roles=['architect']) == {
            'architect': ['C', 'B'],
        }
        assert select_survivors(report, top_k=1, roles=['architect']) == {
            'architect': ['C'],
        }

    def test_the_no_plan_cell_is_gated_not_excluded(self):
        """Its real spend is still reported: it stays in `trials`, in the
        cost/latency pools, and is counted — only the composite is gated."""
        _, rows = self._rows()

        assert rows['A']['trials'] == 3
        assert rows['A']['cost_usd'] == pytest.approx(0.3)
        assert rows['A']['latency_secs'] == pytest.approx(60.0)
        assert rows['A']['plan_quality_no_plan'] == 3
        assert rows['A']['composite'] is not None


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

        # Shifted right by task 3379, which inserted `plan_rate` at 6 and
        # `cost_per_plan` after `cost_usd`, then again by task 3628's `pq_no_ref`
        # at 6. The pins are what MADE each shift visible instead of silently
        # re-aiming the cell assertions below at the new columns.
        assert _COMPOSITE_COLUMNS[8] == 'cost_usd'
        assert _COMPOSITE_COLUMNS[11] == 'latency_secs'
        assert _COMPOSITE_COLUMNS[12] == 'ci95_composite'
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
        assert cells[8] == '-'
        assert cells[11] == '-'
        assert cells[12] == '-'

    def test_cap_window_is_not_rendered_as_a_cost_advantage(self):
        """arch-mixed's measured cell is arch-good's price; only its plan quality
        differs. The table must not present it as half the cost/latency."""
        from orchestrator.evals.report import format_composite_table

        out = format_composite_table(self._arch_report())
        mixed = self._row(out, 'arch-mixed')
        good = self._row(out, 'arch-good')
        # cost_usd / latency_secs, at the indices the sibling test pins (shifted
        # right by task 3379's plan_rate / cost_per_plan insertion, then by task
        # 3628's pq_no_ref).
        assert mixed[8] == good[8] == '0.3000'
        assert mixed[11] == good[11] == '60.00'

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


# ---------------------------------------------------------------------------
# Rendered surfaces for judged_without_reference (eval-revival σ, task 3628)
#
# A count that is loud only in JSON is still mute where an operator reads it.
# Both tables gain a cell, and the θ table's trailer gains a summary line —
# a mean_plan_quality printed WITHOUT saying how many of its cells were judged
# blind is the same silent degradation one layer up.
# ---------------------------------------------------------------------------

class TestJudgedWithoutReferenceRendering:
    @staticmethod
    def _blind_results():
        return [
            _mresult('f1', 'arch-blind', 1, quality=1.0, cost_usd=0.30,
                     duration_ms=60000, role_under_test='architect',
                     plan_quality=0.8, plan_steps=6,
                     judged_without_reference=True),
            _mresult('f2', 'arch-blind', 1, quality=1.0, cost_usd=0.30,
                     duration_ms=60000, role_under_test='architect',
                     plan_quality=0.6, plan_steps=6),
        ]

    @staticmethod
    def _clean_results():
        return [
            _mresult('f1', 'arch-clean', 1, quality=1.0, cost_usd=0.30,
                     duration_ms=60000, role_under_test='architect',
                     plan_quality=0.8, plan_steps=6),
        ]

    def test_composite_column_is_inserted_beside_its_siblings(self):
        from orchestrator.evals.report import _COMPOSITE_COLUMNS

        # Pin the column tuple FIRST (the file's established idiom), so a
        # reorder fails here rather than silently re-aiming cell assertions.
        # The new column is INSERTED after its two siblings, not appended away
        # from them — the three counts describe one pool and must read together.
        assert _COMPOSITE_COLUMNS[4:8] == (
            'pq_excluded', 'pq_no_plan', 'pq_no_ref', 'plan_rate',
        )
        # …and the relative $/plan pin still holds across the insertion.
        assert (
            _COMPOSITE_COLUMNS[_COMPOSITE_COLUMNS.index('cost_usd') + 1]
            == 'cost_per_plan'
        )

    def test_composite_table_renders_the_count(self):
        from orchestrator.evals.report import (
            build_composite_report,
            format_composite_table,
        )

        out = format_composite_table(build_composite_report(self._blind_results()))
        header = out.splitlines()[1].split()
        cells = _table_row_cells(out, 'arch-blind')

        assert cells[header.index('pq_no_ref')] == '1'

    def test_mean_columns_keep_the_mean_last(self):
        from orchestrator.evals.report import _PLAN_QUALITY_MEAN_COLUMNS

        assert 'judged_without_reference' in _PLAN_QUALITY_MEAN_COLUMNS
        # BEFORE the mean it bounds…
        assert (
            _PLAN_QUALITY_MEAN_COLUMNS.index('judged_without_reference')
            < _PLAN_QUALITY_MEAN_COLUMNS.index('mean_plan_quality')
        )
        # …and the mean stays LAST, which other tests depend on positionally
        # (the mean-section `means[-1]` read, and the doomed-config line whose
        # rendering must still `endswith('-')`).
        assert _PLAN_QUALITY_MEAN_COLUMNS[-1] == 'mean_plan_quality'

    def test_plan_quality_mean_section_renders_the_count(self):
        from orchestrator.evals.report import (
            build_plan_quality_report,
            format_plan_quality_table,
        )

        out = format_plan_quality_table(
            build_plan_quality_report(self._blind_results())
        )
        section = 'plan_quality by config:'
        header = out.partition(section)[2].splitlines()[1].split()
        cells = _table_row_cells(out, 'arch-blind', section=section)

        assert cells[header.index('judged_without_reference')] == '1'

    def test_trailer_reports_the_total_against_the_scored_pool(self):
        from orchestrator.evals.report import (
            build_plan_quality_report,
            format_plan_quality_table,
        )

        out = format_plan_quality_table(
            build_plan_quality_report(self._blind_results())
        )
        line = next(
            ln for ln in out.splitlines()
            if ln.startswith('judged without reference:')
        )
        # 1 of the 2 scored cells — the bound stated against its own pool,
        # beside the existing `excluded: N ... of M architect cell(s)` line.
        assert '1' in line and '2' in line
        assert 'excluded:' in out

    def test_a_clean_campaign_does_not_grow_a_scary_line(self):
        """Absent, or reading 0 — never an alarming line on a healthy run."""
        from orchestrator.evals.report import (
            build_plan_quality_report,
            format_plan_quality_table,
        )

        out = format_plan_quality_table(
            build_plan_quality_report(self._clean_results())
        )
        lines = [
            ln for ln in out.splitlines()
            if ln.startswith('judged without reference:')
        ]
        assert lines == [] or lines[0].split(':')[1].strip().startswith('0')
