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
# Task 2477 step-09: build_composite_report — the C4 per-config composite report
# over the UNION of configs (retiring the all-tasks-intersection collapse).
# ---------------------------------------------------------------------------

def _mresult(
    task_id, config_name, trial, *, quality, cost_usd, duration_ms,
    tests_pass=True, cost_source='price_table', recovery_score=None,
    plan_quality=None, role_under_test='implementer',
    judge_invocations=0, judge_cost_usd=0.0,
):
    """Build a synthetic EvalResult with a production-shaped metrics dict."""
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
        judge_invocations=judge_invocations,
        judge_cost_usd=judge_cost_usd,
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

    Per-fixture best cost/latency drives the normalization:
      f1: cost A=2 B=4 C=1 → best 1 (C); latency A=2 B=4 C=1 → best 1 (C)
      f2: cost A=2 B=4     → best 2 (A); latency A=2 B=4     → best 2 (A)

    Hand-computed per-trial blends (quality 1.0 everywhere, weights .6/.2/.2):
      f1 A: blend(1.0, 2→0.5, 2→0.5) = 0.8   f2 A: blend(1.0, 1.0, 1.0) = 1.0
      f1 B: blend(1.0, 4→0.25, 0.25) = 0.7   f2 B: blend(1.0, 0.5, 0.5) = 0.8
      f1 C: blend(1.0, 1.0, 1.0)     = 1.0
    Per-config composite means: A=0.9, B=0.75, C=1.0.

    Results are inserted in scrambled config order (C, A, B) to prove the impl
    sorts rows by config name.
    """
    results = []
    for tr in (1, 2, 3):
        results.append(_mresult(
            'f1', 'C', tr, quality=1.0, cost_usd=1.0, duration_ms=1000,
            role_under_test='architect', plan_quality=0.9,
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
        # composite = mean per-fixture-normalized blend = mean([0.8]*3+[1.0]*3)
        assert a['composite'] == pytest.approx(0.9)
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
        assert a['ci95']['composite']['mean'] == pytest.approx(0.9)

        b = rows['B']
        assert b['composite'] == pytest.approx(0.75)  # mean([0.7]*3+[0.8]*3)
        assert b['cost_usd'] == pytest.approx(4.0)
        assert b['latency_secs'] == pytest.approx(4.0)
        assert b['trials'] == 6
        assert b['fixtures'] == 2

        c = rows['C']
        # C is cheapest+fastest+top-quality in its only fixture → blend 1.0.
        assert c['composite'] == pytest.approx(1.0)
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
