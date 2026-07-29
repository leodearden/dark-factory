"""Tests for shared.memory_eval_limits — the M2 statistical alarm limits.

Built bottom-up in TDD order (see docs/prds/memory-eval-program.md §3 M2, D3):
  - TestBinomialExactTest / TestPoissonExactTest: the two stdlib exact tests.
  - TestDerivedAlpha: significance is DERIVED from a false-alarm budget (G6).
  - TestStructuralTripwireRatchet: rule (a), grandfather-and-ratchet (D1).
  - TestProportionAndCountRules / TestMinSamplesGuard: rules (b) and (c).
  - TestLimitsArtifact: limits-current.json as state AND dashboard source.

Convention note (orchestrator/tests/test_eval_composite_report.py:20): the
loose numeric anchors below are cross-checked against an INDEPENDENT in-test
enumeration rather than against the production function, so a shared bug in
the estimator cannot make the test agree with itself.
"""

from __future__ import annotations

import math

import pytest

from shared.memory_eval_limits import (
    LimitsConfig,
    binomial_two_sided_p,
    derive_alpha,
    evaluate_tripwire,
    grandfather_set_hash,
    poisson_two_sided_p,
)
from shared.memory_eval_metrics import Metric, TripwireItem


def _tripwire(results: dict[str, bool], *, metric_id: str = 'topic-canonical-present') -> Metric:
    """Build a tripwire metric from an item_key -> passed mapping.

    A tripwire's ``value`` IS its failure count (M1), so it is derived here
    rather than passed in — a builder that let a test state an inconsistent
    count would just be testing the M1 validator by accident.
    """
    return Metric(
        metric_id=metric_id,
        kind='tripwire',
        value=float(sum(1 for passed in results.values() if not passed)),
        n=len(results),
        items=[TripwireItem(item_key=key, passed=passed) for key, passed in results.items()],
    )


def _reference_binomial_two_sided_p(k: int, n: int, p0: float) -> float:
    """Independent enumeration of the method of small p-values.

    Deliberately written differently from the production implementation (a
    plain list comprehension over the whole support, no float-slack guard, no
    log-space work) so it is an actual cross-check and not a transcription.
    """
    pmf = [math.comb(n, i) * (p0**i) * ((1 - p0) ** (n - i)) for i in range(n + 1)]
    return min(1.0, sum(p for p in pmf if p <= pmf[k] * (1 + 1e-9)))


def _reference_poisson_two_sided_p(k: int, lam: float) -> float:
    """Independent enumeration of the Poisson small-p sum.

    Deliberately unlike the production implementation: the support is
    unbounded, so where production terminates ADAPTIVELY (walking out from the
    mode until the remaining tail underflows the running sum), this reference
    just sweeps a brutally generous fixed horizon. Different termination
    strategy == a real cross-check, and it is also the direct evidence that
    production's adaptive stop does not truncate mass that matters.
    """
    if lam == 0.0:
        return 1.0 if k == 0 else 0.0
    pmf = [math.exp(i * math.log(lam) - lam - math.lgamma(i + 1)) for i in range(_HORIZON(lam, k))]
    return min(1.0, math.fsum(p for p in pmf if p <= pmf[k] * (1 + 1e-9)))


def _HORIZON(lam: float, k: int) -> int:
    """A horizon far past where Poisson(lam) mass is representable in float64."""
    return max(k, int(lam)) + int(60 * math.sqrt(lam)) + 200


class TestBinomialExactTest:
    """Rule (b)'s engine: an exact two-sided binomial test, stdlib only.

    Anchors are justified by a stated identity rather than being tuned
    constants — see each test.
    """

    def test_observed_mode_yields_exactly_one(self):
        # k=5 is the mode of Binomial(10, 0.5), so every pmf(i) <= pmf(k) and
        # the small-p sum sweeps the whole distribution. Exactness is
        # structural, not tuned — assert it as an exact equality.
        assert binomial_two_sided_p(5, 10, 0.5) == 1.0

    def test_symmetric_extreme_is_exactly_two_over_1024(self):
        # By symmetry only i in {0, 10} have pmf <= pmf(0); both terms are
        # exact binary fractions, so the sum is exact in float.
        assert binomial_two_sided_p(0, 10, 0.5) == 2 / 1024
        assert binomial_two_sided_p(0, 10, 0.5) == 0.001953125

    def test_mode_identity_holds_for_an_asymmetric_p0(self):
        # 30 * 0.8 == 24 is the mode of Binomial(30, 0.8).
        assert binomial_two_sided_p(24, 30, 0.8) == 1.0

    @pytest.mark.parametrize(('k', 'n', 'p0'), [(20, 30, 0.8), (12, 30, 0.8), (18, 30, 0.5)])
    def test_matches_an_independent_enumeration(self, k, n, p0):
        assert binomial_two_sided_p(k, n, p0) == pytest.approx(
            _reference_binomial_two_sided_p(k, n, p0), rel=1e-12
        )

    def test_baseline_window_anchors_straddle_the_derived_alpha(self):
        # The two fixture anchors, against alpha = 1/360 ~= 0.002778. They
        # separate by three-plus orders of magnitude on each side, so neither
        # verdict is knife-edge.
        assert binomial_two_sided_p(12, 30, 0.8) == pytest.approx(1.8424e-06, rel=1e-4)
        assert binomial_two_sided_p(20, 30, 0.8) == pytest.approx(0.10527, rel=1e-4)

    @pytest.mark.parametrize('k', range(31))
    def test_result_is_a_probability(self, k):
        p = binomial_two_sided_p(k, 30, 0.8)
        assert 0.0 < p <= 1.0

    @pytest.mark.parametrize(('k', 'n', 'p0'), [(0, 10, 0.5), (3, 10, 0.5), (12, 30, 0.8)])
    def test_symmetry_under_mirrored_parameters(self, k, n, p0):
        assert binomial_two_sided_p(k, n, p0) == pytest.approx(
            binomial_two_sided_p(n - k, n, 1 - p0), rel=1e-12
        )

    def test_monotone_moving_down_away_from_the_mode(self):
        ps = [binomial_two_sided_p(k, 30, 0.8) for k in range(24, -1, -1)]
        assert ps == sorted(ps, reverse=True)

    def test_monotone_moving_up_away_from_the_mode(self):
        ps = [binomial_two_sided_p(k, 30, 0.8) for k in range(24, 31)]
        assert ps == sorted(ps, reverse=True)

    def test_degenerate_n_zero(self):
        # The only outcome under H0 is the observed one.
        assert binomial_two_sided_p(0, 0, 0.5) == 1.0

    @pytest.mark.parametrize('p0', [0.0, 1.0])
    def test_degenerate_p0_does_not_raise(self, p0):
        # A degenerate H0 must not produce a ZeroDivisionError or a 0**negative
        # domain error. The certain outcome is unsurprising (p == 1.0); an
        # outcome H0 calls impossible is maximally surprising (p == 0.0).
        certain = 0 if p0 == 0.0 else 10
        assert binomial_two_sided_p(certain, 10, p0) == 1.0
        assert binomial_two_sided_p(10 - certain, 10, p0) == 0.0

    @pytest.mark.parametrize(('k', 'n', 'p0'), [(-1, 10, 0.5), (11, 10, 0.5), (0, -1, 0.5)])
    def test_out_of_range_counts_raise(self, k, n, p0):
        with pytest.raises(ValueError):
            binomial_two_sided_p(k, n, p0)

    @pytest.mark.parametrize('p0', [-0.1, 1.1])
    def test_out_of_range_p0_raises(self, p0):
        with pytest.raises(ValueError):
            binomial_two_sided_p(5, 10, p0)


class TestPoissonExactTest:
    """Rule (c)'s engine: an exact two-sided Poisson test, stdlib only.

    Same method of small p-values as the binomial, so the two rule kinds share
    one reviewed idea. The support is unbounded on the right, which is where
    every anchor below earns its keep.
    """

    def test_observed_mode_is_effectively_one(self):
        # lam=5 is an integer, so pmf(4) == pmf(5) and both are the mode: every
        # outcome is at most as probable as the observed one, and the small-p
        # sum sweeps the whole distribution. Asserted as ~1.0 rather than
        # exactly 1.0 because the log-space pmf loses the last few ulps.
        p = poisson_two_sided_p(5, 5.0)
        assert p > 0.99
        assert p <= 1.0

    def test_no_alarm_side_anchor(self):
        assert poisson_two_sided_p(8, 5.0) == pytest.approx(0.17380, rel=1e-4)

    def test_alarm_side_anchor(self):
        assert poisson_two_sided_p(20, 5.0) == pytest.approx(3.4521e-07, rel=1e-4)

    def test_zero_count_is_not_the_naive_one_tail(self):
        """Standing regression guard against the natural wrong implementation.

        The correct small-p value at k=0 also sweeps the FAR RIGHT TAIL, so it
        is neither the one-tail ``exp(-5)`` nor twice it. An implementation
        that doubles a one-tail — the reflex for a two-sided test — fails here,
        which is the whole reason this anchor is pinned.
        """
        p = poisson_two_sided_p(0, 5.0)
        assert p == pytest.approx(0.012191, rel=1e-4)
        assert p != pytest.approx(math.exp(-5.0), rel=1e-3)
        assert p != pytest.approx(2 * math.exp(-5.0), rel=1e-3)

    @pytest.mark.parametrize(
        ('k', 'lam'), [(0, 5.0), (3, 5.0), (12, 5.0), (2, 2.5), (15, 2.5), (0, 0.5), (7, 1.0)]
    )
    def test_matches_an_independent_enumeration(self, k, lam):
        assert poisson_two_sided_p(k, lam) == pytest.approx(
            _reference_poisson_two_sided_p(k, lam), rel=1e-9
        )

    @pytest.mark.parametrize('k', range(41))
    def test_result_is_a_probability(self, k):
        p = poisson_two_sided_p(k, 5.0)
        assert 0.0 < p <= 1.0

    def test_monotone_moving_down_away_from_the_mode(self):
        ps = [poisson_two_sided_p(k, 5.0) for k in range(5, -1, -1)]
        assert ps == sorted(ps, reverse=True)

    def test_monotone_moving_up_away_from_the_mode(self):
        ps = [poisson_two_sided_p(k, 5.0) for k in range(5, 26)]
        assert ps == sorted(ps, reverse=True)

    def test_degenerate_lam_zero_is_answered_not_a_domain_error(self):
        # A baseline window of all-zero counts is a real input (a dangling-
        # pointer count that has never fired). log(0) must never be taken.
        assert poisson_two_sided_p(0, 0.0) == 1.0
        assert poisson_two_sided_p(3, 0.0) == 0.0

    def test_large_k_terminates(self):
        # The right tail is unbounded; an implementation that walks it without
        # an adaptive stop hangs here rather than failing.
        assert poisson_two_sided_p(200, 5.0) == pytest.approx(
            _reference_poisson_two_sided_p(200, 5.0), rel=1e-9
        )

    @pytest.mark.parametrize(('k', 'lam'), [(-1, 5.0), (0, -1.0), (-3, 0.0)])
    def test_out_of_range_arguments_raise(self, k, lam):
        with pytest.raises(ValueError):
            poisson_two_sided_p(k, lam)


class TestDerivedAlpha:
    """G6: significance is DERIVED from a declared false-alarm budget.

    The module holds no a-priori significance constant. What it holds is a
    BUDGET — "how many false alarms per quarter am I willing to be woken for?"
    — and alpha falls out of it by division once the run cadence and the number
    of metrics that can alarm are known.

    Note on the behavioural half of G6: proving alpha reaches the COMPARISON
    (not merely that ``derive_alpha`` is arithmetic) requires the evaluator, so
    that assertion lives in ``TestBudgetMovesTheVerdict`` alongside the rule
    kinds it exercises. The tests here pin the derivation itself.
    """

    def test_alpha_is_the_budget_quotient_exactly(self):
        # budget=1/quarter, the D10 daily cadence (90 runs/quarter), 4 metrics
        # eligible to alarm. Exact rational: no rounding, nothing tuned.
        assert derive_alpha(1.0, 90, 4) == 1 / 360

    def test_alpha_shrinks_as_more_metrics_can_alarm(self):
        # The Bonferroni intent of M2: adding a metric spends from the SAME
        # quarterly budget, so every metric's bar gets stricter. This is why
        # alpha is recomputed per run rather than frozen at authoring time.
        alphas = [derive_alpha(1.0, 90, count) for count in range(1, 9)]
        assert alphas == sorted(alphas, reverse=True)
        assert len(set(alphas)) == len(alphas)

    def test_alpha_shrinks_as_the_run_rate_grows(self):
        # Running an eval more often is more chances to be unlucky, and the
        # budget is per QUARTER, not per run.
        alphas = [derive_alpha(1.0, runs, 4) for runs in (1, 7, 30, 90, 365)]
        assert alphas == sorted(alphas, reverse=True)
        assert len(set(alphas)) == len(alphas)

    def test_alpha_grows_with_the_declared_budget(self):
        # Tolerating more false alarms buys a looser bar. This is the knob an
        # operator actually turns, and it is stated in units they can reason
        # about ("once a quarter"), not in units of p-value.
        alphas = [derive_alpha(budget, 90, 4) for budget in (0.1, 0.5, 1.0, 4.0, 12.0)]
        assert alphas == sorted(alphas)
        assert len(set(alphas)) == len(alphas)

    @pytest.mark.parametrize(
        ('budget', 'runs', 'count'), [(1.0, 0, 4), (1.0, 90, 0), (1.0, -90, 4), (1.0, 90, -4)]
    )
    def test_zero_or_negative_divisors_raise(self, budget, runs, count):
        # A zero divisor here means "nothing runs" or "nothing can alarm" — a
        # caller bug. Raise rather than return an infinite (or negative) alpha
        # that would silently make every run either an alarm or never one.
        with pytest.raises(ValueError):
            derive_alpha(budget, runs, count)

    @pytest.mark.parametrize('budget', [0.0, -1.0])
    def test_non_positive_budget_raises(self, budget):
        # A budget of zero is not "be very strict", it is a contradiction: no
        # finite alpha admits zero false alarms. Say so instead of returning 0.
        with pytest.raises(ValueError):
            derive_alpha(budget, 90, 4)


class TestStructuralTripwireRatchet:
    """Rule (a): pre-existing failures are grandfathered, regressions alarm (D1).

    The whole lifecycle, because the interesting behaviour is not any single
    verdict but the way the grandfather set MOVES between runs. Two properties
    carry the design and each has a test that would catch its inversion:

      * the set only ever SHRINKS — seeded once, and thereafter items leave it
        by being fixed and never join it by failing;
      * consequently a fixed item that regresses alarms again, and a newly
        failing item cannot silence itself by being absorbed into the baseline.
    """

    def test_first_run_snapshots_failures_without_alarming(self):
        # No prior state: today's failures are the starting line, not news.
        # The fix lineage (3111/3112/3200) reads this set as its worklist —
        # these are known-bad, deliberately NOT alarms.
        verdict, grandfather = evaluate_tripwire(
            _tripwire({'t-a': True, 't-b': False, 't-c': False}), None
        )
        assert verdict.status == 'baseline_snapshot'
        assert verdict.alarms == ()
        assert grandfather == frozenset({'t-b', 't-c'})

    def test_first_run_still_reports_the_numbers(self):
        # Canary precedent: the comparison is always computed and exposed, so
        # a non-alarming status is still inspectable rather than opaque.
        verdict, _ = evaluate_tripwire(_tripwire({'t-a': True, 't-b': False}), None)
        assert verdict.metric_id == 'topic-canonical-present'
        assert verdict.rule_kind == 'tripwire'
        assert verdict.value == 1.0
        assert verdict.n == 2

    def test_rerunning_an_unchanged_series_alarms_nothing(self):
        metric = _tripwire({'t-a': True, 't-b': False, 't-c': False})
        _, first = evaluate_tripwire(metric, None)
        verdict, second = evaluate_tripwire(metric, first)
        assert verdict.status == 'ok'
        assert verdict.alarms == ()
        assert second == first
        assert grandfather_set_hash(second) == grandfather_set_hash(first)

    def test_a_grandfathered_failure_does_not_alarm(self):
        verdict, _ = evaluate_tripwire(_tripwire({'t-a': True, 't-b': False}), frozenset({'t-b'}))
        assert verdict.status == 'ok'
        assert verdict.alarms == ()

    def test_a_newly_failing_item_alarms_and_names_itself(self):
        verdict, _ = evaluate_tripwire(
            _tripwire({'t-a': False, 't-b': False, 't-c': True}), frozenset({'t-b'})
        )
        assert verdict.status == 'alarm'
        assert len(verdict.alarms) == 1
        alarm = verdict.alarms[0]
        assert alarm.item_key == 't-a'
        assert alarm.metric_id == 'topic-canonical-present'
        assert 't-a' in alarm.detail

    def test_a_structural_alarm_spends_no_alpha(self):
        # Rule (a) is structural, not statistical: an item newly failing is a
        # fact, not a surprise to be measured. No p-value, no alpha.
        verdict, _ = evaluate_tripwire(_tripwire({'t-a': False}), frozenset())
        assert verdict.status == 'alarm'
        assert verdict.p_value is None
        assert verdict.alpha is None

    def test_multiple_new_failures_alarm_in_a_deterministic_order(self):
        # The alarm list lands verbatim in a committed artifact, so its order
        # must not depend on set iteration order.
        verdict, _ = evaluate_tripwire(
            _tripwire({'t-c': False, 't-a': False, 't-b': False}), frozenset({'t-b'})
        )
        assert [alarm.item_key for alarm in verdict.alarms] == ['t-a', 't-c']

    def test_ratchet_drops_an_item_that_now_passes(self):
        before = frozenset({'t-b', 't-c'})
        verdict, after = evaluate_tripwire(
            _tripwire({'t-a': True, 't-b': True, 't-c': False}), before
        )
        assert verdict.status == 'ok'
        assert after == frozenset({'t-c'})
        assert grandfather_set_hash(after) != grandfather_set_hash(before)

    def test_a_fixed_item_that_regresses_alarms_again(self):
        """The ratchet's entire purpose, asserted end to end."""
        _, after_fix = evaluate_tripwire(
            _tripwire({'t-a': True, 't-b': True, 't-c': False}), frozenset({'t-b', 't-c'})
        )
        assert 't-b' not in after_fix

        verdict, _ = evaluate_tripwire(
            _tripwire({'t-a': True, 't-b': False, 't-c': False}), after_fix
        )
        assert verdict.status == 'alarm'
        assert [alarm.item_key for alarm in verdict.alarms] == ['t-b']

    def test_a_newly_failing_item_is_never_grandfathered(self):
        """The load-bearing negative: an alarm must not silence itself.

        If a newly-failing item were folded into the grandfather set after
        alarming, the very next run would reclassify the regression as
        known-bad and go quiet — precisely the failure mode grandfathering
        exists to avoid. So it alarms again, and again, until someone fixes it.
        """
        before = frozenset({'t-b'})
        first, after = evaluate_tripwire(_tripwire({'t-a': False, 't-b': False}), before)
        assert first.status == 'alarm'
        assert 't-a' not in after
        assert after == before

        second, _ = evaluate_tripwire(_tripwire({'t-a': False, 't-b': False}), after)
        assert second.status == 'alarm'
        assert [alarm.item_key for alarm in second.alarms] == ['t-a']

    def test_the_set_only_ever_shrinks(self):
        # Walk a run sequence that both fixes and breaks items, and assert the
        # set is a subset at every step. This is the invariant the two tests
        # above are each half of.
        runs = [
            {'t-a': True, 't-b': False, 't-c': False},
            {'t-a': False, 't-b': False, 't-c': False},
            {'t-a': False, 't-b': True, 't-c': False},
            {'t-a': False, 't-b': False, 't-c': True},
        ]
        _, grandfather = evaluate_tripwire(_tripwire(runs[0]), None)
        for run in runs[1:]:
            _, nxt = evaluate_tripwire(_tripwire(run), grandfather)
            assert nxt <= grandfather
            grandfather = nxt
        assert grandfather == frozenset()

    def test_hash_is_stable_under_reordering(self):
        assert grandfather_set_hash(['t-b', 't-a', 't-c']) == grandfather_set_hash(
            ['t-c', 't-b', 't-a']
        )
        assert grandfather_set_hash(frozenset({'t-a', 't-b'})) == grandfather_set_hash(
            ['t-b', 't-a']
        )

    def test_hash_changes_when_membership_changes(self):
        assert grandfather_set_hash(['t-a', 't-b']) != grandfather_set_hash(['t-a', 't-b', 't-c'])
        assert grandfather_set_hash([]) != grandfather_set_hash(['t-a'])

    def test_hash_is_hex_and_length_stable(self):
        # It travels in a JSON artifact and gets eyeballed in diffs.
        for keys in ([], ['t-a'], ['t-a', 't-b']):
            digest = grandfather_set_hash(keys)
            assert len(digest) == 64
            assert set(digest) <= set('0123456789abcdef')

    def test_config_defaults_to_one_false_alarm_per_quarter(self):
        # The PRD-sanctioned default (M2) — a budget DECLARATION, not a
        # significance threshold. Everything else must be declared explicitly.
        config = LimitsConfig(runs_per_quarter=90, min_samples=10, baseline_window=3)
        assert config.false_alarm_budget == 1.0

    def test_config_is_frozen(self):
        config = LimitsConfig(runs_per_quarter=90, min_samples=10, baseline_window=3)
        with pytest.raises(Exception):  # noqa: B017 - FrozenInstanceError is a TypeError subclass
            config.false_alarm_budget = 99.0  # type: ignore[misc]

    def test_config_carries_the_derivation_inputs(self):
        # alpha is not stored on the config: it depends on how many metrics
        # alarm in THIS run, which the config cannot know. The config carries
        # the two stable halves and the evaluator supplies the third.
        config = LimitsConfig(
            false_alarm_budget=2.0, runs_per_quarter=90, min_samples=10, baseline_window=3
        )
        assert derive_alpha(config.false_alarm_budget, config.runs_per_quarter, 4) == 2 / 360
