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

import json
import math
import os
from pathlib import Path

import pytest

from shared.memory_eval_limits import (
    GRANDFATHER_KEY_SEPARATOR,
    LimitsConfig,
    MetricVerdict,
    binomial_two_sided_p,
    derive_alpha,
    evaluate_count,
    evaluate_proportion,
    evaluate_series,
    evaluate_tripwire,
    grandfather_set_hash,
    grandfather_slice,
    limits_artifact_path,
    load_limits_artifact,
    poisson_two_sided_p,
    scoped_grandfather_key,
    write_limits_artifact,
)
from shared.memory_eval_metrics import (
    Corpus,
    Metric,
    MetricSeries,
    TripwireItem,
    load_metric_series,
)

_FIXTURES = Path(__file__).parent / 'fixtures' / 'memory_eval'
_BASELINE_STAMPS = ('20260701T031500Z', '20260702T031500Z', '20260703T031500Z')
_REGRESSION_STAMP = '20260704T031500Z'
_QUIET_STAMP = '20260705T031500Z'

# The committed exemplars are the shared corpus for both this suite and the
# boundary suite (the two-consumer precedent of test_invocation_outcome.py +
# ..._boundary.py), so the numbers below are never restated in code.
_ALARMED_METRIC_COUNT = 4  # 1 proportion + 2 counts + 1 tripwire; scalar never alarms
_MIN_SAMPLES = 10


def _series(stamp: str, *, eval_id: str = 'e1-retrieval-health') -> MetricSeries:
    return load_metric_series(_FIXTURES / eval_id / f'metrics-{stamp}.json')


def _baseline() -> list[MetricSeries]:
    return [_series(stamp) for stamp in _BASELINE_STAMPS]


def _metric(series: MetricSeries, metric_id: str) -> Metric:
    return next(m for m in series.metrics if m.metric_id == metric_id)


def _verdict_for(result, metric_id: str) -> MetricVerdict:
    return next(v for v in result.verdicts if v.metric_id == metric_id)


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


def _exact_rational_binomial_two_sided_p(k: int, n: int, a: int, b: int) -> float:
    """The same small-p sum for ``p0 == a/b``, in EXACT integer arithmetic.

    Every pmf term shares the denominator ``b**n``, so the whole comparison and
    the whole sum can be done on integer numerators — no floats, and therefore
    no tie slack and no rounding, at any n. That is what makes it a usable
    cross-check where ``_reference_binomial_two_sided_p`` cannot go: THAT
    reference enumerates the full support in floats, so it raises
    ``OverflowError`` on the central coefficients for any n past ~1030,
    whatever k is.
    """
    numerators = [math.comb(n, i) * a**i * (b - a) ** (n - i) for i in range(n + 1)]
    observed = numerators[k]
    return sum(x for x in numerators if x <= observed) / b**n


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

    @pytest.mark.parametrize(('k', 'n'), [(515, 1030), (550, 1100), (900, 2000)])
    def test_a_corpus_scale_n_yields_a_probability_rather_than_raising(self, k, n):
        """Regression: the pmf must not be assembled from exact bignum coefficients.

        ``math.comb(1030, 515)`` exceeds float64, so a ``comb(n, i) * float``
        pmf raises ``OverflowError`` from here up — taking the whole run, and
        its grandfather-set update, with it. Nothing in the M1 schema bounds a
        proportion's ``n`` (``Metric`` asks only for ``n >= 0`` and
        ``n == denominator``), and a corpus-scale proportion is an ordinary
        metric for this programme: the committed exemplar's own corpus counts
        1204 ``entities_and_relations``.
        """
        p = binomial_two_sided_p(k, n, 0.5)
        assert 0.0 <= p <= 1.0

    def test_a_corpus_scale_n_still_matches_exact_arithmetic(self):
        # Not merely finite — right. Cross-checked against the exact-integer
        # reference, which is the only one that survives this n.
        assert binomial_two_sided_p(900, 1204, 0.8) == pytest.approx(
            _exact_rational_binomial_two_sided_p(900, 1204, 4, 5), rel=1e-9
        )
        assert binomial_two_sided_p(1000, 1204, 0.8) == pytest.approx(
            _exact_rational_binomial_two_sided_p(1000, 1204, 4, 5), rel=1e-9
        )

    def test_the_mode_identity_survives_a_corpus_scale_n(self):
        # 1204 * 0.8 == 963.2, so 963 is the mode: every outcome qualifies and
        # the answer is 1.0 by definition, not by float luck.
        assert binomial_two_sided_p(963, 1204, 0.8) == 1.0


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


class TestGrandfatherScopingAcrossMetrics:
    """The grandfather set is PER-METRIC state, not one flat namespace.

    ``MetricSeries`` explicitly permits several tripwires in one run
    (``test_memory_eval_metrics.py::test_distinct_metric_ids_of_the_same_kind_are_fine``),
    and E1 gains a second structural probe the moment another invariant is
    wired up. Two things break if the state is kept flat, and both are silent:

    * evaluating metric B would REPLACE the persisted set with B's slice,
      dropping A's known-bad items — so the next run alarms on items that never
      changed, the exact phantom alarm the ratchet exists to prevent;
    * item_keys are unique only WITHIN a metric, so a key passing under B would
      release the same key grandfathered under A.

    Every test here evaluates the SAME run twice and asserts the second pass is
    silent, because unchanged data producing an alarm is the observable symptom
    of both defects.
    """

    CONFIG = LimitsConfig(
        false_alarm_budget=1.0, runs_per_quarter=90, min_samples=1, baseline_window=3
    )

    def _run(self, a: dict[str, bool], b: dict[str, bool] | None = None) -> MetricSeries:
        metrics = [_tripwire(a, metric_id='probe-a')]
        if b is not None:
            metrics.append(_tripwire(b, metric_id='probe-b'))
        return MetricSeries(
            schema_version=1,
            eval_id='e1-dual-tripwire',
            run_stamp='20260801T000000Z',
            corpus=Corpus(project_id='dark_factory'),
            metrics=metrics,
        )

    def test_the_first_run_snapshots_every_tripwire_not_just_the_last(self):
        result = evaluate_series(
            self._run({'a1': False, 'a2': True}, {'b1': False}), [], self.CONFIG, None
        )
        assert result.alarms == ()
        assert sorted(result.grandfather) == ['probe-a::a1', 'probe-b::b1']

    def test_an_unchanged_rerun_of_a_two_tripwire_series_alarms_nothing(self):
        run = self._run({'a1': False, 'a2': True}, {'b1': False})
        first = evaluate_series(run, [], self.CONFIG, None)
        second = evaluate_series(run, [], self.CONFIG, first.grandfather)
        assert second.alarms == ()
        assert second.grandfather_hash == first.grandfather_hash

    def test_fixing_one_metric_leaves_the_others_known_bad_alone(self):
        first = evaluate_series(self._run({'a1': False}, {'b1': False}), [], self.CONFIG, None)
        healed = evaluate_series(
            self._run({'a1': True}, {'b1': False}), [], self.CONFIG, first.grandfather
        )
        assert healed.alarms == ()  # a1 fixed (ratchet), b1 still grandfathered
        assert sorted(healed.grandfather) == ['probe-b::b1']

    def test_a_regression_names_only_the_metric_it_happened_in(self):
        first = evaluate_series(self._run({'a1': False}, {'b1': False}), [], self.CONFIG, None)
        regressed = evaluate_series(
            self._run({'a1': False, 'a2': False}, {'b1': False}),
            [],
            self.CONFIG,
            first.grandfather,
        )
        assert [(a.metric_id, a.item_key) for a in regressed.alarms] == [('probe-a', 'a2')]

    def test_colliding_item_keys_stay_independent(self):
        # 'x' fails under probe-a and passes under probe-b. Under a flat
        # namespace probe-b's pass releases probe-a's known-bad 'x', and the
        # very next pass over the SAME data alarms.
        run = self._run({'x': False}, {'x': True})
        first = evaluate_series(run, [], self.CONFIG, None)
        assert sorted(first.grandfather) == ['probe-a::x']
        second = evaluate_series(run, [], self.CONFIG, first.grandfather)
        assert second.alarms == ()

    def test_a_metric_absent_from_a_run_keeps_its_known_bad(self):
        # A probe that did not run this time has not been fixed. Dropping its
        # entries would silently re-arm it, and it would alarm on its way back.
        both = evaluate_series(self._run({'a1': False}, {'b1': False}), [], self.CONFIG, None)
        without_b = evaluate_series(self._run({'a1': False}), [], self.CONFIG, both.grandfather)
        assert without_b.alarms == ()
        assert sorted(without_b.grandfather) == ['probe-a::a1', 'probe-b::b1']

    def test_a_tripwire_added_mid_programme_snapshots_rather_than_alarming(self):
        """M2: pre-existing failures are grandfathered — including a new probe's.

        Adding a metric is an expected event (it is why alpha is recomputed per
        run), and a probe wired up in month three has the same pre-existing
        failures day one had: the fix lineage's worklist, not news. Inferring
        first-run-ness from the grandfather slice cannot work — an empty slice
        means "new" and "all fixed" alike — so the ledger records it.
        """
        first = evaluate_series(self._run({'a1': False}), [], self.CONFIG, None)
        added = evaluate_series(
            self._run({'a1': False}, {'b1': False, 'b2': False, 'b3': False}),
            [],
            self.CONFIG,
            first.grandfather,
            first.snapshotted_metrics,
        )
        assert added.alarms == ()
        assert _verdict_for(added, 'probe-b').status == 'baseline_snapshot'
        # ...and the sibling that HAS state is untouched by the new arrival.
        assert _verdict_for(added, 'probe-a').status == 'ok'
        assert sorted(added.grandfather) == [
            'probe-a::a1',
            'probe-b::b1',
            'probe-b::b2',
            'probe-b::b3',
        ]

    def test_the_newly_added_tripwire_alarms_on_its_next_regression(self):
        # Snapshotting a new probe must not make it permanently quiet: the run
        # after it joins is judged against what it snapshotted.
        first = evaluate_series(self._run({'a1': False}), [], self.CONFIG, None)
        added = evaluate_series(
            self._run({'a1': False}, {'b1': False}),
            [],
            self.CONFIG,
            first.grandfather,
            first.snapshotted_metrics,
        )
        later = evaluate_series(
            self._run({'a1': False}, {'b1': False, 'b2': False}),
            [],
            self.CONFIG,
            added.grandfather,
            added.snapshotted_metrics,
        )
        assert [(a.metric_id, a.item_key) for a in later.alarms] == [('probe-b', 'b2')]

    def test_the_ledger_keeps_a_metric_that_skipped_a_run(self):
        # A probe absent from one run has not become new again — re-snapshotting
        # it on return would swallow whatever regressed while it was away.
        both = evaluate_series(self._run({'a1': False}, {'b1': False}), [], self.CONFIG, None)
        without_b = evaluate_series(
            self._run({'a1': False}), [], self.CONFIG, both.grandfather, both.snapshotted_metrics
        )
        assert without_b.snapshotted_metrics == frozenset({'probe-a', 'probe-b'})
        returned = evaluate_series(
            self._run({'a1': False}, {'b1': False, 'b2': False}),
            [],
            self.CONFIG,
            without_b.grandfather,
            without_b.snapshotted_metrics,
        )
        assert [(a.metric_id, a.item_key) for a in returned.alarms] == [('probe-b', 'b2')]

    def test_a_missing_ledger_is_read_conservatively(self):
        """Half-carried state degrades LOUD, never silent.

        Passing a grandfather set without its ledger is a caller bug. Read the
        other way — assume nothing was snapshotted — a metric that already had
        state would re-snapshot today's failures and swallow a real regression
        as known-bad. Alarming on a new probe's known-bad items is noisy and
        self-correcting; silently muting a regression is neither.
        """
        first = evaluate_series(self._run({'a1': False}), [], self.CONFIG, None)
        no_ledger = evaluate_series(
            self._run({'a1': False}, {'b1': False}), [], self.CONFIG, first.grandfather
        )
        assert [(a.metric_id, a.item_key) for a in no_ledger.alarms] == [('probe-b', 'b1')]

    def test_the_whole_first_run_ignores_a_stale_ledger(self):
        # grandfather=None is the programme's first run by definition, so every
        # tripwire snapshots whatever a caller claims to have snapshotted.
        result = evaluate_series(
            self._run({'a1': False}, {'b1': False}),
            [],
            self.CONFIG,
            None,
            frozenset({'probe-a', 'probe-b'}),
        )
        assert result.alarms == ()
        assert {v.status for v in result.verdicts} == {'baseline_snapshot'}

    def test_the_scoped_key_helpers_round_trip(self):
        keys = {scoped_grandfather_key('probe-a', 'x'), scoped_grandfather_key('probe-b', 'x')}
        assert grandfather_slice(keys, 'probe-a') == frozenset({'x'})
        assert grandfather_slice(keys, 'probe-b') == frozenset({'x'})
        assert grandfather_slice(keys, 'probe-c') == frozenset()

    def test_an_item_key_may_itself_contain_the_separator(self):
        # Only the FIRST separator is structural; the rest belongs to the item.
        keys = {scoped_grandfather_key('probe-a', 'ns::item')}
        assert grandfather_slice(keys, 'probe-a') == frozenset({'ns::item'})

    def test_an_unscoped_grandfather_seed_is_refused(self):
        """The bare-item_key seed is rejected, not silently ignored.

        evaluate_tripwire returns bare item_keys for ONE metric; evaluate_series
        consumes the persisted, metric-scoped set. Feeding the former to the
        latter is the natural mistake — it type-checks, and both are
        frozenset[str] — and it fails silently in the worst way: the seed
        matches no prefix, so every known-bad item alarms and the artifact then
        publishes the unscoped keys as state. Refuse it at the door instead,
        naming the trap, since a downstream runner leaf will be copying this
        seeding recipe.
        """
        run = self._run({'a1': False})
        _, bare = evaluate_tripwire(run.metrics[0], None)
        assert bare == frozenset({'a1'})  # bare, by evaluate_tripwire's contract

        with pytest.raises(ValueError, match='not metric-scoped'):
            evaluate_series(run, [], self.CONFIG, bare)

    def test_a_partly_unscoped_seed_is_refused_too(self):
        # One bad key is enough: a set that is mostly scoped is not a set that
        # can be trusted, and the message names the offenders.
        with pytest.raises(ValueError, match="\\['a1'\\]"):
            evaluate_series(
                self._run({'a1': False}),
                [],
                self.CONFIG,
                frozenset({'probe-a::a1', 'a1'}),
            )

    def test_an_empty_seed_is_not_mistaken_for_an_unscoped_one(self):
        # Nothing to scope is not the same as wrongly scoped: an eval whose
        # every known-bad item has been fixed resumes from an empty set.
        result = evaluate_series(self._run({'a1': True}), [], self.CONFIG, frozenset())
        assert result.alarms == ()

    def test_a_metric_id_containing_the_separator_is_refused(self):
        # Ambiguous by construction: 'probe::a' + '::' + 'x' is also
        # 'probe' + '::' + 'a::x'. Refuse loudly rather than mis-scope quietly.
        with pytest.raises(ValueError):
            scoped_grandfather_key('probe::a', 'x')


class TestProportionAndCountRules:
    """Rules (b) and (c) against the trailing baseline window, at the derived alpha.

    alpha here is 1/360 — budget of 1 false alarm/quarter, the D10 daily
    cadence, 4 alarm-eligible metrics. Every fixture anchor sits three or more
    orders of magnitude clear of it on one side or the other, so no verdict
    below is knife-edge and a small change in the estimator cannot flip one.
    """

    ALPHA = 1 / 360

    def test_regression_run_proportion_alarms(self):
        verdict = evaluate_proportion(
            _metric(_series(_REGRESSION_STAMP), 'canonical-in-top-5'),
            _baseline(),
            self.ALPHA,
            _MIN_SAMPLES,
        )
        assert verdict.status == 'alarm'
        assert verdict.p_value == pytest.approx(1.8424e-06, rel=1e-4)
        assert verdict.p_value is not None and verdict.p_value < self.ALPHA

    def test_quiet_run_proportion_does_not_alarm(self):
        verdict = evaluate_proportion(
            _metric(_series(_QUIET_STAMP), 'canonical-in-top-5'),
            _baseline(),
            self.ALPHA,
            _MIN_SAMPLES,
        )
        assert verdict.status == 'ok'
        assert verdict.p_value == pytest.approx(0.10527, rel=1e-4)
        assert verdict.alarms == ()

    def test_an_unchanged_proportion_is_exactly_unsurprising(self):
        # 24/30 IS the pooled baseline, so it sits at the mode: p == 1.0 by the
        # identity, not by luck.
        unchanged = Metric(
            metric_id='canonical-in-top-5', kind='proportion', value=0.8, n=30, denominator=30
        )
        verdict = evaluate_proportion(unchanged, _baseline(), self.ALPHA, _MIN_SAMPLES)
        assert verdict.status == 'ok'
        assert verdict.p_value == 1.0

    def test_regression_run_count_alarms(self):
        verdict = evaluate_count(
            _metric(_series(_REGRESSION_STAMP), 'dangling-pointers'),
            _baseline(),
            self.ALPHA,
            _MIN_SAMPLES,
        )
        assert verdict.status == 'alarm'
        assert verdict.p_value == pytest.approx(3.4521e-07, rel=1e-4)

    def test_quiet_run_count_does_not_alarm(self):
        verdict = evaluate_count(
            _metric(_series(_QUIET_STAMP), 'dangling-pointers'),
            _baseline(),
            self.ALPHA,
            _MIN_SAMPLES,
        )
        assert verdict.status == 'ok'
        assert verdict.p_value == pytest.approx(0.17380, rel=1e-4)

    def test_proportion_baseline_is_pooled_across_the_window(self):
        # 24/30 three times over pools to 72/90 == 0.8. Pooling (rather than
        # averaging the three ratios) is what gives the exact test a single
        # honest trial count to work from.
        verdict = evaluate_proportion(
            _metric(_series(_QUIET_STAMP), 'canonical-in-top-5'),
            _baseline(),
            self.ALPHA,
            _MIN_SAMPLES,
        )
        assert verdict.baseline == pytest.approx(0.8)

    def test_count_baseline_is_the_window_mean_rate(self):
        # 4, 5, 6 over the three baseline runs -> lam = 5.0.
        verdict = evaluate_count(
            _metric(_series(_QUIET_STAMP), 'dangling-pointers'),
            _baseline(),
            self.ALPHA,
            _MIN_SAMPLES,
        )
        assert verdict.baseline == pytest.approx(5.0)

    @pytest.mark.parametrize('metric_id', ['canonical-in-top-5', 'dangling-pointers'])
    def test_a_verdict_carries_its_provenance(self, metric_id):
        """Provenance travels with the verdict — never a bare boolean.

        Someone reading this hours later must be able to re-derive the call by
        hand: which runs formed the baseline, what the baseline was, what the
        p-value was, and what bar it was held to.
        """
        series = _series(_REGRESSION_STAMP)
        metric = _metric(series, metric_id)
        evaluate = evaluate_proportion if metric.kind == 'proportion' else evaluate_count
        verdict = evaluate(metric, _baseline(), self.ALPHA, _MIN_SAMPLES)

        assert verdict.metric_id == metric_id
        assert verdict.rule_kind == metric.kind
        assert verdict.alpha == self.ALPHA
        assert verdict.p_value is not None
        assert verdict.baseline is not None
        assert verdict.baseline_run_stamps == _BASELINE_STAMPS

    @pytest.mark.parametrize('metric_id', ['canonical-in-top-5', 'dangling-pointers'])
    def test_ok_verdicts_still_expose_the_numbers(self, metric_id):
        # Canary precedent: "fine" must still answer "fine, but how close?".
        series = _series(_QUIET_STAMP)
        metric = _metric(series, metric_id)
        evaluate = evaluate_proportion if metric.kind == 'proportion' else evaluate_count
        verdict = evaluate(metric, _baseline(), self.ALPHA, _MIN_SAMPLES)

        assert verdict.status == 'ok'
        assert verdict.p_value is not None
        assert verdict.baseline is not None
        assert verdict.alpha == self.ALPHA
        assert verdict.value == metric.value


class TestMinSamplesGuard:
    """``insufficient_data`` is a REPORT STATUS and is NEVER an alarm.

    A thin run is not evidence of a regression — it is an absence of evidence,
    and the two must not be conflated. This is the canary's ``min_samples``
    precedent (``canary.py``), and it is a guard on the SAMPLE COUNT, not a
    threshold on the statistic.
    """

    ALPHA = 1 / 360

    def test_a_thin_current_run_is_insufficient(self):
        # The committed e1-thin exemplar, n=6 against min_samples=10. Its
        # baseline is borrowed from e1-retrieval-health, whose metric ids it
        # shares — the guard fires before any of that matters.
        thin = _metric(_series(_REGRESSION_STAMP, eval_id='e1-thin'), 'canonical-in-top-5')
        verdict = evaluate_proportion(thin, _baseline(), self.ALPHA, _MIN_SAMPLES)
        assert verdict.status == 'insufficient_data'
        assert verdict.alarms == ()

    def test_a_catastrophic_looking_thin_run_still_does_not_alarm(self):
        """The assertion that earns this guard its place.

        0/6 against a baseline of 0.8 has p == 6.4e-05, comfortably below
        alpha=1/360 — so WITHOUT the guard this alarms. It must not: six probes
        is too few to distinguish a real collapse from a bad afternoon, and
        crying wolf here is exactly how an eval gets ignored.
        """
        catastrophic = Metric(
            metric_id='canonical-in-top-5', kind='proportion', value=0.0, n=6, denominator=6
        )
        assert binomial_two_sided_p(0, 6, 0.8) < self.ALPHA  # it WOULD alarm

        verdict = evaluate_proportion(catastrophic, _baseline(), self.ALPHA, _MIN_SAMPLES)
        assert verdict.status == 'insufficient_data'
        assert verdict.alarms == ()

    def test_a_thin_run_still_reports_its_numbers(self):
        # Inconclusive is not opaque: the numbers are computed anyway so a
        # reader can see the trend even though it is not actionable yet.
        catastrophic = Metric(
            metric_id='canonical-in-top-5', kind='proportion', value=0.0, n=6, denominator=6
        )
        verdict = evaluate_proportion(catastrophic, _baseline(), self.ALPHA, _MIN_SAMPLES)
        assert verdict.p_value == pytest.approx(6.4e-05, rel=1e-4)
        assert verdict.baseline == pytest.approx(0.8)
        assert verdict.value == 0.0

    def test_a_thin_count_run_is_insufficient(self):
        thin = _metric(_series(_REGRESSION_STAMP, eval_id='e1-thin'), 'dangling-pointers')
        verdict = evaluate_count(thin, _baseline(), self.ALPHA, _MIN_SAMPLES)
        assert verdict.status == 'insufficient_data'
        assert verdict.alarms == ()

    @pytest.mark.parametrize('evaluate', [evaluate_proportion, evaluate_count])
    def test_an_empty_baseline_window_is_insufficient_not_a_division_error(self, evaluate):
        # The very first run of a new eval. There is nothing to compare to, and
        # dividing by an empty window would be a crash where a status belongs.
        metric = _metric(_series(_REGRESSION_STAMP), 'canonical-in-top-5')
        if evaluate is evaluate_count:
            metric = _metric(_series(_REGRESSION_STAMP), 'dangling-pointers')
        verdict = evaluate(metric, [], self.ALPHA, _MIN_SAMPLES)
        assert verdict.status == 'insufficient_data'
        assert verdict.alarms == ()
        assert verdict.p_value is None

    def test_a_baseline_missing_this_metric_is_insufficient(self):
        # A newly added metric: the window exists but has never measured it.
        fresh = Metric(metric_id='brand-new-probe', kind='count', value=99.0, n=30)
        verdict = evaluate_count(fresh, _baseline(), self.ALPHA, _MIN_SAMPLES)
        assert verdict.status == 'insufficient_data'
        assert verdict.alarms == ()


class TestBudgetMovesTheVerdict:
    """The behavioural half of G6, over the committed fixture series.

    A source-text scan for numeric literals would pin authoring habits, not
    runtime behaviour, and any literal spelled as an expression would defeat
    it. Showing that the SAME data yields a DIFFERENT verdict when only the
    declared false-alarm budget changes is strictly stronger: it is direct
    evidence that alpha flows from config, through ``derive_alpha``, into the
    comparison — rather than being baked in anywhere along the way.
    """

    def _config(self, budget: float) -> LimitsConfig:
        return LimitsConfig(
            false_alarm_budget=budget,
            runs_per_quarter=90,
            min_samples=_MIN_SAMPLES,
            baseline_window=3,
        )

    def _evaluate(self, budget: float, stamp: str):
        return evaluate_series(_series(stamp), _baseline(), self._config(budget), None)

    def test_a_stricter_budget_silences_a_regression(self):
        # p == 1.84e-06 for the regression run's proportion. At a budget of 1
        # false alarm/quarter (alpha = 2.8e-03) that alarms; at 1e-4/quarter
        # (alpha = 2.8e-07) the same data does not clear the bar.
        loud = _verdict_for(self._evaluate(1.0, _REGRESSION_STAMP), 'canonical-in-top-5')
        strict = _verdict_for(self._evaluate(1e-4, _REGRESSION_STAMP), 'canonical-in-top-5')

        assert loud.status == 'alarm'
        assert strict.status == 'ok'
        assert loud.p_value == strict.p_value  # identical data, identical statistic
        assert strict.alpha is not None and loud.alpha is not None
        assert strict.alpha < loud.alpha  # only the bar moved

    def test_a_looser_budget_alarms_on_a_quiet_run(self):
        # And in the other direction: p == 0.105 is unremarkable at a budget of
        # 1/quarter, but an operator who declares they will tolerate 100 false
        # alarms a quarter (alpha = 0.278) has asked to hear about it.
        quiet = _verdict_for(self._evaluate(1.0, _QUIET_STAMP), 'canonical-in-top-5')
        loose = _verdict_for(self._evaluate(100.0, _QUIET_STAMP), 'canonical-in-top-5')

        assert quiet.status == 'ok'
        assert loose.status == 'alarm'
        assert quiet.p_value == loose.p_value

    def test_the_result_alpha_is_the_documented_derivation(self):
        result = self._evaluate(1.0, _REGRESSION_STAMP)
        assert result.alarmed_metric_count == _ALARMED_METRIC_COUNT
        assert result.alpha == derive_alpha(1.0, 90, _ALARMED_METRIC_COUNT)
        assert result.alpha == 1 / 360

    def test_scalar_metrics_are_reported_but_never_alarm(self):
        # A latency scalar has no rule attached yet, so it rides along in the
        # report and spends none of the budget — which is why the alarm-eligible
        # count is 4 and not 5.
        result = self._evaluate(100.0, _REGRESSION_STAMP)
        latency = _verdict_for(result, 'search-latency-p50-ms')
        assert latency.rule_kind == 'scalar'
        assert latency.status != 'alarm'
        assert latency.alarms == ()
        assert latency.value == 44.0

    def test_every_metric_in_the_series_gets_a_verdict(self):
        result = self._evaluate(1.0, _REGRESSION_STAMP)
        assert {v.metric_id for v in result.verdicts} == {
            m.metric_id for m in _series(_REGRESSION_STAMP).metrics
        }

    def test_the_alarm_list_is_the_union_of_the_verdicts_alarms(self):
        result = self._evaluate(1.0, _REGRESSION_STAMP)
        assert list(result.alarms) == [a for v in result.verdicts for a in v.alarms]

    def test_only_the_trailing_baseline_window_is_used(self):
        # Given more history than configured, the evaluator judges against the
        # most recent `baseline_window` runs — a baseline that silently grew
        # without bound would drift away from current behaviour.
        config = LimitsConfig(runs_per_quarter=90, min_samples=_MIN_SAMPLES, baseline_window=2)
        result = evaluate_series(_series(_QUIET_STAMP), _baseline(), config, None)
        verdict = _verdict_for(result, 'dangling-pointers')
        # Trailing two runs are 5 and 6 -> lam == 5.5, not the 5.0 of all three.
        assert verdict.baseline == pytest.approx(5.5)
        assert verdict.baseline_run_stamps == _BASELINE_STAMPS[1:]


def _seeded_state() -> tuple[frozenset[str], frozenset[str]]:
    """Resumable state as a runner actually holds it: an earlier run's output.

    Seeding by calling ``evaluate_tripwire`` directly is a trap worth naming,
    because it looks right: that function speaks BARE item_keys for one metric,
    while the persisted set ``evaluate_series`` consumes is metric-scoped. A
    hand-built bare seed matches no prefix, so every known-bad item alarms and
    the artifact publishes the unscoped keys as state. ``evaluate_series``
    rejects such a seed now, and chaining a previous ``EvaluationResult`` — the
    recipe the downstream runner leaves should copy — cannot get it wrong.
    """
    seed = evaluate_series(_series(_BASELINE_STAMPS[-1]), [], _config(), None)
    return seed.grandfather, seed.snapshotted_metrics


def _config(budget: float = 1.0, *, baseline_window: int = 3) -> LimitsConfig:
    return LimitsConfig(
        false_alarm_budget=budget,
        runs_per_quarter=90,
        min_samples=_MIN_SAMPLES,
        baseline_window=baseline_window,
    )


class TestLimitsArtifact:
    """``limits-current.json`` is ONE file doing two jobs.

    It is the evaluator's persisted state (the grandfather set it resumes
    from) AND the dashboard's alarm source. Deliberately not two files:
    splitting them would let the published alarms drift from the limits that
    actually produced them, which is precisely the INV-1/INV-5 re-implementation
    hazard the PRD names. Making one file safe for both jobs is what
    ``test_state_continuity_...`` below is for.
    """

    def test_path_is_the_documented_shape(self, tmp_path):
        assert (
            limits_artifact_path(tmp_path, 'e1-retrieval-health')
            == tmp_path / 'e1-retrieval-health' / 'limits-current.json'
        )

    def _write(self, tmp_path, stamp=_QUIET_STAMP, budget=1.0, grandfather=None):
        result = evaluate_series(_series(stamp), _baseline(), _config(budget), grandfather)
        path = limits_artifact_path(tmp_path, result.eval_id)
        write_limits_artifact(result, path)
        return result, path

    def test_artifact_records_the_alpha_derivation(self, tmp_path):
        """Not just the derived alpha — the inputs it came from.

        G6 asks for "a calibration output with recorded provenance". An alpha
        alone is a magic number in a file; alpha alongside the budget, cadence
        and metric count that produced it can be re-derived and argued with.
        """
        _, path = self._write(tmp_path)
        data = json.loads(path.read_text())

        assert data['alpha'] == pytest.approx(1 / 360)
        assert data['false_alarm_budget'] == 1.0
        assert data['runs_per_quarter'] == 90
        assert data['alarmed_metric_count'] == _ALARMED_METRIC_COUNT
        assert data['alpha'] == pytest.approx(
            data['false_alarm_budget'] / (data['runs_per_quarter'] * data['alarmed_metric_count'])
        )

    def test_artifact_records_its_provenance(self, tmp_path):
        _, path = self._write(tmp_path)
        data = json.loads(path.read_text())

        assert data['schema_version'] == 1
        assert data['eval_id'] == 'e1-retrieval-health'
        assert data['run_stamp'] == _QUIET_STAMP
        assert data['generator']
        assert data['baseline_run_stamps'] == list(_BASELINE_STAMPS)

    def test_artifact_records_a_rule_kind_per_metric(self, tmp_path):
        result, path = self._write(tmp_path)
        data = json.loads(path.read_text())

        by_id = {v['metric_id']: v for v in data['verdicts']}
        assert by_id.keys() == {v.metric_id for v in result.verdicts}
        assert by_id['canonical-in-top-5']['rule_kind'] == 'proportion'
        assert by_id['dangling-pointers']['rule_kind'] == 'count'
        assert by_id['topic-canonical-present']['rule_kind'] == 'tripwire'
        assert by_id['search-latency-p50-ms']['rule_kind'] == 'scalar'

    def test_artifact_records_the_grandfather_set_and_its_hash(self, tmp_path):
        result, path = self._write(tmp_path)
        data = json.loads(path.read_text())

        assert data['grandfather_set'] == sorted(result.grandfather)
        assert data['grandfather_set_hash'] == grandfather_set_hash(result.grandfather)

    def test_artifact_records_the_snapshotted_metric_ledger(self, tmp_path):
        # The other half of the resumable state: without it a tripwire added
        # mid-programme cannot be told from one whose every failure was fixed.
        result, path = self._write(tmp_path)
        data = json.loads(path.read_text())

        assert data['snapshotted_metric_ids'] == sorted(result.snapshotted_metrics)
        assert data['snapshotted_metric_ids'] == ['topic-canonical-present']

    def test_artifact_records_the_current_alarms(self, tmp_path):
        # The regression run, evaluated against a seeded grandfather set, so
        # all three alarm kinds are present at once.
        grandfather, snapshotted = _seeded_state()
        result = evaluate_series(
            _series(_REGRESSION_STAMP), _baseline(), _config(), grandfather, snapshotted
        )
        path = limits_artifact_path(tmp_path, result.eval_id)
        write_limits_artifact(result, path)
        data = json.loads(path.read_text())

        # An equality, not an `any`: a membership check would still pass with
        # the per-metric scoping deleted, because the extra alarms a mis-seeded
        # run produces are additions rather than removals.
        assert {(a['metric_id'], a['item_key']) for a in data['alarms']} == {
            ('canonical-in-top-5', None),
            ('dangling-pointers', None),
            ('topic-canonical-present', 't-worktree-lifecycle'),
        }
        assert len(data['alarms']) == len(result.alarms)
        assert all('detail' in a for a in data['alarms'])
        # t-recon-watcher-triage fails in every run and is grandfathered, so its
        # ABSENCE above is the seed actually taking effect.
        assert 'topic-canonical-present::t-recon-watcher-triage' in data['grandfather_set']
        assert all(GRANDFATHER_KEY_SEPARATOR in key for key in data['grandfather_set'])

    def test_round_trip_restores_the_grandfather_set_exactly(self, tmp_path):
        result, path = self._write(tmp_path)
        restored = load_limits_artifact(path)

        assert restored.grandfather_set == sorted(result.grandfather)
        assert restored.grandfather_set_hash == result.grandfather_hash
        assert restored.snapshotted_metric_ids == sorted(result.snapshotted_metrics)
        assert restored.alpha == result.alpha
        assert restored.run_stamp == result.run_stamp

    def test_state_continuity_a_rerun_from_the_reloaded_state_alarms_nothing(self, tmp_path):
        """The property that makes one file safe to use as two things.

        Evaluate, persist, reload, re-evaluate the SAME run from the reloaded
        state: nothing may alarm and the known-bad list may not move. If a
        round-trip through the artifact perturbed the state, the file would be
        unusable as resumable state — and every restart would produce a burst
        of phantom alarms.
        """
        first, path = self._write(tmp_path)
        assert first.alarms == ()

        restored = load_limits_artifact(path)
        second = evaluate_series(
            _series(_QUIET_STAMP),
            _baseline(),
            _config(),
            frozenset(restored.grandfather_set),
            restored.snapshotted_metric_ids,
        )

        assert second.alarms == ()
        assert second.grandfather == first.grandfather
        assert second.grandfather_hash == first.grandfather_hash

    def test_writer_creates_parent_directories(self, tmp_path):
        result = evaluate_series(_series(_QUIET_STAMP), _baseline(), _config(), None)
        path = limits_artifact_path(tmp_path / 'nested' / 'deeper', result.eval_id)
        write_limits_artifact(result, path)
        assert path.is_file()

    def test_writer_leaves_no_temp_file(self, tmp_path):
        _, path = self._write(tmp_path)
        assert sorted(p.name for p in path.parent.iterdir()) == ['limits-current.json']

    def test_bytes_are_stable_across_two_identical_writes(self, tmp_path):
        result, path = self._write(tmp_path)
        first = path.read_bytes()
        write_limits_artifact(result, path)
        assert path.read_bytes() == first

    def test_a_failed_write_does_not_corrupt_the_previous_artifact(self, tmp_path, monkeypatch):
        """A crashed run must not cost the operator their grandfather set.

        The artifact is resumable state, so a torn write is worse than no
        write: it would take the known-bad list with it. The mkstemp+replace
        pattern means the old file is either wholly replaced or wholly intact.
        """
        first, path = self._write(tmp_path)
        original = path.read_bytes()

        def boom(*args, **kwargs):
            raise OSError('disk full')

        monkeypatch.setattr(os, 'replace', boom)
        later = evaluate_series(_series(_REGRESSION_STAMP), _baseline(), _config(), None)
        with pytest.raises(OSError):
            write_limits_artifact(later, path)
        monkeypatch.undo()

        assert path.read_bytes() == original
        assert load_limits_artifact(path).grandfather_set_hash == first.grandfather_hash
        assert sorted(p.name for p in path.parent.iterdir()) == ['limits-current.json']

    def test_a_wrong_schema_version_is_rejected(self, tmp_path):
        # A pinned Literal, so a future version is a loud validation failure
        # rather than a silent misread of fields that moved.
        _, path = self._write(tmp_path)
        data = json.loads(path.read_text())
        data['schema_version'] = 2
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError):
            load_limits_artifact(path)

    def test_an_unknown_field_is_rejected(self, tmp_path):
        # extra='forbid': the evaluator's own state file gets the same strict
        # treatment as the metric series it judges.
        _, path = self._write(tmp_path)
        data = json.loads(path.read_text())
        data['grandfather_sett'] = []  # a typo an author would want caught
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError):
            load_limits_artifact(path)

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


class TestARunWithNothingAlarmEligible:
    """A run that cannot alarm must still REPORT, not crash.

    ``derive_alpha`` rightly rejects a zero metric count — no finite alpha
    splits a budget zero ways — but ``evaluate_series`` used to hand it
    ``len(alarm_eligible)`` unconditionally, so two series that are perfectly
    valid under the M1 schema blew up instead of producing verdicts: one whose
    metrics are all ``kind='scalar'`` (the module's own contract says scalars
    are "reported but never alarmed" — unreportable is not that), and one whose
    ``metrics`` list is empty (M1 sets no ``min_length``, so a probe set on its
    first run before any metric is wired is a legal series).

    ``result.alpha`` is ``None`` in this state rather than a stand-in number:
    any fabricated bar would be exactly the a-priori threshold G6 forbids.
    """

    CONFIG = LimitsConfig(
        false_alarm_budget=1.0, runs_per_quarter=90, min_samples=1, baseline_window=3
    )

    def _series(self, metrics: list[Metric]) -> MetricSeries:
        return MetricSeries(
            schema_version=1,
            eval_id='e1-scalar-only',
            run_stamp='20260801T000000Z',
            corpus=Corpus(project_id='dark_factory'),
            metrics=metrics,
        )

    def _scalars(self) -> list[Metric]:
        return [
            Metric(metric_id='mean-latency-ms', kind='scalar', value=42.5, n=30),
            Metric(metric_id='index-size-mb', kind='scalar', value=17.0, n=1),
        ]

    def test_a_scalar_only_series_reports_verdicts_and_never_alarms(self):
        result = evaluate_series(self._series(self._scalars()), [], self.CONFIG, None)
        assert result.alarms == ()
        assert result.alpha is None
        assert result.alarmed_metric_count == 0
        assert [v.metric_id for v in result.verdicts] == ['mean-latency-ms', 'index-size-mb']
        assert {v.rule_kind for v in result.verdicts} == {'scalar'}
        assert {v.status for v in result.verdicts} == {'ok'}
        # The reported values survive — the point of a scalar is the trend.
        assert [v.value for v in result.verdicts] == [42.5, 17.0]

    def test_an_empty_series_evaluates_to_no_verdicts_and_no_alarms(self):
        result = evaluate_series(self._series([]), [], self.CONFIG, None)
        assert result.verdicts == ()
        assert result.alarms == ()
        assert result.alpha is None
        assert result.alarmed_metric_count == 0

    def test_derive_alpha_itself_still_rejects_a_zero_count(self):
        # The short-circuit belongs to the caller, which knows an empty split is
        # a legitimate state; the pure function keeps failing loudly.
        with pytest.raises(ValueError, match='alarmed_metric_count'):
            derive_alpha(1.0, 90, 0)

    def test_a_scalar_only_run_round_trips_through_the_artifact_with_a_null_alpha(self, tmp_path):
        result = evaluate_series(self._series(self._scalars()), [], self.CONFIG, None)
        path = write_limits_artifact(result, limits_artifact_path(tmp_path, result.eval_id))

        # `null`, not an omitted key and not a substituted default: a dashboard
        # reading this must render "no alarm rules in this run".
        raw = json.loads(path.read_text(encoding='utf-8'))
        assert 'alpha' in raw
        assert raw['alpha'] is None
        assert raw['alarmed_metric_count'] == 0
        assert raw['alarms'] == []

        loaded = load_limits_artifact(path)
        assert loaded.alpha is None
        assert loaded.alarmed_metric_count == 0
        assert [v.metric_id for v in loaded.verdicts] == ['mean-latency-ms', 'index-size-mb']

    def test_adding_a_first_alarming_metric_starts_deriving_an_alpha(self):
        # The state is transitional, not terminal: alpha appears as soon as
        # anything can spend the budget, at the full undivided share.
        metrics = [*self._scalars(), _tripwire({'t1': True}, metric_id='probe-a')]
        result = evaluate_series(self._series(metrics), [], self.CONFIG, None)
        assert result.alarmed_metric_count == 1
        assert result.alpha == derive_alpha(1.0, 90, 1)
        assert result.alarms == ()
