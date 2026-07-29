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

from shared.memory_eval_limits import binomial_two_sided_p, poisson_two_sided_p


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
