"""shared.quality_series.stats: the pure statistics behind the quality series (S6 worked numbers)."""

from __future__ import annotations

import hashlib
import math
import statistics
from dataclasses import astuple
from datetime import UTC, datetime, timedelta

import pytest

from shared.quality_series.stats import (
    BaselineState,
    CostEffect,
    Effect,
    NotYet,
    RateState,
    RegimeEvent,
    Segment,
    baseline_rate,
    before_after_cost,
    before_after_rate,
    block_rate_datum,
    bootstrap_interval,
    cap_occurrences,
    content_seed,
    cost_fit,
    cusum_alarms,
    cusum_design,
    ewma_limits,
    ewma_series,
    explain_alarm,
    log_residuals,
    newcombe_difference,
    reference_cost,
    regime_segments,
    theil_sen,
    wilson_interval,
)

_T0 = datetime(2026, 10, 1, tzinfo=UTC)


def _hourly(n: int) -> list[datetime]:
    return [_T0 + timedelta(hours=i) for i in range(n)]


class TestWilsonInterval:
    @pytest.mark.parametrize(
        ('k', 'n', 'lo', 'hi'),
        [(10, 100, 0.0552, 0.1744), (3, 30, 0.0346, 0.2562)],
    )
    def test_matches_the_published_values(self, k: int, n: int, lo: float, hi: float) -> None:
        assert wilson_interval(k, n) == pytest.approx((lo, hi), abs=5e-4)

    def test_zero_successes_has_a_zero_floor_and_an_honest_ceiling(self) -> None:
        lo, hi = wilson_interval(0, 100)
        assert lo == 0.0
        assert hi == pytest.approx(0.0370, abs=5e-4)

    @pytest.mark.parametrize(('k', 'n'), [(0, 0), (-1, 10), (11, 10)])
    def test_impossible_counts_are_refused_naming_k_and_n(self, k: int, n: int) -> None:
        with pytest.raises(ValueError) as excinfo:
            wilson_interval(k, n)
        assert f'k={k}' in str(excinfo.value)
        assert f'n={n}' in str(excinfo.value)


class TestNewcombeDifference:
    def test_a_halving_on_100_per_side_is_not_a_finding(self) -> None:
        assert newcombe_difference(10, 100, 5, 100) == pytest.approx(
            (-0.05, -0.1296, 0.0263), abs=5e-4
        )

    def test_a_halving_on_300_per_side_is_a_finding(self) -> None:
        assert newcombe_difference(30, 300, 15, 300) == pytest.approx(
            (-0.05, -0.0937, -0.0076), abs=5e-4
        )


class TestContentSeed:
    def test_is_insensitive_to_order(self) -> None:
        assert content_seed(['3', '1', '2']) == content_seed(['2', '3', '1'])

    def test_differs_for_different_id_sets(self) -> None:
        assert content_seed(['1', '2', '3']) != content_seed(['1', '2', '4'])

    def test_is_the_first_eight_sha256_bytes_of_the_sorted_joined_ids(self) -> None:
        expected = int.from_bytes(hashlib.sha256(b'1,2,3').digest()[:8], 'big')
        assert content_seed({1, 2, 3}) == expected


class TestBootstrapInterval:
    _SYMMETRIC = [float(x) for x in range(1, 42)]

    def test_is_deterministic_for_equal_inputs(self) -> None:
        first = bootstrap_interval(self._SYMMETRIC, statistics.median, seed=7, B=1000)
        second = bootstrap_interval(self._SYMMETRIC, statistics.median, seed=7, B=1000)
        assert first == second

    def test_brackets_the_sample_statistic_of_a_symmetric_sample(self) -> None:
        lo, hi = bootstrap_interval(self._SYMMETRIC, statistics.median, seed=7, B=1000)
        assert lo < statistics.median(self._SYMMETRIC) < hi

    def test_a_constant_sample_has_a_degenerate_interval(self) -> None:
        assert bootstrap_interval([2.5] * 20, statistics.median, seed=1, B=200) == (2.5, 2.5)

    def test_an_empty_sample_is_refused(self) -> None:
        with pytest.raises(ValueError):
            bootstrap_interval([], statistics.median, seed=1)


class TestEwma:
    def test_series_follows_the_recursion_from_z0(self) -> None:
        assert ewma_series([1, 0, 0], 0.5, 0.0) == [0.5, 0.25, 0.125]

    def test_asymptotic_limits_at_a_ten_percent_baseline(self) -> None:
        limits = ewma_limits(0.10, 0.05, 3, t=10_000)
        assert limits.ucl == pytest.approx(0.2441, abs=1e-4)
        assert limits.lcl is None

    def test_a_positive_lower_limit_is_reported(self) -> None:
        limits = ewma_limits(0.30, 0.05, 3, t=10_000)
        assert limits.lcl == pytest.approx(0.0799, abs=1e-4)

    def test_early_limits_are_narrower_than_asymptotic_ones(self) -> None:
        early = ewma_limits(0.30, 0.05, 3, t=1)
        late = ewma_limits(0.30, 0.05, 3, t=10_000)
        assert early.ucl < late.ucl
        assert early.lcl is not None and late.lcl is not None
        assert early.lcl > late.lcl


class TestCusum:
    def test_upward_alarms_reset_and_report_the_excursion_onset(self) -> None:
        assert cusum_alarms([0] * 200 + [1] * 20, p0=0.1, p1=0.2, h=5) == [(207, 200), (215, 208)]

    def test_downward_design_accumulates_on_zeros(self) -> None:
        assert cusum_alarms([0] * 100, 0.1, 0.05, 5) == [(92, 0)]

    @pytest.mark.parametrize(
        ('p0', 'p1', 'h'),
        [(0.0, 0.2, 5), (1.0, 0.2, 5), (0.1, 0.0, 5), (0.1, 1.0, 5), (0.1, 0.1, 5), (0.1, 0.2, 0)],
    )
    def test_invalid_designs_are_refused(self, p0: float, p1: float, h: float) -> None:
        with pytest.raises(ValueError):
            cusum_alarms([0, 1], p0, p1, h)

    def test_design_doubles_and_halves_below_one_third(self) -> None:
        assert astuple(cusum_design(0.10)) == pytest.approx((0.2, 0.05, 5))

    def test_design_up_shift_stays_a_probability_at_one_half(self) -> None:
        assert astuple(cusum_design(0.5)) == pytest.approx((0.75, 0.25, 5))

    @pytest.mark.parametrize('p0', [0.0, 1.0, -0.1])
    def test_design_refuses_a_degenerate_baseline(self, p0: float) -> None:
        with pytest.raises(ValueError):
            cusum_design(p0)


class TestExplainAlarm:
    def test_an_event_ten_landings_before_the_onset_explains_it(self) -> None:
        landed_at = _hourly(200)
        event = RegimeEvent('e1', landed_at[140])
        assert explain_alarm(150, landed_at[150], [event], landed_at) == 'e1'

    def test_an_event_far_in_landings_and_days_explains_nothing(self) -> None:
        landed_at = _hourly(200)
        event = RegimeEvent('e1', landed_at[50])
        assert explain_alarm(150, landed_at[150], [event], landed_at) is None

    def test_the_day_rule_reaches_past_the_landing_rule(self) -> None:
        landed_at = _hourly(200)
        event = RegimeEvent('e1', landed_at[110])
        assert explain_alarm(150, landed_at[150], [event], landed_at) == 'e1'

    def test_the_nearest_event_wins(self) -> None:
        landed_at = _hourly(200)
        events = [RegimeEvent('far', landed_at[140]), RegimeEvent('near', landed_at[145])]
        assert explain_alarm(150, landed_at[150], events, landed_at) == 'near'

    def test_equally_near_events_resolve_by_event_id(self) -> None:
        landed_at = _hourly(200)
        events = [RegimeEvent('zeta', landed_at[145]), RegimeEvent('alpha', landed_at[155])]
        assert explain_alarm(150, landed_at[150], events, landed_at) == 'alpha'


class TestTheilSen:
    _XS = [float(x) for x in range(1, 21)]

    def test_recovers_an_exact_line(self) -> None:
        ys = [0.7 + 0.57 * x for x in self._XS]
        assert theil_sen(self._XS, ys) == pytest.approx((0.7, 0.57), abs=1e-9)

    def test_one_gross_outlier_moves_nothing(self) -> None:
        ys = [0.7 + 0.57 * x for x in self._XS]
        ys[7] = 500.0
        assert theil_sen(self._XS, ys) == pytest.approx((0.7, 0.57), abs=1e-9)

    @pytest.mark.parametrize('xs', [[3.0], [3.0, 3.0, 3.0]])
    def test_fewer_than_two_distinct_xs_are_refused(self, xs: list[float]) -> None:
        with pytest.raises(ValueError):
            theil_sen(xs, [1.0] * len(xs))


class TestCostModel:
    _LINES = [10, 50, 100, 400, 1000, 2500, 2]
    _COSTS = [2 * n**0.5 for n in _LINES[:-1]] + [40.0]

    def test_reference_cost_reads_the_power_law_at_100_lines(self) -> None:
        assert reference_cost(self._LINES, self._COSTS) == pytest.approx(20.0, abs=1e-9)

    def test_reference_cost_at_another_size(self) -> None:
        assert reference_cost(self._LINES, self._COSTS, ref=400) == pytest.approx(40.0, abs=1e-9)

    def test_fit_excludes_landings_below_min_lines(self) -> None:
        a, b = cost_fit(self._LINES, self._COSTS)
        assert (a, b) == pytest.approx((math.log(2), 0.5), abs=1e-9)

    def test_log_residuals_measure_against_the_fit_and_skip_unmeasurable_points(self) -> None:
        a, b = math.log(2), 0.5
        residuals = log_residuals([100, 0, 100, 400], [20.0, 5.0, 0.0, 40.0 * math.e], a, b)
        assert residuals[0] == pytest.approx(0.0, abs=1e-9)
        assert residuals[1] is None
        assert residuals[2] is None
        assert residuals[3] == pytest.approx(1.0, abs=1e-9)


class TestCapOccurrences:
    def test_bearing_flags_and_mass_events_at_the_floor(self) -> None:
        capped = cap_occurrences([1, 3, 0, 25, 478], baseline_median=2, floor=20)
        assert capped.bearing == (True, True, False, True, True)
        assert capped.mass_event_indices == (3, 4)
        assert capped.cap == 20

    def test_the_baseline_relative_term_raises_the_cap(self) -> None:
        capped = cap_occurrences([1, 3, 0, 25, 478], baseline_median=5, floor=20)
        assert capped.mass_event_indices == (4,)
        assert capped.cap == 50


class TestRegimeSegments:
    def test_series_tags_and_artefact_ranges(self) -> None:
        landed_at = _hourly(50)
        events = [
            RegimeEvent('e1', landed_at[10]),
            RegimeEvent('e2', landed_at[20], series=('reviewer_blocking',)),
            RegimeEvent(
                'a1', landed_at[30], until=landed_at[35], series=('flakes_confirmed',),
                artefact=True,
            ),
        ]
        segmented = regime_segments(landed_at, events, series='flakes_confirmed')
        assert segmented.segments == (Segment(0, 10, None), Segment(10, 50, 'e1'))
        assert segmented.excluded == frozenset(range(30, 36))

    def test_an_event_before_every_landing_opens_the_first_segment(self) -> None:
        landed_at = _hourly(5)
        events = [RegimeEvent('e0', _T0 - timedelta(days=1))]
        segmented = regime_segments(landed_at, events, series='x')
        assert segmented.segments == (Segment(0, 5, 'e0'),)

    def test_an_event_after_every_landing_adds_no_boundary(self) -> None:
        landed_at = _hourly(5)
        events = [RegimeEvent('late', _T0 + timedelta(days=30))]
        segmented = regime_segments(landed_at, events, series='x')
        assert segmented.segments == (Segment(0, 5, None),)

    def test_an_artefact_needs_an_end(self) -> None:
        with pytest.raises(ValueError):
            RegimeEvent('a', _T0, artefact=True)


class TestBaselineRate:
    def test_two_hundred_observations_freeze_the_baseline(self) -> None:
        assert baseline_rate([1] * 20 + [0] * 180) == (pytest.approx(0.1), BaselineState.FROZEN)

    def test_only_the_first_n_base_observations_count(self) -> None:
        xs = [1] * 20 + [0] * 180 + [1] * 100
        assert baseline_rate(xs) == (pytest.approx(0.1), BaselineState.FROZEN)

    def test_fewer_than_n_base_is_provisional_over_everything(self) -> None:
        xs = [1] * 15 + [0] * 135
        assert baseline_rate(xs) == (pytest.approx(0.1), BaselineState.PROVISIONAL)

    def test_below_n_min_there_is_no_baseline(self) -> None:
        assert baseline_rate([0] * 99) == (None, None)

    def test_unmeasured_landings_are_skipped_not_counted(self) -> None:
        assert baseline_rate([0] * 99 + [None] * 50) == (None, None)
        xs: list[int | None] = [None, 1] * 20 + [0] * 180
        assert baseline_rate(xs) == (pytest.approx(0.1), BaselineState.FROZEN)


class TestBlockRateDatum:
    def test_a_full_block_is_a_rate_over_the_last_n_block(self) -> None:
        xs = [1] * 50 + [1] * 10 + [0] * 90
        ids = [str(i) for i in range(len(xs))]
        datum = block_rate_datum(xs, ids)
        assert datum.state is RateState.RATE
        assert (datum.k, datum.n) == (10, 100)
        assert datum.rate == pytest.approx(0.1)
        assert (datum.lo, datum.hi) == pytest.approx(wilson_interval(10, 100))
        assert datum.window_task_ids == tuple(ids[50:])
        assert datum.reason is None

    def test_between_n_min_and_n_block_is_k_of_n(self) -> None:
        datum = block_rate_datum([1] * 4 + [0] * 36, [str(i) for i in range(40)])
        assert datum.state is RateState.K_OF_N
        assert (datum.k, datum.n) == (4, 40)
        assert (datum.lo, datum.hi) == pytest.approx(wilson_interval(4, 40))

    def test_below_n_min_is_unknown_with_a_reason(self) -> None:
        datum = block_rate_datum([1] + [0] * 28, [str(i) for i in range(29)])
        assert datum.state is RateState.UNKNOWN
        assert (datum.rate, datum.lo, datum.hi) == (None, None, None)
        assert datum.reason is not None
        assert '29' in datum.reason and '30' in datum.reason

    def test_unmeasured_landings_drop_from_k_and_n(self) -> None:
        xs: list[int | None] = [1, None, 0, None] * 10
        datum = block_rate_datum(xs, [str(i) for i in range(40)])
        assert (datum.k, datum.n) == (10, 20)
        assert datum.state is RateState.UNKNOWN

    def test_xs_and_task_ids_must_pair(self) -> None:
        with pytest.raises(ValueError):
            block_rate_datum([0, 1], ['a'])


class TestBeforeAfterRate:
    def test_a_full_after_window_is_a_newcombe_effect_on_capped_windows(self) -> None:
        before = [1] * 100 + [1] * 30 + [0] * 270
        after = [1] * 5 + [0] * 95 + [1] * 50 + [0] * 250 + [1] * 10
        effect = before_after_rate(before, after)
        assert isinstance(effect, Effect)
        assert (effect.n_before, effect.n_after) == (300, 300)
        d, lo, hi = newcombe_difference(30, 300, 55, 300)
        assert (effect.d, effect.lo, effect.hi) == pytest.approx((d, lo, hi))

    def test_exactly_n_min_after_is_enough(self) -> None:
        effect = before_after_rate([1] * 10 + [0] * 90, [1] * 5 + [0] * 95)
        assert isinstance(effect, Effect)
        assert (effect.d, effect.lo, effect.hi) == pytest.approx(
            newcombe_difference(10, 100, 5, 100)
        )

    def test_a_short_after_window_is_not_yet_with_an_eta(self) -> None:
        verdict = before_after_rate([1] * 10 + [0] * 90, [0] * 40, landings_per_day=20)
        assert isinstance(verdict, NotYet)
        assert verdict.n_after == 40
        assert verdict.eta_days == pytest.approx(3.0)
        assert verdict.min_detectable is not None and verdict.min_detectable > 0

    def test_without_a_velocity_there_is_no_eta(self) -> None:
        verdict = before_after_rate([1] * 10 + [0] * 90, [0] * 40)
        assert isinstance(verdict, NotYet)
        assert verdict.eta_days is None


class TestBeforeAfterCost:
    _LINES = [10 + 7 * i for i in range(120)]

    def _costs(self, scale: float) -> list[float]:
        return [scale * n**0.5 for n in self._LINES]

    def test_a_uniform_cost_increase_is_its_ratio_with_a_degenerate_band(self) -> None:
        effect = before_after_cost(
            self._LINES, self._costs(2.0), self._LINES, self._costs(3.0), seed=11, B=200
        )
        assert isinstance(effect, CostEffect)
        assert (effect.ratio, effect.lo, effect.hi) == pytest.approx((1.5, 1.5, 1.5), abs=1e-9)

    def test_is_deterministic_for_one_seed(self) -> None:
        jitter = [1.0 + 0.1 * ((i * 7) % 5) for i in range(len(self._LINES))]
        after = [c * j for c, j in zip(self._costs(3.0), jitter, strict=True)]
        first = before_after_cost(self._LINES, self._costs(2.0), self._LINES, after, seed=5, B=200)
        second = before_after_cost(self._LINES, self._costs(2.0), self._LINES, after, seed=5, B=200)
        assert first == second

    def test_a_short_after_window_is_not_yet(self) -> None:
        verdict = before_after_cost(
            self._LINES, self._costs(2.0), self._LINES[:50], self._costs(3.0)[:50], seed=1
        )
        assert isinstance(verdict, NotYet)
        assert verdict.n_after == 50
