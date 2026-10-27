"""The shadow coordinator's storm escape (plans/live-shadow-eval-prd.md decision 10, INV-4).

Part 1 covers the pure storm policy; the coordinator escape follows in part 2.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta

import pytest

from orchestrator.event_store import ShadowSkipReason
from orchestrator.shadow_eval import (
    PRODUCTION_COST_BASIS,
    TRAILING_WINDOW,
    CostPair,
    CostRatioVerdict,
    HostVerdict,
    OpenInputs,
    RatioSkip,
    RunEndedCell,
    StormFacts,
    StormTrip,
    budget_exhausted,
    cost_ratio_verdict,
    decide_open,
    evaluate_storm,
    failure_streak_tripped,
    storm_window_start,
)
from orchestrator.shadow_types import CostBasis, ShadowCellState, ShadowShape, ShadowStormCause

_S = ShadowCellState
_M = CostBasis.METERED
_SUB = CostBasis.SUBSCRIPTION


class TestFailureStreak:
    def test_the_newest_threshold_cells_all_failed_or_expired_trip(self):
        states = [_S.FAILED, _S.EXPIRED, _S.FAILED, _S.FAILED, _S.EXPIRED, _S.SETTLED]
        assert failure_streak_tripped(states, 5) is True

    @pytest.mark.parametrize('success', [_S.AWAITING_REFERENCE, _S.SETTLED])
    def test_a_successful_run_inside_the_window_breaks_the_streak(self, success):
        states = [_S.FAILED, _S.FAILED, success, _S.FAILED, _S.FAILED, _S.FAILED]
        assert failure_streak_tripped(states, 5) is False

    def test_fewer_than_threshold_cells_never_trip(self):
        assert failure_streak_tripped([_S.FAILED] * 4, 5) is False


def _pair(cell: float, cell_basis: CostBasis, production: float) -> CostPair:
    return CostPair(
        cell_cost_usd=cell,
        cell_basis=cell_basis,
        production_cost_usd=production,
        production_basis=PRODUCTION_COST_BASIS,
    )


class TestCostRatioVerdict:
    def test_production_is_token_metered(self):
        assert PRODUCTION_COST_BASIS is CostBasis.METERED

    def test_metered_pairs_give_a_measured_ratio(self):
        verdict = cost_ratio_verdict([_pair(3.0, _M, 1.0), _pair(1.0, _M, 1.0)], ceiling=3.0)

        assert verdict.ratio == 2.0
        assert verdict.skipped is None
        assert verdict.tripped is False

    def test_a_ratio_over_the_ceiling_trips(self):
        verdict = cost_ratio_verdict([_pair(7.0, _M, 2.0)], ceiling=3.0)

        assert verdict.ratio == 3.5
        assert verdict.tripped is True

    def test_a_ratio_at_the_ceiling_does_not_trip(self):
        assert cost_ratio_verdict([_pair(6.0, _M, 2.0)], ceiling=3.0).tripped is False

    def test_a_subscription_arm_is_a_named_skip_never_a_zero_ratio(self):
        verdict = cost_ratio_verdict([_pair(0.0001, _SUB, 1.0)], ceiling=3.0)

        assert verdict.skipped is RatioSkip.COST_BASIS_MISMATCH
        assert verdict.ratio is None
        assert verdict.tripped is False

    def test_mixed_bases_compare_only_the_matching_pairs(self):
        verdict = cost_ratio_verdict(
            [_pair(8.0, _M, 2.0), _pair(0.0001, _SUB, 100.0)], ceiling=3.0,
        )

        assert verdict.ratio == 4.0
        assert verdict.tripped is True

    def test_no_pairs_is_a_named_skip(self):
        verdict = cost_ratio_verdict([], ceiling=3.0)

        assert (verdict.ratio, verdict.skipped) == (None, RatioSkip.NO_PAIRS)

    def test_a_zero_production_sum_is_a_named_skip(self):
        verdict = cost_ratio_verdict([_pair(1.0, _M, 0.0)], ceiling=3.0)

        assert (verdict.ratio, verdict.skipped) == (None, RatioSkip.NO_PRODUCTION_COST)

    @pytest.mark.parametrize(
        'ratio,skipped',
        [(None, None), (1.0, RatioSkip.NO_PAIRS)],
        ids=['neither', 'both'],
    )
    def test_exactly_one_of_ratio_and_skip_is_set(self, ratio, skipped):
        with pytest.raises(ValueError, match='ratio'):
            CostRatioVerdict(ratio=ratio, skipped=skipped, ceiling=3.0)


def _ended(cell_id: str, state: ShadowCellState) -> RunEndedCell:
    return RunEndedCell(cell_id=cell_id, state=state)


_CALM = StormFacts(
    spent_usd=0.5,
    daily_budget_usd=1.0,
    run_ends_newest_first=(_ended('c1', _S.SETTLED),),
    failure_streak_pause=3,
    ratios={'glm-5.3-endpoint': cost_ratio_verdict([_pair(1.0, _SUB, 1.0)], ceiling=3.0)},
)
_OVER_BUDGET = {'spent_usd': 1.2}
_STREAK = {
    'run_ends_newest_first': (
        _ended('c9', _S.FAILED),
        _ended('c8', _S.EXPIRED),
        _ended('c7', _S.FAILED),
    ),
}
_RATIO = {'ratios': {'minimax-m3-endpoint': cost_ratio_verdict([_pair(9.0, _M, 2.0)], ceiling=3.0)}}


class TestEvaluateStorm:
    def test_calm_facts_trip_nothing(self):
        assert evaluate_storm(_CALM) is None

    def test_the_budget_trip_names_spend_and_cap(self):
        trip = evaluate_storm(dataclasses.replace(_CALM, **_OVER_BUDGET))

        assert isinstance(trip, StormTrip)
        assert trip.cause is ShadowStormCause.DAILY_BUDGET
        assert '1.20' in trip.detail
        assert '1.00' in trip.detail

    def test_the_streak_trip_names_its_cells(self):
        trip = evaluate_storm(dataclasses.replace(_CALM, **_STREAK))

        assert trip is not None
        assert trip.cause is ShadowStormCause.FAILURE_STREAK
        assert all(cell_id in trip.detail for cell_id in ('c9', 'c8', 'c7'))

    def test_the_ratio_trip_names_candidate_ratio_and_ceiling(self):
        trip = evaluate_storm(dataclasses.replace(_CALM, **_RATIO))

        assert trip is not None
        assert trip.cause is ShadowStormCause.COST_RATIO
        assert 'minimax-m3-endpoint' in trip.detail
        assert '4.50' in trip.detail
        assert '3.00' in trip.detail

    def test_budget_outranks_streak_outranks_ratio(self):
        everything = dataclasses.replace(_CALM, **_OVER_BUDGET, **_STREAK, **_RATIO)
        streak_and_ratio = dataclasses.replace(_CALM, **_STREAK, **_RATIO)

        causes = [evaluate_storm(facts) for facts in (everything, streak_and_ratio)]

        assert [trip and trip.cause for trip in causes] == [
            ShadowStormCause.DAILY_BUDGET,
            ShadowStormCause.FAILURE_STREAK,
        ]


def _inputs_with_budget(spent: float, cap: float) -> OpenInputs:
    return OpenInputs(
        task_id='5387',
        shape=ShadowShape.IMPLEMENTER,
        candidate='c',
        stratum=('df', 'feature', 'full'),
        base_sha='a' * 40,
        seed=0,
        sample_rate=1.0,
        enabled=True,
        enabled_shapes=frozenset({ShadowShape.IMPLEMENTER}),
        end_to_end_candidates=frozenset(),
        paused=False,
        already_open=False,
        has_credentials=True,
        isolation_available=True,
        credits_exhausted_recently=False,
        spent_usd=spent,
        daily_budget_usd=cap,
        held_cells=0,
        max_concurrent=1,
        host=HostVerdict.CLEAR,
    )


@pytest.mark.parametrize('spent,cap', [(0.0, 0.0), (0.99, 1.0), (1.0, 1.0), (1.01, 1.0), (5.0, 50.0)])
def test_the_open_gate_and_the_storm_share_one_budget_rule(spent, cap):
    expected = budget_exhausted(spent, cap)
    storm = evaluate_storm(dataclasses.replace(_CALM, spent_usd=spent, daily_budget_usd=cap))
    decision = decide_open(_inputs_with_budget(spent, cap))

    assert (storm is not None and storm.cause is ShadowStormCause.DAILY_BUDGET) is expected
    assert (decision.reason is ShadowSkipReason.BUDGET_EXHAUSTED) is expected


_NOW = datetime(2026, 10, 27, 12, 0, tzinfo=UTC)


class TestStormWindow:
    def test_the_window_is_the_trailing_day(self):
        assert timedelta(hours=24) == TRAILING_WINDOW
        assert storm_window_start(_NOW, None) == _NOW - TRAILING_WINDOW

    def test_a_recent_resume_restarts_the_window(self):
        resumed = _NOW - timedelta(hours=2)
        assert storm_window_start(_NOW, resumed) == resumed

    def test_an_old_resume_leaves_the_trailing_day(self):
        assert storm_window_start(_NOW, _NOW - timedelta(days=3)) == _NOW - TRAILING_WINDOW
