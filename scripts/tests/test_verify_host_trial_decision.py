"""verify_host_trial_decision.decide: the fleet on-against-off decision line."""
from __future__ import annotations

import pytest
from verify_host_trial_decision import BlockLandings, Decision, decide


def on(landings: int, days: float, host_verifies: int = 5) -> BlockLandings:
    return BlockLandings(on=True, landings=landings, days=days, host_verifies=host_verifies)


def off(landings: int, days: float) -> BlockLandings:
    return BlockLandings(on=False, landings=landings, days=days, host_verifies=0)


def run(blocks: list[BlockLandings], *, resamples: int = 2000, seed: int = 0) -> Decision:
    return decide(blocks, host_cost_usd=252, seat_bill_usd=1375, resamples=resamples, seed=seed)


class TestTheComparison:
    def test_equal_per_block_rates_give_an_exact_effect_and_a_degenerate_interval(self):
        decision = run([on(30, 3), on(30, 3), off(24, 3), off(24, 3)])

        assert decision.on_rate == 10.0
        assert decision.off_rate == 8.0
        assert decision.effect == 0.25
        assert decision.interval is not None
        assert (decision.interval.low, decision.interval.high) == (0.25, 0.25)
        assert decision.reason is None
        assert (decision.on_blocks, decision.off_blocks) == (2, 2)

    def test_an_on_block_where_no_host_verify_ran_is_excluded_and_listed(self):
        decision = run([on(30, 3), on(99, 3, host_verifies=0), on(30, 3), off(24, 3), off(24, 3)])

        assert decision.excluded_on_blocks == (1,)
        assert decision.on_blocks == 2
        assert decision.on_rate == 10.0
        assert decision.effect == 0.25

    def test_the_same_seed_gives_the_same_interval(self):
        blocks = [on(30, 3), on(45, 3), on(20, 2), off(24, 3), off(18, 3), off(30, 4)]

        first, second = run(blocks, seed=7), run(blocks, seed=7)

        assert first.interval == second.interval

    def test_with_unequal_rates_the_interval_brackets_the_point_estimate(self):
        decision = run([on(30, 3), on(45, 3), on(20, 2), off(24, 3), off(18, 3), off(30, 4)])

        assert decision.effect is not None
        assert decision.interval is not None
        assert decision.interval.low < decision.effect < decision.interval.high


class TestNoComparison:
    def test_no_on_block_with_a_host_verify_gives_no_effect_never_zero(self):
        decision = run([on(30, 3, host_verifies=0), off(24, 3), off(24, 3)])

        assert decision.effect is None
        assert decision.interval is None
        assert decision.on_rate is None
        assert decision.reason is not None
        assert 'no verify ran on this host' in decision.reason

    def test_no_off_block_gives_no_effect(self):
        decision = run([on(30, 3), on(30, 3)])

        assert decision.effect is None
        assert decision.off_rate is None
        assert decision.reason is not None
        assert 'no off-block' in decision.reason

    def test_off_blocks_that_landed_nothing_give_no_effect(self):
        decision = run([on(30, 3), on(30, 3), off(0, 3), off(0, 3)])

        assert decision.effect is None
        assert decision.off_rate == 0.0
        assert decision.reason is not None
        assert 'landed nothing' in decision.reason

    def test_one_block_in_a_condition_gives_a_point_estimate_and_no_interval(self):
        decision = run([on(30, 3), off(24, 3), off(24, 3)])

        assert decision.effect == 0.25
        assert decision.interval is None
        assert decision.reason is not None
        assert '1 on-block' in decision.reason
        assert '2 off-blocks' in decision.reason


class TestCostAndValidation:
    def test_the_host_cost_share_is_its_cost_over_the_seat_bill(self):
        assert run([on(30, 3)]).host_cost_share == 252 / 1375

    @pytest.mark.parametrize('seat_bill_usd', [0, -1375])
    def test_a_seat_bill_that_is_not_positive_is_refused(self, seat_bill_usd):
        with pytest.raises(ValueError, match='seat'):
            decide([], host_cost_usd=252, seat_bill_usd=seat_bill_usd, resamples=10, seed=0)

    def test_a_negative_host_cost_is_refused(self):
        with pytest.raises(ValueError, match='host cost'):
            decide([], host_cost_usd=-1, seat_bill_usd=1375, resamples=10, seed=0)
