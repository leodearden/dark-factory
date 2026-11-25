"""verify_host_trial_allowance: account cycles and the allowance expired unused, an estimate."""
from __future__ import annotations

from datetime import UTC, datetime

from verify_host_trial_allowance import ALLOWANCE_RULE, CYCLE, account_cycles, expired_in_block
from verify_host_trial_blocks import Block, BlockOrigin, HostState
from verify_host_trial_stores import CapHit, Invocation

SINCE = datetime(2026, 11, 11, tzinfo=UTC)
UNTIL = datetime(2026, 11, 25, tzinfo=UTC)


def at(month_day: int, hour: int = 0) -> datetime:
    return datetime(2026, 11, month_day, hour, tzinfo=UTC)


def spend(account: str, when: datetime, usd: float) -> Invocation:
    return Invocation(when, account, usd)


R_A = at(16, 21)
"""max-a's capped reset; its lattice puts the next reset at the 23rd, 21:00."""
R_B = at(20, 10)
"""max-b's capped reset; its lattice puts the previous reset at the 13th, 10:00."""

TWO_ACCOUNTS_CAPS = (CapHit('max-a', at(15, 10), R_A), CapHit('max-b', at(19, 4), R_B))
TWO_ACCOUNTS_SPEND = (
    spend('max-a', at(12), 1500.0),
    spend('max-a', at(15), 500.0),
    spend('max-a', at(20), 500.0),
    spend('max-b', at(10), 1000.0),
    spend('max-b', at(14), 2400.0),
)


def cycles_by_reset(reading):
    return {(cycle.account, cycle.reset): cycle for cycle in reading.cycles}


class TestCappedCycles:
    def test_a_cap_hit_makes_its_cycle_capped_with_nothing_expired(self):
        reading = account_cycles(TWO_ACCOUNTS_CAPS, TWO_ACCOUNTS_SPEND, SINCE, UNTIL)

        cycle = cycles_by_reset(reading)['max-a', R_A]
        assert cycle.capped
        assert cycle.capped_at == at(15, 10)
        assert cycle.start == R_A - CYCLE
        assert cycle.spend_usd == 2000.0
        assert cycle.expired_usd == 0.0
        assert cycle.hours_capped == 35.0

    def test_one_cap_recorded_by_several_stores_is_one_cycle(self):
        caps = (*TWO_ACCOUNTS_CAPS, CapHit('max-a', at(15, 11), R_A), CapHit('max-a', at(15, 10), R_A))

        reading = account_cycles(caps, TWO_ACCOUNTS_SPEND, SINCE, UNTIL)

        assert [c.reset for c in reading.cycles if c.account == 'max-a' and c.capped] == [R_A]
        assert cycles_by_reset(reading)['max-a', R_A].capped_at == at(15, 10)


class TestTheLattice:
    def test_lattice_resets_in_the_window_without_a_cap_are_uncapped_cycles(self):
        reading = account_cycles(TWO_ACCOUNTS_CAPS, TWO_ACCOUNTS_SPEND, SINCE, UNTIL)

        assert sorted(cycles_by_reset(reading)) == [
            ('max-a', R_A),
            ('max-a', R_A + CYCLE),
            ('max-b', R_B - CYCLE),
            ('max-b', R_B),
        ]
        assert not cycles_by_reset(reading)['max-a', R_A + CYCLE].capped

    def test_a_capped_cycle_near_a_lattice_point_replaces_it(self):
        """A reset an hour off the lattice (a clock change) is the same cycle, not a second one."""
        caps = (CapHit('max-c', at(24, 1), at(24, 21)), CapHit('max-c', at(17, 2), at(17, 20)))

        reading = account_cycles(caps, (spend('max-c', at(15), 10.0),), SINCE, UNTIL)

        assert sorted(cycles_by_reset(reading)) == [('max-c', at(17, 20)), ('max-c', at(24, 21))]
        assert all(cycle.capped for cycle in reading.cycles)

    def test_an_account_with_no_activity_near_the_window_is_dormant_and_unplaced(self):
        caps = (*TWO_ACCOUNTS_CAPS, CapHit('max-old', at(1, 5), at(2, 21)))

        reading = account_cycles(caps, TWO_ACCOUNTS_SPEND, SINCE, UNTIL)

        assert 'max-old' not in {cycle.account for cycle in reading.cycles}
        assert reading.dormant_accounts == ('max-old',)


class TestTheEstimate:
    def test_the_estimate_is_the_median_capped_spend_and_uncapped_cycles_expire_the_rest(self):
        reading = account_cycles(TWO_ACCOUNTS_CAPS, TWO_ACCOUNTS_SPEND, SINCE, UNTIL)

        assert reading.estimate_usd == 2200.0
        cycles = cycles_by_reset(reading)
        assert cycles['max-a', R_A + CYCLE].spend_usd == 500.0
        assert cycles['max-a', R_A + CYCLE].expired_usd == 1700.0
        assert cycles['max-b', R_B - CYCLE].expired_usd == 1200.0

    def test_spend_above_the_estimate_expires_nothing(self):
        heavy = (*TWO_ACCOUNTS_SPEND, spend('max-a', at(21), 5000.0))

        reading = account_cycles(TWO_ACCOUNTS_CAPS, heavy, SINCE, UNTIL)

        assert cycles_by_reset(reading)['max-a', R_A + CYCLE].expired_usd == 0.0

    def test_with_no_capped_cycle_in_the_window_the_estimate_and_expiry_are_unknown(self):
        caps = (CapHit('max-a', at(1, 5), at(2, 21)),)

        reading = account_cycles(caps, (spend('max-a', at(12), 100.0),), SINCE, UNTIL)

        assert reading.estimate_usd is None
        assert [cycle.reset for cycle in reading.cycles] == [at(16, 21), at(23, 21)]
        assert all(cycle.expired_usd is None for cycle in reading.cycles)


class TestUnplaced:
    def test_an_account_with_spend_but_no_recorded_reset_is_listed(self):
        reading = account_cycles(TWO_ACCOUNTS_CAPS, (*TWO_ACCOUNTS_SPEND, spend('max-z', at(12), 50.0)), SINCE, UNTIL)

        assert reading.unplaced_accounts == ('max-z',)

    def test_a_cap_hit_without_a_reset_is_counted_as_unplaceable(self):
        caps = (*TWO_ACCOUNTS_CAPS, CapHit('max-a', at(13), None))

        assert account_cycles(caps, TWO_ACCOUNTS_SPEND, SINCE, UNTIL).unplaceable_cap_hits == 1


class TestExpiredInBlock:
    def block(self, start: datetime, end: datetime) -> Block:
        return Block(HostState.ON, start, end, BlockOrigin.ADMITTED, start)

    def test_sums_the_expiry_of_cycles_resetting_in_the_block(self):
        reading = account_cycles(TWO_ACCOUNTS_CAPS, TWO_ACCOUNTS_SPEND, SINCE, UNTIL)

        assert expired_in_block(reading, self.block(SINCE, at(20))) == 1200.0
        assert expired_in_block(reading, self.block(at(20), UNTIL)) == 1700.0

    def test_one_unknown_cycle_makes_the_block_unknown(self):
        reading = account_cycles((CapHit('max-a', at(1, 5), at(2, 21)),), (spend('max-a', at(12), 1.0),), SINCE, UNTIL)

        assert expired_in_block(reading, self.block(SINCE, UNTIL)) is None

    def test_a_block_with_no_reset_expired_nothing(self):
        reading = account_cycles(TWO_ACCOUNTS_CAPS, TWO_ACCOUNTS_SPEND, SINCE, UNTIL)

        assert expired_in_block(reading, self.block(SINCE, at(12))) == 0.0


def test_the_rule_says_it_is_an_estimate_in_list_usd_counting_every_model_once():
    for words in ('estimate', 'list USD', 'every model counted once'):
        assert words in ALLOWANCE_RULE
