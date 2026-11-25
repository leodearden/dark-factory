"""verify_host_trial_allowance: account cycles and the allowance expired unused, an estimate."""
from __future__ import annotations

from datetime import UTC, datetime

from verify_host_trial_allowance import CYCLE, account_cycles, expired_in_block
from verify_host_trial_blocks import Block, BlockOrigin, HostState
from verify_host_trial_stores import CapHit, CapKind, Invocation

SINCE = datetime(2026, 11, 11, tzinfo=UTC)
UNTIL = datetime(2026, 11, 25, tzinfo=UTC)


def at(month_day: int, hour: int = 0) -> datetime:
    return datetime(2026, 11, month_day, hour, tzinfo=UTC)


def spend(account: str, when: datetime, usd: float) -> Invocation:
    return Invocation(when, account, usd)


def weekly(account: str, when: datetime, reset: datetime | None) -> CapHit:
    return CapHit(account, when, reset, CapKind.WEEKLY)


R_A = at(16, 21)
"""max-a's capped reset; its lattice puts the next reset at the 23rd, 21:00."""
R_B = at(20, 10)
"""max-b's capped reset; its lattice puts the previous reset at the 13th, 10:00."""

TWO_ACCOUNTS_CAPS = (weekly('max-a', at(15, 10), R_A), weekly('max-b', at(19, 4), R_B))
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
        caps = (*TWO_ACCOUNTS_CAPS, weekly('max-a', at(15, 11), R_A), weekly('max-a', at(15, 10), R_A))

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
        caps = (weekly('max-c', at(24, 1), at(24, 21)), weekly('max-c', at(17, 2), at(17, 20)))

        reading = account_cycles(caps, (spend('max-c', at(15), 10.0),), SINCE, UNTIL)

        assert sorted(cycles_by_reset(reading)) == [('max-c', at(17, 20)), ('max-c', at(24, 21))]
        assert all(cycle.capped for cycle in reading.cycles)

    def test_an_account_with_no_activity_near_the_window_is_dormant_and_unplaced(self):
        caps = (*TWO_ACCOUNTS_CAPS, weekly('max-old', at(1, 5), at(2, 21)))

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
        caps = (weekly('max-a', at(1, 5), at(2, 21)),)

        reading = account_cycles(caps, (spend('max-a', at(12), 100.0),), SINCE, UNTIL)

        assert reading.estimate_usd is None
        assert [cycle.reset for cycle in reading.cycles] == [at(16, 21), at(23, 21)]
        assert all(cycle.expired_usd is None for cycle in reading.cycles)


class TestUnplaced:
    def test_an_account_with_spend_but_no_recorded_reset_is_listed(self):
        reading = account_cycles(TWO_ACCOUNTS_CAPS, (*TWO_ACCOUNTS_SPEND, spend('max-z', at(12), 50.0)), SINCE, UNTIL)

        assert reading.unplaced_accounts == ('max-z',)

    def test_a_cap_hit_without_a_reset_is_counted_as_unplaceable(self):
        caps = (*TWO_ACCOUNTS_CAPS, weekly('max-a', at(13), None))

        assert account_cycles(caps, TWO_ACCOUNTS_SPEND, SINCE, UNTIL).unplaceable_cap_hits == 1


class TestExpiredInBlock:
    def block(self, start: datetime, end: datetime) -> Block:
        return Block(HostState.ON, start, end, BlockOrigin.ADMITTED, start)

    def test_sums_the_expiry_of_cycles_resetting_in_the_block(self):
        reading = account_cycles(TWO_ACCOUNTS_CAPS, TWO_ACCOUNTS_SPEND, SINCE, UNTIL)

        assert expired_in_block(reading, self.block(SINCE, at(20))) == 1200.0
        assert expired_in_block(reading, self.block(at(20), UNTIL)) == 1700.0

    def test_one_unknown_cycle_makes_the_block_unknown(self):
        reading = account_cycles((weekly('max-a', at(1, 5), at(2, 21)),), (spend('max-a', at(12), 1.0),), SINCE, UNTIL)

        assert expired_in_block(reading, self.block(SINCE, UNTIL)) is None

    def test_a_block_with_no_reset_expired_nothing(self):
        reading = account_cycles(TWO_ACCOUNTS_CAPS, TWO_ACCOUNTS_SPEND, SINCE, UNTIL)

        assert expired_in_block(reading, self.block(SINCE, at(12))) == 0.0


class TestOnlyWeeklyCapsPlaceCycles:
    WINDOW = Block(HostState.ON, SINCE, UNTIL, BlockOrigin.ADMITTED, SINCE)

    def reading(self, *extra: CapHit):
        return account_cycles((*TWO_ACCOUNTS_CAPS, *extra), TWO_ACCOUNTS_SPEND, SINCE, UNTIL)

    def test_a_session_cap_creates_no_cycle(self):
        weekly_only = self.reading()
        with_sessions = self.reading(
            CapHit('max-a', at(18, 13), at(18, 18), CapKind.SESSION),
            CapHit('max-b', at(21, 17), at(21, 19), CapKind.SESSION),
        )

        assert with_sessions.cycles == weekly_only.cycles
        assert with_sessions.estimate_usd == weekly_only.estimate_usd
        assert expired_in_block(with_sessions, self.WINDOW) == expired_in_block(weekly_only, self.WINDOW)

    def test_a_session_cap_does_not_move_the_lattice(self):
        reading = self.reading(CapHit('max-a', at(24, 13), at(24, 18), CapKind.SESSION))

        cycles = cycles_by_reset(reading)
        assert sorted(reset for account, reset in cycles if account == 'max-a') == [R_A, R_A + CYCLE]
        assert not cycles['max-a', R_A + CYCLE].capped
        assert ('max-a', at(24, 18)) not in cycles

    def test_a_per_model_cap_does_not_retime_a_weekly_cap(self):
        reading = self.reading(CapHit('max-b', at(17, 3), R_B, CapKind.PER_MODEL))

        assert cycles_by_reset(reading)['max-b', R_B].capped_at == at(19, 4)
        assert reading.estimate_usd == 2200.0

    def test_a_per_model_cap_does_not_cap_an_uncapped_cycle(self):
        reading = self.reading(CapHit('max-a', at(22, 2), R_A + CYCLE, CapKind.PER_MODEL))

        cycle = cycles_by_reset(reading)['max-a', R_A + CYCLE]
        assert not cycle.capped
        assert cycle.expired_usd == 1700.0
        assert reading.estimate_usd == 2200.0

    def test_one_other_cap_recorded_by_two_stores_counts_once(self):
        reading = self.reading(
            CapHit('max-a', at(18, 13), at(18, 18), CapKind.SESSION),
            CapHit('max-a', at(18, 14), at(18, 18), CapKind.SESSION),
        )

        assert reading.other_caps == {CapKind.SESSION: 1}

    def test_hits_without_a_reset_each_count_alone(self):
        reading = self.reading(
            CapHit('max-b', at(21, 17), None, CapKind.SESSION),
            CapHit('max-b', at(21, 23), None, CapKind.SESSION),
        )

        assert reading.other_caps == {CapKind.SESSION: 2}

    def test_a_hit_more_than_a_cycle_before_since_is_not_counted(self):
        reading = self.reading(CapHit('max-a', at(3, 23), at(4, 4), CapKind.SESSION))

        assert reading.other_caps == {}

    def test_the_count_is_by_kind(self):
        reading = self.reading(
            CapHit('max-a', at(18, 13), at(18, 18), CapKind.SESSION),
            CapHit('max-a', at(18, 14), at(18, 18), CapKind.SESSION),
            CapHit('max-b', at(21, 17), None, CapKind.SESSION),
            CapHit('max-b', at(17, 3), R_B, CapKind.PER_MODEL),
            CapHit('max-a', at(13), None, CapKind.UNRECOGNISED),
            CapHit('max-a', at(3, 23), at(4, 4), CapKind.SESSION),
        )

        assert reading.other_caps == {CapKind.SESSION: 2, CapKind.PER_MODEL: 1, CapKind.UNRECOGNISED: 1}

    def test_weekly_caps_alone_count_no_other_cap(self):
        assert self.reading().other_caps == {}

    def test_only_a_weekly_cap_without_a_reset_is_unplaceable(self):
        reading = self.reading(weekly('max-a', at(13), None), CapHit('max-b', at(14), None, CapKind.SESSION))

        assert reading.unplaceable_cap_hits == 1
        assert reading.other_caps == {CapKind.SESSION: 1}
