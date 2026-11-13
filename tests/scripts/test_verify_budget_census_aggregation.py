"""_series, summarise_legs, day_bucket, parse_window and within_window: how the
selected legs become a dated distribution.

The synthetic-corpus norm lives in tests/scripts/verify_budget_census_fixture.py.
"""
from __future__ import annotations

import pytest
from verify_budget_census_fixture import (
    FULL_SUITE,
    command_entry,
    write_corpus,
)


class TestSeriesAggregation:
    """p50/p90/max over a duration series, with every number computed by hand.

    Linear interpolation between the two nearest order statistics — the
    `numpy.percentile` default, and the same one
    `merge_lane_throughput._percentile` uses. Shape copied from that sibling,
    not imported: `scripts/` modules do not import one another here.
    """

    def test_an_empty_series_is_null_never_zero(self):
        """The single most consequential misreading a budget report can make.

        `_percentile`'s own docstring states it for the sibling: a `0.0` p50
        would render "no run in this window" as an instantaneous suite. In a
        BUDGET report that is worse than wrong — a p50 of zero invites a reader
        to conclude the suite got fast, on evidence that says nothing at all.
        """
        from verify_budget_census import _series  # noqa: PLC0415

        assert _series([]) == {'n': 0, 'p50': None, 'p90': None, 'max': None}

    def test_a_single_value_series(self):
        from verify_budget_census import _series  # noqa: PLC0415

        assert _series([3300.0]) == {
            'n': 1, 'p50': 3300.0, 'p90': 3300.0, 'max': 3300.0,
        }

    def test_percentiles_are_interpolated_by_hand_checked_arithmetic(self):
        """values = [100, 200, 300, 400, 500], n=5.

        p50: k = (5-1) * 0.50 = 2.0      -> exactly s[2]           = 300
        p90: k = (5-1) * 0.90 = 3.6      -> s[3] + 0.6*(s[4]-s[3]) = 460
        """
        from verify_budget_census import _series  # noqa: PLC0415

        assert _series([100.0, 200.0, 300.0, 400.0, 500.0]) == {
            'n': 5, 'p50': 300.0, 'p90': 460.0, 'max': 500.0,
        }

    def test_the_input_order_does_not_matter(self):
        from verify_budget_census import _series  # noqa: PLC0415

        assert _series([500.0, 100.0, 400.0, 200.0, 300.0])['p50'] == 300.0

    def test_an_even_length_series_interpolates_the_median(self):
        """values = [10, 20, 30, 40], n=4. p50: k = 3 * 0.5 = 1.5 -> 20 + 0.5*10 = 25."""
        from verify_budget_census import _series  # noqa: PLC0415

        assert _series([10.0, 20.0, 30.0, 40.0])['p50'] == 25.0


class TestTimedOutAndFailedAreCountedApart:
    """A timed-out run's duration is the BUDGET, not the suite.

    Folding it into the percentiles measures the ceiling and calls it the
    workload — and then a budget derived from that distribution is derived from
    itself, which is the one circularity a census must not have. Counted, and
    excluded.
    """

    def _legs(self, tmp_path, *entries):
        from verify_budget_census import load_records, select_full_suite_legs  # noqa: PLC0415

        write_corpus(tmp_path, *entries)
        return select_full_suite_legs(
            load_records([tmp_path]), expected=FULL_SUITE, prefix='orchestrator',
        ).legs

    def test_a_timed_out_leg_is_counted_and_excluded(self, tmp_path):
        from verify_budget_census import summarise_legs  # noqa: PLC0415

        legs = self._legs(
            tmp_path,
            command_entry(duration_secs=3300.0),
            command_entry(duration_secs=7200.0, timed_out=True, rc=-9),
        )

        summary = summarise_legs(legs)

        assert summary['timed_out'] == 1
        assert summary['durations'] == {
            'n': 1, 'p50': 3300.0, 'p90': 3300.0, 'max': 3300.0,
        }

    def test_a_failing_leg_is_counted_and_excluded(self, tmp_path):
        """A red run stopped at the first failure, so its duration is not the
        suite's either — a different fact from a timeout, counted separately."""
        from verify_budget_census import summarise_legs  # noqa: PLC0415

        legs = self._legs(
            tmp_path, command_entry(duration_secs=3300.0), command_entry(duration_secs=120.0, rc=1),
        )

        summary = summarise_legs(legs)

        assert summary['failed'] == 1
        assert summary['timed_out'] == 0
        assert summary['durations']['max'] == 3300.0

    def test_a_timed_out_leg_is_not_double_counted_as_failed(self, tmp_path):
        """A timeout carries a non-zero rc too, so the two buckets must not
        both claim it — n would stop reconciling."""
        from verify_budget_census import summarise_legs  # noqa: PLC0415

        legs = self._legs(tmp_path, command_entry(duration_secs=7200.0, timed_out=True, rc=-9))

        summary = summarise_legs(legs)

        assert (summary['timed_out'], summary['failed']) == (1, 0)

    def test_every_leg_is_accounted_for(self, tmp_path):
        from verify_budget_census import summarise_legs  # noqa: PLC0415

        legs = self._legs(
            tmp_path,
            command_entry(duration_secs=3300.0),
            command_entry(duration_secs=3400.0),
            command_entry(duration_secs=7200.0, timed_out=True, rc=-9),
            command_entry(duration_secs=120.0, rc=1),
        )

        summary = summarise_legs(legs)

        assert (
            summary['durations']['n'] + summary['timed_out'] + summary['failed']
        ) == len(legs) == 4


class TestDayBucketing:
    """Buckets are the entry's `started_at`, normalised to UTC, then `.date()`."""

    def test_a_non_utc_offset_is_normalised_before_bucketing(self):
        """23:30-04:00 is 03:30Z the NEXT day — the bucket must say so.

        Bucketing on the raw string's leading 10 characters would file it under
        the local date, silently sliding runs across a day boundary and
        smearing the trailing-window edge this census's floor depends on.
        """
        from verify_budget_census import day_bucket  # noqa: PLC0415

        assert str(day_bucket('2026-09-13T23:30:00-04:00')) == '2026-09-14'

    def test_a_utc_timestamp_buckets_on_its_own_date(self):
        from verify_budget_census import day_bucket  # noqa: PLC0415

        assert str(day_bucket('2026-09-14T03:30:00+00:00')) == '2026-09-14'

    def test_a_naive_timestamp_is_read_as_utc(self):
        from verify_budget_census import day_bucket  # noqa: PLC0415

        assert str(day_bucket('2026-09-14T03:30:00')) == '2026-09-14'

    @pytest.mark.parametrize('stamp', ['', 'not a date', '2026-13-45T99:99:99+00:00'])
    def test_an_unparseable_timestamp_is_none_not_today(self, stamp):
        """Defaulting to the current date would invent a run inside the window."""
        from verify_budget_census import day_bucket  # noqa: PLC0415

        assert day_bucket(stamp) is None


class TestParseWindow:
    """Both `--window` forms, resolved against an INJECTED clock."""

    def test_the_relative_form(self):
        from datetime import UTC, datetime  # noqa: PLC0415

        from verify_budget_census import parse_window  # noqa: PLC0415

        now = datetime(2026, 9, 14, 12, 0, tzinfo=UTC)
        lo, hi = parse_window('14d', now)

        assert hi == now
        assert lo == datetime(2026, 8, 31, 12, 0, tzinfo=UTC)

    def test_the_dated_form_is_exactly_those_instants(self):
        """The mechanism for a report whose header carries a fixed date."""
        from datetime import UTC, datetime  # noqa: PLC0415

        from verify_budget_census import parse_window  # noqa: PLC0415

        lo, hi = parse_window(
            '2026-09-12T08:00:00+00:00..2026-09-14T00:00:00+00:00',
            datetime(2026, 9, 14, 12, 0, tzinfo=UTC),
        )

        assert lo == datetime(2026, 9, 12, 8, 0, tzinfo=UTC)
        assert hi == datetime(2026, 9, 14, 0, 0, tzinfo=UTC)

    def test_a_naive_endpoint_is_read_as_utc(self):
        from datetime import UTC, datetime  # noqa: PLC0415

        from verify_budget_census import parse_window  # noqa: PLC0415

        lo, _hi = parse_window(
            '2026-09-12T08:00:00..2026-09-14T00:00:00',
            datetime(2026, 9, 14, 12, 0, tzinfo=UTC),
        )

        assert lo == datetime(2026, 9, 12, 8, 0, tzinfo=UTC)

    @pytest.mark.parametrize(
        'spec',
        ['', '0d', '-3d', 'fortnight', '..', '2026-09-14T00:00:00+00:00..',
         'nonsense..alsononsense'],
    )
    def test_a_malformed_spec_is_rejected_echoing_the_spec(self, spec):
        import argparse  # noqa: PLC0415
        from datetime import UTC, datetime  # noqa: PLC0415

        from verify_budget_census import parse_window  # noqa: PLC0415

        with pytest.raises(argparse.ArgumentTypeError) as caught:
            parse_window(spec, datetime(2026, 9, 14, 12, 0, tzinfo=UTC))

        assert repr(spec) in str(caught.value)

    def test_a_reversed_range_is_rejected_not_swapped(self):
        """Far more often a pasted-backwards pair than a request for one instant."""
        import argparse  # noqa: PLC0415
        from datetime import UTC, datetime  # noqa: PLC0415

        from verify_budget_census import parse_window  # noqa: PLC0415

        with pytest.raises(argparse.ArgumentTypeError):
            parse_window(
                '2026-09-14T00:00:00+00:00..2026-09-12T08:00:00+00:00',
                datetime(2026, 9, 14, 12, 0, tzinfo=UTC),
            )


class TestLegsAreFilteredToTheResolvedWindow:
    def test_only_legs_inside_the_window_are_summarised(self, tmp_path):
        from datetime import UTC, datetime  # noqa: PLC0415

        from verify_budget_census import (  # noqa: PLC0415
            load_records,
            select_full_suite_legs,
            within_window,
        )

        write_corpus(
            tmp_path,
            command_entry(started_at='2026-09-13T04:00:00+00:00', duration_secs=3300.0),
            command_entry(started_at='2026-08-25T04:00:00+00:00', duration_secs=4991.0),
        )
        legs = select_full_suite_legs(
            load_records([tmp_path]), expected=FULL_SUITE, prefix='orchestrator',
        ).legs

        window = (
            datetime(2026, 9, 12, 8, 0, tzinfo=UTC),
            datetime(2026, 9, 14, 12, 0, tzinfo=UTC),
        )
        kept = within_window(legs, window)

        assert [leg.duration_secs for leg in kept] == [3300.0]

    def test_a_leg_with_an_unparseable_timestamp_is_excluded(self, tmp_path):
        """It cannot be placed in the window, so it cannot be counted in it."""
        from datetime import UTC, datetime  # noqa: PLC0415

        from verify_budget_census import (  # noqa: PLC0415
            load_records,
            select_full_suite_legs,
            within_window,
        )

        write_corpus(tmp_path, command_entry(started_at='', duration_secs=3300.0))
        legs = select_full_suite_legs(
            load_records([tmp_path]), expected=FULL_SUITE, prefix='orchestrator',
        ).legs

        window = (
            datetime(2026, 1, 1, tzinfo=UTC), datetime(2027, 1, 1, tzinfo=UTC),
        )

        assert within_window(legs, window) == ()
