"""Tests for scripts/merge_lane_throughput.py — the merge-lane baseline reporter.

Every number asserted here is a KNOWN ANSWER computed by hand from a
programmatically-built fixture ``runs.db``. Deliberately NO test asserts a
number read from a live store: ``data/orchestrator/runs.db`` is mutated
continuously by the running orchestrator, so a live-number assertion would be
non-hermetic and would self-invalidate within hours (plan decision 3).
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta

import merge_lane_throughput as mlt
import pytest

# ---------------------------------------------------------------------------
# parse_window — clock-free: ``now`` is always injected, never read inside.
# ---------------------------------------------------------------------------

NOW = datetime(2026, 9, 5, 12, 30, tzinfo=UTC)


def test_parse_window_relative_14d_is_now_minus_14_days():
    lo, hi = mlt.parse_window('14d', NOW)
    assert hi == NOW
    assert lo == NOW - timedelta(days=14)
    assert lo.tzinfo is not None and hi.tzinfo is not None
    assert lo.utcoffset() == timedelta(0)
    assert hi.utcoffset() == timedelta(0)


def test_parse_window_relative_30d_is_now_minus_30_days():
    lo, hi = mlt.parse_window('30d', NOW)
    assert (lo, hi) == (NOW - timedelta(days=30), NOW)


def test_parse_window_dated_range_is_exactly_those_two_instants():
    # The dated form is how a caller reproduces the PRD's dated baseline;
    # `14d` resolves relative to NOW and therefore cannot (plan decision 3).
    lo, hi = mlt.parse_window(
        '2026-08-20T16:10:00+00:00..2026-09-03T16:10:00+00:00', NOW
    )
    assert lo == datetime(2026, 8, 20, 16, 10, tzinfo=UTC)
    assert hi == datetime(2026, 9, 3, 16, 10, tzinfo=UTC)


def test_parse_window_naive_iso_endpoint_is_interpreted_as_utc():
    lo, hi = mlt.parse_window('2026-08-20T16:10:00..2026-09-03T16:10:00', NOW)
    assert lo == datetime(2026, 8, 20, 16, 10, tzinfo=UTC)
    assert hi == datetime(2026, 9, 3, 16, 10, tzinfo=UTC)
    assert lo.utcoffset() == timedelta(0)


def test_parse_window_mixed_naive_and_aware_endpoints_both_utc():
    lo, hi = mlt.parse_window('2026-08-20T16:10:00..2026-09-03T16:10:00+00:00', NOW)
    assert lo == datetime(2026, 8, 20, 16, 10, tzinfo=UTC)
    assert hi == datetime(2026, 9, 3, 16, 10, tzinfo=UTC)


def test_parse_window_date_only_endpoints_are_midnight_utc():
    lo, hi = mlt.parse_window('2026-08-20..2026-09-03', NOW)
    assert lo == datetime(2026, 8, 20, 0, 0, tzinfo=UTC)
    assert hi == datetime(2026, 9, 3, 0, 0, tzinfo=UTC)


@pytest.mark.parametrize('spec', ['', 'd', '14', '14x', 'a..b', '0d', '-3d'])
def test_parse_window_rejects_malformed_specs_loudly(spec):
    with pytest.raises(argparse.ArgumentTypeError) as exc:
        mlt.parse_window(spec, NOW)
    # The offending text must be echoed back — a bare "bad window" would make
    # the operator guess which of several --window flags was wrong.
    assert repr(spec) in str(exc.value)


def test_parse_window_rejects_reversed_range():
    spec = '2026-09-03T16:10:00+00:00..2026-08-20T16:10:00+00:00'
    with pytest.raises(argparse.ArgumentTypeError) as exc:
        mlt.parse_window(spec, NOW)
    assert repr(spec) in str(exc.value)


def test_parse_window_rejects_empty_range_lo_equals_hi():
    spec = '2026-09-03T16:10:00+00:00..2026-09-03T16:10:00+00:00'
    with pytest.raises(argparse.ArgumentTypeError) as exc:
        mlt.parse_window(spec, NOW)
    assert repr(spec) in str(exc.value)


def test_parse_window_rejects_three_part_range():
    spec = '2026-08-20..2026-08-25..2026-09-03'
    with pytest.raises(argparse.ArgumentTypeError) as exc:
        mlt.parse_window(spec, NOW)
    assert repr(spec) in str(exc.value)


def test_iso_formats_the_spelling_the_events_table_stores():
    # Window bounds are used directly as SQL string comparands against the
    # `timestamp` TEXT column, so the spelling must match byte-for-byte.
    assert mlt._iso(datetime(2026, 8, 20, 16, 10, tzinfo=UTC)) == (
        '2026-08-20T16:10:00+00:00'
    )
