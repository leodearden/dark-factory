"""Tests for reconciliation drain/inflow throughput analysis (task 3049).

Covers `fused_memory.reconciliation.throughput`: UTC hour bucketing, inflow
readers over the `event_arrival_hourly` rollup unioned with live
`event_buffer` rows, per-mode drain statistics over the `runs` table, the
remediation duty cycle, and the pure capacity arithmetic.
"""

from __future__ import annotations

import pytest
import pytest_asyncio  # noqa: F401  — strict-mode async fixtures land here in later steps

from fused_memory.reconciliation.throughput import utc_hour_bucket

# ── utc_hour_bucket ────────────────────────────────────────────────────
#
# METHOD NOTE (task 3049): `event_buffer.timestamp` is an ISO8601 string that
# CARRIES AN OFFSET ('2026-07-25T13:23:22.383113+00:00').  SQLite's
# `datetime('now', ...)` renders space-separated and offset-free
# ('2026-07-25 13:23:22'), so `timestamp < datetime('now', ...)` is a *string*
# comparison in which 'T' (0x54) > ' ' (0x20) — essentially every same-day row
# compares greater and the whole day collapses into one bucket.  Bucketing must
# therefore go through real parsing, which is what these cases pin.


def test_utc_hour_bucket_offset_bearing_utc_timestamp() -> None:
    """The canonical event_buffer shape truncates to its own UTC hour."""
    assert utc_hour_bucket('2026-07-25T13:23:22.383113+00:00') == '2026-07-25T13'


def test_utc_hour_bucket_converts_non_utc_offset_rather_than_truncating() -> None:
    """A +02:00 timestamp is CONVERTED to UTC, not truncated at its local hour."""
    assert utc_hour_bucket('2026-07-25T13:23:22+02:00') == '2026-07-25T11'


def test_utc_hour_bucket_treats_naive_timestamp_as_utc() -> None:
    """A naive timestamp is interpreted as UTC (the buffer's own convention)."""
    assert utc_hour_bucket('2026-07-25T13:23:22') == '2026-07-25T13'


def test_utc_hour_bucket_separates_hour_23_from_hour_00_same_day() -> None:
    """Hour resolution is real: 00 and 23 on one calendar day are distinct buckets.

    A naive `date`-only or string-prefix bucketing would collapse these, which
    is exactly the failure mode the METHOD NOTE above describes.
    """
    early = utc_hour_bucket('2026-07-25T00:00:00+00:00')
    late = utc_hour_bucket('2026-07-25T23:59:59+00:00')
    assert early == '2026-07-25T00'
    assert late == '2026-07-25T23'
    assert early != late


def test_utc_hour_bucket_is_sortable_across_a_day_boundary() -> None:
    """Bucket keys sort lexicographically in true chronological order."""
    keys = [
        utc_hour_bucket('2026-07-26T00:15:00+00:00'),
        utc_hour_bucket('2026-07-25T23:45:00+00:00'),
        utc_hour_bucket('2026-07-25T09:00:00+00:00'),
    ]
    assert sorted(keys) == ['2026-07-25T09', '2026-07-25T23', '2026-07-26T00']


def test_utc_hour_bucket_rejects_unparseable_input() -> None:
    """Garbage in raises rather than silently bucketing to a wrong hour."""
    with pytest.raises(ValueError):
        utc_hour_bucket('not-a-timestamp')
