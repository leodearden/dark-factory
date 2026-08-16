"""Tests for reconciliation drain/inflow throughput analysis (task 3049).

Covers `fused_memory.reconciliation.throughput`: UTC hour bucketing, inflow
readers over the `event_arrival_hourly` rollup unioned with live
`event_buffer` rows, per-mode drain statistics over the `runs` table, the
remediation duty cycle, and the pure capacity arithmetic.
"""

from __future__ import annotations

import pytest
import pytest_asyncio  # noqa: F401  — strict-mode async fixtures land here in later steps

import uuid
from datetime import UTC, datetime

from fused_memory.models.reconciliation import (
    EventSource,
    EventType,
    ReconciliationEvent,
)
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.reconciliation.throughput import (
    inflow_daily,
    inflow_hourly,
    utc_hour_bucket,
)


def _make_event(
    project_id: str = 'reify',
    event_type: EventType = EventType.task_status_changed,
    timestamp: datetime | None = None,
) -> ReconciliationEvent:
    return ReconciliationEvent(
        id=str(uuid.uuid4()),
        type=event_type,
        source=EventSource.agent,
        project_id=project_id,
        timestamp=timestamp or datetime.now(UTC),
        payload={'test': True},
        agent_id=None,
    )


@pytest_asyncio.fixture
async def recon_db(tmp_path):
    """A real reconciliation.db with the EventBuffer schema applied."""
    buf = EventBuffer(db_path=tmp_path / 'reconciliation.db')
    await buf.initialize()
    yield buf
    await buf.close()

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


# ── inflow readers ─────────────────────────────────────────────────────
#
# Inflow must be read from the UNION of the live `event_buffer` rows (any
# status) and the rolled-up `event_arrival_hourly` aggregate.  The
# same-transaction rollup in cleanup_drained guarantees a row is in exactly
# one of the two, so the union is exact — no double count, no loss — and the
# report is useful from its first run rather than after a week of warm-up.

_H13 = datetime(2026, 7, 25, 13, 30, tzinfo=UTC)
_H14 = datetime(2026, 7, 25, 14, 15, tzinfo=UTC)
_H00_NEXT_DAY = datetime(2026, 7, 26, 0, 30, tzinfo=UTC)


async def _seed_inflow(buf) -> None:
    """3x status@13, 1x memory@13, 2x modified@14, 1x status@next-day-00."""
    for _ in range(3):
        await buf.push(_make_event(
            timestamp=_H13, event_type=EventType.task_status_changed))
    await buf.push(_make_event(timestamp=_H13, event_type=EventType.memory_added))
    for _ in range(2):
        await buf.push(_make_event(
            timestamp=_H14, event_type=EventType.task_modified))
    await buf.push(_make_event(
        timestamp=_H00_NEXT_DAY, event_type=EventType.task_status_changed))


@pytest.mark.asyncio
async def test_inflow_hourly_counts_live_rows_with_no_aggregate(recon_db):
    """(a) With only live event_buffer rows and an empty aggregate, counts are right."""
    await _seed_inflow(recon_db)

    assert inflow_hourly(recon_db._db_path, 'reify') == {
        '2026-07-25T13': 4,
        '2026-07-25T14': 2,
        '2026-07-26T00': 1,
    }


@pytest.mark.asyncio
async def test_inflow_hourly_total_is_unchanged_across_the_rollup_boundary(recon_db):
    """(b) A cleanup_drained sweep moves rows into the aggregate without changing totals.

    This is the property the same-transaction rollup exists to provide: each
    event is counted exactly once whether it is still live or already rolled
    up.  A double count or a loss at the boundary would show up here.
    """
    await _seed_inflow(recon_db)
    before = inflow_hourly(recon_db._db_path, 'reify')
    assert sum(before.values()) == 7

    # Drain + sweep only SOME of them, so the union spans both sources.
    await recon_db.drain('reify')
    # A late arrival stays live in event_buffer.
    await recon_db.push(_make_event(
        timestamp=_H14, event_type=EventType.task_modified))
    moved = await recon_db.cleanup_drained(max_age_seconds=0)
    assert moved == 7, 'the seeded rows should have been rolled up and deleted'

    after = inflow_hourly(recon_db._db_path, 'reify')
    assert sum(after.values()) == 8, 'no double count and no loss across the boundary'
    assert after == {
        '2026-07-25T13': 4,
        '2026-07-25T14': 3,
        '2026-07-26T00': 1,
    }


@pytest.mark.asyncio
async def test_inflow_hourly_since_filters_by_hour_bucket(recon_db):
    """(d) `since` filters on the normalised hour bucket, inclusive."""
    await _seed_inflow(recon_db)

    assert inflow_hourly(recon_db._db_path, 'reify', since='2026-07-25T14') == {
        '2026-07-25T14': 2,
        '2026-07-26T00': 1,
    }
    assert inflow_hourly(recon_db._db_path, 'reify', since='2026-07-26T00') == {
        '2026-07-26T00': 1,
    }
    # An offset-bearing ISO timestamp normalises to its bucket rather than
    # being string-compared (the METHOD NOTE trap again).
    assert inflow_hourly(
        recon_db._db_path, 'reify', since='2026-07-25T14:59:59+00:00',
    ) == {'2026-07-25T14': 2, '2026-07-26T00': 1}


@pytest.mark.asyncio
async def test_inflow_hourly_scopes_to_the_requested_project(recon_db):
    """Another project's arrivals never leak into the report."""
    await _seed_inflow(recon_db)
    await recon_db.push(_make_event(project_id='other', timestamp=_H13))
    await recon_db.drain('other')
    await recon_db.cleanup_drained(max_age_seconds=0)

    assert inflow_hourly(recon_db._db_path, 'reify')['2026-07-25T13'] == 4
    assert inflow_hourly(recon_db._db_path, 'other') == {'2026-07-25T13': 1}


@pytest.mark.asyncio
async def test_inflow_daily_rolls_hours_into_days_with_event_type_composition(recon_db):
    """(c) Per-day totals plus the per-event_type composition lever 3 needs to size itself."""
    await _seed_inflow(recon_db)

    daily = inflow_daily(recon_db._db_path, 'reify')

    assert daily['daily'] == {'2026-07-25': 6, '2026-07-26': 1}
    assert daily['total_events'] == 7
    assert daily['composition'] == {
        'task_status_changed': 4,
        'memory_added': 1,
        'task_modified': 2,
    }


@pytest.mark.asyncio
async def test_inflow_daily_composition_survives_the_rollup_boundary(recon_db):
    """Composition is exact across live rows and the aggregate alike."""
    await _seed_inflow(recon_db)
    await recon_db.drain('reify')
    await recon_db.cleanup_drained(max_age_seconds=0)
    await recon_db.push(_make_event(
        timestamp=_H14, event_type=EventType.memory_added))

    daily = inflow_daily(recon_db._db_path, 'reify')
    assert daily['total_events'] == 8
    assert daily['composition'] == {
        'task_status_changed': 4,
        'memory_added': 2,
        'task_modified': 2,
    }
