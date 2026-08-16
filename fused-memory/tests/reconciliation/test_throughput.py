"""Tests for reconciliation drain/inflow throughput analysis (task 3049).

Covers `fused_memory.reconciliation.throughput`: UTC hour bucketing, inflow
readers over the `event_arrival_hourly` rollup unioned with live
`event_buffer` rows, per-mode drain statistics over the `runs` table, the
remediation duty cycle, and the pure capacity arithmetic.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import UTC, datetime

import pytest
import pytest_asyncio  # noqa: F401  — strict-mode async fixtures land here in later steps

from fused_memory.models.reconciliation import (
    EventSource,
    EventType,
    ReconciliationEvent,
)
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.reconciliation.journal import ReconciliationJournal
from fused_memory.reconciliation.throughput import (
    drain_stats,
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


def _insert_run(
    db_path,
    *,
    project_id: str = 'reify',
    run_type: str = 'full',
    trigger_reason: str = 'quiescence',
    started_at: str,
    completed_at: str | None,
    events_processed: int = 0,
) -> None:
    """Insert one `runs` row directly, for exact control over the timestamps.

    The schema comes from the real ReconciliationJournal (see the recon_db
    fixture), so these rows are shaped exactly like production ones.
    """
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """INSERT INTO runs
                   (id, project_id, run_type, trigger_reason, started_at,
                    completed_at, events_processed, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                str(uuid.uuid4()), project_id, run_type, trigger_reason,
                started_at, completed_at, events_processed,
                'running' if completed_at is None else 'completed',
            ),
        )
        conn.commit()
    finally:
        conn.close()


@pytest_asyncio.fixture
async def recon_db(tmp_path):
    """A real reconciliation.db with BOTH the journal and EventBuffer schemas.

    Production puts `runs` (journal.py) and `event_buffer` (event_buffer.py)
    in the same reconciliation.db file, so the report reads one path; this
    fixture reproduces that layout rather than a synthetic one.
    """
    journal = ReconciliationJournal(tmp_path)
    await journal.initialize()
    await journal.close()

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


# ── drain_stats ────────────────────────────────────────────────────────
#
# ADDENDUM 2's observed sequence, verbatim (2026-07-25, project reify).  Each
# backlog chunk is immediately followed by a remediation pass that drained
# zero events — the inline remediation tail of run_full_cycle — which is the
# duty-cycle loss lever 1 exists to recover.
_ADDENDUM_2_RUNS = [
    # (run_type, trigger_reason, started_at, completed_at, events_processed)
    ('full', 'backlog_chunk:1:393', '13:40:30', '13:56:24', 393),   # 954s
    ('remediation', 'integrity_findings:2', '13:56:24', '14:09:00', 0),  # 756s
    ('full', 'backlog_chunk:1:326', '14:13:36', '14:29:36', 326),   # 960s
    ('remediation', 'integrity_findings:5', '14:29:41', '14:42:00', 0),  # 739s
]


def _t(hms: str) -> str:
    """'13:40:30' -> the offset-bearing ISO8601 form the journal actually writes."""
    return f'2026-07-25T{hms}+00:00'


def _seed_addendum_2(db_path) -> None:
    for run_type, reason, start, end, events in _ADDENDUM_2_RUNS:
        _insert_run(
            db_path, run_type=run_type, trigger_reason=reason,
            started_at=_t(start), completed_at=_t(end), events_processed=events,
        )


@pytest.mark.asyncio
async def test_drain_stats_classifies_the_four_run_modes(recon_db):
    """Chunks, remediation, targeted and steady state land in distinct buckets."""
    db = recon_db._db_path
    _seed_addendum_2(db)
    _insert_run(
        db, trigger_reason='quiescence',
        started_at=_t('15:00:00'), completed_at=_t('15:15:45'), events_processed=50,
    )  # 945s steady state
    _insert_run(
        db, run_type='targeted', trigger_reason='task_status_changed',
        started_at=_t('15:01:00'), completed_at=_t('15:01:30'), events_processed=1,
    )

    stats = drain_stats(db, 'reify')
    modes = stats['modes']

    assert modes['backlog_chunk']['run_count'] == 2
    assert modes['backlog_chunk']['events'] == 719
    assert modes['backlog_chunk']['wall_clock_seconds'] == pytest.approx(1914.0)

    assert modes['remediation']['run_count'] == 2
    assert modes['remediation']['events'] == 0
    assert modes['remediation']['wall_clock_seconds'] == pytest.approx(1495.0)

    assert modes['steady_state']['run_count'] == 1
    assert modes['steady_state']['events'] == 50
    assert modes['steady_state']['wall_clock_seconds'] == pytest.approx(945.0)
    assert modes['steady_state']['seconds_per_event'] == pytest.approx(18.9)

    assert modes['targeted']['run_count'] == 1
    assert modes['targeted']['wall_clock_seconds'] == pytest.approx(30.0)


@pytest.mark.asyncio
async def test_drain_stats_does_not_classify_final_consolidation_as_a_chunk(recon_db):
    """'backlog_final_consolidation' is a steady-state run, not a chunk.

    Only trigger_reason values that literally start with 'backlog_chunk:' are
    chunks.  The final consolidation pass runs AFTER the chunks with a drained
    buffer, so folding it into the chunk bucket would understate chunk
    throughput and hide the pass whose whole purpose is to let remediation
    resume.
    """
    db = recon_db._db_path
    _insert_run(
        db, trigger_reason='backlog_final_consolidation',
        started_at=_t('16:00:00'), completed_at=_t('16:10:00'), events_processed=12,
    )

    modes = drain_stats(db, 'reify')['modes']
    assert modes['backlog_chunk']['run_count'] == 0
    assert modes['steady_state']['run_count'] == 1
    assert modes['steady_state']['events'] == 12


@pytest.mark.asyncio
async def test_drain_stats_excludes_in_flight_runs_from_wall_clock(recon_db):
    """(a) completed_at IS NULL contributes no wall-clock but IS counted in-flight."""
    db = recon_db._db_path
    _insert_run(
        db, trigger_reason='quiescence',
        started_at=_t('15:00:00'), completed_at=_t('15:15:45'), events_processed=50,
    )
    _insert_run(
        db, trigger_reason='quiescence',
        started_at=_t('15:20:00'), completed_at=None, events_processed=0,
    )

    stats = drain_stats(db, 'reify')
    assert stats['runs_in_flight'] == 1
    assert stats['modes']['steady_state']['wall_clock_seconds'] == pytest.approx(945.0)
    assert stats['modes']['steady_state']['run_count'] == 1, (
        'an unfinished run has no measurable wall-clock and must not dilute the average'
    )


@pytest.mark.asyncio
async def test_drain_stats_excludes_targeted_from_the_drain_capacity_totals(recon_db):
    """(b) Targeted runs hold no lock and execute concurrently (ADDENDUM 2 finding 5).

    They are reported separately so an operator can see them, but they consume
    no drain capacity, so counting their wall-clock against the drain budget
    would understate the pipeline's real throughput.
    """
    db = recon_db._db_path
    _seed_addendum_2(db)
    baseline = drain_stats(db, 'reify')

    for i in range(5):
        _insert_run(
            db, run_type='targeted', trigger_reason='task_status_changed',
            started_at=_t(f'15:0{i}:00'), completed_at=_t(f'15:0{i}:40'),
            events_processed=3,
        )
    after = drain_stats(db, 'reify')

    assert after['drain_wall_clock_seconds'] == baseline['drain_wall_clock_seconds']
    assert after['drain_events'] == baseline['drain_events']
    assert after['modes']['targeted']['run_count'] == 5
    assert after['modes']['targeted']['wall_clock_seconds'] == pytest.approx(200.0)
    assert after['drain_wall_clock_seconds'] == pytest.approx(3409.0)
    assert after['drain_events'] == 719


@pytest.mark.asyncio
async def test_drain_stats_seconds_per_event_is_none_for_zero_event_modes(recon_db):
    """(c) A zero-event remediation mode reports None, never a ZeroDivisionError."""
    db = recon_db._db_path
    _seed_addendum_2(db)

    modes = drain_stats(db, 'reify')['modes']
    assert modes['remediation']['events'] == 0
    assert modes['remediation']['seconds_per_event'] is None
    # An entirely absent mode is present-but-empty, also with None.
    assert modes['targeted']['run_count'] == 0
    assert modes['targeted']['seconds_per_event'] is None
    assert modes['backlog_chunk']['seconds_per_event'] == pytest.approx(1914.0 / 719)


@pytest.mark.asyncio
async def test_drain_stats_scopes_by_project_and_since(recon_db):
    """Rows are filtered by project and by an inclusive started_at floor."""
    db = recon_db._db_path
    _seed_addendum_2(db)
    _insert_run(
        db, project_id='other', trigger_reason='quiescence',
        started_at=_t('13:40:30'), completed_at=_t('13:56:24'), events_processed=999,
    )

    assert drain_stats(db, 'reify')['drain_events'] == 719
    assert drain_stats(db, 'other')['drain_events'] == 999

    later = drain_stats(db, 'reify', since='2026-07-25T14:00:00+00:00')
    assert later['modes']['backlog_chunk']['run_count'] == 1
    assert later['modes']['backlog_chunk']['events'] == 326
    assert later['modes']['remediation']['run_count'] == 1
