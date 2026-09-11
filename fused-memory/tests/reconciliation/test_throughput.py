"""Tests for reconciliation drain/inflow throughput analysis (task 3049).

Covers `fused_memory.reconciliation.throughput`: UTC hour bucketing, inflow
readers over the `event_arrival_hourly` rollup unioned with live
`event_buffer` rows, per-mode drain statistics over the `runs` table, the
remediation duty cycle, and the pure capacity arithmetic.
"""

from __future__ import annotations

import json
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
    FITTED_CYCLE_FIXED_SECONDS,
    FITTED_CYCLE_MARGINAL_SECONDS,
    OBSERVED_BURST_EVENTS_PER_DAY,
    STEADY_STATE_AMORTISATION_MIN_BATCH,
    build_report,
    capacity_verdict,
    cycle_seconds,
    drain_stats,
    inflow_daily,
    inflow_hourly,
    main,
    remediation_duty_cycle,
    seconds_per_event,
    sustainable_events_per_day,
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


# ── the remediation-free drain rate ────────────────────────────────────
#
# `drain_seconds_per_event` is REMEDIATION-INCLUSIVE: remediation wall-clock
# lands in the numerator while remediation's zero events add nothing to the
# denominator, so the remediation penalty is ALREADY applied once in that rate.
# A caller that multiplies it by (1 - duty_cycle) discounts twice.  These tests
# pin the remediation-FREE companion rate that exists so a caller can pick the
# right one, and pin the algebraic identity linking the two.


@pytest.mark.asyncio
async def test_drain_stats_exposes_a_remediation_free_rate(recon_db):
    """(a)(b) The drain-only totals cover backlog_chunk + steady_state only.

    On the ADDENDUM 2 fixture that is the two chunks: 954 + 960 = 1914s over
    719 events = 2.662 s/event, against the inclusive 3409s / 719 = 4.741.
    """
    db = recon_db._db_path
    _seed_addendum_2(db)

    stats = drain_stats(db, 'reify')

    assert stats['drain_only_wall_clock_seconds'] == pytest.approx(1914.0)
    assert stats['drain_only_seconds_per_event'] == pytest.approx(1914.0 / 719)
    assert stats['drain_only_seconds_per_event'] == pytest.approx(2.6620, abs=0.0001)


@pytest.mark.asyncio
async def test_drain_stats_remediation_inclusive_keys_are_unchanged(recon_db):
    """(c) The new keys are purely additive — no existing value may move.

    The inclusive rate is the OBSERVED status quo the measured window actually
    sustained, so it stays the headline; adding the remediation-free companion
    must not silently redefine it.
    """
    db = recon_db._db_path
    _seed_addendum_2(db)

    stats = drain_stats(db, 'reify')

    assert stats['drain_wall_clock_seconds'] == pytest.approx(3409.0)
    assert stats['drain_events'] == 719
    assert stats['drain_seconds_per_event'] == pytest.approx(3409.0 / 719)
    assert stats['drain_seconds_per_event'] == pytest.approx(4.7413, abs=0.0001)


@pytest.mark.asyncio
async def test_drain_only_rate_excludes_remediation_and_keeps_steady_state(recon_db):
    """(d) The mode partition is pinned by NUMBER, not by an algebraic identity.

    Asserting `drain_only == inclusive * (1 - duty)` would be vacuous: both
    rates share the same event denominator and _CAPACITY_MODES is derived from
    _DRAIN_ONLY_MODES, so that identity holds for ANY partition — including a
    wrong one that dropped steady state or kept remediation.  Hand-computed
    figures do catch that.

    Fixture: backlog 1914s / 719 events, remediation 1495s / 0 events, plus one
    945s / 50-event steady-state run.
      drain_only = (1914 + 945) / 769 = 2859 / 769 = 3.7178 s/event
      inclusive  = (1914 + 945 + 1495) / 769 = 4354 / 769 = 5.6619 s/event
    """
    db = recon_db._db_path
    _seed_addendum_2(db)
    _insert_run(
        db, trigger_reason='quiescence',
        started_at=_t('15:00:00'), completed_at=_t('15:15:45'), events_processed=50,
    )  # a steady-state run, so drain_only covers more than the chunks alone

    stats = drain_stats(db, 'reify')

    # Steady state joins the drain-only wall-clock; remediation still does not.
    assert stats['drain_only_wall_clock_seconds'] == pytest.approx(2859.0)
    assert stats['drain_events'] == 769
    assert stats['drain_only_seconds_per_event'] == pytest.approx(3.7178, abs=0.0001)
    assert stats['drain_wall_clock_seconds'] == pytest.approx(4354.0)
    assert stats['drain_seconds_per_event'] == pytest.approx(5.6619, abs=0.0001)


@pytest.mark.asyncio
async def test_drain_only_rate_is_none_when_no_events_drained(recon_db):
    """(e) An empty DB and a remediation-only window both read None, never raise."""
    db = recon_db._db_path

    empty = drain_stats(db, 'reify')
    assert empty['drain_only_seconds_per_event'] is None
    assert empty['drain_only_wall_clock_seconds'] == pytest.approx(0.0)

    # A window whose only completed runs are zero-event remediation passes:
    # non-zero inclusive wall-clock, zero events, zero drain-only wall-clock.
    _insert_run(
        db, run_type='remediation', trigger_reason='integrity_findings:2',
        started_at=_t('13:56:24'), completed_at=_t('14:09:00'), events_processed=0,
    )
    remediation_only = drain_stats(db, 'reify')

    assert remediation_only['drain_wall_clock_seconds'] == pytest.approx(756.0)
    assert remediation_only['drain_events'] == 0
    assert remediation_only['drain_only_wall_clock_seconds'] == pytest.approx(0.0)
    assert remediation_only['drain_only_seconds_per_event'] is None
    assert remediation_only['drain_seconds_per_event'] is None


@pytest.mark.asyncio
async def test_drain_only_rate_ignores_targeted_runs(recon_db):
    """(f) Targeted runs hold no lock (ADDENDUM 2 finding 5), so neither key moves."""
    db = recon_db._db_path
    _seed_addendum_2(db)
    before = drain_stats(db, 'reify')

    for i in range(5):
        _insert_run(
            db, run_type='targeted', trigger_reason='task_status_changed',
            started_at=_t(f'15:0{i}:00'), completed_at=_t(f'15:0{i}:40'),
            events_processed=3,
        )
    after = drain_stats(db, 'reify')

    assert after['drain_only_wall_clock_seconds'] == pytest.approx(
        before['drain_only_wall_clock_seconds'])
    assert after['drain_only_seconds_per_event'] == pytest.approx(
        before['drain_only_seconds_per_event'])
    assert after['modes']['targeted']['run_count'] == 5


# ── remediation duty cycle ─────────────────────────────────────────────
#
# The number lever 1 exists to recover.  Remediation is an unconditional
# inline tail of every completed run_full_cycle (harness.py:2963), and
# BacklogIterator runs each chunk as a full cycle — so every chunk drags in
# its own zero-event remediation pass.  ADDENDUM 2 measured that as roughly
# half the backlog-mode drain wall-clock.


@pytest.mark.asyncio
async def test_remediation_duty_cycle_reproduces_the_addendum_2_measurement(recon_db):
    """Exact arithmetic over the observed 2026-07-25 reify sequence.

    remediation 756 + 739 = 1495s, over 1495 + 954 + 960 = 3409s total
    lock-held drain wall-clock => 0.4385.  This is fixture arithmetic, not a
    measurement tolerance, so it pins the definition rather than the weather.
    """
    db = recon_db._db_path
    _seed_addendum_2(db)

    duty = remediation_duty_cycle(drain_stats(db, 'reify'))
    assert duty == pytest.approx(1495.0 / 3409.0)
    assert pytest.approx(round(duty, 2)) == 0.44


@pytest.mark.asyncio
async def test_remediation_duty_cycle_ignores_targeted_runs(recon_db):
    """Targeted runs hold no lock, so they must not move the duty cycle.

    If they leaked into the denominator, adding concurrent targeted work would
    make remediation preemption look cheaper than it is — exactly the wrong
    signal for sizing lever 1.
    """
    db = recon_db._db_path
    _seed_addendum_2(db)
    before = remediation_duty_cycle(drain_stats(db, 'reify'))

    for i in range(4):
        _insert_run(
            db, run_type='targeted', trigger_reason='task_status_changed',
            started_at=_t(f'15:1{i}:00'), completed_at=_t(f'15:1{i}:50'),
            events_processed=2,
        )

    assert remediation_duty_cycle(drain_stats(db, 'reify')) == pytest.approx(before)


@pytest.mark.asyncio
async def test_remediation_duty_cycle_counts_steady_state_wall_clock(recon_db):
    """Steady-state runs are lock-held drain too, so they sit in the denominator."""
    db = recon_db._db_path
    _seed_addendum_2(db)
    _insert_run(
        db, trigger_reason='quiescence',
        started_at=_t('15:00:00'), completed_at=_t('15:15:45'), events_processed=50,
    )  # 945s

    duty = remediation_duty_cycle(drain_stats(db, 'reify'))
    assert duty == pytest.approx(1495.0 / (3409.0 + 945.0))


@pytest.mark.asyncio
async def test_remediation_duty_cycle_is_zero_for_an_empty_denominator(recon_db):
    """No drain wall-clock at all returns 0.0 rather than raising."""
    assert remediation_duty_cycle(drain_stats(recon_db._db_path, 'reify')) == 0.0


# ── capacity arithmetic ────────────────────────────────────────────────
#
# Pure functions over caller-supplied inputs — nothing here measures a live
# system, so these assertions are exact arithmetic rather than tolerances.
#
# The cost model is T(B) = F + c*B.  The two measured 2026-07-25 reify points
# (50 events/945s steady state; 393 events/954s backlog chunk) fit it with
# c ~ 0.03 s/event, i.e. essentially ALL fixed cost — but they ran on
# different tiers, so that fit mixes regimes and would over-claim.  The
# shipped constants take chunk 2's 2.94 s/event as an upper bound on marginal
# cost instead: F = 900, c = 2.5.


def test_cycle_seconds_implements_the_two_parameter_cost_model() -> None:
    """T(B) = F + c*B, literally."""
    assert cycle_seconds(50, 900.0, 2.5) == pytest.approx(1025.0)
    assert cycle_seconds(150, 900.0, 2.5) == pytest.approx(1275.0)
    # A zero-size batch still pays the full fixed cost — that IS the finding.
    assert cycle_seconds(0, 900.0, 2.5) == pytest.approx(900.0)


def test_seconds_per_event_reproduces_the_measured_steady_state_point() -> None:
    """(a) The conservative fit predicts the observed steady-state point within 10%.

    Measured: 50 events / 945s = 18.9 s/event.  Predicted: 1025/50 = 20.5.
    A fit that could not reproduce its own input point would not be usable to
    extrapolate to a larger batch.
    """
    measured = 945.0 / 50
    predicted = seconds_per_event(50, FITTED_CYCLE_FIXED_SECONDS,
                                  FITTED_CYCLE_MARGINAL_SECONDS)
    assert predicted == pytest.approx(20.5)
    assert measured == pytest.approx(18.9)
    assert abs(predicted - measured) / measured < 0.10


def test_seconds_per_event_predicts_the_amortised_cost_at_the_target_batch() -> None:
    """(a) B=150 costs 8.5 s/event — a 2.2x amortisation over the measured point."""
    predicted = seconds_per_event(STEADY_STATE_AMORTISATION_MIN_BATCH,
                                  FITTED_CYCLE_FIXED_SECONDS,
                                  FITTED_CYCLE_MARGINAL_SECONDS)
    assert STEADY_STATE_AMORTISATION_MIN_BATCH == 150
    assert predicted == pytest.approx(8.5)
    assert (945.0 / 50) / predicted == pytest.approx(2.22, abs=0.01)


def test_seconds_per_event_rejects_a_non_positive_batch() -> None:
    """Per-event cost is undefined for an empty batch; raise rather than divide."""
    with pytest.raises(ValueError):
        seconds_per_event(0, 900.0, 2.5)


def test_sustainable_events_per_day_applies_utilisation_and_duty_cycle() -> None:
    """(b) 86400 * utilisation * (1 - duty_cycle) / seconds_per_event.

    Hand-computed, not re-implemented: 86400 * 0.562 / 2.43 = 19982.2, halved
    to 9991.1 at 50% utilisation.  Restating the formula in the assertion would
    only detect a change made in one place and not the other.
    """
    # ADDENDUM 2 backlog-chunk rate, with the measured remediation duty cycle.
    assert sustainable_events_per_day(2.43, 0.438, 1.0) == pytest.approx(19982.2, abs=0.1)
    # Utilisation scales linearly.
    assert sustainable_events_per_day(2.43, 0.438, 0.5) == pytest.approx(9991.1, abs=0.1)


def test_sustainable_events_per_day_recovers_the_duty_cycle_lever_1_removes() -> None:
    """(b) Driving the duty cycle to 0 is worth 1/(1-0.438) ~ 1.8x on the same rate.

    That is the whole of lever 1: a scheduling change, no per-event work saved.
    Both endpoints are pinned by value (86400/2.43 = 35555.6 with no
    remediation, 19982.2 with it), so the ratio cannot be satisfied by two
    equally-wrong figures.
    """
    with_remediation = sustainable_events_per_day(2.43, 0.438, 1.0)
    without = sustainable_events_per_day(2.43, 0.0, 1.0)
    assert with_remediation == pytest.approx(19982.2, abs=0.1)
    assert without == pytest.approx(35555.6, abs=0.1)
    assert without / with_remediation == pytest.approx(1.7794, abs=0.0001)


def test_sustainable_events_per_day_reproduces_the_addendum_2_observed_gross_rate() -> None:
    """(b) End-to-end validation: the formula reproduces the fixture it was fitted against.

    esc-3049-1 observed that the plan's cross-check figure of "~655 events/h
    gross" does not fall out of the stated formula, and no reading of the
    ADDENDUM 2 fixture produces it.  Both halves of that are true, but the
    conclusion is narrower than "the figure does not reproduce": the formula
    reproduces the fixture EXACTLY once its two inputs are taken at the same
    scope.

    The plan paired 2.43 s/event -- which is chunk 1 ALONE (393 events / 954s)
    -- with duty 0.438, which is the aggregate over BOTH chunks.  Mixing a
    single-chunk rate with a two-chunk duty is what breaks the cross-check.
    Taken consistently over the whole fixture:

        drain-only rate = (954 + 960)s / (393 + 326) events = 2.662 s/event
        duty            = 1495 / (1495 + 1914)              = 0.4386

    and the formula returns the observed gross rate, computed independently as
    total events over total lock-held wall-clock, to within a rounding step:

        719 events / 3409s = 759.3 events/h

    This is the strongest evidence available that the capacity claim is sound,
    so it is pinned here rather than left as prose.  The "~655/h" figure is a
    scope-mix in the opposite direction -- the AVERAGE event count (359.5) over
    chunk 1's cycle period (33.1 min) = 652/h -- which is why no consistent
    reading of the fixture reaches it.
    """
    events, drain_s, remediation_s = 393 + 326, 954.0 + 960.0, 756.0 + 739.0

    drain_only_rate = drain_s / events
    duty = remediation_s / (remediation_s + drain_s)
    assert drain_only_rate == pytest.approx(2.662, abs=0.001)
    assert duty == pytest.approx(0.4386, abs=0.0001)

    # Observed, computed WITHOUT the formula: events over total lock-held time.
    observed_per_hour = events / (drain_s + remediation_s) * 3600.0
    assert observed_per_hour == pytest.approx(759.3, abs=0.1)

    predicted_per_hour = sustainable_events_per_day(drain_only_rate, duty, 1.0) / 24.0
    assert predicted_per_hour == pytest.approx(observed_per_hour, rel=1e-9), (
        'the formula must reproduce the fixture it was fitted against when its '
        'rate and duty are taken at the same scope; a mismatch here means the '
        'capacity claim no longer rests on the ADDENDUM 2 measurement'
    )

    # The plan's scope-mixed pairing is the higher figure, and is NOT the
    # observed rate -- pinned so the two are never conflated again.
    assert sustainable_events_per_day(2.43, 0.438, 1.0) / 24.0 == pytest.approx(
        832.6, abs=0.1)


def test_sustainable_events_per_day_rejects_impossible_inputs() -> None:
    """A non-positive rate or an out-of-range duty/utilisation is a caller bug."""
    with pytest.raises(ValueError):
        sustainable_events_per_day(0.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        sustainable_events_per_day(2.43, 1.5, 1.0)
    with pytest.raises(ValueError):
        sustainable_events_per_day(2.43, 0.0, 0.0)


def test_capacity_verdict_is_insufficient_for_the_pre_fix_ceiling() -> None:
    """(c) B=50 with the measured 0.438 duty cycle does not clear the 3.5k/day burst.

    2368.6/day sustainable against OBSERVED_BURST_EVENTS_PER_DAY = 3500 — the
    overrun this task exists to fix.
    """
    pre_fix = sustainable_events_per_day(
        seconds_per_event(50, FITTED_CYCLE_FIXED_SECONDS,
                          FITTED_CYCLE_MARGINAL_SECONDS),
        0.438, 1.0,
    )
    assert pre_fix == pytest.approx(2368.6, abs=0.1)

    verdict = capacity_verdict(pre_fix, OBSERVED_BURST_EVENTS_PER_DAY)
    assert verdict['verdict'] == 'insufficient'
    # 2368.6 / 3500, hand-computed rather than re-derived from pre_fix.
    assert verdict['headroom_ratio'] == pytest.approx(0.6767, abs=0.0001)
    assert verdict['headroom_ratio'] < 1.0


def test_capacity_verdict_is_sufficient_for_the_post_fix_figure() -> None:
    """(c) Both levers together clear the burst with headroom.

    Lever 2 raises the batch to 150 (8.5 s/event) and lever 1 removes the
    remediation duty cycle, giving 10164.7/day against a 3500/day burst.
    """
    post_fix = sustainable_events_per_day(
        seconds_per_event(STEADY_STATE_AMORTISATION_MIN_BATCH,
                          FITTED_CYCLE_FIXED_SECONDS,
                          FITTED_CYCLE_MARGINAL_SECONDS),
        0.0, 1.0,
    )
    assert post_fix == pytest.approx(10164.7, abs=0.1)

    verdict = capacity_verdict(post_fix, OBSERVED_BURST_EVENTS_PER_DAY)
    assert verdict['verdict'] == 'sufficient'
    assert verdict['headroom_ratio'] == pytest.approx(post_fix / 3500)
    assert verdict['headroom_ratio'] > 2.9


def test_capacity_verdict_boundary_is_sufficient_at_exactly_break_even() -> None:
    """Exactly meeting the burst counts as sufficient; a hair under does not."""
    assert capacity_verdict(3500.0, 3500)['verdict'] == 'sufficient'
    assert capacity_verdict(3499.0, 3500)['verdict'] == 'insufficient'
    assert capacity_verdict(3500.0, 3500)['headroom_ratio'] == pytest.approx(1.0)


def test_observed_burst_constant_matches_the_task_measurement() -> None:
    """The burst figure the capacity claim is checked against is ~3.5k/day."""
    assert OBSERVED_BURST_EVENTS_PER_DAY == 3500


# ── build_report / CLI ─────────────────────────────────────────────────
#
# The report is the deliverable for ask (a): an operator points it at a live
# reconciliation.db and gets inflow, drain and the resulting capacity claim
# out, without a live harness process.  It must also state its own retention
# caveat — inflow BEFORE the event_arrival_hourly rollup landed is gone
# forever, because cleanup_drained used to just DELETE those rows.


@pytest.mark.asyncio
async def test_build_report_carries_every_section(recon_db):
    """The report is a JSON-serialisable superset of the component readers."""
    db = recon_db._db_path
    await _seed_inflow(recon_db)
    _seed_addendum_2(db)

    report = build_report(db, 'reify')

    assert report['project_id'] == 'reify'
    assert report['inflow']['hourly'] == inflow_hourly(db, 'reify')
    expected_daily = inflow_daily(db, 'reify')
    assert report['inflow']['daily'] == expected_daily['daily']
    assert report['inflow']['composition'] == expected_daily['composition']
    assert report['inflow']['total_events'] == 7

    expected_drain = drain_stats(db, 'reify')
    assert report['drain'] == expected_drain
    assert report['remediation_duty_cycle'] == pytest.approx(
        remediation_duty_cycle(expected_drain))
    assert report['remediation_duty_cycle'] == pytest.approx(0.438, abs=0.001)

    # HAND-COMPUTED, not re-executed from the implementation's own expression.
    # Fixture: 719 events; backlog 1914s; remediation 1495s; total 3409s.
    #   observed status quo  = 86400 * 719 / 3409 = 18222.8/day
    #   post-lever-1         = 86400 * 719 / 1914 = 32456.4/day
    # Each figure applies the remediation penalty exactly once, in its own rate.
    assert report['sustainable_events_per_day'] == pytest.approx(18222.8, abs=1.0)
    assert report['remediation_free_events_per_day'] == pytest.approx(32456.4, abs=1.0)

    assert report['capacity_verdict']['verdict'] in {'sufficient', 'insufficient'}
    assert report['capacity_verdict']['observed_burst_per_day'] == (
        OBSERVED_BURST_EVENTS_PER_DAY)

    assert isinstance(report['retention_note'], str) and report['retention_note']

    # The whole thing must survive a JSON round-trip — it is a CLI payload.
    assert json.loads(json.dumps(report)) == report


@pytest.mark.asyncio
async def test_build_report_applies_the_remediation_penalty_exactly_once(recon_db):
    """Regression: the headline must not be the double-discounted figure.

    `drain_seconds_per_event` (4.7413) already carries the remediation penalty
    — remediation wall-clock is in its numerator, remediation's zero events add
    nothing to its denominator.  Feeding it to
    `sustainable_events_per_day(rate, duty)` with the measured duty applies the
    same penalty a SECOND time and yields 86400*(1-0.4385)/4.7413 = 10231.3,
    understating the observed capacity by 1/(1-duty) ~ 1.78x.
    """
    db = recon_db._db_path
    _seed_addendum_2(db)

    report = build_report(db, 'reify')

    assert report['sustainable_events_per_day'] != pytest.approx(10231.3, abs=1.0), (
        'sustainable_events_per_day is the double-discounted figure — the '
        'remediation penalty applied twice, once in the inclusive rate and '
        'again via the duty cycle argument'
    )
    assert report['sustainable_events_per_day'] == pytest.approx(18222.8, abs=1.0)


@pytest.mark.asyncio
async def test_build_report_lever_1_delta_is_exactly_the_duty_cycle(recon_db):
    """The gap between the two capacity figures IS the remediation duty cycle.

    Pinned by hand-computed numbers rather than by `ratio == 1/(1-duty)`: that
    identity is true of the implementation's own algebra whatever the mode
    partition, so it would survive a dropped remediation exclusion.  These
    won't.

    Fixture: 719 events; backlog 1914s; remediation 1495s; total 3409s.
      duty      = 1495 / 3409                = 0.4385
      observed  = 86400 * 719 / 3409         = 18222.8/day
      lever 1   = 86400 * 719 / 1914         = 32456.4/day
      ratio     = 3409 / 1914                = 1.7811
    """
    db = recon_db._db_path
    _seed_addendum_2(db)

    report = build_report(db, 'reify')

    assert report['remediation_duty_cycle'] == pytest.approx(0.4385, abs=0.0001)
    assert report['sustainable_events_per_day'] == pytest.approx(18222.8, abs=1.0)
    assert report['remediation_free_events_per_day'] == pytest.approx(32456.4, abs=1.0)
    assert (
        report['remediation_free_events_per_day']
        / report['sustainable_events_per_day']
    ) == pytest.approx(1.7811, abs=0.0001)


@pytest.mark.asyncio
async def test_build_report_renders_each_verdict_from_its_own_figure(recon_db):
    """Two capacity figures, two verdicts — neither may be rendered from the other."""
    db = recon_db._db_path
    _seed_addendum_2(db)

    report = build_report(db, 'reify')

    assert report['capacity_verdict']['sustainable_per_day'] == pytest.approx(
        report['sustainable_events_per_day'])
    assert report['remediation_free_capacity_verdict'][
        'sustainable_per_day'] == pytest.approx(
        report['remediation_free_events_per_day'])

    # On this fixture both clear the 3500/day burst: ~5.2x and ~9.3x headroom.
    assert report['capacity_verdict']['verdict'] == 'sufficient'
    assert report['capacity_verdict']['headroom_ratio'] == pytest.approx(5.21, abs=0.01)
    assert report['remediation_free_capacity_verdict']['verdict'] == 'sufficient'
    assert report['remediation_free_capacity_verdict'][
        'headroom_ratio'] == pytest.approx(9.27, abs=0.01)
    assert report['remediation_free_capacity_verdict'][
        'observed_burst_per_day'] == OBSERVED_BURST_EVENTS_PER_DAY


@pytest.mark.asyncio
async def test_build_report_degrades_on_an_empty_database(recon_db):
    """No runs at all: every section present, capacity reported as unknown.

    The capacity arithmetic raises on a non-positive rate by design, so the
    report must decline to compute rather than propagate that — an empty DB is
    an operator pointing the tool somewhere new, not a bug.
    """
    report = build_report(recon_db._db_path, 'reify')

    assert report['inflow']['hourly'] == {}
    assert report['drain']['drain_seconds_per_event'] is None
    assert report['remediation_duty_cycle'] == 0.0
    assert report['sustainable_events_per_day'] is None
    assert report['capacity_verdict'] is None
    # The post-lever-1 pair degrades the same way — no rate, no claim.
    assert report['drain']['drain_only_seconds_per_event'] is None
    assert report['remediation_free_events_per_day'] is None
    assert report['remediation_free_capacity_verdict'] is None
    assert report['retention_note']


@pytest.mark.asyncio
async def test_build_report_threads_since_into_both_halves(recon_db):
    """`since` filters inflow and drain alike, so the two halves stay comparable."""
    db = recon_db._db_path
    await _seed_inflow(recon_db)
    _seed_addendum_2(db)

    report = build_report(db, 'reify', since='2026-07-25T14:00:00+00:00')

    assert report['since'] == '2026-07-25T14:00:00+00:00'
    assert report['since_effective'] == '2026-07-25T14:00:00+00:00'
    assert report['inflow']['hourly'] == {'2026-07-25T14': 2, '2026-07-26T00': 1}
    # Only the 14:13 chunk and the 14:29 remediation started at/after the floor.
    assert report['drain']['modes']['backlog_chunk']['run_count'] == 1
    assert report['drain']['modes']['remediation']['run_count'] == 1


@pytest.mark.asyncio
async def test_build_report_since_covers_one_window_off_the_hour(recon_db):
    """An off-the-hour `since` must not give the two halves different windows.

    Inflow reads hourly-bucketed data, so it can only floor to the hour; drain
    compares an exact instant against runs.started_at.  Left unaligned,
    `--since 14:30` counted arrivals from 14:00 while ignoring every run that
    started before 14:30 — including the 14:13 chunk and the 14:29:41
    remediation — so the inflow-vs-capacity comparison, which is the whole
    point of the tool, silently compared unequal windows.

    Both halves now floor to 14:00, so an off-the-hour `since` produces exactly
    the same window as the on-the-hour one.
    """
    db = recon_db._db_path
    await _seed_inflow(recon_db)
    _seed_addendum_2(db)

    off_hour = build_report(db, 'reify', since='2026-07-25T14:30:00+00:00')
    on_hour = build_report(db, 'reify', since='2026-07-25T14:00:00+00:00')

    assert off_hour['since'] == '2026-07-25T14:30:00+00:00', (
        'the requested value is echoed verbatim'
    )
    assert off_hour['since_effective'] == '2026-07-25T14:00:00+00:00', (
        'the widening must be visible in the payload, not implicit'
    )

    # The drain half honours the SAME floor the inflow half was already using:
    # the 14:13:36 chunk (326 events) and the 14:29:41 remediation are in.
    assert off_hour['drain']['modes']['backlog_chunk']['run_count'] == 1
    assert off_hour['drain']['modes']['backlog_chunk']['events'] == 326
    assert off_hour['drain']['modes']['remediation']['run_count'] == 1
    assert off_hour['inflow']['hourly'] == {'2026-07-25T14': 2, '2026-07-26T00': 1}

    assert off_hour['drain'] == on_hour['drain']
    assert off_hour['inflow'] == on_hour['inflow']
    assert off_hour['sustainable_events_per_day'] == pytest.approx(
        on_hour['sustainable_events_per_day'])


@pytest.mark.asyncio
async def test_main_prints_the_report_as_json_and_returns_zero(recon_db, capsys):
    """--json prints exactly the build_report payload and exits 0."""
    db = recon_db._db_path
    await _seed_inflow(recon_db)
    _seed_addendum_2(db)

    rc = main(['--db', str(db), '--project', 'reify', '--json'])
    assert rc == 0

    printed = json.loads(capsys.readouterr().out)
    assert printed == build_report(db, 'reify')
    # Pinned by value too: equality with build_report alone would pass for any
    # report, including a wrong one.
    assert printed['remediation_duty_cycle'] == pytest.approx(0.438, abs=0.001)
    assert printed['sustainable_events_per_day'] == pytest.approx(18222.8, abs=1.0)
    assert printed['inflow']['total_events'] == 7


@pytest.mark.asyncio
async def test_main_threads_since_through_to_the_report(recon_db, capsys):
    """--since reaches build_report rather than being parsed and dropped."""
    db = recon_db._db_path
    await _seed_inflow(recon_db)
    _seed_addendum_2(db)

    rc = main([
        '--db', str(db), '--project', 'reify',
        '--since', '2026-07-25T14:00:00+00:00', '--json',
    ])
    assert rc == 0

    printed = json.loads(capsys.readouterr().out)
    assert printed == build_report(db, 'reify', since='2026-07-25T14:00:00+00:00')
    assert printed['inflow']['hourly'] == {'2026-07-25T14': 2, '2026-07-26T00': 1}


@pytest.mark.asyncio
async def test_main_without_json_prints_a_readable_summary(recon_db, capsys):
    """The default output is human-readable but still names the headline figures."""
    db = recon_db._db_path
    await _seed_inflow(recon_db)
    _seed_addendum_2(db)

    rc = main(['--db', str(db), '--project', 'reify'])
    assert rc == 0

    out = capsys.readouterr().out
    assert 'reify' in out
    assert 'remediation' in out.lower()
    assert out.strip(), 'the default output must not be empty'


@pytest.mark.asyncio
async def test_main_reports_an_unparseable_since_as_a_since_error(recon_db, capsys):
    """A bad --since names --since, and never reaches the database."""
    db = recon_db._db_path
    _seed_addendum_2(db)

    rc = main(['--db', str(db), '--project', 'reify', '--since', 'yesterday-ish'])

    assert rc != 0
    captured = capsys.readouterr()
    message = captured.err + captured.out
    assert '--since' in message
    assert 'yesterday-ish' in message, 'the message must quote the value that failed'
    assert 'Traceback' not in message


@pytest.mark.asyncio
async def test_main_reports_a_corrupt_row_as_a_database_error(recon_db, capsys):
    """A malformed stored timestamp must not be misreported as a --since failure.

    build_report raises ValueError from utc_hour_bucket on a corrupt
    event_buffer.timestamp, exactly as it does for an unparseable --since.  A
    blanket `except ValueError` blamed --since for both, so an operator who
    never passed --since was told `could not parse --since None` and never saw
    the corrupt-row diagnosis.
    """
    db = recon_db._db_path
    _seed_addendum_2(db)
    await _seed_inflow(recon_db)

    conn = sqlite3.connect(db)
    try:
        conn.execute("UPDATE event_buffer SET timestamp = 'not-a-timestamp'")
        conn.commit()
    finally:
        conn.close()

    rc = main(['--db', str(db), '--project', 'reify', '--json'])

    assert rc != 0
    captured = capsys.readouterr()
    message = captured.err + captured.out
    assert str(db) in message, 'the message must name the database that is corrupt'
    assert '--since' not in message, (
        'the operator passed no --since; blaming it hides the real diagnosis'
    )
    assert 'Traceback' not in message


def test_main_reports_a_missing_database_without_a_traceback(tmp_path, capsys):
    """A wrong --db is an operator typo, not a crash."""
    missing = tmp_path / 'nope' / 'reconciliation.db'

    rc = main(['--db', str(missing), '--project', 'reify', '--json'])

    assert rc != 0, 'a missing database must be a non-zero exit'
    captured = capsys.readouterr()
    message = captured.err + captured.out
    assert str(missing) in message, 'the message must name the path that was tried'
    assert 'Traceback' not in message
