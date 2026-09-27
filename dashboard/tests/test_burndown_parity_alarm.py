"""The burndown parity alarm — task 3543 / PRD ι, spec E12.

The historical defect this module pins: the factory ran up to 33 tasks
in-progress against a configured ``max_concurrent_tasks`` of 24, and every
surface reported it as ordinary throughput.  The datum existed; nothing
projected it.

Two halves:

* the read side carries ``in_progress_live`` / ``in_progress_stranded`` /
  ``concurrency_cap`` out of the snapshot table, and
* :func:`compute_parity_alarm` is a PURE function over that series, so the
  alarm is fixture-testable with no DB, no clock and no config I/O.

The alarm's numerator is the rows' ``in_progress_live``, not the census
``in_progress`` (plans/dashboard-one-datum-one-path-prd.md decision 9).

The cap is compared PER SNAPSHOT against the cap stored on that snapshot.
``max_concurrent_tasks`` is restart-only (red-tier), but a burndown window
spans restarts and the cap also varies between projects, so it is
TIME-VARYING across the window regardless: re-deriving one "current" cap and
applying it across history would forgive a past breach after a cap raise and
invent one after a cut.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import aiosqlite
import pytest

from dashboard.data import redux_api
from dashboard.data.burndown import (
    BURNDOWN_SCHEMA,
    aggregate_burndown_projects,
    aggregate_burndown_series,
    compute_parity_alarm,
    get_burndown_series,
)

_LEGACY_BURNDOWN_SCHEMA = """\
CREATE TABLE IF NOT EXISTS snapshots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id  TEXT    NOT NULL,
    ts          TEXT    NOT NULL,
    pending     INTEGER NOT NULL DEFAULT 0,
    in_progress INTEGER NOT NULL DEFAULT 0,
    blocked     INTEGER NOT NULL DEFAULT 0,
    deferred    INTEGER NOT NULL DEFAULT 0,
    cancelled   INTEGER NOT NULL DEFAULT 0,
    done        INTEGER NOT NULL DEFAULT 0
);
"""

_SPLIT_KEYS = ('in_progress_live', 'in_progress_stranded', 'concurrency_cap')

_LOGGER = 'dashboard.data.burndown'


def _ts(minute: int) -> str:
    """A recent timestamp inside the default 7-day window."""
    return (datetime.now(UTC) - timedelta(minutes=minute)).isoformat()


_FIXTURE_COLUMNS = (
    'project_id', 'ts', 'pending', 'in_progress', 'blocked', 'deferred', 'cancelled',
    'done', 'in_progress_live', 'in_progress_stranded', 'concurrency_cap', 'state',
)
_INSERT_FIXTURE_ROW = (
    f'INSERT INTO snapshots ({", ".join(_FIXTURE_COLUMNS)}) '
    f'VALUES ({", ".join("?" for _ in _FIXTURE_COLUMNS)})'
)


def _make_db(path: Path, rows: list[dict]) -> None:
    conn = sqlite3.connect(str(path))
    conn.executescript(BURNDOWN_SCHEMA)
    for row in rows:
        conn.execute(
            _INSERT_FIXTURE_ROW,
            (
                row.get('project_id', 'p1'),
                row['ts'],
                row.get('pending', 0),
                row.get('in_progress', 0),
                row.get('blocked', 0),
                row.get('deferred', 0),
                row.get('cancelled', 0),
                row.get('done', 0),
                row.get('in_progress_live', row.get('in_progress', 0)),
                row.get('in_progress_stranded', 0),
                row.get('concurrency_cap'),
                # NULL, a legacy measured row, unless the fixture says 'value'.
                row.get('state'),
            ),
        )
    conn.commit()
    conn.close()


def _add_gap_row(path: Path, project_id: str, ts: str) -> None:
    """A gap row exactly as the collector writes one: no count column named."""
    conn = sqlite3.connect(str(path))
    conn.execute(
        'INSERT INTO snapshots (project_id, ts, state, reason) VALUES (?, ?, ?, ?)',
        (project_id, ts, 'gap', 'rows: offline'),
    )
    conn.commit()
    conn.close()


def _series(
    live: list[int],
    caps: list[int | None],
    *,
    stranded: list[int] | None = None,
) -> dict:
    """A minimal read-side series shaped like ``get_burndown_series`` returns.

    ``in_progress`` is the census total ``live + stranded``, so a fixture can
    make the census and the live count differ.
    """
    n = len(live)
    stranded = list(stranded) if stranded is not None else [0] * n
    return {
        'labels': [_ts(n - i) for i in range(n)],
        'in_progress_live': list(live),
        'in_progress_stranded': stranded,
        'in_progress': [count + held for count, held in zip(live, stranded, strict=True)],
        'concurrency_cap': list(caps),
    }


# ---------------------------------------------------------------------------
# Read side: the three columns reach the series
# ---------------------------------------------------------------------------


class TestSeriesCarriesSplitAndCap:
    @pytest.mark.asyncio
    async def test_get_burndown_series_returns_the_new_keys(self, tmp_path):
        db_path = tmp_path / 'burndown.db'
        _make_db(db_path, [
            {'ts': _ts(2), 'in_progress': 5, 'in_progress_live': 3,
             'in_progress_stranded': 2, 'concurrency_cap': 24},
        ])

        async with aiosqlite.connect(str(db_path)) as db:
            series = await get_burndown_series(db, 'p1')

        for key in _SPLIT_KEYS:
            assert key in series, f'{key} missing from the series'
        assert series['in_progress'] == [5]
        assert series['in_progress_live'] == [3]
        assert series['in_progress_stranded'] == [2]
        assert series['concurrency_cap'] == [24]

    @pytest.mark.asyncio
    async def test_empty_series_default_is_widened(self):
        """A caller iterating the new keys must not KeyError on a missing DB."""
        series = await get_burndown_series(None, 'p1')
        for key in _SPLIT_KEYS:
            assert series[key] == [], f'{key} missing from the empty default'

    @pytest.mark.asyncio
    async def test_legacy_unmigrated_db_still_yields_its_zones(self, tmp_path):
        """An un-migrated peer DB must degrade to unknown, NOT to a blank chart.

        ``_burndown_dbs`` opens OTHER projects' burndown.db files through the
        read-only pool, and nothing migrates those.  A bare widened SELECT
        would raise ``no such column`` there, hit the except-and-return-empty
        guard, and silently blank that project's whole burndown — losing the
        six zones that ARE present to report three that are not.
        """
        db_path = tmp_path / 'legacy.db'
        conn = sqlite3.connect(str(db_path))
        conn.executescript(_LEGACY_BURNDOWN_SCHEMA)
        conn.execute(
            'INSERT INTO snapshots (project_id, ts, pending, in_progress, blocked, '
            'deferred, cancelled, done) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
            ('p1', _ts(2), 5, 3, 1, 0, 0, 20),
        )
        conn.commit()
        conn.close()

        async with aiosqlite.connect(str(db_path)) as db:
            series = await get_burndown_series(db, 'p1')

        assert series['labels'], 'the legacy rows must still be returned'
        assert series['in_progress'] == [3]
        assert series['done'] == [20]
        # Unknown split defaults to all-live so conservation still holds, and an
        # unknown cap is NULL — never 0, which would alarm on every row.
        assert series['in_progress_live'] == [3]
        assert series['in_progress_stranded'] == [0]
        assert series['concurrency_cap'] == [None]

    @pytest.mark.asyncio
    async def test_aggregate_merges_the_new_keys(self, tmp_path):
        a = tmp_path / 'a.db'
        b = tmp_path / 'b.db'
        _make_db(a, [
            {'ts': _ts(3), 'in_progress': 4, 'in_progress_live': 4,
             'in_progress_stranded': 0, 'concurrency_cap': 24},
        ])
        _make_db(b, [
            {'ts': _ts(2), 'in_progress': 7, 'in_progress_live': 2,
             'in_progress_stranded': 5, 'concurrency_cap': 6},
        ])

        async with aiosqlite.connect(str(a)) as da, aiosqlite.connect(str(b)) as db_b:
            merged = await aggregate_burndown_series([da, db_b], 'p1')

        assert merged['in_progress'] == [4, 7]
        assert merged['in_progress_live'] == [4, 2]
        assert merged['in_progress_stranded'] == [0, 5]
        assert merged['concurrency_cap'] == [24, 6]

    @pytest.mark.asyncio
    async def test_aggregate_empty_default_is_widened(self):
        merged = await aggregate_burndown_series([], 'p1')
        for key in _SPLIT_KEYS:
            assert merged[key] == [], f'{key} missing from the aggregate empty default'

    @pytest.mark.asyncio
    async def test_aggregate_last_writer_wins_covers_the_new_keys(self, tmp_path):
        """A colliding timestamp must not leave the new keys behind."""
        shared_ts = _ts(2)
        a = tmp_path / 'a.db'
        b = tmp_path / 'b.db'
        _make_db(a, [{'ts': shared_ts, 'in_progress': 4, 'in_progress_live': 4,
                      'in_progress_stranded': 0, 'concurrency_cap': 24}])
        _make_db(b, [{'ts': shared_ts, 'in_progress': 9, 'in_progress_live': 1,
                      'in_progress_stranded': 8, 'concurrency_cap': 6}])

        async with aiosqlite.connect(str(a)) as da, aiosqlite.connect(str(b)) as db_b:
            merged = await aggregate_burndown_series([da, db_b], 'p1')

        assert merged['labels'] == [shared_ts]
        assert merged['in_progress'] == [9], 'later DB wins'
        assert merged['in_progress_stranded'] == [8]
        assert merged['concurrency_cap'] == [6]

    @pytest.mark.asyncio
    async def test_legacy_unmigrated_db_series_does_not_alarm(self, tmp_path):
        """The census fill of an un-migrated DB never reaches the alarm with a cap.

        ``get_burndown_series`` fills ``in_progress_live`` with the census so
        the stacked chart still conserves, but that DB has no cap column, so
        every cap is NULL and the filled count is never compared.
        """
        db_path = tmp_path / 'legacy.db'
        conn = sqlite3.connect(str(db_path))
        conn.executescript(_LEGACY_BURNDOWN_SCHEMA)
        conn.execute(
            'INSERT INTO snapshots (project_id, ts, pending, in_progress, blocked, '
            'deferred, cancelled, done) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
            ('p1', _ts(2), 5, 99, 1, 0, 0, 20),
        )
        conn.commit()
        conn.close()

        async with aiosqlite.connect(str(db_path)) as db:
            series = await get_burndown_series(db, 'p1')

        result = compute_parity_alarm(series)

        assert result['parity_peak'] is None
        assert result['parity_alarm'] is False


# ---------------------------------------------------------------------------
# compute_parity_alarm — pure
# ---------------------------------------------------------------------------


class TestComputeParityAlarm:
    def test_returns_the_documented_keys(self):
        result = compute_parity_alarm(_series([], []))

        assert set(result) == {
            'parity_alarm', 'parity_cap', 'parity_peak', 'parity_breach_count',
        }

    def test_live_shaped_regression_peak_33_against_cap_24(self):
        """The historical defect: 33 in-progress against a cap of 24.

        Two stranded rows ride on every snapshot, so the census peaks at 35
        with four breaches: the asserts below hold only for the live count.
        """
        series = _series(
            [18, 24, 29, 33, 27],
            [24, 24, 24, 24, 24],
            stranded=[2] * 5,
        )

        result = compute_parity_alarm(series)

        assert result['parity_alarm'] is True
        assert result['parity_cap'] == 24
        assert result['parity_peak'] == 33
        assert result['parity_breach_count'] == 3, '29, 33 and 27 all exceed 24'

    def test_equal_to_cap_does_not_alarm(self):
        """The cap is INCLUSIVE — running exactly at capacity is healthy."""
        result = compute_parity_alarm(_series([24, 24], [24, 24]))

        assert result['parity_alarm'] is False
        assert result['parity_breach_count'] == 0
        assert result['parity_peak'] == 24
        assert result['parity_cap'] == 24

    def test_stranded_pile_up_over_the_cap_does_not_alarm(self):
        """A stranded row holds no scheduler slot and is surfaced by the
        stranded band; counting it would fire the over-dispatch alarm on a
        strand pile-up (PRD decision 9).

        The census reads 30 and 33 against a cap of 24; the live count never
        exceeds it.
        """
        result = compute_parity_alarm(_series([20, 24], [24, 24], stranded=[10, 9]))

        assert result['parity_alarm'] is False
        assert result['parity_breach_count'] == 0
        assert result['parity_peak'] == 24
        assert result['parity_cap'] == 24

    @pytest.mark.parametrize('series', [
        pytest.param(
            {'labels': ['a', 'b'], 'in_progress': [30, 31],
             'in_progress_live': [None, None], 'concurrency_cap': [24, 24]},
            id='live-entries-none',
        ),
        pytest.param(
            {'labels': ['a', 'b'], 'in_progress': [30, 31],
             'concurrency_cap': [24, 24]},
            id='live-key-absent',
        ),
    ])
    def test_a_snapshot_with_no_live_count_is_unmeasured(self, series):
        """No live count means unmeasured: never the census, never 0.

        Reading the census would alarm on 30/31 against 24; reading the hole
        as 0 would publish peak 0 against cap 24.  Both fail this equality.
        """
        assert compute_parity_alarm(series) == {
            'parity_alarm': False,
            'parity_cap': None,
            'parity_peak': None,
            'parity_breach_count': 0,
        }

    def test_pre_split_history_is_skipped_and_the_measured_suffix_still_alarms(self):
        """Unmeasured history does not mask a breach the split did measure."""
        result = compute_parity_alarm({
            'labels': ['a', 'b', 'c'],
            'in_progress': [40, 40, 26],
            'in_progress_live': [None, None, 26],
            'concurrency_cap': [24, 24, 24],
        })

        assert result['parity_alarm'] is True
        assert result['parity_breach_count'] == 1
        assert result['parity_peak'] == 26
        assert result['parity_cap'] == 24

    def test_all_caps_null_is_unknown_not_a_breach(self, caplog):
        """Unknown must never render as 'not breaching' by accident, and the
        collapse must leave a trace rather than being silent."""
        with caplog.at_level(logging.DEBUG, logger=_LOGGER):
            result = compute_parity_alarm(_series([99, 99], [None, None]))

        assert result['parity_alarm'] is False
        assert result['parity_cap'] is None
        assert result['parity_peak'] is None, 'a peak with nothing to compare against is not a peak'
        assert result['parity_breach_count'] == 0
        assert [r for r in caplog.records if _LOGGER in r.name], (
            'an all-unknown cap series must leave a log line'
        )

    def test_missing_cap_key_entirely_is_unknown(self):
        """A pre-3543 series (six zones only) must not raise."""
        result = compute_parity_alarm({'labels': ['a'], 'in_progress_live': [50]})

        assert result == {
            'parity_alarm': False,
            'parity_cap': None,
            'parity_peak': None,
            'parity_breach_count': 0,
        }

    def test_each_snapshot_is_judged_against_its_own_cap(self):
        """The load-bearing case: a cap RAISE must not retro-forgive a breach.

        in_progress is a flat 20 throughout.  Against the historical cap of 10
        the first two snapshots breached; against today's cap of 40 they did
        not.  Judging the series by the latest cap alone would report zero
        breaches and erase the incident.
        """
        series = _series([20, 20, 20, 20], [10, 10, 40, 40])

        result = compute_parity_alarm(series)

        assert result['parity_breach_count'] == 2
        assert result['parity_alarm'] is True
        # The pair is read off the breaching snapshot, not assembled from the
        # peak of one snapshot and the cap of another: 20 breached the cap of
        # 10 that was in force at the time.  Publishing today's cap of 40 here
        # would render the breach as 20-under-40, i.e. no breach at all.
        assert result['parity_peak'] == 20
        assert result['parity_cap'] == 10

    def test_peak_and_cap_come_from_the_same_snapshot_when_the_cap_changes(self):
        """A mid-window cap change must not misstate the breach severity.

        The real breach is 50 against the cap of 40 in force at the time
        (margin 10).  Pairing the series peak with the LATEST cap would render
        it as 50-against-24 — a margin of 26, 2.6x the truth.
        """
        result = compute_parity_alarm(_series([50, 10], [40, 24]))

        assert result['parity_alarm'] is True
        assert result['parity_breach_count'] == 1
        assert result['parity_peak'] == 50
        assert result['parity_cap'] == 40

    def test_healthy_window_never_publishes_a_breach_looking_pair(self):
        """No snapshot breached, so no pair may read as a breach.

        33-under-40 and 10-under-24 are both healthy.  Pairing the peak of 33
        with the latest cap of 24 would print '33 in flight, cap 24' next to
        ``parity_alarm: False`` — a 9-over breach that never happened.
        """
        result = compute_parity_alarm(_series([33, 10], [40, 24]))

        assert result['parity_alarm'] is False
        assert result['parity_breach_count'] == 0
        assert result['parity_peak'] == 33
        assert result['parity_cap'] == 40
        assert result['parity_peak'] <= result['parity_cap'], (
            'a non-alarming series must not publish a pair that reads as a breach'
        )

    def test_the_widest_margin_breach_wins_not_the_highest_count(self):
        """Severity is the margin over the cap in force, not the raw count."""
        result = compute_parity_alarm(_series([50, 30], [49, 10]))

        assert result['parity_breach_count'] == 2
        assert result['parity_peak'] == 30, '30 over a cap of 10 is the worse breach'
        assert result['parity_cap'] == 10

    def test_cap_cut_does_not_retro_alarm_a_healthy_window(self):
        """The mirror case: a cap CUT must not invent a historical breach."""
        series = _series([20, 20, 5, 5], [40, 40, 10, 10])

        result = compute_parity_alarm(series)

        assert result['parity_breach_count'] == 0
        assert result['parity_alarm'] is False

    def test_peak_ignores_snapshots_with_no_cap(self):
        """A peak is only meaningful next to the cap it is compared against."""
        series = _series([99, 5, 3], [None, 10, 10])

        result = compute_parity_alarm(series)

        assert result['parity_peak'] == 5, 'the capless 99 has nothing to breach'
        assert result['parity_cap'] == 10
        assert result['parity_alarm'] is False

    def test_partial_caps_still_alarm_on_the_known_ones(self):
        series = _series([3, 30, 4], [None, 24, 24])

        result = compute_parity_alarm(series)

        assert result['parity_alarm'] is True
        assert result['parity_breach_count'] == 1
        assert result['parity_peak'] == 30

    def test_empty_series_is_quiet(self):
        result = compute_parity_alarm(_series([], []))

        assert result['parity_alarm'] is False
        assert result['parity_cap'] is None
        assert result['parity_peak'] is None
        assert result['parity_breach_count'] == 0

    def test_a_short_cap_list_truncates_rather_than_padding(self):
        """A short cap list TRUNCATES to the shortest input — it does not pad.

        The policy is ``zip(in_progress_live, caps, strict=False)``: an index with no
        cap is not comparable, so it is dropped rather than judged against a
        forward-filled or last-seen cap.  A cap that was never recorded is
        UNKNOWN, and inventing one would let this surface alarm on a comparison
        that was never made.

        The fixture is built so the policy is OBSERVABLE.  ``30`` sits at index
        1 — past the end of the single-entry cap list — and is far ABOVE that
        cap, so any padding or forward-fill policy would compare ``(30, 24)``
        and report alarm True / breach_count 1 / peak 30.  (The previous
        fixture, ``[1, 2, 3]`` against ``[24]``, had every count below the cap,
        so truncation, padding and forward-fill were indistinguishable — it
        pinned "does not raise", not the policy its name claimed.)

        ``parity_peak == 1`` is the load-bearing assertion: it proves index 1
        was never considered at all, which ``breach_count == 0`` alone cannot.
        With only ``(1, 24)`` comparable and no breach, the published pair falls
        back to ``highest``, which is that single pair.
        """
        result = compute_parity_alarm({
            'labels': ['a', 'b', 'c'],
            'in_progress_live': [1, 30, 3],
            'concurrency_cap': [24],
        })

        assert result['parity_alarm'] is False
        assert result['parity_breach_count'] == 0
        assert result['parity_peak'] == 1, (
            'the uncapped index 1 (count 30) must not be compared at all; a '
            f"peak of {result['parity_peak']} means a cap was invented for it"
        )
        assert result['parity_cap'] == 24, 'the matched half of the same pair'

    def test_ragged_lists_do_not_raise(self):
        """Defensive: a ragged series must degrade, not explode a route.

        Kept separate from the truncation-policy test above: this one is about
        not raising at all (a longer cap list than count list, the other
        direction), and it must stay true whatever the comparison policy is.
        """
        result = compute_parity_alarm({
            'labels': ['a'],
            'in_progress_live': [1],
            'concurrency_cap': [24, 24, 24],
        })

        assert result['parity_alarm'] is False
        assert result['parity_breach_count'] == 0

    def test_non_numeric_entries_are_skipped_not_fatal(self):
        result = compute_parity_alarm({
            'labels': ['a', 'b'],
            'in_progress_live': [None, 30],
            'concurrency_cap': [24, 24],
        })

        assert result['parity_breach_count'] == 1
        assert result['parity_peak'] == 30

    def test_is_pure_and_does_not_mutate_the_series(self):
        series = _series([30, 5], [24, 24])
        before = {k: list(v) for k, v in series.items()}

        compute_parity_alarm(series)

        assert {k: list(v) for k, v in series.items()} == before


# ---------------------------------------------------------------------------
# Sketch #7: a ragged two-project store, read and shaped as the route does
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 9, 20, 12, 0, tzinfo=UTC)
_T2 = _NOW - timedelta(minutes=5)
_T1 = _T2 - timedelta(days=1)


def _day(index: int) -> datetime:
    """The *index*-th daily sample, index 7 being t2."""
    return _T2 - timedelta(days=7 - index)


def _project_a_row(index: int) -> dict:
    return {
        'project_id': 'A', 'ts': _day(index).isoformat(), 'state': 'value',
        'done': 10 + 2 * index, 'pending': 40 - 2 * index, 'blocked': 1,
        'in_progress': 3, 'in_progress_live': 3, 'in_progress_stranded': 0,
        'concurrency_cap': 24,
    }


def _project_b_row(index: int) -> dict:
    """B's last row (index 6, t1) breaches on the LIVE count: 30 against 24."""
    live = 30 if _day(index) == _T1 else 5
    return {
        'project_id': 'B', 'ts': _day(index).isoformat(), 'state': 'value',
        'done': 100 + index, 'pending': 20 - index, 'blocked': 0,
        'in_progress': live, 'in_progress_live': live, 'in_progress_stranded': 0,
        'concurrency_cap': 24,
    }


def _ragged_store(path: Path, *, with_b_gap: bool) -> None:
    """A measured daily t-7d..t2; B measured t-7d..t1, then (optionally) a gap at t2."""
    _make_db(path, [
        *(_project_a_row(index) for index in range(8)),
        *(_project_b_row(index) for index in range(7)),
    ])
    if with_b_gap:
        _add_gap_row(path, 'B', _T2.isoformat())


async def _shaped_as_the_route_does(path: Path) -> dict:
    async with aiosqlite.connect(str(path)) as db:
        projects = await aggregate_burndown_projects([db])
        per_pid = await asyncio.gather(
            *(aggregate_burndown_series([db], pid, days=30, now=_NOW) for pid in projects)
        )
    return redux_api.shape_burndown(dict(zip(projects, per_pid, strict=True)), served_at=_T2)


class TestRaggedTwoProjectStore:
    """B's newest instant is a gap: B is carried, never re-measured or re-judged."""

    @pytest.fixture
    async def shaped(self, tmp_path):
        with_gap, without_gap = tmp_path / 'with_gap.db', tmp_path / 'without_gap.db'
        _ragged_store(with_gap, with_b_gap=True)
        _ragged_store(without_gap, with_b_gap=False)
        return (
            await _shaped_as_the_route_does(with_gap),
            await _shaped_as_the_route_does(without_gap),
        )

    @pytest.mark.asyncio
    async def test_the_gap_row_changes_nothing(self, shaped):
        with_gap, without_gap = shaped
        assert with_gap == without_gap

    @pytest.mark.asyncio
    async def test_the_newest_point_carries_b_from_t1(self, shaped):
        agg = shaped[0]['BURNDOWN']
        at_t2 = agg['labels'].index(_T2.isoformat())
        assert agg['pending'][at_t2] == _project_a_row(7)['pending'] + _project_b_row(6)['pending']

    @pytest.mark.asyncio
    async def test_b_is_judged_once_at_t1_never_again_where_only_carried(self, shaped):
        body = shaped[0]
        assert body['BURNDOWN_BY_PROJECT']['B']['parity_breach_count'] == 1
        assert body['BURNDOWN']['parity_breach_count'] == 1

    @pytest.mark.asyncio
    async def test_b_is_served_stale_as_of_t1(self, shaped):
        latest = shaped[0]['BURNDOWN_BY_PROJECT']['B']['latest']
        assert latest['state'] == 'stale'
        assert datetime.fromisoformat(latest['as_of']) == _T1
        assert _T2 - datetime.fromisoformat(latest['as_of']) == _T2 - _T1

    @pytest.mark.asyncio
    async def test_a_is_served_fresh(self, shaped):
        assert shaped[0]['BURNDOWN_BY_PROJECT']['A']['latest']['state'] == 'fresh'

    @pytest.mark.asyncio
    async def test_the_aggregate_is_stale_as_of_b_and_names_it(self, shaped):
        latest = shaped[0]['BURNDOWN']['latest']
        assert latest['state'] == 'stale'
        assert datetime.fromisoformat(latest['as_of']) == _T1
        assert 'B' in latest['reason']

    @pytest.mark.asyncio
    async def test_the_aggregate_forecast_is_measured_so_equality_is_not_vacuous(self, shaped):
        """The window spans eight days, so the payloads compared above hold a forecast."""
        assert shaped[0]['BURNDOWN']['forecast']['value'] is not None

    @pytest.mark.asyncio
    async def test_a_gap_only_project_is_listed_and_served_unknown(self, tmp_path):
        """C's sampler never measured it: C renders unknown and the total a lower bound."""
        path = tmp_path / 'with_gap_only_project.db'
        _ragged_store(path, with_b_gap=True)
        _add_gap_row(path, 'C', _T1.isoformat())
        _add_gap_row(path, 'C', _T2.isoformat())

        body = await _shaped_as_the_route_does(path)

        assert body['BURNDOWN_BY_PROJECT']['C']['latest']['state'] == 'unknown'
        aggregate = body['BURNDOWN']['latest']
        assert aggregate['state'] == 'lower_bound'
        assert 'C' in aggregate['reason']
