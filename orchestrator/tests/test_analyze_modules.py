"""Tests for the analyze_modules per-module conflict helper."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import _hold_history_fixtures as F
import pytest

from orchestrator.analyze_modules import (
    ModuleStats,
    _iter_events,
    _parse_since,
    aggregate,
    main,
    render_json,
    render_table,
    suggest_max_per_module,
)
from orchestrator.event_store import EventStore, EventType


@pytest.fixture
def event_store(tmp_path: Path) -> EventStore:
    return EventStore(tmp_path / 'runs.db', run_id='test')


#: ``since`` that admits every fixed 2026-08-01 fixture timestamp.
_BEFORE_TRACE = F.BASE_TS - timedelta(seconds=1)


def _trace_db(tmp_path: Path, rows: list[dict] | None = None, name: str = 'trace.db') -> Path:
    """A real ``runs.db`` carrying *rows* (default: the canonical 21-row trace).

    ``EventStore.__init__`` only creates the schema, so the fixture's explicit
    ids cannot collide with anything.  ``EventStore.emit`` cannot stand in:
    it stamps its own wall-clock timestamp and its own ``run_id``, and this
    trace needs fixed offsets and TWO run ids.
    """
    store = EventStore(tmp_path / name, run_id=F.RUN_A)
    F.write_trace(store, rows=rows)
    return store.db_path


@pytest.fixture
def canonical_trace_db(tmp_path: Path) -> Path:
    return _trace_db(tmp_path, name='canonical.db')


def test_parse_since_duration_shorthand():
    # '7d' shouldn't raise and must yield a past timestamp.
    from datetime import UTC, datetime
    cut = _parse_since('7d')
    assert cut < datetime.now(UTC)


def test_parse_since_invalid_raises():
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_since('nonsense')


# ===========================================================================
# The grouping key — the depth-coarsened lock module, verbatim
# ===========================================================================
#
# `max_per_module` is enforced in `ModuleLockTable._limit_for`
# (scheduler.py:1505-1517) under `normalize_lock(module, lock_depth)` — the
# FULL depth-`lock_depth` path — and `module_overrides` is read under that same
# key.  No first-path-component key exists anywhere in the lock layer, so a
# `suggest` column published under one names something the enforcement layer
# never consults.  These tests pin the key space the CLI's whole reason to
# exist depends on.


def test_aggregate_counts_dispatches_and_skips(tmp_path: Path, event_store: EventStore):
    event_store.emit(
        EventType.lock_acquired,
        task_id='1',
        data={'modules': ['crates/foo/src']},
    )
    event_store.emit(
        EventType.lock_released,
        task_id='1',
        data={'modules': ['crates/foo/src']},
    )
    event_store.emit(
        EventType.task_skipped,
        task_id='2',
        data={'modules': ['crates/foo/src']},
    )
    event_store.emit(
        EventType.task_skipped,
        task_id='2',
        data={'modules': ['crates/foo/src']},
    )
    from datetime import UTC, datetime, timedelta
    stats = aggregate(event_store.db_path, datetime.now(UTC) - timedelta(days=1))
    assert 'crates/foo/src' in stats
    assert 'crates' not in stats, 'the first path component is not a lock key'
    assert stats['crates/foo/src'].dispatches == 1
    assert stats['crates/foo/src'].skipped_waiting == 2
    assert stats['crates/foo/src'].conflict_rate() == pytest.approx(2.0)


def test_aggregate_keeps_sibling_modules_under_one_parent_distinct(event_store: EventStore):
    """Two modules sharing a parent are two lock keys, not one.

    `ModuleLockTable` limits `orchestrator/src` and `orchestrator/tests`
    independently, so rolling them into an `orchestrator` row would report a
    contention figure for a key no override can address.
    """
    event_store.emit(
        EventType.lock_acquired,
        task_id='1',
        data={'modules': ['orchestrator/src', 'orchestrator/tests']},
    )

    from datetime import UTC, datetime, timedelta
    stats = aggregate(event_store.db_path, datetime.now(UTC) - timedelta(days=1))

    assert sorted(stats) == ['orchestrator/src', 'orchestrator/tests']
    assert stats['orchestrator/src'].dispatches == 1
    assert stats['orchestrator/tests'].dispatches == 1


def test_aggregate_counts_a_repeated_module_once_per_event(event_store: EventStore):
    """One event is one dispatch for a module however many times it names it."""
    event_store.emit(
        EventType.lock_acquired,
        task_id='1',
        data={'modules': ['shared/src', 'shared/src']},
    )

    from datetime import UTC, datetime, timedelta
    stats = aggregate(event_store.db_path, datetime.now(UTC) - timedelta(days=1))

    assert stats['shared/src'].dispatches == 1


# ===========================================================================
# Hold accounting comes from hold_history.iter_hold_spans (PRD INV-5)
# ===========================================================================
#
# Every expected duration below is the hand-computed literal in
# ``_hold_history_fixtures`` -- nothing is read back out of the code under
# test.  The canonical trace covers all four residue classes named in
# PARKING_MODEL_REPORT.md:100-104, and the CLI's own pairing loop got three of
# them wrong: it closed no span at an era boundary, LOST the first hold of a
# double-acquire to a dict overwrite, and keyed its open slot per (module-key,
# task) so a partial release closed spans it never named.


def test_aggregate_hold_stats_match_the_hand_computed_trace(canonical_trace_db: Path):
    """The whole oracle at once: samples, totals and means for all three modules."""
    stats = aggregate(canonical_trace_db, _BEFORE_TRACE)

    assert stats['orchestrator/src'].hold_samples == 5
    assert stats['orchestrator/src'].total_hold_secs == pytest.approx(520.0)
    assert stats['orchestrator/src'].avg_hold_secs() == pytest.approx(104.0)

    assert stats['shared/src'].hold_samples == 3
    assert stats['shared/src'].total_hold_secs == pytest.approx(360.0)
    assert stats['shared/src'].avg_hold_secs() == pytest.approx(120.0)

    assert stats['fused-memory/src'].hold_samples == 3
    assert stats['fused-memory/src'].total_hold_secs == pytest.approx(600.0)
    assert stats['fused-memory/src'].avg_hold_secs() == pytest.approx(200.0)


def test_aggregate_counts_dispatches_across_the_whole_trace(canonical_trace_db: Path):
    """Dispatches are per lock_acquired event, independent of how it closes."""
    stats = aggregate(canonical_trace_db, _BEFORE_TRACE)

    assert stats['orchestrator/src'].dispatches == 6
    assert stats['shared/src'].dispatches == 3
    assert stats['fused-memory/src'].dispatches == 3


def test_aggregate_reports_truncated_holds_per_module(canonical_trace_db: Path):
    """Era- and double-acquire-closed spans are right-censored LOWER BOUNDS.

    They are counted -- PARKING_MODEL_REPORT.md:102, "the lock did block
    others until then" -- but blending them into `avg_hold_s` with no way to
    tell would trade one silent distortion for another.
    """
    stats = aggregate(canonical_trace_db, _BEFORE_TRACE)

    # One per module, matching F.EXPECTED_TRUNCATED_SPANS.
    assert {m: stats[m].truncated_holds for m in F.EXPECTED_SAMPLES} == {
        'orchestrator/src': 1,
        'shared/src': 1,
        'fused-memory/src': 1,
    }
    assert len(F.EXPECTED_TRUNCATED_SPANS) == 3, 'the fixture oracle still says three'
    assert stats['orchestrator/src'].truncated_fraction() == pytest.approx(1 / 5)


# --- the two era-boundary sources, one named test each ---------------------


def test_a_service_restart_row_closes_the_hold_it_interrupted(tmp_path: Path):
    """Era boundary #1.  T5 acquires at 800 and the process is restarted at
    1000; today's loop drops the hold entirely because no release ever
    arrives."""
    db = _trace_db(tmp_path, rows=[
        F.acquire(10, 800, 'T5', ['fused-memory/src']),
        F.service_restart(11, 1000),
    ])

    stats = aggregate(db, _BEFORE_TRACE)

    assert stats['fused-memory/src'].hold_samples == 1
    assert stats['fused-memory/src'].total_hold_secs == pytest.approx(200.0)
    assert stats['fused-memory/src'].truncated_holds == 1
    # The restart row's task_id is the TRIGGER task, not a lock holder.
    assert F.SERVICE_RESTART_TRIGGER_TASK not in {'T5'}
    assert list(stats) == ['fused-memory/src'], 'the restart names no module of its own'


def test_a_run_transition_closes_at_the_previous_rows_timestamp(tmp_path: Path):
    """Era boundary #2, and the timestamp it closes at is the point.

    T7 holds shared/src from 1300; run-a's last row is at 1400 and run-b opens
    at 1500.  Closing at 1500 would charge the whole inter-run gap -- however
    many days of it in production -- to a hold nobody observed.
    """
    db = _trace_db(tmp_path, rows=[
        F.acquire(14, 1300, 'T7', ['shared/src']),
        F.acquire(15, 1360, 'T8', ['orchestrator/src']),
        F.release(16, 1400, 'T8', ['orchestrator/src']),
        F.acquire(17, 1500, 'T9', ['shared/src'], run_id=F.RUN_B),
    ])

    stats = aggregate(db, _BEFORE_TRACE)

    assert stats['shared/src'].hold_samples == 1
    assert stats['shared/src'].total_hold_secs == pytest.approx(100.0), \
        'closed at 1400 (last evidence of the hold), not at 1500'
    assert stats['shared/src'].truncated_holds == 1


def test_a_double_acquire_keeps_the_hold_it_force_closes(tmp_path: Path):
    """T4 acquires at 400 and again at 580 without releasing.

    The old `open_acquires[task_id] = start` OVERWROTE the first start, so the
    180s hold vanished and only the 120s tail was ever counted.
    """
    db = _trace_db(tmp_path, rows=[
        F.acquire(7, 400, 'T4', ['orchestrator/src']),
        F.acquire(8, 580, 'T4', ['orchestrator/src']),
        F.release(9, 700, 'T4', ['orchestrator/src']),
    ])

    stats = aggregate(db, _BEFORE_TRACE)

    assert stats['orchestrator/src'].hold_samples == 2
    assert stats['orchestrator/src'].total_hold_secs == pytest.approx(300.0), '180 + 120'
    assert stats['orchestrator/src'].truncated_holds == 1


# --- the two refusals that must SURVIVE the convergence --------------------


def test_an_orphan_release_contributes_no_hold(tmp_path: Path):
    """A release with no open span (3,594 DF / 5,780 reify in the measured
    trace) is skipped, not paired with whatever came before it."""
    db = _trace_db(tmp_path, rows=[
        F.acquire(1, 0, 'T1', ['orchestrator/src']),
        F.release(2, 60, 'T1', ['orchestrator/src']),
        F.release(3, 350, 'T3', ['orchestrator/src']),   # ORPHAN
    ])

    stats = aggregate(db, _BEFORE_TRACE)

    assert stats['orchestrator/src'].hold_samples == 1
    assert stats['orchestrator/src'].total_hold_secs == pytest.approx(60.0)
    assert stats['orchestrator/src'].truncated_holds == 0


def test_a_hold_still_open_at_the_end_of_the_window_is_dropped(tmp_path: Path):
    """Unlike an era boundary, nothing here observed an end.  Closing at "now"
    would invent a duration, so the span is dropped -- while the acquire still
    counts as a dispatch, because it really did happen."""
    db = _trace_db(tmp_path, rows=[
        F.acquire(1, 0, 'T1', ['orchestrator/src']),
        F.release(2, 60, 'T1', ['orchestrator/src']),
        F.acquire(3, 100, 'T10', ['orchestrator/src']),   # open at end of stream
    ])

    stats = aggregate(db, _BEFORE_TRACE)

    assert stats['orchestrator/src'].dispatches == 2
    assert stats['orchestrator/src'].hold_samples == 1
    assert stats['orchestrator/src'].total_hold_secs == pytest.approx(60.0)


def test_the_canonical_trace_drops_its_end_of_stream_hold(canonical_trace_db: Path):
    """T10's acquire at id 19 is the 6th orchestrator/src dispatch and the 5th
    -- not 6th -- sample."""
    stats = aggregate(canonical_trace_db, _BEFORE_TRACE)

    assert stats['orchestrator/src'].dispatches == 6
    assert stats['orchestrator/src'].hold_samples == 5


def test_aggregate_ignores_events_before_since(tmp_path: Path, event_store: EventStore):
    from datetime import UTC, datetime, timedelta
    event_store.emit(
        EventType.lock_acquired,
        task_id='1',
        data={'modules': ['crates/foo/src']},
    )
    # Pretend these happened in the past: cut is "now + 1 hour" so nothing qualifies.
    stats = aggregate(event_store.db_path, datetime.now(UTC) + timedelta(hours=1))
    assert stats == {}


def test_suggest_max_per_module_tiers():
    assert suggest_max_per_module(ModuleStats(dispatches=1, skipped_waiting=2)) == 1
    assert suggest_max_per_module(ModuleStats(dispatches=2, skipped_waiting=1)) == 2
    assert suggest_max_per_module(ModuleStats(dispatches=10, skipped_waiting=1)) == 3
    assert suggest_max_per_module(ModuleStats(dispatches=100, skipped_waiting=0)) == 4


def test_render_table_orders_by_conflict_desc():
    stats = {
        'low': ModuleStats(dispatches=10, skipped_waiting=0),
        'hot': ModuleStats(dispatches=5, skipped_waiting=20),
    }
    table = render_table(stats)
    lines = table.splitlines()
    # Header + two rows, hot first.
    assert lines[1].startswith('hot')
    assert lines[2].startswith('low')


def test_render_json_is_machine_readable():
    # Asymmetric values so a dispatches<->skipped swap would be caught:
    # swapped input would be (7, 4) → ratio 4/7 ≈ 0.57 → still suggest 2,
    # so the conflict_rate assertion does the distinguishing.
    stats = {
        'crates/foo/src': ModuleStats(
            dispatches=4, skipped_waiting=7,
            total_hold_secs=300.0, hold_samples=3, truncated_holds=2,
        ),
    }
    payload = json.loads(render_json(stats))
    assert payload['crates/foo/src']['dispatches'] == 4
    assert payload['crates/foo/src']['skipped_waiting'] == 7
    # conflict = 7/4 = 1.75 (>= 0.5 but < 2.0) → suggest 2.
    assert payload['crates/foo/src']['conflict_rate'] == 1.75
    assert payload['crates/foo/src']['suggested_max_per_module'] == 2
    assert payload['crates/foo/src']['avg_hold_secs'] == 100.0
    # Two of the three samples are censored lower bounds — a consumer reading
    # avg_hold_secs alone cannot tell, so the count travels with it.
    assert payload['crates/foo/src']['truncated_holds'] == 2


# --- the table must not collapse two distinct lock keys into one row -------

#: Two REAL sibling paths in this repo that are distinct lock keys and share a
#: 32-character prefix ('fused-memory/src/fused_memory/re').  The old 32-column
#: truncation rendered both as the same label — a defect the first-path-
#: component key space hid, because it never produced a key this long.
_LONG_KEY = 'fused-memory/src/fused_memory/reconciliation'
_LONGER_KEY = 'fused-memory/src/fused_memory/reconciliation/prompts'


def test_render_table_prints_a_long_lock_module_in_full():
    assert _LONG_KEY[:32] == _LONGER_KEY[:32], 'the fixture must actually collide at 32'

    table = render_table({
        _LONG_KEY: ModuleStats(dispatches=10, skipped_waiting=2),
        _LONGER_KEY: ModuleStats(dispatches=4, skipped_waiting=8),
    })

    assert _LONG_KEY in table
    assert _LONGER_KEY in table
    labels = [line.split()[0] for line in table.splitlines()[1:]]
    assert sorted(labels) == sorted([_LONG_KEY, _LONGER_KEY]), \
        'each lock key needs its own distinguishable row'


def test_render_table_reports_truncated_holds():
    """The censoring is visible in the human surface too, not just --json."""
    table = render_table({
        'orchestrator/src': ModuleStats(
            dispatches=5, skipped_waiting=1,
            total_hold_secs=400.0, hold_samples=4, truncated_holds=2,
        ),
    })

    header, row = table.splitlines()
    assert 'trunc' in header
    # Distinct values, so a trunc/suggest column swap cannot pass:
    # conflict 1/5 = 0.2 → suggest 3; truncated_holds is 2.
    assert row.split()[-3:] == ['100.0', '2', '3']


def test_iter_events_opens_runs_db_read_only(event_store: EventStore) -> None:
    """_iter_events must open runs.db via a read-only file:/// URI (uri=True, ?mode=ro).

    Modelled on dashboard/tests/test_db.py::test_get_builds_sqlite_uri_via_path_as_uri
    and fused-memory/tests/test_scheduler_state_tools.py.

    The spy delegates to the real sqlite3.connect so the SELECT still executes
    and the generator's finally-block closes the connection normally.
    """
    event_store.emit(
        EventType.lock_acquired,
        task_id='1',
        data={'modules': ['crates/foo/src']},
    )

    real_connect = sqlite3.connect
    captured_uri: str | None = None
    captured_uri_kwarg: bool | None = None

    def spy(*args, **kwargs):
        nonlocal captured_uri, captured_uri_kwarg
        captured_uri = args[0] if args else None
        captured_uri_kwarg = kwargs.get('uri')
        return real_connect(*args, **kwargs)

    since = datetime.now(UTC) - timedelta(days=1)
    with patch('orchestrator.analyze_modules.sqlite3.connect', spy):
        rows = list(_iter_events(event_store.db_path, since))

    # URI contract — single exact-match subsumes all substring checks.
    # Enforces: file:/// prefix, .resolve() present, ?mode=ro suffix, uri=True kwarg.
    expected_uri = event_store.db_path.resolve().as_uri() + '?mode=ro'
    assert captured_uri == expected_uri, (
        f'Expected read-only URI {expected_uri!r}, got {captured_uri!r}'
    )
    assert captured_uri_kwarg is True, (
        f'Expected uri=True kwarg, got uri={captured_uri_kwarg!r}'
    )

    # Reads still work — seeded event is returned.
    assert len(rows) == 1
    assert rows[0]['event_type'] == 'lock_acquired'


# ===========================================================================
# _iter_events row shape — the contract that lets iter_hold_spans consume it
# ===========================================================================
#
# The CLI keeps its own read (``fetch_events_by_type_all_runs`` has no ``since``
# predicate and returns one list per type), so the rows it yields must be
# shaped like the ones that fetch returns or ``iter_hold_spans`` cannot read
# them.  Every key below is one the span helper actually reads: drop ``run_id``
# and the run-transition era boundary goes silent; drop ``service_restart`` at
# the SQL level and the restart boundary never arrives at all.


def test_iter_events_yields_event_store_shaped_dicts(tmp_path: Path, event_store: EventStore):
    """Rows carry exactly the columns iter_hold_spans reads, in ``id`` order."""
    F.write_trace(event_store, rows=[
        F.acquire(1, 0, 'T1', ['orchestrator/src'], run_id='run-a'),
        F.release(2, 60, 'T1', ['orchestrator/src'], run_id='run-a'),
        F.row(3, 90, 'task_skipped', task_id='T2',
              run_id='run-a', data={'modules': ['orchestrator/src']}),
        F.service_restart(4, 120, run_id='run-a'),
    ])

    rows = list(_iter_events(event_store.db_path, F.BASE_TS - timedelta(seconds=1)))

    assert [r['id'] for r in rows] == [1, 2, 3, 4], 'ascending id order is required'
    assert [r['event_type'] for r in rows] == [
        'lock_acquired', 'lock_released', 'task_skipped', 'service_restart',
    ], 'service_restart must survive the SQL filter — it is an era boundary'
    for r in rows:
        assert set(r) == {'id', 'timestamp', 'run_id', 'task_id', 'event_type', 'data'}
        assert r['run_id'] == 'run-a', 'run_id carries the store value, not None'
        assert isinstance(r['data'], dict), 'data is already JSON-decoded'
    assert rows[0]['task_id'] == 'T1'
    assert rows[0]['data']['modules'] == ['orchestrator/src']
    assert rows[0]['timestamp'] == F.ts(0)


def test_iter_events_tolerates_malformed_data(event_store: EventStore):
    """A payload that is not JSON costs its own ``data``, not the row."""
    conn = sqlite3.connect(str(event_store.db_path))
    try:
        conn.execute(
            'INSERT INTO events (id, timestamp, run_id, task_id, event_type, data) '
            'VALUES (?, ?, ?, ?, ?, ?)',
            (1, F.ts(0), 'run-a', 'T1', 'lock_acquired', 'not-json{'),
        )
        conn.commit()
    finally:
        conn.close()

    rows = list(_iter_events(event_store.db_path, F.BASE_TS - timedelta(seconds=1)))

    assert len(rows) == 1
    assert rows[0]['data'] == {}
    assert rows[0]['event_type'] == 'lock_acquired'


# ===========================================================================
# main() — the CLI interface, which this task does NOT change
# ===========================================================================
#
# The key space and every per-key number move; --since, --json and
# --min-dispatches must not.  End to end over a real runs.db so the whole
# path (read -> aggregate -> render -> exit code) is covered at once.


def test_main_json_reports_the_full_lock_modules(canonical_trace_db: Path, capsys):
    assert main([str(canonical_trace_db), '--json', '--since', '2026-07-01']) == 0

    payload = json.loads(capsys.readouterr().out)

    assert sorted(payload) == ['fused-memory/src', 'orchestrator/src', 'shared/src']
    assert not {'fused-memory', 'orchestrator', 'shared'} & set(payload), \
        'no bare first-path-component keys survive'
    assert payload['orchestrator/src']['dispatches'] == 6
    assert payload['orchestrator/src']['avg_hold_secs'] == 104.0
    assert payload['orchestrator/src']['truncated_holds'] == 1


def test_main_min_dispatches_still_filters(canonical_trace_db: Path, capsys):
    assert main([
        str(canonical_trace_db), '--json', '--since', '2026-07-01',
        '--min-dispatches', '4',
    ]) == 0

    payload = json.loads(capsys.readouterr().out)

    # orchestrator/src has 6; the other two have 3 each.
    assert list(payload) == ['orchestrator/src']


def test_main_reports_an_empty_window_without_failing(canonical_trace_db: Path, capsys):
    """An empty window is a fact about the window, not an error."""
    assert main([str(canonical_trace_db), '--since', '2026-09-01']) == 0

    captured = capsys.readouterr()
    assert '(no module events in window)' in captured.err
    assert captured.out == ''


def test_main_table_mode_prints_the_full_lock_modules(canonical_trace_db: Path, capsys):
    assert main([str(canonical_trace_db), '--since', '2026-07-01']) == 0

    out = capsys.readouterr().out
    assert 'orchestrator/src' in out
    assert 'fused-memory/src' in out


def test_main_returns_2_for_a_missing_db(tmp_path: Path, capsys):
    assert main([str(tmp_path / 'nope.db')]) == 2
    assert 'runs.db not found' in capsys.readouterr().err
