"""Tests for scripts/legibility/session_ledger.py — the host-local record of
successfully coded sessions that the census uses as its mining-skip key
(plans/census-incremental-prd.md §4.2).

Every case runs against a REAL sqlite file under tmp_path; the state root is
redirected through trickle_state's one supported lever.
"""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta, timezone

import pytest
from legibility import session_ledger, trickle_state
from legibility.session_ledger import (
    CodedBy,
    LedgerRow,
    LedgerState,
    Outcome,
)

_NOW = datetime(2026, 10, 5, 12, 0, 0, tzinfo=UTC)
_GARBAGE = b'this is not an sqlite database, just bytes\n' * 8


@pytest.fixture(autouse=True)
def _isolate_state_root(tmp_path, monkeypatch):
    monkeypatch.setenv(trickle_state.STATE_ROOT_ENV, str(tmp_path / 'state'))


def _record(session, *, matches=(), candidates=()):
    return {
        'session': session,
        'date': '2026-10-04',
        'project': 'p',
        'agent_class': 'interactive',
        'matches': list(matches),
        'candidates': list(candidates),
    }


def _row(session, *, coded_by=CodedBy.TRICKLE, coded_at=_NOW, outcome=Outcome.EMPTY):
    return LedgerRow(
        session=session,
        instrument_version=5,
        coded_by=coded_by,
        run_ref=f'{coded_by}-run',
        outcome=outcome,
        coded_at=coded_at,
    )


def _garbage_ledger(tmp_path):
    path = tmp_path / 'ledger' / 'coded-sessions.sqlite'
    path.parent.mkdir(parents=True)
    path.write_bytes(_GARBAGE)
    return path


# ---------------------------------------------------------------------------
# (a) location
# ---------------------------------------------------------------------------

def test_ledger_path_lives_in_the_project_state_dir():
    assert session_ledger.ledger_path('p') == (
        trickle_state.project_state_dir('p') / 'coded-sessions.sqlite'
    )


# ---------------------------------------------------------------------------
# (b) outcome_of
# ---------------------------------------------------------------------------

def test_outcome_of_prefers_candidate_over_matched():
    record = _record('s', matches=[{'entry_id': 'e1'}], candidates=[{'title': 't'}])
    assert session_ledger.outcome_of(record) is Outcome.CANDIDATE


def test_outcome_of_matches_only_is_matched():
    assert session_ledger.outcome_of(_record('s', matches=[{'entry_id': 'e1'}])) is Outcome.MATCHED


def test_outcome_of_neither_is_empty():
    assert session_ledger.outcome_of(_record('s')) is Outcome.EMPTY


# ---------------------------------------------------------------------------
# (c) rows_for + LedgerRow validation
# ---------------------------------------------------------------------------

def test_rows_for_builds_one_row_per_record_and_skips_unusable_sessions():
    records = [
        _record('s1', matches=[{'entry_id': 'e1'}]),
        _record('s2', candidates=[{'title': 't'}]),
        _record(None),
        _record(''),
        _record('unknown'),
        {k: v for k, v in _record('s3').items() if k != 'session'},
        _record('s4'),
    ]
    rows = session_ledger.rows_for(
        records,
        coded_by=CodedBy.TRICKLE,
        run_ref='trickle-p-20261005',
        instrument_version=5,
        coded_at=_NOW,
    )
    assert [(r.session, r.outcome) for r in rows] == [
        ('s1', Outcome.MATCHED),
        ('s2', Outcome.CANDIDATE),
        ('s4', Outcome.EMPTY),
    ]
    assert {(r.coded_by, r.run_ref, r.instrument_version, r.coded_at) for r in rows} == {
        (CodedBy.TRICKLE, 'trickle-p-20261005', 5, _NOW),
    }


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('session', ''),
        ('coded_at', datetime(2026, 10, 5, 12, 0, 0)),
        ('coded_by', 'trickle'),
        ('outcome', 'matched'),
    ],
)
def test_ledger_row_rejects_invalid_fields_naming_field_and_value(field, value):
    kwargs = {
        'session': 's',
        'instrument_version': 5,
        'coded_by': CodedBy.CENSUS,
        'run_ref': 'census-p-20261005',
        'outcome': Outcome.EMPTY,
        'coded_at': _NOW,
    }
    kwargs[field] = value
    with pytest.raises(ValueError) as excinfo:
        LedgerRow(**kwargs)
    assert field in str(excinfo.value)
    assert repr(value) in str(excinfo.value) or str(value) in str(excinfo.value)


def test_ledger_row_is_frozen():
    row = _row('s')
    with pytest.raises(AttributeError):
        row.session = 'other'  # type: ignore[misc]


# ---------------------------------------------------------------------------
# (d) read_ledger on an absent file never creates it
# ---------------------------------------------------------------------------

def test_read_ledger_on_absent_path_is_absent_and_creates_nothing(tmp_path):
    path = tmp_path / 'nowhere' / 'coded-sessions.sqlite'
    snapshot = session_ledger.read_ledger(path)
    assert snapshot.state is LedgerState.ABSENT
    assert snapshot.total_rows is None
    assert snapshot.sessions == frozenset()
    assert snapshot.has_census_rows is False
    assert not path.exists()
    assert not path.parent.exists()


# ---------------------------------------------------------------------------
# (e) record_codings: create-if-absent, nothing ledgered twice
# ---------------------------------------------------------------------------

def test_record_codings_creates_ledger_and_never_ledgers_a_session_twice(tmp_path):
    path = tmp_path / 'deep' / 'dir' / 'coded-sessions.sqlite'

    assert session_ledger.record_codings(path, [_row('s1'), _row('s2')]) == 2
    assert path.is_file()

    again = _row('s1', coded_by=CodedBy.CENSUS)
    assert session_ledger.record_codings(path, [again]) == 0

    with sqlite3.connect(path) as conn:
        stored = conn.execute(
            "SELECT coded_by, run_ref FROM coded_sessions WHERE session = 's1'",
        ).fetchall()
    assert stored == [('trickle', 'trickle-run')]


def test_record_codings_of_no_rows_returns_zero(tmp_path):
    path = tmp_path / 'coded-sessions.sqlite'
    assert session_ledger.record_codings(path, []) == 0


# ---------------------------------------------------------------------------
# (f) read_ledger after writes
# ---------------------------------------------------------------------------

def test_read_ledger_counts_rows_by_producer(tmp_path):
    path = tmp_path / 'coded-sessions.sqlite'
    session_ledger.record_codings(path, [_row('t1'), _row('t2')])

    trickle_only = session_ledger.read_ledger(path)
    assert trickle_only.state is LedgerState.OK
    assert trickle_only.has_census_rows is False
    assert trickle_only.rows_by_coded_by == {CodedBy.TRICKLE: 2, CodedBy.CENSUS: 0}

    session_ledger.record_codings(path, [_row('c1', coded_by=CodedBy.CENSUS)])
    snapshot = session_ledger.read_ledger(path)

    assert snapshot.state is LedgerState.OK
    assert isinstance(snapshot.sessions, frozenset)
    assert snapshot.sessions == frozenset({'t1', 't2', 'c1'})
    assert snapshot.rows_by_coded_by == {CodedBy.TRICKLE: 2, CodedBy.CENSUS: 1}
    assert snapshot.has_census_rows is True
    assert snapshot.total_rows == 3
    assert snapshot.pruned == 0
    assert snapshot.error is None


def test_snapshot_rows_by_coded_by_is_read_only(tmp_path):
    path = tmp_path / 'coded-sessions.sqlite'
    session_ledger.record_codings(path, [_row('t1')])
    snapshot = session_ledger.read_ledger(path)
    with pytest.raises(TypeError):
        snapshot.rows_by_coded_by[CodedBy.CENSUS] = 9  # type: ignore[index]


# ---------------------------------------------------------------------------
# (g) open_for_census: create when absent, prune before the cutoff
# ---------------------------------------------------------------------------

def test_open_for_census_creates_an_absent_ledger(tmp_path):
    path = tmp_path / 'fresh' / 'coded-sessions.sqlite'
    snapshot = session_ledger.open_for_census(path, prune_before=_NOW)
    assert snapshot.state is LedgerState.CREATED
    assert path.is_file()
    assert snapshot.total_rows == 0
    assert snapshot.pruned == 0
    assert snapshot.has_census_rows is False

    reopened = session_ledger.open_for_census(path, prune_before=_NOW)
    assert reopened.state is LedgerState.OK
    assert reopened.total_rows == 0


def test_open_for_census_prunes_exactly_the_rows_before_the_cutoff(tmp_path):
    path = tmp_path / 'coded-sessions.sqlite'
    cutoff = datetime(2026, 9, 6, tzinfo=UTC)
    session_ledger.record_codings(path, [
        _row('old-1', coded_at=cutoff - timedelta(days=3)),
        _row('old-2', coded_at=cutoff - timedelta(seconds=1)),
        _row('at-cutoff', coded_at=cutoff),
        _row('new', coded_at=cutoff + timedelta(days=2), coded_by=CodedBy.CENSUS),
    ])

    snapshot = session_ledger.open_for_census(path, prune_before=cutoff)

    assert snapshot.state is LedgerState.OK
    assert snapshot.pruned == 2
    assert snapshot.sessions == frozenset({'at-cutoff', 'new'})
    assert snapshot.total_rows == 2
    assert session_ledger.read_ledger(path).sessions == frozenset({'at-cutoff', 'new'})


# ---------------------------------------------------------------------------
# (h) a corrupt file is UNREADABLE, never rewritten
# ---------------------------------------------------------------------------

def test_read_ledger_reports_a_corrupt_file_unreadable(tmp_path):
    path = _garbage_ledger(tmp_path)
    snapshot = session_ledger.read_ledger(path)
    assert snapshot.state is LedgerState.UNREADABLE
    assert str(path) in snapshot.error
    assert snapshot.total_rows is None
    assert snapshot.sessions == frozenset()
    assert path.read_bytes() == _GARBAGE


def test_open_for_census_leaves_a_corrupt_file_untouched(tmp_path):
    path = _garbage_ledger(tmp_path)
    snapshot = session_ledger.open_for_census(path, prune_before=_NOW)
    assert snapshot.state is LedgerState.UNREADABLE
    assert str(path) in snapshot.error
    assert snapshot.total_rows is None
    assert snapshot.has_census_rows is False
    assert path.read_bytes() == _GARBAGE


def test_record_codings_on_a_corrupt_file_raises_naming_the_path(tmp_path):
    path = _garbage_ledger(tmp_path)
    with pytest.raises(session_ledger.LedgerError) as excinfo:
        session_ledger.record_codings(path, [_row('s1')])
    assert str(path) in str(excinfo.value)
    assert path.read_bytes() == _GARBAGE


# ---------------------------------------------------------------------------
# (i) coded_at is stored as ISO-8601 UTC seconds, so text order is time order
# ---------------------------------------------------------------------------

def test_coded_at_is_stored_as_utc_iso_seconds(tmp_path):
    path = tmp_path / 'coded-sessions.sqlite'
    plus_two = timezone(timedelta(hours=2))
    session_ledger.record_codings(path, [
        _row('s1', coded_at=datetime(2026, 10, 1, 1, 0, 0, 123456, tzinfo=plus_two)),
    ])
    with sqlite3.connect(path) as conn:
        (stored,) = conn.execute('SELECT coded_at FROM coded_sessions').fetchone()
    assert stored == '2026-09-30T23:00:00+00:00'


def test_prune_compares_instants_not_local_clock_text(tmp_path):
    path = tmp_path / 'coded-sessions.sqlite'
    plus_two = timezone(timedelta(hours=2))
    session_ledger.record_codings(path, [
        _row('before', coded_at=datetime(2026, 10, 1, 1, 0, tzinfo=plus_two)),
        _row('after', coded_at=datetime(2026, 9, 30, 20, 0, tzinfo=timezone(timedelta(hours=-5)))),
    ])
    snapshot = session_ledger.open_for_census(
        path, prune_before=datetime(2026, 10, 1, tzinfo=UTC),
    )
    assert snapshot.pruned == 1
    assert snapshot.sessions == frozenset({'after'})
