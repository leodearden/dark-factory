"""Tests for scripts/legibility/census_window.py — the session window one
census run mines (plans/census-incremental-prd.md §4.2)."""
from __future__ import annotations

import json
import logging
import random
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

import pytest
from legibility import (
    census_window,
    coder,
    config,
    inventory,
    sampling,
    session_ledger,
    trickle_state,
)
from legibility.census_window import MiningWindow
from legibility.session_ledger import CodedBy, LedgerRow, LedgerState, Outcome

_NOW = datetime(2026, 10, 6, 15, 0, tzinfo=UTC)
_START = datetime(2026, 9, 6, tzinfo=UTC)
_END = datetime(2026, 10, 6, tzinfo=UTC)


# ---------------------------------------------------------------------------
# window maths
# ---------------------------------------------------------------------------

def test_retention_window_is_the_utc_days_before_today():
    window = census_window.retention_window(_NOW, 30)
    assert window == MiningWindow(start=_START, end=_END)
    assert window.first_day == date(2026, 9, 6)
    assert window.last_day == date(2026, 10, 5)
    assert window.is_empty is False


def test_window_record_is_its_half_open_iso_bounds():
    window = census_window.retention_window(_NOW, 30)
    assert window.to_record() == ['2026-09-06T00:00:00+00:00', '2026-10-06T00:00:00+00:00']


def test_transition_raises_the_start_to_the_last_census_day():
    window = census_window.retention_window(_NOW, 30)
    moved = census_window.transition_window(window, last_census_at=date(2026, 10, 1))
    assert moved == MiningWindow(start=datetime(2026, 10, 1, tzinfo=UTC), end=_END)


@pytest.mark.parametrize('last_census_at', [None, date(2026, 8, 1), date(2026, 9, 6)])
def test_transition_leaves_the_window_when_the_last_census_is_older(last_census_at):
    window = census_window.retention_window(_NOW, 30)
    assert census_window.transition_window(window, last_census_at=last_census_at) == window


def test_a_census_already_run_today_leaves_an_empty_window():
    window = census_window.retention_window(_NOW, 30)
    moved = census_window.transition_window(window, last_census_at=date(2026, 10, 6))
    assert moved.start == moved.end == _END
    assert moved.is_empty is True
    assert moved.last_day < moved.first_day


def test_mining_window_rejects_naive_bounds():
    naive = datetime(2026, 9, 6)
    with pytest.raises(ValueError) as excinfo:
        MiningWindow(start=naive, end=_END)
    assert str(naive) in str(excinfo.value) or repr(naive) in str(excinfo.value)


def test_mining_window_rejects_start_after_end():
    with pytest.raises(ValueError) as excinfo:
        MiningWindow(start=_END, end=_START)
    message = str(excinfo.value)
    assert _END.isoformat() in message
    assert _START.isoformat() in message


# ---------------------------------------------------------------------------
# WindowBatchSource over real transcripts and a real sqlite ledger
# ---------------------------------------------------------------------------

_SOURCE_NOW = datetime(2026, 10, 6, 12, 0, tzinfo=UTC)
_ARCHIVE_ROOT = 'data/orchestrator/agent-transcripts'


@pytest.fixture(autouse=True)
def _isolate_state_root(tmp_path, monkeypatch):
    monkeypatch.setenv(trickle_state.STATE_ROOT_ENV, str(tmp_path / 'state'))


def _days_ago(days: int) -> date:
    return _SOURCE_NOW.date() - timedelta(days=days)


def _write_session(directory: Path, cwd: str, sid: str, *, day: date, signal=True) -> Path:
    base = {'cwd': cwd, 'sessionId': sid, 'timestamp': f'{day.isoformat()}T10:00:00.000Z'}
    records = [{
        **base, 'type': 'user', 'isSidechain': False, 'isMeta': False,
        'message': {'role': 'user', 'content': f'Please look at {sid}.'},
    }]
    if signal:
        records.append({
            **base, 'type': 'user',
            'message': {'role': 'user', 'content': [{
                'type': 'tool_result', 'tool_use_id': 't1', 'is_error': True,
                'content': 'cat: /x: No such file or directory',
            }]},
        })
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{sid}.jsonl'
    path.write_text(''.join(json.dumps(r) + '\n' for r in records), encoding='utf-8')
    return path


class _Fixture:
    def __init__(self, tmp_path: Path):
        self.root = tmp_path / 'project'
        self.root.mkdir()
        self.cwd = str(self.root)
        self.projects_root = tmp_path / 'projects'
        self.projects_root.mkdir()
        self.ledger_path = tmp_path / 'ledger' / 'coded-sessions.sqlite'
        self.cfg = config.LegibilityConfig(
            project_id='p',
            project_root=str(self.root),
            escalation_port=8103,
            cwd_prefixes=[self.cwd],
            agent_transcript_roots=[_ARCHIVE_ROOT],
        )

    def session(self, sid, *, days_ago=3, signal=True):
        directory = self.projects_root / inventory.encode_cwd(self.cwd)
        return _write_session(directory, self.cwd, sid, day=_days_ago(days_ago), signal=signal)

    def archived_session(self, sid, *, days_ago=3):
        directory = (
            self.root / _ARCHIVE_ROOT / '6397'
            / inventory.encode_cwd(f'{self.cwd}/.worktrees/6397')
        )
        return _write_session(
            directory, f'{self.cwd}/.worktrees/6397', sid, day=_days_ago(days_ago),
        )

    def ledger_row(self, sid, *, coded_by=CodedBy.TRICKLE, coded_at=_SOURCE_NOW):
        session_ledger.record_codings(self.ledger_path, [LedgerRow(
            session=sid, instrument_version=5, coded_by=coded_by,
            run_ref=f'{coded_by}-run', outcome=Outcome.EMPTY, coded_at=coded_at,
        )])

    def source(self, *, last_census_at=None, batch_size=census_window.DEFAULT_BATCH_SIZE):
        return census_window.WindowBatchSource(
            self.cfg,
            projects_root=self.projects_root,
            now=_SOURCE_NOW,
            ledger_path=self.ledger_path,
            last_census_at=last_census_at,
            batch_size=batch_size,
            rng=random.Random(0),
        )


def _mined_sessions(batches) -> list[str]:
    return sorted(
        coder.parse_frontmatter(text)['session'] for batch in batches for text in batch
    )


@pytest.fixture
def fx(tmp_path):
    return _Fixture(tmp_path)


def test_selection_is_unknown_until_the_source_is_iterated(fx):
    fx.session('T')
    assert fx.source().selection is None


def test_a_ledgered_session_is_never_mined(fx):
    fx.session('S')
    fx.session('T')
    fx.ledger_row('S')

    source = fx.source()
    assert _mined_sessions(list(source)) == ['T']

    selection = source.selection
    assert selection is not None
    assert selection.sessions_enumerated == 2
    assert selection.skipped_coded == 1
    assert selection.skipped_zero_signal == 0


def test_a_ledgered_session_costs_no_full_transcript_scan(fx, monkeypatch):
    fx.session('S')
    t_path = fx.session('T')
    fx.ledger_row('S')
    scanned = []
    score_session = sampling.score_session

    def recording_score_session(session):
        scanned.append(session.path)
        return score_session(session)

    monkeypatch.setattr(sampling, 'score_session', recording_score_session)

    list(fx.source())

    assert scanned == [t_path]


def test_a_zero_signal_session_is_never_digested(fx):
    fx.session('T')
    fx.session('Z', signal=False)

    source = fx.source()
    assert _mined_sessions(list(source)) == ['T']
    assert source.selection is not None
    assert source.selection.skipped_zero_signal == 1
    assert source.selection.skipped_coded == 0


def test_an_absent_ledger_is_created(fx):
    fx.session('T')
    source = fx.source()
    list(source)
    assert source.selection is not None
    assert source.selection.ledger.state is LedgerState.CREATED
    assert fx.ledger_path.is_file()


def test_an_unreadable_ledger_excludes_nothing_and_takes_the_transition_window(fx, caplog):
    fx.ledger_path.parent.mkdir(parents=True)
    garbage = b'not an sqlite database\n' * 8
    fx.ledger_path.write_bytes(garbage)
    fx.session('OLD', days_ago=5)
    fx.session('NEW', days_ago=1)

    source = fx.source(last_census_at=_days_ago(2))
    with caplog.at_level(logging.WARNING, logger='legibility.census_window'):
        assert _mined_sessions(list(source)) == ['NEW']

    [warning] = [r for r in caplog.records if r.name == 'legibility.census_window']
    assert warning.levelno == logging.WARNING
    assert str(fx.ledger_path) in warning.getMessage()

    selection = source.selection
    assert selection is not None
    assert selection.ledger.state is LedgerState.UNREADABLE
    assert selection.skipped_coded == 0
    assert selection.window.start == datetime.combine(_days_ago(2), time(), UTC)
    assert fx.ledger_path.read_bytes() == garbage


def test_rows_coded_before_the_retention_window_are_pruned(fx):
    fx.session('T')
    fx.ledger_row('ANCIENT', coded_at=_SOURCE_NOW - timedelta(days=45))
    fx.ledger_row('RECENT')

    source = fx.source()
    list(source)
    assert source.selection is not None
    assert source.selection.ledger.pruned == 1
    assert source.selection.ledger.sessions == frozenset({'RECENT'})


def test_the_transition_floor_holds_until_the_ledger_has_census_rows(fx):
    fx.session('OLD', days_ago=5)
    fx.session('NEW', days_ago=1)

    before = fx.source(last_census_at=_days_ago(2))
    assert _mined_sessions(list(before)) == ['NEW']
    assert before.selection is not None
    assert before.selection.sessions_enumerated == 1

    fx.ledger_row('C', coded_by=CodedBy.CENSUS)
    after = fx.source(last_census_at=_days_ago(2))
    assert _mined_sessions(list(after)) == ['NEW', 'OLD']
    assert after.selection is not None
    assert after.selection.window == census_window.retention_window(
        _SOURCE_NOW, fx.cfg.census.ledger_retention_days,
    )


def test_a_session_dated_today_is_never_enumerated(fx):
    fx.session('TODAY', days_ago=0)
    fx.session('T')

    source = fx.source()
    assert _mined_sessions(list(source)) == ['T']
    assert source.selection is not None
    assert source.selection.sessions_enumerated == 1


def test_archived_agent_transcripts_are_enumerated(fx):
    fx.archived_session('ARCHIVED')

    source = fx.source()
    assert _mined_sessions(list(source)) == ['ARCHIVED']
    assert source.selection is not None
    assert source.selection.sessions_enumerated == 1


def test_batch_size_bounds_every_batch(fx):
    for i in range(5):
        fx.session(f'S{i}')

    batches = list(fx.source(batch_size=2))

    assert [len(batch) for batch in batches] == [2, 2, 1]
    assert _mined_sessions(batches) == [f'S{i}' for i in range(5)]
