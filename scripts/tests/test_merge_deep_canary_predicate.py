"""Drives the shipped scripts/merge-deep-canary-predicate.sh as a subprocess over a synthetic runs.db.

Rows are seeded in EventStore's own merge_verify / merge_finalized shape: the
payload carries every fact and the events.duration_ms column stays NULL.
"""
from __future__ import annotations

import sqlite3
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from canary_predicate_summary import metric, n_deep_verifies, verdict
from cli_subprocess_timeout import cli_timeout_from_env
from orchestrator.event_store import EventStore, EventType

_PREDICATE = Path(__file__).resolve().parents[1] / 'merge-deep-canary-predicate.sh'
_PREDICATE_TIMEOUT = cli_timeout_from_env('MERGE_DEEP_CANARY_PREDICATE_CLI_TIMEOUT_SECS')

_ABSENT = object()

_AS_OF = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)
_NOW_ARGS = ('--now', str(_AS_OF.timestamp()), '--window-days', '7')
_CUTOFF = _AS_OF - timedelta(days=7)


def _run(db_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ['bash', str(_PREDICATE), '--runs-db', str(db_path), *args],
        capture_output=True, text=True, timeout=_PREDICATE_TIMEOUT,
    )


def _seed_verify(conn, ts: datetime, *, passed: bool, duration_ms=_ABSENT, chain_items: int = 2):
    data = {'runner': 'local', 'passed': passed, 'chain_items': chain_items}
    if duration_ms is not _ABSENT:
        data['duration_ms'] = duration_ms
    conn.seed_event(ts.isoformat(), 'merge_verify', data=data)


def _seed_finalized(conn, ts: datetime, landed_via_chain: int):
    conn.seed_event(
        ts.isoformat(), 'merge_finalized',
        data={'state': 'done', 'landed_via_chain': landed_via_chain},
    )


def _recent(i: int) -> datetime:
    return datetime.now(UTC) - timedelta(minutes=i + 1)


def _seed_inside(conn, verify_ts: datetime, finalized_ts: datetime | None = None):
    _seed_verify(conn, verify_ts, passed=True, duration_ms=600_000)
    _seed_finalized(conn, finalized_ts or verify_ts + timedelta(minutes=1), landed_via_chain=2)


def _seed_intruder(conn, verify_ts: datetime, finalized_ts: datetime | None = None):
    """A row that, once admitted, makes the run HOLD on every leg."""
    _seed_verify(conn, verify_ts, passed=False, duration_ms=9_000_000)
    _seed_finalized(conn, finalized_ts or verify_ts + timedelta(minutes=1), landed_via_chain=5)


class TestVerifyP90ReadsThePayloadDuration:
    def test_p90_comes_from_the_merge_verify_payload(self, runs_db_path, runs_db):
        for k in range(1, 11):
            _seed_verify(runs_db, _recent(k), passed=True, duration_ms=60_000 * k)
            _seed_finalized(runs_db, _recent(k), landed_via_chain=2)

        result = _run(runs_db_path)

        assert result.returncode == 0, f'{result.stdout=} {result.stderr=}'
        assert metric(result.stdout, 'verify_p90_secs') == 540.0
        assert metric(result.stdout, 'deep_fail_rate') == 0.0
        assert metric(result.stdout, 'items_per_verify') == 2.0

    def test_p90_over_the_bound_holds(self, runs_db_path, runs_db):
        for i, duration_ms in enumerate((6_000_000, 7_000_000)):
            _seed_verify(runs_db, _recent(i), passed=True, duration_ms=duration_ms)
            _seed_finalized(runs_db, _recent(i), landed_via_chain=2)

        result = _run(runs_db_path)

        assert result.returncode == 1, f'{result.stdout=} {result.stderr=}'
        assert metric(result.stdout, 'verify_p90_secs') == 7000.0
        assert verdict(result.stdout, 'verify_p90_secs') == 'FAIL'
        assert verdict(result.stdout, 'deep_fail_rate') == 'ok'
        assert verdict(result.stdout, 'items_per_verify') == 'ok'

    def test_non_numeric_payload_durations_are_not_durations(self, runs_db_path, runs_db):
        for i, duration_ms in enumerate((True, None, '600000', _ABSENT, float('nan'), float('inf'), 300_000)):
            _seed_verify(runs_db, _recent(i), passed=True, duration_ms=duration_ms)
            _seed_finalized(runs_db, _recent(i), landed_via_chain=2)

        result = _run(runs_db_path)

        assert result.returncode == 0, f'{result.stdout=} {result.stderr=}'
        assert metric(result.stdout, 'verify_p90_secs') == 300.0


class TestWindowEdges:
    def test_boundary_day_rows_split_at_the_cutoff_hour(self, runs_db_path, runs_db):
        _seed_inside(runs_db, _CUTOFF + timedelta(hours=2))
        _seed_intruder(runs_db, _CUTOFF - timedelta(hours=2))

        result = _run(runs_db_path, *_NOW_ARGS)

        assert result.returncode == 0, f'{result.stdout=} {result.stderr=}'
        assert n_deep_verifies(result.stdout) == 1
        assert metric(result.stdout, 'deep_fail_rate') == 0.0
        assert metric(result.stdout, 'items_per_verify') == 2.0
        assert metric(result.stdout, 'verify_p90_secs') == 600.0

    def test_window_is_half_open_at_the_cutoff_and_closed_at_now(self, runs_db_path, runs_db):
        _seed_intruder(runs_db, _CUTOFF, finalized_ts=_CUTOFF)
        _seed_inside(runs_db, _AS_OF, finalized_ts=_AS_OF)

        result = _run(runs_db_path, *_NOW_ARGS)

        assert result.returncode == 0, f'{result.stdout=} {result.stderr=}'
        assert n_deep_verifies(result.stdout) == 1

    def test_rows_after_now_are_outside_the_window(self, runs_db_path, runs_db):
        _seed_inside(runs_db, _AS_OF - timedelta(hours=1))
        _seed_intruder(runs_db, _AS_OF + timedelta(hours=1))

        result = _run(runs_db_path, *_NOW_ARGS)

        assert result.returncode == 0, f'{result.stdout=} {result.stderr=}'
        assert n_deep_verifies(result.stdout) == 1

    def test_default_now_is_the_wall_clock(self, runs_db_path, runs_db):
        now = datetime.now(UTC)
        _seed_inside(runs_db, now - timedelta(days=6))
        _seed_intruder(runs_db, now - timedelta(days=8))

        result = _run(runs_db_path, '--window-days', '7')

        assert result.returncode == 0, f'{result.stdout=} {result.stderr=}'
        assert n_deep_verifies(result.stdout) == 1


def _emit_unlanded_deep_verify(db_path: Path) -> datetime:
    """Write one deep verify through the real EventStore.emit; return the instant it stamped."""
    EventStore(db_path, 'run-writer').emit(
        EventType.merge_verify, data={'passed': True, 'chain_items': 2, 'duration_ms': 600_000},
    )
    conn = sqlite3.connect(db_path)
    try:
        (stamp,), = conn.execute('SELECT timestamp FROM events').fetchall()
    finally:
        conn.close()
    return datetime.fromisoformat(stamp)


class TestWindowBoundsMeetTheWritersStamp:
    """The bounds compare exactly against the timestamp layout EventStore.emit itself writes."""

    @pytest.mark.parametrize(('now_after_stamp', 'admitted'), [
        pytest.param(timedelta(0), 1, id='stamped-at-now'),
        pytest.param(-timedelta(microseconds=1), 0, id='stamped-after-now'),
        pytest.param(timedelta(days=7), 0, id='stamped-at-the-cutoff'),
        pytest.param(timedelta(days=7, microseconds=-1), 1, id='stamped-just-inside-the-cutoff'),
    ])
    def test_an_emitted_row_meets_the_bounds_exactly(self, tmp_path, now_after_stamp, admitted):
        db_path = tmp_path / 'runs.db'
        now = _emit_unlanded_deep_verify(db_path) + now_after_stamp

        result = _run(db_path, '--now', repr(now.timestamp()), '--window-days', '7')

        assert result.returncode == 1, f'{result.stdout=} {result.stderr=}'
        assert n_deep_verifies(result.stdout) == admitted


class TestUsageFaultsExitTwo:
    @pytest.mark.parametrize(('flag', 'value'), [
        ('--now', 'yesterday'),
        ('--now', 'nan'),
        ('--now', 'inf'),
        ('--now', '1e20'),
        ('--window-days', 'seven'),
        ('--window-days', '0'),
        ('--window-days', '-3'),
        ('--window-days', '1e12'),
        ('--deep-fail-rate-max', 'loose'),
        ('--items-per-verify-min', 'x'),
        ('--verify-p90-max-secs', 'nan'),
    ])
    def test_malformed_numeric_flag_is_a_usage_fault(self, runs_db_path, flag, value):
        result = _run(runs_db_path, flag, value)

        assert result.returncode == 2, f'{result.stdout=} {result.stderr=}'
        assert flag in result.stderr
