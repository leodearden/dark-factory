"""Drives the shipped scripts/merge-deep-canary-predicate.sh as a subprocess over a synthetic runs.db.

Rows are seeded in EventStore's own merge_verify / merge_finalized shape: the
payload carries every fact and the events.duration_ms column stays NULL.
"""
from __future__ import annotations

import itertools
import sqlite3
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from canary_predicate_summary import (
    insufficient_signal,
    metric,
    n_deep_verifies,
    population,
    store_n_deep,
    store_population,
    verdict,
)
from cli_subprocess_timeout import cli_timeout_from_env
from orchestrator.event_store import EventStore, EventType

_PREDICATE = Path(__file__).resolve().parents[1] / 'merge-deep-canary-predicate.sh'
_PREDICATE_TIMEOUT = cli_timeout_from_env('MERGE_DEEP_CANARY_PREDICATE_CLI_TIMEOUT_SECS')

_ABSENT = object()

_AS_OF = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)
_NOW_ARGS = ('--now', str(_AS_OF.timestamp()), '--window-days', '7')
_CUTOFF = _AS_OF - timedelta(days=7)


def _run_stores(db_paths: list[Path], *args: str) -> subprocess.CompletedProcess[str]:
    store_args = [arg for db_path in db_paths for arg in ('--runs-db', str(db_path))]
    return subprocess.run(
        ['bash', str(_PREDICATE), *store_args, *args],
        capture_output=True, text=True, timeout=_PREDICATE_TIMEOUT,
    )


def _run(db_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return _run_stores([db_path], *args)


_FRESH_SHA = object()
_sha_serial = itertools.count()


def _event(ts: datetime, event_type: str, data: dict[str, object], task_id: str | None = None) -> dict[str, object]:
    """One events row as both make_runs_db and SeedingConnection.seed_event(**row) accept it."""
    return {'timestamp': ts.isoformat(), 'event_type': event_type, 'task_id': task_id, 'data': data}


def _verify_event(
    ts: datetime,
    *,
    passed: object,
    chain_items: object = 2,
    runner: object = 'local',
    duration_ms: object = 600_000,
    merge_sha: object = _FRESH_SHA,
    attempt: object = 0,
    task_id: str | None = None,
) -> dict[str, object]:
    """A merge_verify row in VerifyRunnerPool.dispatch's payload shape; ``_ABSENT`` omits a key."""
    fields = {
        'runner': runner,
        'merge_sha': f'sha-{next(_sha_serial)}' if merge_sha is _FRESH_SHA else merge_sha,
        'passed': passed,
        'duration_ms': duration_ms,
        'attempt': attempt,
        'chain_items': chain_items,
    }
    data = {key: value for key, value in fields.items() if value is not _ABSENT}
    return _event(ts, 'merge_verify', data, task_id)


def _finalized_event(ts: datetime, landed_via_chain: object) -> dict[str, object]:
    return _event(ts, 'merge_finalized', {'state': 'done', 'landed_via_chain': landed_via_chain})


def _healthy_round_events(
    ts: datetime,
    *,
    runner: str = 'local',
    deep_ms: int = 600_000,
    baseline_ms: int | None = None,
    task_id: str | None = None,
    finalized_ts: datetime | None = None,
) -> list[dict[str, object]]:
    """A passed deep verify, its same-runner passed solo baseline a minute before, and its landing a minute after."""
    return [
        _verify_event(ts, passed=True, runner=runner, duration_ms=deep_ms, task_id=task_id),
        _verify_event(
            ts - timedelta(minutes=1), passed=True, chain_items=1, runner=runner,
            duration_ms=deep_ms if baseline_ms is None else baseline_ms,
        ),
        _finalized_event(finalized_ts or ts + timedelta(minutes=1), landed_via_chain=2),
    ]


def _seed_verify(conn, ts: datetime, **fields):
    conn.seed_event(**_verify_event(ts, **fields))


def _seed_finalized(conn, ts: datetime, landed_via_chain: int):
    conn.seed_event(**_finalized_event(ts, landed_via_chain))


def _recent(i: int) -> datetime:
    return datetime.now(UTC) - timedelta(minutes=i + 1)


def _seed_inside(conn, verify_ts: datetime, finalized_ts: datetime | None = None):
    for event in _healthy_round_events(verify_ts, finalized_ts=finalized_ts):
        conn.seed_event(**event)


def _healthy_rounds(count: int, *, first: datetime = _AS_OF - timedelta(days=3), **round_fields):
    """*count* healthy rounds an hour apart from *first*, flattened into one event list."""
    return [
        event
        for k in range(count)
        for event in _healthy_round_events(first + timedelta(hours=k), **round_fields)
    ]


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

        result = _run(runs_db_path, '--min-deep-verifies', '2')

        assert result.returncode == 1, f'{result.stdout=} {result.stderr=}'
        assert metric(result.stdout, 'verify_p90_secs') == 7000.0
        assert verdict(result.stdout, 'verify_p90_secs') == 'FAIL'
        assert verdict(result.stdout, 'deep_fail_rate') == 'ok'
        assert verdict(result.stdout, 'items_per_verify') == 'ok'

    def test_non_numeric_payload_durations_are_not_durations(self, runs_db_path, runs_db):
        for i, duration_ms in enumerate((True, None, '600000', _ABSENT, float('nan'), float('inf'), 300_000)):
            _seed_verify(runs_db, _recent(i), passed=True, duration_ms=duration_ms)
            _seed_finalized(runs_db, _recent(i), landed_via_chain=2)

        result = _run(runs_db_path, '--min-deep-verifies', '1')

        assert result.returncode == 0, f'{result.stdout=} {result.stderr=}'
        assert metric(result.stdout, 'verify_p90_secs') == 300.0


class TestWindowEdges:
    def test_boundary_day_rows_split_at_the_cutoff_hour(self, runs_db_path, runs_db):
        _seed_inside(runs_db, _CUTOFF + timedelta(hours=2))
        _seed_intruder(runs_db, _CUTOFF - timedelta(hours=2))

        result = _run(runs_db_path, *_NOW_ARGS, '--min-deep-verifies', '1')

        assert result.returncode == 0, f'{result.stdout=} {result.stderr=}'
        assert n_deep_verifies(result.stdout) == 1
        assert metric(result.stdout, 'deep_fail_rate') == 0.0
        assert metric(result.stdout, 'items_per_verify') == 2.0
        assert metric(result.stdout, 'verify_p90_secs') == 600.0

    def test_window_is_half_open_at_the_cutoff_and_closed_at_now(self, runs_db_path, runs_db):
        _seed_intruder(runs_db, _CUTOFF, finalized_ts=_CUTOFF)
        _seed_inside(runs_db, _AS_OF, finalized_ts=_AS_OF)

        result = _run(runs_db_path, *_NOW_ARGS, '--min-deep-verifies', '1')

        assert result.returncode == 0, f'{result.stdout=} {result.stderr=}'
        assert n_deep_verifies(result.stdout) == 1

    def test_rows_after_now_are_outside_the_window(self, runs_db_path, runs_db):
        _seed_inside(runs_db, _AS_OF - timedelta(hours=1))
        _seed_intruder(runs_db, _AS_OF + timedelta(hours=1))

        result = _run(runs_db_path, *_NOW_ARGS, '--min-deep-verifies', '1')

        assert result.returncode == 0, f'{result.stdout=} {result.stderr=}'
        assert n_deep_verifies(result.stdout) == 1

    def test_default_now_is_the_wall_clock(self, runs_db_path, runs_db):
        now = datetime.now(UTC)
        _seed_inside(runs_db, now - timedelta(days=6))
        _seed_intruder(runs_db, now - timedelta(days=8))

        result = _run(runs_db_path, '--window-days', '7', '--min-deep-verifies', '1')

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
        ('--min-deep-verifies', '0'),
        ('--min-deep-verifies', '-1'),
        ('--min-deep-verifies', 'eight'),
        ('--min-deep-verifies', '2.5'),
    ])
    def test_malformed_numeric_flag_is_a_usage_fault(self, runs_db_path, flag, value):
        result = _run(runs_db_path, flag, value)

        assert result.returncode == 2, f'{result.stdout=} {result.stderr=}'
        assert flag in result.stderr


class TestPooledStores:
    def test_metrics_pool_across_repeated_runs_db(self, make_runs_db):
        failed_tip = _verify_event(_AS_OF - timedelta(days=1), passed=False, task_id='9001')
        reify = make_runs_db([*_healthy_rounds(4), failed_tip], name='reify.db')
        df = make_runs_db(_healthy_rounds(4), name='df.db')

        result = _run_stores([reify, df], *_NOW_ARGS)

        assert result.returncode == 0, f'{result.stdout=} {result.stderr=}'
        assert n_deep_verifies(result.stdout) == 9
        assert metric(result.stdout, 'deep_fail_rate') == 0.1111
        assert store_n_deep(result.stdout) == {str(reify): 5, str(df): 4}

    @pytest.mark.parametrize('respell', [
        pytest.param(lambda path: path, id='verbatim'),
        pytest.param(lambda path: path.parent / '.' / path.name, id='respelled'),
    ])
    def test_a_store_named_twice_is_a_usage_fault(self, make_runs_db, respell):
        reify = make_runs_db(_healthy_rounds(8), name='reify.db')

        result = _run_stores([reify, respell(reify)], *_NOW_ARGS)

        assert result.returncode == 2, f'{result.stdout=} {result.stderr=}'
        assert '--runs-db' in result.stderr

    @pytest.mark.parametrize('missing_first', [False, True], ids=['missing-second', 'missing-first'])
    def test_a_missing_store_is_a_usage_fault(self, make_runs_db, tmp_path, missing_first):
        reify = make_runs_db(_healthy_rounds(8), name='reify.db')
        missing = tmp_path / 'absent.db'
        stores = [missing, reify] if missing_first else [reify, missing]

        result = _run_stores(stores, *_NOW_ARGS)

        assert result.returncode == 2, f'{result.stdout=} {result.stderr=}'
        assert str(missing) in result.stderr

    @pytest.mark.parametrize('bare_first', [False, True], ids=['bare-second', 'bare-first'])
    def test_a_store_without_an_events_table_is_an_infra_fault(self, make_runs_db, tmp_path, bare_first):
        reify = make_runs_db(_healthy_rounds(8), name='reify.db')
        bare = tmp_path / 'bare.db'
        sqlite3.connect(bare).close()
        stores = [bare, reify] if bare_first else [reify, bare]

        result = _run_stores(stores, *_NOW_ARGS)

        assert result.returncode == 2, f'{result.stdout=} {result.stderr=}'
        assert str(bare) in result.stderr


class TestMinimumSample:
    @pytest.mark.parametrize(('reify_rounds', 'df_rounds'), [
        pytest.param(4, 3, id='7-pooled'),
        pytest.param(4, 4, id='8-pooled'),
        pytest.param(8, 0, id='8-in-one-store'),
    ])
    def test_pooled_n_below_the_minimum_is_insufficient_signal(self, make_runs_db, reify_rounds, df_rounds):
        reify = make_runs_db(_healthy_rounds(reify_rounds), name='reify.db')
        df = make_runs_db(_healthy_rounds(df_rounds), name='df.db')
        pooled = reify_rounds + df_rounds

        result = _run_stores([reify, df], *_NOW_ARGS)

        assert store_n_deep(result.stdout) == {str(reify): reify_rounds, str(df): df_rounds}
        if pooled < 8:
            assert result.returncode == 1, f'{result.stdout=} {result.stderr=}'
            assert result.stdout.splitlines()[-1] == f'insufficient_signal n_deep_verifies={pooled} (< 8)'
            assert '-> PASS' not in result.stdout and '-> HOLD' not in result.stdout
            assert '(ok ' not in result.stdout and '(FAIL ' not in result.stdout
        else:
            assert result.returncode == 0, f'{result.stdout=} {result.stderr=}'
            assert '-> PASS' in result.stdout
            assert insufficient_signal(result.stdout) is None

    def test_an_explicit_minimum_lowers_the_bar(self, make_runs_db):
        reify = make_runs_db(_healthy_rounds(3), name='reify.db')

        result = _run(reify, *_NOW_ARGS, '--min-deep-verifies', '3')

        assert result.returncode == 0, f'{result.stdout=} {result.stderr=}'
        assert result.stdout.splitlines()[-1].endswith('-> PASS')

    def test_no_deep_verifies_is_insufficient_signal(self, runs_db_path):
        result = _run(runs_db_path, *_NOW_ARGS)

        assert result.returncode == 1, f'{result.stdout=} {result.stderr=}'
        assert result.stdout.splitlines()[-1] == 'insufficient_signal n_deep_verifies=0 (< 8)'
        assert insufficient_signal(result.stdout) == (0, 8)

    def test_p90_is_reported_only_when_the_timed_population_reaches_the_minimum(self, make_runs_db):
        untimed_at = _AS_OF - timedelta(days=1)
        untimed_round = [
            _verify_event(untimed_at, passed=True, duration_ms=_ABSENT),
            _finalized_event(untimed_at + timedelta(minutes=1), landed_via_chain=2),
        ]
        reify = make_runs_db([*_healthy_rounds(7), *untimed_round], name='reify.db')

        result = _run(reify, *_NOW_ARGS)

        assert result.returncode == 1, f'{result.stdout=} {result.stderr=}'
        assert n_deep_verifies(result.stdout) == 8
        assert metric(result.stdout, 'verify_p90_secs') is None
        assert verdict(result.stdout, 'verify_p90_secs') == 'FAIL'


class TestItemsPerPassedDeepRound:
    def test_denominator_counts_distinct_passed_first_attempt_tips(self, make_runs_db):
        first = _AS_OF - timedelta(days=2)
        tips = [
            dict(merge_sha='tip-a', attempt=0, passed=True),
            dict(merge_sha='tip-a', attempt=1, passed=True),
            dict(merge_sha='tip-b', attempt=0, passed=False),
            dict(merge_sha='tip-c', attempt=0, passed=True),
            dict(merge_sha=_ABSENT, attempt=0, passed=True),
        ]
        events = []
        for k, tip in enumerate(tips):
            at = first + timedelta(hours=k)
            events.append(_verify_event(at, **tip))
            events.append(_verify_event(at - timedelta(minutes=1), passed=True, chain_items=1))
        events += [_finalized_event(first + timedelta(hours=k, minutes=30), landed_via_chain=1) for k in range(4)]
        store = make_runs_db(events, name='reify.db')

        result = _run(store, *_NOW_ARGS, '--min-deep-verifies', '1')

        assert result.returncode == 0, f'{result.stdout=} {result.stderr=}'
        assert metric(result.stdout, 'items_per_verify') == 2.0
        assert population(result.stdout, 'landed_via_chain') == 4
        assert population(result.stdout, 'passed_deep_rounds') == 2
        assert store_population(result.stdout, str(store), 'passed_deep_rounds') == 2

    def test_a_window_with_no_passed_deep_round_has_no_items_signal(self, make_runs_db):
        first = _AS_OF - timedelta(days=2)
        events = [
            *(_verify_event(first + timedelta(hours=k), passed=False) for k in range(2)),
            _finalized_event(first + timedelta(hours=3), landed_via_chain=2),
        ]
        store = make_runs_db(events, name='reify.db')

        result = _run(store, *_NOW_ARGS, '--min-deep-verifies', '1')

        assert result.returncode == 1, f'{result.stdout=} {result.stderr=}'
        assert metric(result.stdout, 'items_per_verify') is None
        assert verdict(result.stdout, 'items_per_verify') == 'FAIL'
