"""Drives the shipped scripts/merge-deep-canary-predicate.sh as a subprocess over a synthetic runs.db.

Rows are seeded in EventStore's own merge_verify / merge_finalized shape: the
payload carries every fact and the events.duration_ms column stays NULL.
"""
from __future__ import annotations

import re
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

from cli_subprocess_timeout import cli_timeout_from_env

_PREDICATE = Path(__file__).resolve().parents[1] / 'merge-deep-canary-predicate.sh'
_PREDICATE_TIMEOUT = cli_timeout_from_env('MERGE_DEEP_CANARY_PREDICATE_CLI_TIMEOUT_SECS')

_ABSENT = object()


def _run(db_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ['bash', str(_PREDICATE), '--runs-db', str(db_path), *args],
        capture_output=True, text=True, timeout=_PREDICATE_TIMEOUT,
    )


def _metric(stdout: str, name: str) -> float | None:
    match = re.search(rf'{name}=([^(]+)\(', stdout)
    assert match, stdout
    raw = match.group(1)
    return None if raw == 'n/a' else float(raw)


def _verdict(stdout: str, name: str) -> str:
    match = re.search(rf'{name}=[^(]+\((\w+) ', stdout)
    assert match, stdout
    return match.group(1)


def _n_deep(stdout: str) -> int:
    match = re.search(r'n_deep_verifies=(\d+)', stdout)
    assert match, stdout
    return int(match.group(1))


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


class TestVerifyP90ReadsThePayloadDuration:
    def test_p90_comes_from_the_merge_verify_payload(self, runs_db_path, runs_db):
        for k in range(1, 11):
            _seed_verify(runs_db, _recent(k), passed=True, duration_ms=60_000 * k)
            _seed_finalized(runs_db, _recent(k), landed_via_chain=2)

        result = _run(runs_db_path)

        assert result.returncode == 0, f'{result.stdout=} {result.stderr=}'
        assert _metric(result.stdout, 'verify_p90_secs') == 540.0
        assert _metric(result.stdout, 'deep_fail_rate') == 0.0
        assert _metric(result.stdout, 'items_per_verify') == 2.0

    def test_p90_over_the_bound_holds(self, runs_db_path, runs_db):
        for i, duration_ms in enumerate((6_000_000, 7_000_000)):
            _seed_verify(runs_db, _recent(i), passed=True, duration_ms=duration_ms)
            _seed_finalized(runs_db, _recent(i), landed_via_chain=2)

        result = _run(runs_db_path)

        assert result.returncode == 1, f'{result.stdout=} {result.stderr=}'
        assert _metric(result.stdout, 'verify_p90_secs') == 7000.0
        assert _verdict(result.stdout, 'verify_p90_secs') == 'FAIL'
        assert _verdict(result.stdout, 'deep_fail_rate') == 'ok'
        assert _verdict(result.stdout, 'items_per_verify') == 'ok'

    def test_non_numeric_payload_durations_are_not_durations(self, runs_db_path, runs_db):
        for i, duration_ms in enumerate((True, None, '600000', _ABSENT, 300_000)):
            _seed_verify(runs_db, _recent(i), passed=True, duration_ms=duration_ms)
            _seed_finalized(runs_db, _recent(i), landed_via_chain=2)

        result = _run(runs_db_path)

        assert result.returncode == 0, f'{result.stdout=} {result.stderr=}'
        assert _metric(result.stdout, 'verify_p90_secs') == 300.0
