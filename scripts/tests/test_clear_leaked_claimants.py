"""Tests for scripts/clear_leaked_claimants.py (task 4625).

Every expectation is hand-derived from the tier table in
docs/prds/claimant-invariant-enforcement.md D3, never computed by calling the
shared predicates, and every store is a synthetic temp tasks.db.
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from _task_db_scan import UNREADABLE_STORE_ERRORS
from clear_leaked_claimants import Tier, take_census
from cli_subprocess_timeout import cli_timeout_from_env

NOW = datetime(2026, 10, 11, 12, tzinfo=UTC)
STALE = NOW - timedelta(hours=2)
FRESH = NOW - timedelta(minutes=1)


def _iso(moment: datetime) -> str:
    return moment.isoformat()


def _claimed(task_id, status, claimant, heartbeat, **extra):
    heartbeat_at = None if heartbeat is None else _iso(heartbeat)
    return {
        'id': task_id, 'status': status,
        'claimant_run_id': claimant, 'heartbeat_at': heartbeat_at, **extra,
    }


POSITIVE_CONTROL = _claimed(4028, 'done', 'run-a/4028-x/pid=1', NOW - timedelta(days=50))
FREEFORM_CANCELLED = _claimed(5225, 'cancelled', 'agent-esc-5053-2-docs-fix', None)
FRESH_DONE = _claimed(101, 'done', 'run-c/101-x/pid=3', FRESH)
STALE_PENDING = _claimed(103, 'pending', 'run-e/103-x/pid=5', STALE)
STALE_MERGE_DEFERRED = _claimed(109, 'merge-deferred', 'run-k/109-x/pid=11', STALE)

TIER_TABLE_ROWS = [
    POSITIVE_CONTROL,
    FREEFORM_CANCELLED,
    FRESH_DONE,
    _claimed(102, 'done', None, FRESH),
    STALE_PENDING,
    _claimed(104, 'pending', 'run-f/104-x/pid=6', FRESH),
    _claimed(105, 'in-progress', 'run-g/105-x/pid=7', STALE),
    _claimed(106, 'infra-hold', 'run-h/106-x/pid=8', STALE),
    _claimed(107, 'blocked', 'run-i/107-x/pid=9', STALE),
    _claimed(108, 'deferred', 'run-j/108-x/pid=10', STALE, metadata={'infra_hold': True}),
    STALE_MERGE_DEFERRED,
    _claimed(110, 'pending', '  ', STALE),
]


@pytest.fixture
def seed_root(tmp_path, make_tasks_db):
    def _seed(rows, name='proj') -> str:
        root = tmp_path / name
        store_dir = root / '.taskmaster' / 'tasks'
        store_dir.mkdir(parents=True)
        make_tasks_db(rows, directory=store_dir)
        return str(root)

    return _seed


class TestCensus:
    def test_classifies_each_row_by_the_d3_tier_table(self, seed_root):
        census = take_census(seed_root(TIER_TABLE_ROWS), NOW)

        assert {(v.task_id, v.tier) for v in census.violations} == {
            (4028, Tier.TERMINAL),
            (5225, Tier.TERMINAL),
            (101, Tier.TERMINAL),
            (103, Tier.HYGIENE),
            (109, Tier.HYGIENE),
        }

    def test_violation_carries_the_row_verbatim_and_the_measurement_time(self, seed_root):
        census = take_census(seed_root(TIER_TABLE_ROWS), NOW)
        by_id = {v.task_id: v for v in census.violations}

        for row in (POSITIVE_CONTROL, FREEFORM_CANCELLED, STALE_PENDING):
            violation = by_id[row['id']]
            assert violation.tag == 'master'
            assert violation.status == row['status']
            assert violation.claimant_run_id == row['claimant_run_id']
            assert violation.heartbeat_at == row['heartbeat_at']
            assert violation.measured_at == NOW

    def test_reports_how_many_rows_it_read(self, seed_root):
        census = take_census(seed_root(TIER_TABLE_ROWS), NOW)

        assert census.rows_in_store == len(TIER_TABLE_ROWS)

    def test_totals_by_tier_and_by_tier_and_status(self, seed_root):
        totals = take_census(seed_root(TIER_TABLE_ROWS), NOW).totals()

        assert totals['by_tier'] == {Tier.TERMINAL: 3, Tier.HYGIENE: 2}
        assert totals['by_tier_status'] == {
            Tier.TERMINAL: {'done': 2, 'cancelled': 1},
            Tier.HYGIENE: {'pending': 1, 'merge-deferred': 1},
        }

    def test_zero_tiers_are_reported_not_omitted(self, seed_root):
        totals = take_census(seed_root([POSITIVE_CONTROL]), NOW).totals()

        assert totals['by_tier'] == {Tier.TERMINAL: 1, Tier.HYGIENE: 0}

    def test_an_empty_stub_store_is_unreadable_not_a_clean_zero(self, tmp_path):
        root = tmp_path / 'stub'
        store = root / '.taskmaster' / 'tasks' / 'tasks.db'
        store.parent.mkdir(parents=True)
        store.write_bytes(b'')

        with pytest.raises(UNREADABLE_STORE_ERRORS):
            take_census(str(root), NOW)



_SCRIPT = str(Path(__file__).resolve().parent.parent / 'clear_leaked_claimants.py')
# Nothing listens here, so a dry run that dialled the server could not exit 0.
_CLOSED_PORT_URL = 'http://127.0.0.1:9'
CLI_TIMEOUT = cli_timeout_from_env('CLEAR_LEAKED_CLAIMANTS_TEST_TIMEOUT')

# The subprocess classifies against the wall clock, so its hygiene row is stale by any reading.
ABANDONED_PENDING = _claimed(103, 'pending', 'run-e/103-x/pid=5', NOW - timedelta(days=50))
CLI_ROWS = [POSITIVE_CONTROL, FREEFORM_CANCELLED, ABANDONED_PENDING]
VIOLATION_JSON_KEYS = {
    'status', 'tier', 'task_id', 'tag', 'claimant_run_id', 'heartbeat_at', 'measured_at',
}


def _run_cli(*args: str, timeout: float = CLI_TIMEOUT) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, _SCRIPT, *args], capture_output=True, text=True, timeout=timeout,
    )


def _store_rows(root: str) -> list[tuple]:
    store = Path(root) / '.taskmaster' / 'tasks' / 'tasks.db'
    with closing(sqlite3.connect(f'file:{store}?mode=ro', uri=True)) as conn:
        return conn.execute('SELECT * FROM tasks ORDER BY tag, id').fetchall()


def _violation_line(stdout: str, task_id: int) -> str:
    token = f'task_id={task_id}'
    matches = [line for line in stdout.splitlines() if token in line.split()]
    assert len(matches) == 1, stdout
    return matches[0]


class TestDryRunCli:
    def test_reports_both_tiers_writes_nothing_and_never_dials(self, seed_root):
        root = seed_root(CLI_ROWS)
        before = _store_rows(root)

        result = _run_cli('--project-root', root, '--server-url', _CLOSED_PORT_URL)

        assert result.returncode == 0, result.stderr
        assert 'dry-run' in result.stdout
        assert 'tier=terminal' in _violation_line(result.stdout, 4028).split()
        assert 'tier=terminal' in _violation_line(result.stdout, 5225).split()
        assert 'tier=hygiene' in _violation_line(result.stdout, 103).split()
        tokens = result.stdout.split()
        assert 'terminal=2' in tokens
        assert 'hygiene=1' in tokens
        assert _store_rows(root) == before

    def test_json_carries_the_census_and_no_apply_section(self, seed_root):
        root = seed_root(CLI_ROWS)

        result = _run_cli('--project-root', root, '--json')

        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        assert payload['mode'] == 'dry-run'
        assert payload['repair_tiers'] == ['terminal']
        assert payload['unreadable'] == []
        project = payload['projects'][0]
        assert project['project_root'] == root
        assert project['apply'] is None
        census = project['census']
        assert {'measured_at', 'rows_in_store', 'totals', 'violations'} <= census.keys()
        assert census['rows_in_store'] == len(CLI_ROWS)
        assert census['totals']['by_tier'] == {'terminal': 2, 'hygiene': 1}
        assert all(set(v) == VIOLATION_JSON_KEYS for v in census['violations'])
        by_id = {v['task_id']: v for v in census['violations']}
        assert by_id[4028]['tier'] == 'terminal'
        assert by_id[4028]['claimant_run_id'] == POSITIVE_CONTROL['claimant_run_id']
        assert by_id[4028]['heartbeat_at'] == POSITIVE_CONTROL['heartbeat_at']
        assert by_id[5225]['heartbeat_at'] is None

    def test_repair_hygiene_widens_the_repair_tiers(self, seed_root):
        root = seed_root(CLI_ROWS)

        result = _run_cli('--project-root', root, '--repair-hygiene', '--json')

        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)['repair_tiers'] == ['terminal', 'hygiene']

    def test_one_unreadable_root_is_warned_about_and_skipped(self, seed_root, tmp_path):
        root = seed_root(CLI_ROWS)
        missing = str(tmp_path / 'no-such-project')

        result = _run_cli('--project-root', missing, '--project-root', root, '--json')

        assert result.returncode == 0, result.stderr
        assert missing in result.stderr
        payload = json.loads(result.stdout)
        assert [p['project_root'] for p in payload['projects']] == [root]
        assert payload['unreadable'] == [missing]

    def test_every_root_unreadable_is_not_a_clean_run(self, tmp_path):
        result = _run_cli('--project-root', str(tmp_path / 'no-such-project'))

        assert result.returncode == 3
        assert any(line.startswith('error:') for line in result.stderr.splitlines())

    def test_project_root_is_required(self):
        assert _run_cli().returncode == 2
