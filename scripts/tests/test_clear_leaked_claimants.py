"""Tests for scripts/clear_leaked_claimants.py (task 4625).

Every expectation is hand-derived from the tier table in
docs/prds/claimant-invariant-enforcement.md D3, never computed by calling the
shared predicates, and every store is a synthetic temp tasks.db.
"""
from __future__ import annotations

import asyncio
import itertools
import json
import sqlite3
import subprocess
import sys
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from _task_db_scan import UNREADABLE_STORE_ERRORS
from clear_leaked_claimants import (
    STILL_VIOLATING_DETAIL,
    Bucket,
    Tier,
    apply_repairs,
    take_census,
)
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


def _store_path(root: str) -> Path:
    return Path(root) / '.taskmaster' / 'tasks' / 'tasks.db'


def _store_rows(root: str) -> list[tuple]:
    with closing(sqlite3.connect(f'file:{_store_path(root)}?mode=ro', uri=True)) as conn:
        return conn.execute('SELECT * FROM tasks ORDER BY tag, id').fetchall()


def _claimants(root: str) -> dict[int, tuple[str | None, str | None]]:
    with closing(sqlite3.connect(f'file:{_store_path(root)}?mode=ro', uri=True)) as conn:
        rows = conn.execute('SELECT id, claimant_run_id, heartbeat_at FROM tasks')
        return {task_id: (claimant, heartbeat) for task_id, claimant, heartbeat in rows}


def _mutate_store(root: str, sql: str, params: tuple = ()) -> None:
    with closing(sqlite3.connect(_store_path(root))) as conn:
        conn.execute(sql, params)
        conn.commit()


def _seeded_claimant(row: dict) -> tuple[str | None, str | None]:
    return (row['claimant_run_id'], row['heartbeat_at'])


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


class _FakeClaimantServer:
    """Answers set_task_claimant by really writing the named project's store, as the backend does."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.error_replies: dict[int, dict] = {}
        self.raises: dict[int, Exception] = {}
        self.ack_without_write: set[int] = set()

    async def call_tool(self, name: str, arguments: dict) -> dict:
        self.calls.append((name, dict(arguments)))
        if name != 'set_task_claimant':
            raise AssertionError(f'unexpected tool call: {name}')
        task_id = int(arguments['id'])
        if task_id in self.raises:
            raise self.raises[task_id]
        if task_id in self.error_replies:
            return self.error_replies[task_id]
        if task_id not in self.ack_without_write:
            _mutate_store(
                arguments['project_root'],
                'UPDATE tasks SET claimant_run_id = ?, heartbeat_at = ? WHERE tag = ? AND id = ?',
                (arguments['claimant_run_id'], arguments['heartbeat_at'], arguments['tag'], task_id),
            )
        return {'id': arguments['id'], 'message': f"Updated claimant fields for task {arguments['id']}"}

    def written_ids(self) -> set[int]:
        return {int(arguments['id']) for _, arguments in self.calls}


TERMINAL_ONLY = frozenset({Tier.TERMINAL})
BOTH_TIERS = frozenset({Tier.TERMINAL, Tier.HYGIENE})
TERMINAL_IDS = {4028, 5225, 101}
HYGIENE_IDS = {103, 109}


def _apply(census, client, tiers=TERMINAL_ONLY, clock=lambda: NOW):
    return asyncio.run(apply_repairs(census, client, tiers=tiers, clock=clock))


def _buckets(report) -> dict[int, Bucket]:
    return {outcome.violation.task_id: outcome.bucket for outcome in report.outcomes}


class TestApplyRepairs:
    def test_clears_each_terminal_row_with_one_set_task_claimant_call(self, seed_root):
        root = seed_root(TIER_TABLE_ROWS)
        server = _FakeClaimantServer()

        _apply(take_census(root, NOW), server)

        calls = sorted(server.calls, key=lambda call: int(call[1]['id']))
        assert calls == [
            ('set_task_claimant', {
                'id': str(task_id), 'tag': 'master', 'project_root': root,
                'claimant_run_id': None, 'heartbeat_at': None,
            })
            for task_id in sorted(TERMINAL_IDS)
        ]

    def test_terminal_rows_end_cleared_and_repaired_and_hygiene_rows_untouched(self, seed_root):
        root = seed_root(TIER_TABLE_ROWS)

        report = _apply(take_census(root, NOW), _FakeClaimantServer())

        live = _claimants(root)
        assert all(live[task_id] == (None, None) for task_id in TERMINAL_IDS)
        assert live[103] == _seeded_claimant(STALE_PENDING)
        assert live[109] == _seeded_claimant(STALE_MERGE_DEFERRED)
        assert _buckets(report) == {task_id: Bucket.REPAIRED for task_id in TERMINAL_IDS}

    def test_verification_census_shows_no_terminal_violation_and_keeps_the_hygiene_rows(
        self, seed_root,
    ):
        root = seed_root(TIER_TABLE_ROWS)

        report = _apply(take_census(root, NOW), _FakeClaimantServer())

        verification = report.verification_census
        assert verification.totals()['by_tier'][Tier.TERMINAL] == 0
        assert {v.task_id for v in verification.violations} == HYGIENE_IDS

    def test_freeform_claimant_with_no_heartbeat_is_repaired_not_skipped(self, seed_root):
        root = seed_root([FREEFORM_CANCELLED])

        report = _apply(take_census(root, NOW), _FakeClaimantServer())

        assert _buckets(report) == {5225: Bucket.REPAIRED}
        assert _claimants(root)[5225] == (None, None)

    def test_hygiene_tier_is_cleared_only_when_selected(self, seed_root):
        root = seed_root(TIER_TABLE_ROWS)
        server = _FakeClaimantServer()

        report = _apply(take_census(root, NOW), server, tiers=BOTH_TIERS)

        assert server.written_ids() == TERMINAL_IDS | HYGIENE_IDS
        assert _buckets(report) == {
            task_id: Bucket.REPAIRED for task_id in TERMINAL_IDS | HYGIENE_IDS
        }
        live = _claimants(root)
        assert live[103] == (None, None)
        assert live[109] == (None, None)
        for row in TIER_TABLE_ROWS:
            if row['id'] not in TERMINAL_IDS | HYGIENE_IDS:
                assert live[row['id']] == _seeded_claimant(row)

    def test_refuses_to_run_without_a_client(self, seed_root):
        census = take_census(seed_root(TIER_TABLE_ROWS), NOW)

        with pytest.raises(ValueError):
            _apply(census, None)


def _outcome_for(report, task_id: int):
    (outcome,) = [o for o in report.outcomes if o.violation.task_id == task_id]
    return outcome


class TestPerRowCorroboration:
    def test_a_row_reclaimed_after_the_census_keeps_its_live_claim(self, seed_root):
        root = seed_root(TIER_TABLE_ROWS)
        census = take_census(root, NOW)
        live_claim = ('run-new/4028-y/pid=2', _iso(NOW))
        _mutate_store(
            root,
            "UPDATE tasks SET status = 'in-progress', claimant_run_id = ?, heartbeat_at = ? WHERE id = 4028",
            live_claim,
        )
        server = _FakeClaimantServer()

        report = _apply(census, server)

        assert 4028 not in server.written_ids()
        outcome = _outcome_for(report, 4028)
        assert outcome.bucket is Bucket.SKIPPED_CHANGED_UNDER_US
        assert outcome.delta['status'] == ('done', 'in-progress')
        assert outcome.delta['claimant_run_id'] == (POSITIVE_CONTROL['claimant_run_id'], live_claim[0])
        assert _claimants(root)[4028] == live_claim
        assert _buckets(report)[5225] is Bucket.REPAIRED
        assert _buckets(report)[101] is Bucket.REPAIRED

    def test_a_restamped_claimant_on_the_same_status_is_skipped(self, seed_root):
        root = seed_root(TIER_TABLE_ROWS)
        census = take_census(root, NOW)
        _mutate_store(root, "UPDATE tasks SET claimant_run_id = 'run-other/101-z/pid=4' WHERE id = 101")
        server = _FakeClaimantServer()

        report = _apply(census, server)

        assert 101 not in server.written_ids()
        outcome = _outcome_for(report, 101)
        assert outcome.bucket is Bucket.SKIPPED_CHANGED_UNDER_US
        assert outcome.delta['claimant_run_id'] == (FRESH_DONE['claimant_run_id'], 'run-other/101-z/pid=4')

    def test_a_refreshed_hygiene_heartbeat_declines_the_write(self, seed_root):
        root = seed_root(TIER_TABLE_ROWS)
        census = take_census(root, NOW)
        _mutate_store(root, 'UPDATE tasks SET heartbeat_at = ? WHERE id = 103', (_iso(FRESH),))
        server = _FakeClaimantServer()

        report = _apply(census, server, tiers=BOTH_TIERS)

        assert 103 not in server.written_ids()
        outcome = _outcome_for(report, 103)
        assert outcome.bucket is Bucket.SKIPPED_CHANGED_UNDER_US
        assert outcome.delta['heartbeat_at'] == (STALE_PENDING['heartbeat_at'], _iso(FRESH))
        assert outcome.delta['tier'] == ('hygiene', None)

    def test_a_row_deleted_after_the_census_is_skipped_as_absent(self, seed_root):
        root = seed_root(TIER_TABLE_ROWS)
        census = take_census(root, NOW)
        _mutate_store(root, 'DELETE FROM tasks WHERE id = 4028')
        server = _FakeClaimantServer()

        report = _apply(census, server)

        assert 4028 not in server.written_ids()
        outcome = _outcome_for(report, 4028)
        assert outcome.bucket is Bucket.SKIPPED_CHANGED_UNDER_US
        assert outcome.delta == {'row': ('present', 'absent')}

    def test_an_infra_hold_flag_set_after_the_census_declines_the_write(self, seed_root):
        root = seed_root(TIER_TABLE_ROWS)
        census = take_census(root, NOW)
        _mutate_store(root, 'UPDATE tasks SET metadata = ? WHERE id = 103', ('{"infra_hold": true}',))
        server = _FakeClaimantServer()

        report = _apply(census, server, tiers=BOTH_TIERS)

        assert 103 not in server.written_ids()
        outcome = _outcome_for(report, 103)
        assert outcome.bucket is Bucket.SKIPPED_CHANGED_UNDER_US
        assert outcome.delta == {'tier': ('hygiene', None)}


ARRIVAL_CLAIM = ('run-z/9001-q/pid=9', _iso(NOW - timedelta(days=3)))


def _insert_arrival(root: str) -> None:
    _mutate_store(
        root,
        'INSERT INTO tasks (tag, id, title, status, updated_at, claimant_run_id, heartbeat_at) '
        "VALUES ('master', 9001, 'arrival', 'done', ?, ?, ?)",
        (_iso(NOW), *ARRIVAL_CLAIM),
    )


def _ticking_clock():
    ticks = (NOW + timedelta(seconds=n) for n in itertools.count())
    return lambda: next(ticks)


def _reclaim_4028(root: str) -> None:
    _mutate_store(
        root,
        "UPDATE tasks SET status = 'in-progress', claimant_run_id = ?, heartbeat_at = ? WHERE id = 4028",
        ('run-new/4028-y/pid=2', _iso(NOW)),
    )


class TestBucketAccounting:
    def test_a_violation_that_arrived_after_the_census_is_reported_not_written(self, seed_root):
        root = seed_root(TIER_TABLE_ROWS)
        census = take_census(root, NOW)
        _insert_arrival(root)
        server = _FakeClaimantServer()

        report = _apply(census, server, clock=_ticking_clock())

        assert 9001 not in server.written_ids()
        arrival = _outcome_for(report, 9001)
        assert arrival.bucket is Bucket.ARRIVED_DURING_WINDOW
        assert arrival.violation.measured_at == report.verification_census.measured_at
        assert arrival.violation.measured_at > NOW
        assert _claimants(root)[9001] == ARRIVAL_CLAIM

    def test_a_rejected_write_is_failed_with_the_server_error(self, seed_root):
        root = seed_root(TIER_TABLE_ROWS)
        server = _FakeClaimantServer()
        server.error_replies[5225] = {'error': 'boom', 'error_type': 'TaskmasterError'}

        report = _apply(take_census(root, NOW), server)

        rejected = _outcome_for(report, 5225)
        assert rejected.bucket is Bucket.FAILED
        assert 'TaskmasterError' in rejected.detail
        assert _buckets(report)[4028] is Bucket.REPAIRED
        assert _buckets(report)[101] is Bucket.REPAIRED

    def test_a_raised_write_is_failed_and_the_batch_continues(self, seed_root):
        root = seed_root(TIER_TABLE_ROWS)
        server = _FakeClaimantServer()
        server.raises[4028] = OSError('connection reset')

        report = _apply(take_census(root, NOW), server)

        raised = _outcome_for(report, 4028)
        assert raised.bucket is Bucket.FAILED
        assert 'OSError' in raised.detail
        assert 5225 in server.written_ids()
        assert _buckets(report)[5225] is Bucket.REPAIRED

    def test_an_acknowledged_write_that_did_not_stick_is_failed(self, seed_root):
        root = seed_root(TIER_TABLE_ROWS)
        server = _FakeClaimantServer()
        server.ack_without_write.add(101)

        report = _apply(take_census(root, NOW), server)

        unstuck = _outcome_for(report, 101)
        assert unstuck.bucket is Bucket.FAILED
        assert unstuck.detail == STILL_VIOLATING_DETAIL

    def test_every_row_lands_in_exactly_one_bucket_and_no_repaired_row_still_violates(
        self, seed_root,
    ):
        root = seed_root(TIER_TABLE_ROWS)
        census = take_census(root, NOW)
        _insert_arrival(root)
        _reclaim_4028(root)
        server = _FakeClaimantServer()
        server.error_replies[5225] = {'error': 'boom', 'error_type': 'TaskmasterError'}

        report = _apply(census, server, clock=_ticking_clock())

        keys = [outcome.violation.key for outcome in report.outcomes]
        assert len(keys) == len(set(keys))
        selected = {v.key for v in census.violations if v.tier is Tier.TERMINAL}
        remaining = {
            v.key for v in report.verification_census.violations if v.tier is Tier.TERMINAL
        }
        assert set(keys) == selected | remaining
        by_bucket = {
            bucket: {o.violation.key for o in report.outcomes if o.bucket is bucket}
            for bucket in Bucket
        }
        arrived = by_bucket[Bucket.ARRIVED_DURING_WINDOW]
        assert not arrived & (by_bucket[Bucket.REPAIRED] | by_bucket[Bucket.SKIPPED_CHANGED_UNDER_US])
        assert not by_bucket[Bucket.REPAIRED] & remaining
        assert _buckets(report) == {
            101: Bucket.REPAIRED,
            4028: Bucket.SKIPPED_CHANGED_UNDER_US,
            5225: Bucket.FAILED,
            9001: Bucket.ARRIVED_DURING_WINDOW,
        }


def _write_scheduler_state(root: str, text: str) -> None:
    state = Path(root) / 'data' / 'orchestrator' / 'scheduler_state.json'
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(text)


def _holders(**holders: str) -> str:
    return json.dumps({'current_holders': holders, 'parks': {}})


class TestHolderPreflight:
    def test_a_selected_row_held_by_the_scheduler_refuses_the_whole_project(self, seed_root):
        root = seed_root(TIER_TABLE_ROWS)
        _write_scheduler_state(root, _holders(orchestrator_src='4028'))
        census = take_census(root, NOW)
        before = _store_rows(root)
        server = _FakeClaimantServer()

        report = _apply(census, server)

        assert report.refused is True
        assert report.holder_intersection == (4028,)
        assert server.calls == []
        assert _store_rows(root) == before
        assert report.outcomes == ()
        assert report.verification_census is None

    def test_holders_outside_the_selected_rows_do_not_refuse(self, seed_root):
        root = seed_root(TIER_TABLE_ROWS)
        _write_scheduler_state(root, _holders(scripts='103', docs='7777'))
        server = _FakeClaimantServer()

        report = _apply(take_census(root, NOW), server)

        assert report.refused is False
        assert report.holder_intersection == ()
        assert server.written_ids() == TERMINAL_IDS

    def test_no_scheduler_snapshot_means_no_holders(self, seed_root):
        root = seed_root(TIER_TABLE_ROWS)
        server = _FakeClaimantServer()

        report = _apply(take_census(root, NOW), server)

        assert report.refused is False
        assert server.written_ids() == TERMINAL_IDS

    def test_holders_are_read_at_apply_time_not_census_time(self, seed_root):
        root = seed_root(TIER_TABLE_ROWS)
        census = take_census(root, NOW)
        _write_scheduler_state(root, _holders(shared_src='5225'))
        server = _FakeClaimantServer()

        report = _apply(census, server)

        assert report.refused is True
        assert report.holder_intersection == (5225,)
        assert server.calls == []

    @pytest.mark.parametrize('snapshot', ['[]', '{"current_holders": ["4028"]}'])
    def test_a_malformed_snapshot_is_treated_as_no_holders(self, seed_root, snapshot):
        root = seed_root(TIER_TABLE_ROWS)
        _write_scheduler_state(root, snapshot)
        server = _FakeClaimantServer()

        report = _apply(take_census(root, NOW), server)

        assert report.refused is False
        assert server.written_ids() == TERMINAL_IDS
