"""These tests guard the shared ``make_runs_db`` / ``runs_db_path`` fixtures in scripts/tests/conftest.py."""
import sqlite3
from pathlib import Path

import pytest


def _read(db_path, sql):
    conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def test_make_runs_db_inserts_events_in_list_order_with_ascending_ids(make_runs_db):
    db_path = make_runs_db([
        {'event_type': 'set_to_plan', 'task_id': 7, 'data': {'files': ['m']}},
        {'event_type': 'phase_skipped', 'task_id': 7, 'data': None},
        {'event_type': 'merge_finalized', 'task_id': None, 'data': '{bad'},
    ])

    assert _read(db_path, 'SELECT id, task_id, event_type, data FROM events ORDER BY id') == [
        (1, '7', 'set_to_plan', '{"files": ["m"]}'),
        (2, '7', 'phase_skipped', None),
        (3, None, 'merge_finalized', '{bad'),
    ]


def test_make_runs_db_json_encodes_a_list_payload(make_runs_db):
    db_path = make_runs_db([{'event_type': 'merge_verify', 'data': [1, 2]}])

    assert _read(db_path, 'SELECT data FROM events') == [('[1, 2]',)]


def test_make_runs_db_round_trips_every_explicit_column(make_runs_db):
    db_path = make_runs_db([{
        'event_type': 'merge_verify',
        'timestamp': '2026-08-11T09:30:00+00:00',
        'run_id': 'run-9',
        'task_id': '42',
        'phase': 'merge',
        'role': 'merger',
        'data': {'runner': 'local'},
        'cost_usd': 1.5,
        'duration_ms': 600_000,
    }])

    assert _read(
        db_path,
        'SELECT timestamp, run_id, task_id, event_type, phase, role, data, '
        'cost_usd, duration_ms FROM events',
    ) == [(
        '2026-08-11T09:30:00+00:00',
        'run-9',
        '42',
        'merge_verify',
        'merge',
        'merger',
        '{"runner": "local"}',
        1.5,
        600_000,
    )]


def test_make_runs_db_defaults_run_id_and_leaves_optional_columns_null(make_runs_db):
    db_path = make_runs_db([{'event_type': 'merge_verify'}])

    assert _read(
        db_path, 'SELECT run_id, phase, role, cost_usd, duration_ms FROM events'
    ) == [('run-1', None, None, None, None)]


def test_make_runs_db_default_timestamps_are_distinct_and_ascend_as_text_past_100_rows(
    make_runs_db,
):
    """The scripts under test compare ``timestamp`` as TEXT, so the defaults
    must keep sorting as text well past two-digit row counts."""
    db_path = make_runs_db([{'event_type': 'merge_verify'}] * 120)

    timestamps = [row[0] for row in _read(db_path, 'SELECT timestamp FROM events ORDER BY id')]

    assert len(timestamps) == 120
    assert len(set(timestamps)) == 120
    assert timestamps == sorted(timestamps)


def test_make_runs_db_builds_by_name_and_in_an_explicit_directory(make_runs_db, tmp_path):
    first = make_runs_db(name='a.db')
    second = make_runs_db(name='b.db')

    assert first != second
    assert first.parent == second.parent == tmp_path
    assert first.exists() and second.exists()

    target = tmp_path / 'root' / 'data' / 'orchestrator'
    target.mkdir(parents=True)

    db_path = make_runs_db([{'event_type': 'merge_verify'}], directory=target)

    assert isinstance(db_path, Path)
    assert db_path == target / 'runs.db'
    assert _read(db_path, 'SELECT event_type FROM events') == [('merge_verify',)]


def test_make_runs_db_refuses_to_build_over_an_existing_db(make_runs_db):
    """Re-seeding one path must fail loudly rather than append to its rows."""
    make_runs_db([{'event_type': 'merge_verify'}])

    with pytest.raises(sqlite3.OperationalError):
        make_runs_db([{'event_type': 'merge_verify'}])


def test_runs_db_path_and_make_runs_db_carry_one_schema(runs_db_path, make_runs_db):
    schema_sql = 'SELECT type, name, sql FROM sqlite_master ORDER BY name'

    assert _read(runs_db_path, schema_sql) == _read(make_runs_db(name='seeded.db'), schema_sql)
