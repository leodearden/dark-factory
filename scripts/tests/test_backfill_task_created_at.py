"""Tests for scripts/backfill_task_created_at.py (task 8558).

PRD ``plans/instruments-honesty-prd.md`` decision 3 and boundary row 7: the
backfill fills only rows whose ``created_at`` is NULL, from the task's first
runs.db event, else its un-back-filled ``metadata.pending_since``, else its
``updated_at``; it prints per-source counts and a second run writes nothing.

Each test builds a project root holding a real-shape tasks.db (from
``shared.task_store_schema.TASK_STORE_DDL``) and a runs.db (from conftest's
``make_runs_db``), drives the CLI seam ``main``, and reads the store back.

To run just this suite:

    uv run --project shared pytest scripts/tests/test_backfill_task_created_at.py -q
"""
from __future__ import annotations

import contextlib
import json
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path

import backfill_task_created_at as backfill
import pytest
from shared.project_layout import runs_db, tasks_db
from shared.task_store_schema import TASK_STORE_DDL

_ROW_3_UPDATED_AT = '2026-05-03T00:00:00.000Z'

_ROWS: tuple[Mapping[str, object], ...] = (
    {'id': 1, 'updated_at': '2026-04-20T00:00:00.000Z'},
    {'id': 2, 'metadata': json.dumps({'pending_since': '2026-05-01T00:00:00.000Z'})},
    {
        'id': 3,
        'metadata': json.dumps(
            {'pending_since': '2026-05-02T00:00:00.000Z', 'pending_since_backfilled': True}
        ),
        'updated_at': _ROW_3_UPDATED_AT,
    },
    {'id': 4, 'updated_at': '2026-05-10T19:04:47.522233+00:00'},
    {'id': 5, 'updated_at': ''},
    {
        'id': 6,
        'created_at': '2026-11-05T00:00:00.000Z',
        'created_at_source': 'stamped',
    },
)

_EVENTS = (
    {'task_id': 1, 'event_type': 'task_started', 'timestamp': '2026-04-10T09:00:00+00:00'},
    # Sorts first as TEXT, but names 20:00 UTC: a TEXT MIN would pick it.
    {'task_id': 1, 'event_type': 'task_started', 'timestamp': '2026-04-09T15:00:00-05:00'},
    {'task_id': 1, 'event_type': 'task_started', 'timestamp': '2026-04-09T16:53:21.119808+00:00'},
    {'task_id': 2, 'event_type': 'task_started', 'timestamp': 'not a time'},
    {'task_id': 6, 'event_type': 'task_started', 'timestamp': '2026-01-01T00:00:00+00:00'},
    {'task_id': 'main-sweep-x', 'event_type': 'sweep', 'timestamp': '2026-01-01T00:00:00+00:00'},
    {'task_id': None, 'event_type': 'sweep', 'timestamp': '2026-01-01T00:00:00+00:00'},
)

_FILLED = {
    1: ('2026-04-09T16:53:21.119Z', 'first_event'),
    2: ('2026-05-01T00:00:00.000Z', 'pending_since'),
    3: (_ROW_3_UPDATED_AT, 'updated_at'),
    4: ('2026-05-10T19:04:47.522Z', 'updated_at'),
    5: (None, None),
    6: ('2026-11-05T00:00:00.000Z', 'stamped'),
}


def _build_tasks_db(
    path: Path, rows: Sequence[Mapping[str, object]], *, dropped: Sequence[str] = ()
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.closing(sqlite3.connect(path)) as conn:
        conn.executescript(TASK_STORE_DDL)
        for column in dropped:
            conn.execute(f'ALTER TABLE tasks DROP COLUMN {column}')
        for row in rows:
            full = {
                'tag': 'master',
                'title': f'task {row["id"]}',
                'status': 'pending',
                'updated_at': '2026-04-01T00:00:00.000Z',
                **row,
            }
            conn.execute(
                f'INSERT INTO tasks ({", ".join(full)}) VALUES ({", ".join("?" for _ in full)})',
                tuple(full.values()),
            )
        conn.commit()


@pytest.fixture
def project(tmp_path: Path, make_runs_db) -> Path:
    root = tmp_path / 'project'
    _build_tasks_db(tasks_db(root), _ROWS)
    runs_db(root).parent.mkdir(parents=True)
    make_runs_db(_EVENTS, directory=runs_db(root).parent)
    return root


def _all_rows(root: Path) -> list[tuple[object, ...]]:
    with contextlib.closing(sqlite3.connect(tasks_db(root))) as conn:
        return conn.execute('SELECT * FROM tasks ORDER BY tag, id').fetchall()


def _created_at(root: Path) -> dict[int, tuple[str | None, str | None]]:
    with contextlib.closing(sqlite3.connect(tasks_db(root))) as conn:
        return {
            task_id: (created_at, source)
            for task_id, created_at, source in conn.execute(
                'SELECT id, created_at, created_at_source FROM tasks ORDER BY id'
            )
        }


def _report(capsys, root: Path, *flags: str) -> dict[str, object]:
    assert backfill.main(['--project-root', str(root), *flags]) == 0
    return json.loads(capsys.readouterr().out)


def _refusal(capsys, root: Path) -> str:
    assert backfill.main(['--project-root', str(root)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ''
    return captured.err


def test_the_first_run_fills_each_null_row_from_its_best_source(capsys, project):
    report = _report(capsys, project)

    assert report == {
        'project_root': str(project),
        'tasks_db': str(tasks_db(project)),
        'runs_db': 'present',
        'dry_run': False,
        'counts': {'first_event': 1, 'pending_since': 1, 'updated_at': 2},
        'skipped_unusable_updated_at': 1,
        'unparseable_event_timestamps': 1,
        'rows_written': 4,
    }
    assert _created_at(project) == _FILLED


def test_a_second_run_writes_nothing(capsys, project):
    _report(capsys, project)
    after_first = _all_rows(project)

    report = _report(capsys, project)

    assert report['counts'] == {'first_event': 0, 'pending_since': 0, 'updated_at': 0}
    assert report['rows_written'] == 0
    assert _all_rows(project) == after_first


def test_a_dry_run_reports_the_same_counts_and_writes_nothing(capsys, project):
    before = _all_rows(project)

    report = _report(capsys, project, '--dry-run')

    assert report['dry_run'] is True
    assert report['counts'] == {'first_event': 1, 'pending_since': 1, 'updated_at': 2}
    assert _all_rows(project) == before


def test_without_a_runs_db_every_row_falls_through_to_its_metadata_or_updated_at(
    capsys, project
):
    runs_db(project).unlink()

    report = _report(capsys, project)

    assert report['runs_db'] == 'absent'
    assert report['counts'] == {'first_event': 0, 'pending_since': 1, 'updated_at': 3}
    assert _created_at(project)[1] == ('2026-04-20T00:00:00.000Z', 'updated_at')


def test_a_store_fused_memory_has_not_migrated_is_refused_untouched(capsys, tmp_path):
    root = tmp_path / 'project'
    _build_tasks_db(tasks_db(root), _ROWS[:1], dropped=('created_at', 'created_at_source'))
    before = tasks_db(root).read_bytes()

    err = _refusal(capsys, root)

    assert 'created_at' in err
    assert 'get_tasks' in err
    assert tasks_db(root).read_bytes() == before


def test_a_project_root_without_a_task_store_is_refused_and_no_file_is_created(
    capsys, tmp_path
):
    root = tmp_path / 'project'
    root.mkdir()

    _refusal(capsys, root)

    assert list(root.iterdir()) == []


def test_a_runs_db_that_is_not_a_database_is_refused_rather_than_skipped(capsys, project):
    runs_db(project).write_bytes(b'not a sqlite database, only some bytes ' * 4)
    before = _all_rows(project)

    _refusal(capsys, project)

    assert _all_rows(project) == before
