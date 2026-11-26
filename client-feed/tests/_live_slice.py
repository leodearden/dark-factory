"""A bounded, read-only slice of a live factory root's stores, copied under a fixture root.

The slice is one PRD's master tasks with their dependencies, events, task results, verify archives and open
escalation records. Every live database is attached through a `file:...?mode=ro` URI, so no live store is written.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Final

from _stores import EscalationQueueDir, RunsDb, TasksDb
from shared.project_layout import escalation_queue_dir, runs_db, tasks_db, verify_logs_dir

_TAG: Final = 'master'
_AMONG_IDS: Final = 'IN (SELECT value FROM json_each(:ids))'


def _read_only_uri(path: Path) -> str:
    return f'{path.as_uri()}?mode=ro'


@contextmanager
def _with_live_attached(fixture: Path, live: Path) -> Iterator[sqlite3.Connection]:
    """A connection to the fixture database with `live` attached read-only as schema `live`, committed on exit.

    Its temporary files stay in memory: SQLite's first temp-dir candidate, /var/tmp, can pass access() and still
    refuse the open under a sandbox, which fails a statement as 'unable to open database file'.
    """
    with closing(sqlite3.connect(fixture.as_uri(), uri=True)) as conn:
        conn.execute('PRAGMA temp_store = MEMORY')
        conn.execute('ATTACH DATABASE ? AS live', (_read_only_uri(live),))
        with conn:
            yield conn


def _shared_columns(conn: sqlite3.Connection, table: str, *, leave_out: frozenset[str] = frozenset()) -> str:
    """The columns `table` has in both the fixture and the live schema, in fixture order, as a SQL list."""
    live = {row[1] for row in conn.execute(f'PRAGMA live.table_info({table})')}
    fixture = [row[1] for row in conn.execute(f'PRAGMA main.table_info({table})')]
    return ', '.join(name for name in fixture if name in live and name not in leave_out)


def _copy_tasks(conn: sqlite3.Connection, *, prd_path: str, marker: str) -> tuple[int, ...]:
    columns = _shared_columns(conn, 'tasks')
    conn.execute(
        f'INSERT INTO main.tasks ({columns}) SELECT {columns} FROM live.tasks'
        " WHERE tag = :tag AND json_valid(metadata) AND json_extract(metadata, '$.prd_path') = :prd_path",
        {'tag': _TAG, 'prd_path': prd_path},
    )
    ids = tuple(row[0] for row in conn.execute('SELECT id FROM main.tasks ORDER BY id'))
    conn.execute(
        'INSERT INTO main.dependencies (tag, task_id, depends_on)'
        f' SELECT tag, task_id, depends_on FROM live.dependencies WHERE tag = :tag AND task_id {_AMONG_IDS}',
        {'tag': _TAG, 'ids': json.dumps(ids)},
    )
    conn.execute(
        "UPDATE main.tasks SET description = coalesce(description, '') || ' ' || :marker,"
        " details = coalesce(details, '') || ' ' || :marker,"
        " test_strategy = coalesce(test_strategy, '') || ' ' || :marker,"
        " metadata = json_set(CASE WHEN json_valid(metadata) THEN metadata ELSE '{}' END, '$.x_canary', :marker)",
        {'marker': marker},
    )
    return ids


def _copy_runs(conn: sqlite3.Connection, ids: tuple[int, ...]) -> None:
    among = {'ids': json.dumps([str(task_id) for task_id in ids])}
    events = _shared_columns(conn, 'events', leave_out=frozenset({'id'}))
    conn.execute(
        f'INSERT INTO main.events ({events}) SELECT {events} FROM live.events'
        f' WHERE task_id {_AMONG_IDS} ORDER BY id',
        among,
    )
    results = _shared_columns(conn, 'task_results')
    conn.execute(
        f'INSERT INTO main.task_results ({results}) SELECT {results} FROM live.task_results'
        f' WHERE task_id {_AMONG_IDS}',
        among,
    )
    runs = _shared_columns(conn, 'runs')
    conn.execute(
        f'INSERT OR IGNORE INTO main.runs ({runs}) SELECT {runs} FROM live.runs'
        ' WHERE run_id IN (SELECT run_id FROM main.task_results)'
    )


def _copy_verify_logs(live: Path, root: Path, ids: tuple[int, ...]) -> None:
    for task_id in ids:
        archive = verify_logs_dir(live) / str(task_id)
        if archive.is_dir():
            shutil.copytree(archive, verify_logs_dir(root) / str(task_id), symlinks=True)


def _record_task_id(path: Path) -> str | None:
    try:
        record = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    return str(record.get('task_id')) if isinstance(record, dict) else None


def _copy_escalations(live: Path, queue: EscalationQueueDir, ids: tuple[int, ...]) -> None:
    wanted = {str(task_id) for task_id in ids}
    live_queue = escalation_queue_dir(live)
    if not live_queue.is_dir():
        return
    for path in sorted(live_queue.glob('*.json')):
        if _record_task_id(path) in wanted:
            queue.write_raw(path.name, path.read_text())


def copy_live_slice(live: Path, root: Path, *, prd_path: str, marker: str) -> tuple[int, ...]:
    """Copy `prd_path`'s slice of `live` under `root`, `marker` appended to every hidden task field; the task ids.

    No ids, and nothing copied, when `live` holds no master task under `prd_path`.
    """
    tasks, runs = TasksDb.create(root), RunsDb.create(root)
    queue = EscalationQueueDir.create(root)
    with _with_live_attached(tasks.path, tasks_db(live)) as conn:
        ids = _copy_tasks(conn, prd_path=prd_path, marker=marker)
    if not ids:
        return ()
    if runs_db(live).is_file():
        with _with_live_attached(runs.path, runs_db(live)) as conn:
            _copy_runs(conn, ids)
    _copy_verify_logs(live, root, ids)
    _copy_escalations(live, queue, ids)
    return ids


def live_titles(live: Path, ids: tuple[int, ...]) -> dict[int, str]:
    """The live master titles of `ids`, read back read-only."""
    with closing(sqlite3.connect(_read_only_uri(tasks_db(live)), uri=True)) as conn:
        rows = conn.execute(
            f'SELECT id, title FROM tasks WHERE tag = :tag AND id {_AMONG_IDS}',
            {'tag': _TAG, 'ids': json.dumps(ids)},
        )
        return dict(rows.fetchall())
