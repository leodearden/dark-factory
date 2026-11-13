"""``tasks.created_at`` and ``tasks.created_at_source`` in the task store (task 8558).

PRD ``plans/instruments-honesty-prd.md`` decision 3 and its contract section
"Task creation time". The v5 -> v6 step adds both columns and never invents a
value; ``add_task`` stamps them; the wire carries them when the columns exist.

Every store here is opened through a real ``SqliteTaskBackend`` and read back
with plain sqlite PRAGMAs on the file.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import re
import sqlite3
from collections.abc import AsyncIterator, Mapping
from pathlib import Path

import pytest
from _task_store_shapes import CREATED_AT_COLUMNS, store_without_created_at
from shared.project_layout import tasks_db
from shared.task_created_at import CreatedAtSource

from fused_memory.backends import sqlite_task_backend
from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.config.schema import TaskmasterConfig
from fused_memory.middleware.candidate_key import compute_candidate_key

_TASK_STORE_TIMESTAMP_FORM = re.compile(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z')


def _row(task_id: int, **columns: object) -> dict[str, object]:
    return {
        'tag': 'master',
        'id': task_id,
        'title': f'task {task_id}',
        'status': 'pending',
        'updated_at': '2026-01-01T00:00:00.000Z',
        **columns,
    }


@contextlib.contextmanager
def _read(db_path: Path):
    conn = sqlite3.connect(db_path)
    try:
        yield conn
    finally:
        conn.close()


def _user_version(db_path: Path) -> int:
    with _read(db_path) as conn:
        return conn.execute('PRAGMA user_version').fetchone()[0]


def _created_at_column_shapes(db_path: Path) -> dict[str, tuple[str, int]]:
    with _read(db_path) as conn:
        return {
            name: (declared_type, notnull)
            for _cid, name, declared_type, notnull, _default, _pk in conn.execute(
                'PRAGMA table_info(tasks)'
            )
            if name in CREATED_AT_COLUMNS
        }


def _created_at_values(db_path: Path) -> dict[int, tuple[str | None, str | None]]:
    with _read(db_path) as conn:
        return {
            task_id: (created_at, source)
            for task_id, created_at, source in conn.execute(
                'SELECT id, created_at, created_at_source FROM tasks'
            )
        }


def _index_names(db_path: Path) -> set[str]:
    with _read(db_path) as conn:
        return {row[1] for row in conn.execute('PRAGMA index_list(tasks)')}


def _store_held_at_v3(db_path: Path) -> None:
    """A v3 store whose mixed-status duplicate group keeps the v3 -> v4 step self-gated."""
    ambiguous = {
        'title': 'Ambiguous task',
        'metadata': json.dumps({'files': ['x.py']}),
        'candidate_key': compute_candidate_key('Ambiguous task', ['x.py']),
    }
    store_without_created_at(
        db_path,
        user_version=3,
        rows=[_row(1, status='done', **ambiguous), _row(2, status='pending', **ambiguous)],
    )


def _created_at_pair(task: Mapping[str, object]) -> dict[str, object]:
    """The created_at keys a wire task carries; an absent key stays absent."""
    return {key: task[key] for key in CREATED_AT_COLUMNS if key in task}


@contextlib.asynccontextmanager
async def _backend(
    tmp_path: Path, flagged_groups: list[object] | None = None
) -> AsyncIterator[SqliteTaskBackend]:
    """A started backend whose residual-duplicate escalations are recorded, never filed."""
    recorded = flagged_groups if flagged_groups is not None else []
    backend = SqliteTaskBackend(
        TaskmasterConfig(project_root=str(tmp_path)),
        residual_dup_escalation_cb=lambda _root, groups: recorded.append(groups),
    )
    await backend.start()
    try:
        yield backend
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_a_v5_store_migrates_to_v6_and_the_open_invents_no_value(tmp_path):
    project_root = tmp_path / 'proj'
    db_path = tasks_db(project_root)
    store_without_created_at(db_path, user_version=5, rows=[_row(1)])

    async with _backend(tmp_path) as backend:
        await backend.get_tasks(project_root=str(project_root))

    assert _user_version(db_path) == 6
    assert _created_at_column_shapes(db_path) == {
        'created_at': ('TEXT', 0),
        'created_at_source': ('TEXT', 0),
    }
    assert _created_at_values(db_path) == {1: (None, None)}


@pytest.mark.asyncio
async def test_a_store_held_at_v3_by_the_candidate_key_self_gate_still_gains_both_columns(
    tmp_path,
):
    project_root = tmp_path / 'proj'
    db_path = tasks_db(project_root)
    _store_held_at_v3(db_path)
    flagged_groups: list[object] = []

    async with _backend(tmp_path, flagged_groups) as backend:
        await backend.get_tasks(project_root=str(project_root))

    assert 'ux_tasks_candidate_key' not in _index_names(db_path), (
        'precondition: the mixed-status duplicate group must hold the store below v4'
    )
    assert len(flagged_groups) == 1, flagged_groups
    assert _user_version(db_path) == 3
    assert set(_created_at_column_shapes(db_path)) == set(CREATED_AT_COLUMNS)


@pytest.mark.asyncio
async def test_a_fresh_store_opened_by_the_backend_lands_at_v6(tmp_path):
    project_root = tmp_path / 'proj'

    async with _backend(tmp_path) as backend:
        await backend.get_tasks(project_root=str(project_root))

    assert _user_version(tasks_db(project_root)) == 6


@pytest.mark.asyncio
async def test_a_new_pending_task_is_stamped_with_the_instant_its_anchor_and_updated_at_carry(
    tmp_path,
):
    project_root = str(tmp_path / 'proj')

    async with _backend(tmp_path) as backend:
        added = await backend.add_task(project_root=project_root, title='new', status='pending')
        task = await backend.get_task(added['id'], project_root)

    assert _TASK_STORE_TIMESTAMP_FORM.fullmatch(task['created_at']), task['created_at']
    assert task['created_at'] == task['metadata']['pending_since'] == task['updatedAt']
    assert task['created_at_source'] == CreatedAtSource.STAMPED


@pytest.mark.asyncio
async def test_a_deferred_insert_is_stamped_and_no_later_write_rewrites_its_filing_time(
    tmp_path, monkeypatch,
):
    readings = (f'2026-01-01T00:00:{second:02d}.000Z' for second in itertools.count())
    monkeypatch.setattr(sqlite_task_backend, 'task_timestamp_now', lambda: next(readings))
    project_root = str(tmp_path / 'proj')
    db_path = tasks_db(project_root)

    async with _backend(tmp_path) as backend:
        added = await backend.add_task(project_root=project_root, title='planned', status='deferred')
        task_id = int(added['id'])
        filed = _created_at_values(db_path)[task_id]
        await backend.set_task_status(added['id'], 'pending', project_root)
        await backend.update_task(added['id'], project_root, title='planned, renamed')
        task = await backend.get_task(added['id'], project_root)

    assert filed == ('2026-01-01T00:00:00.000Z', CreatedAtSource.STAMPED)
    assert (task['status'], task['title']) == ('pending', 'planned, renamed')
    assert task['updatedAt'] > filed[0], 'precondition: the later writes read a later clock'
    assert _created_at_values(db_path)[task_id] == filed


@pytest.mark.asyncio
async def test_a_migrated_row_not_yet_backfilled_is_served_with_both_keys_null(tmp_path):
    project_root = tmp_path / 'proj'
    store_without_created_at(tasks_db(project_root), user_version=5, rows=[_row(1)])

    async with _backend(tmp_path) as backend:
        rows = (await backend.get_tasks(project_root=str(project_root)))['tasks']

    assert [_created_at_pair(row) for row in rows] == [
        {'created_at': None, 'created_at_source': None},
    ]


@pytest.mark.asyncio
async def test_a_store_held_at_v3_accepts_add_task_and_stamps_the_new_row(tmp_path):
    project_root = tmp_path / 'proj'
    db_path = tasks_db(project_root)
    _store_held_at_v3(db_path)

    async with _backend(tmp_path) as backend:
        added = await backend.add_task(project_root=str(project_root), title='filed while degraded')
        task = await backend.get_task(added['id'], str(project_root))

    assert _TASK_STORE_TIMESTAMP_FORM.fullmatch(task['created_at']), task['created_at']
    assert task['created_at_source'] == CreatedAtSource.STAMPED
    assert _user_version(db_path) == 3


@pytest.mark.asyncio
async def test_a_store_whose_columns_never_landed_is_served_without_the_keys(tmp_path):
    project_root = tmp_path / 'proj'
    store_without_created_at(tasks_db(project_root), user_version=6, rows=[_row(1)])

    async with _backend(tmp_path) as backend:
        rows = (await backend.get_tasks(project_root=str(project_root)))['tasks']
        task = await backend.get_task('1', str(project_root))
        projected = await backend.get_tasks(
            project_root=str(project_root), fields=['created_at', 'created_at_source'],
        )

    assert [_created_at_pair(row) for row in rows] == [{}]
    assert _created_at_pair(task) == {}
    assert projected['tasks'] == [{'id': '1'}]
