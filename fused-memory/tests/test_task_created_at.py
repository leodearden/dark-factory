"""``tasks.created_at`` and ``tasks.created_at_source`` in the task store (task 8558).

PRD ``plans/instruments-honesty-prd.md`` decision 3 and its contract section
"Task creation time". The v5 -> v6 step adds both columns and never invents a
value; ``add_task`` stamps them; the wire carries them when the columns exist.

Every store here is opened through a real ``SqliteTaskBackend`` and read back
with plain sqlite PRAGMAs on the file.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path

import pytest
from shared.project_layout import tasks_db
from shared.task_store_schema import TASK_STORE_DDL

from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.config.schema import TaskmasterConfig
from fused_memory.middleware.candidate_key import compute_candidate_key

_CREATED_AT_COLUMNS = ('created_at', 'created_at_source')


def _row(task_id: int, **columns: object) -> dict[str, object]:
    return {
        'tag': 'master',
        'id': task_id,
        'title': f'task {task_id}',
        'status': 'pending',
        'updated_at': '2026-01-01T00:00:00.000Z',
        **columns,
    }


def _store_without_created_at(
    db_path: Path, *, user_version: int, rows: Sequence[Mapping[str, object]]
) -> None:
    """The real store shape minus the two created_at columns, stamped *user_version*."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(TASK_STORE_DDL)
        for column in _CREATED_AT_COLUMNS:
            conn.execute(f'ALTER TABLE tasks DROP COLUMN {column}')
        for row in rows:
            columns = ', '.join(row)
            placeholders = ', '.join('?' for _ in row)
            conn.execute(
                f'INSERT INTO tasks ({columns}) VALUES ({placeholders})', tuple(row.values())
            )
        conn.execute(f'PRAGMA user_version = {user_version}')
        conn.commit()
    finally:
        conn.close()


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
            if name in _CREATED_AT_COLUMNS
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
    _store_without_created_at(db_path, user_version=5, rows=[_row(1)])

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
    ambiguous = {
        'title': 'Ambiguous task',
        'metadata': json.dumps({'files': ['x.py']}),
        'candidate_key': compute_candidate_key('Ambiguous task', ['x.py']),
    }
    _store_without_created_at(
        db_path,
        user_version=3,
        rows=[_row(1, status='done', **ambiguous), _row(2, status='pending', **ambiguous)],
    )
    flagged_groups: list[object] = []

    async with _backend(tmp_path, flagged_groups) as backend:
        await backend.get_tasks(project_root=str(project_root))

    assert 'ux_tasks_candidate_key' not in _index_names(db_path), (
        'precondition: the mixed-status duplicate group must hold the store below v4'
    )
    assert len(flagged_groups) == 1, flagged_groups
    assert _user_version(db_path) == 3
    assert set(_created_at_column_shapes(db_path)) == set(_CREATED_AT_COLUMNS)


@pytest.mark.asyncio
async def test_a_fresh_store_opened_by_the_backend_lands_at_v6(tmp_path):
    project_root = tmp_path / 'proj'

    async with _backend(tmp_path) as backend:
        await backend.get_tasks(project_root=str(project_root))

    assert _user_version(tasks_db(project_root)) == 6
