"""A store the backend opens has exactly the columns ``shared.task_store_schema.TASK_STORE_DDL`` declares.

Fixtures elsewhere are guarded as subsets of that DDL; this pins the DDL itself to what the backend
really creates, so a migration that adds or renames a column without editing it fails here.
"""

from __future__ import annotations

import sqlite3

import pytest
from _task_store_shapes import store_without_created_at
from shared.project_layout import tasks_db
from shared.task_store_schema import TASK_STORE_DDL

from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.config.schema import TaskmasterConfig

ColumnShape = tuple[str, str, str, int, int]


def _column_shapes(conn: sqlite3.Connection) -> set[ColumnShape]:
    tables = [
        name
        for (name,) in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        )
    ]
    return {
        (table, name, declared_type, notnull, pk)
        for table in tables
        for _cid, name, declared_type, notnull, _default, pk in conn.execute(
            f'PRAGMA table_info("{table}")'
        )
    }


def _ddl_column_shapes() -> set[ColumnShape]:
    conn = sqlite3.connect(':memory:')
    try:
        conn.executescript(TASK_STORE_DDL)
        return _column_shapes(conn)
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_a_store_the_backend_opens_has_exactly_the_columns_task_store_ddl_declares(tmp_path):
    project_root = tmp_path / 'proj'
    backend = SqliteTaskBackend(TaskmasterConfig(project_root=str(tmp_path)))
    await backend.start()
    try:
        await backend.add_task(project_root=str(project_root), title='t')
    finally:
        await backend.close()

    conn = sqlite3.connect(tasks_db(project_root))
    try:
        opened = _column_shapes(conn)
    finally:
        conn.close()
    declared = _ddl_column_shapes()

    assert opened == declared, (
        f'only in the opened store: {sorted(opened - declared)}; '
        f'only in TASK_STORE_DDL: {sorted(declared - opened)}'
    )


@pytest.mark.asyncio
async def test_a_v5_store_the_backend_migrates_ends_with_exactly_the_columns_task_store_ddl_declares(
    tmp_path,
):
    project_root = tmp_path / 'proj'
    db_path = tasks_db(project_root)
    store_without_created_at(db_path, user_version=5)

    backend = SqliteTaskBackend(TaskmasterConfig(project_root=str(tmp_path)))
    await backend.start()
    try:
        await backend.get_tasks(project_root=str(project_root))
    finally:
        await backend.close()

    conn = sqlite3.connect(db_path)
    try:
        migrated = _column_shapes(conn)
    finally:
        conn.close()
    declared = _ddl_column_shapes()

    assert migrated == declared, (
        f'only in the migrated store: {sorted(migrated - declared)}; '
        f'only in TASK_STORE_DDL: {sorted(declared - migrated)}'
    )
