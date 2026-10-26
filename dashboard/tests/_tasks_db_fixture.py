"""Real on-disk ``tasks.db`` files for tests that read the task store directly.

The claimant-invariant gauge reads ``<root>/.taskmaster/tasks/tasks.db`` with
sqlite, so its tests seed genuine sqlite files rather than mocking a cursor.
Each file is created from the task store's own DDL, read out of its source:
the dashboard does not depend on fused-memory, and a hand-kept copy would let
these tests pass against a schema production no longer has.
"""

from __future__ import annotations

import ast
import functools
import json
import sqlite3
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

_TASK_STORE_SOURCE = (
    Path(__file__).resolve().parents[2]
    / 'fused-memory/src/fused_memory/backends/sqlite_task_backend.py'
)
_TASK_STORE_DDL_NAME = '_SCHEMA_SQL'

_INSERT_SQL = (
    'INSERT INTO tasks (tag, id, title, status, metadata, updated_at, claimant_run_id, heartbeat_at) '
    'VALUES (?, ?, ?, ?, ?, ?, ?, ?)'
)
_DEFAULT_UPDATED_AT = '2026-10-10T00:00:00+00:00'


@functools.cache
def _task_store_ddl() -> str:
    """``sqlite_task_backend.py::_SCHEMA_SQL``, the DDL the task store creates a fresh DB with."""
    for node in ast.parse(_TASK_STORE_SOURCE.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == _TASK_STORE_DDL_NAME
            for target in node.targets
        ):
            ddl = ast.literal_eval(node.value)
            assert isinstance(ddl, str), f'{_TASK_STORE_DDL_NAME} is not a str literal'
            return ddl
    raise AssertionError(f'{_TASK_STORE_SOURCE} no longer assigns {_TASK_STORE_DDL_NAME}')


def task_row(
    id: int,
    status: str,
    claimant: str | None = None,
    heartbeat: str | None = None,
    *,
    tag: str = 'master',
    metadata: Mapping[str, Any] | str | None = None,
) -> dict[str, Any]:
    return {
        'id': id,
        'status': status,
        'tag': tag,
        'claimant_run_id': claimant,
        'heartbeat_at': heartbeat,
        'metadata': metadata,
    }


def _metadata_text(metadata: Mapping[str, Any] | str | None) -> str | None:
    if metadata is None or isinstance(metadata, str):
        return metadata
    return json.dumps(dict(metadata))


def _insert_values(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row.get('tag', 'master'),
        row['id'],
        row.get('title', f'task {row["id"]}'),
        row['status'],
        _metadata_text(row.get('metadata')),
        row.get('updated_at', _DEFAULT_UPDATED_AT),
        row.get('claimant_run_id'),
        row.get('heartbeat_at'),
    )


def write_tasks_db(path: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    """Create the task store's schema at *path* holding *rows*; a str ``metadata`` is written verbatim."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(_task_store_ddl())
        conn.executemany(_INSERT_SQL, [_insert_values(row) for row in rows])
        conn.commit()
    finally:
        conn.close()
    return path
