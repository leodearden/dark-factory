"""Builders for task stores in the shapes older schema versions left on disk.

Each builder starts from ``shared.task_store_schema.TASK_STORE_DDL`` and removes
what a later step added, so a new column in the DDL reaches every older shape
without an edit here.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path

from shared.task_store_schema import TASK_STORE_DDL

CREATED_AT_COLUMNS = ('created_at', 'created_at_source')


def store_without_created_at(
    db_path: Path, *, user_version: int, rows: Sequence[Mapping[str, object]] = ()
) -> None:
    """The real store shape minus the v6 created_at pair, holding *rows*, stamped *user_version*."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(TASK_STORE_DDL)
        for column in CREATED_AT_COLUMNS:
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
