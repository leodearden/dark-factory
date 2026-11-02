"""The task store re-runs its fresh-store DDL on every connection open, so the DDL must be idempotent."""

from __future__ import annotations

import sqlite3

from shared.task_store_schema import TASK_STORE_DDL


def _schema_objects(conn: sqlite3.Connection) -> set[tuple[str, str]]:
    return set(conn.execute('SELECT type, name FROM sqlite_master'))


def test_running_the_ddl_again_on_an_existing_store_changes_nothing():
    conn = sqlite3.connect(':memory:')
    try:
        conn.executescript(TASK_STORE_DDL)
        first = _schema_objects(conn)
        conn.executescript(TASK_STORE_DDL)
        assert _schema_objects(conn) == first
    finally:
        conn.close()
