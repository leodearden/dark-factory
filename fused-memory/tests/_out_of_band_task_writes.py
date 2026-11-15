"""Set a task's status or metadata with stdlib sqlite3, bypassing every backend write gate."""

from __future__ import annotations

import sqlite3

from shared.project_layout import tasks_db


def write_task_out_of_band(
    project_root: str, task_id: str, *, status: str | None = None, metadata: str | None = None,
) -> None:
    conn = sqlite3.connect(tasks_db(project_root))
    try:
        with conn:
            if status is not None:
                conn.execute('UPDATE tasks SET status=? WHERE id=?', (status, int(task_id)))
            if metadata is not None:
                conn.execute('UPDATE tasks SET metadata=? WHERE id=?', (metadata, int(task_id)))
    finally:
        conn.close()
