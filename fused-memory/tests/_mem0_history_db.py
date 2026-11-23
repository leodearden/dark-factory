"""A synthetic mem0 ``history.db`` in mem0's own schema, for tests of the scripts reading one
through ``scripts/mem0_history_at.py``."""
from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from pathlib import Path

HistoryRow = tuple[str, str | None, str, str, str]
"""One history event: (memory_id, old_memory, new_memory, event, updated_at)."""

CREATED_AT = '2026-10-01T00:00:00+00:00'


def mem0_history_db(path: Path, rows: Iterable[HistoryRow], *, created_at: str = CREATED_AT) -> Path:
    """Write mem0's history table to *path*, one event per row, each created at *created_at*."""
    connection = sqlite3.connect(path)
    connection.execute(
        'CREATE TABLE history (id TEXT PRIMARY KEY, memory_id TEXT, old_memory TEXT,'
        ' new_memory TEXT, event TEXT, created_at DATETIME, updated_at DATETIME,'
        ' is_deleted INTEGER, actor_id TEXT, role TEXT)'
    )
    for index, (memory_id, old, new, event, updated_at) in enumerate(rows):
        connection.execute(
            'INSERT INTO history VALUES (?, ?, ?, ?, ?, ?, ?, 0, NULL, NULL)',
            (f'h{index}', memory_id, old, new, event, created_at, updated_at),
        )
    connection.commit()
    connection.close()
    return path
