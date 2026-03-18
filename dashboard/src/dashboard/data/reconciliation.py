"""Async SQLite queries for reconciliation metrics.

Each function accepts a db_path, opens the database read-only via URI mode,
and returns structured data. Missing database files return empty defaults.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import aiosqlite


async def get_recent_runs(db_path: Path, *, limit: int = 10) -> list[dict]:
    """Return recent reconciliation runs ordered by started_at DESC.

    Each dict contains: id, run_type, trigger_reason, started_at, completed_at,
    events_processed, status, duration_seconds.
    """
    try:
        async with aiosqlite.connect(f'file:{db_path}?mode=ro', uri=True) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                'SELECT id, run_type, trigger_reason, started_at, completed_at,'
                ' events_processed, status'
                ' FROM runs ORDER BY started_at DESC LIMIT ?',
                (limit,),
            ) as cursor:
                rows = await cursor.fetchall()
    except (FileNotFoundError, sqlite3.OperationalError):
        return []

    results = []
    for row in rows:
        duration = None
        if row['completed_at'] is not None:
            started = datetime.fromisoformat(row['started_at'])
            completed = datetime.fromisoformat(row['completed_at'])
            duration = (completed - started).total_seconds()

        results.append(
            {
                'id': row['id'],
                'run_type': row['run_type'],
                'trigger_reason': row['trigger_reason'],
                'started_at': row['started_at'],
                'completed_at': row['completed_at'],
                'events_processed': row['events_processed'],
                'status': row['status'],
                'duration_seconds': duration,
            }
        )
    return results
