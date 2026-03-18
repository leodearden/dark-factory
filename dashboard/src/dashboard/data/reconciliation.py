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


async def get_watermarks(db_path: Path, *, project_id: str = 'dark_factory') -> dict:
    """Return watermark timestamps for a given project.

    Returns a dict with keys: last_full_run_completed, last_episode_timestamp,
    last_memory_timestamp, last_task_change_timestamp. Returns {} if not found.
    """
    try:
        async with aiosqlite.connect(f'file:{db_path}?mode=ro', uri=True) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                'SELECT last_full_run_completed, last_episode_timestamp,'
                ' last_memory_timestamp, last_task_change_timestamp'
                ' FROM watermarks WHERE project_id = ?',
                (project_id,),
            ) as cursor:
                row = await cursor.fetchone()
    except (FileNotFoundError, sqlite3.OperationalError):
        return {}

    if row is None:
        return {}

    return {
        'last_full_run_completed': row['last_full_run_completed'],
        'last_episode_timestamp': row['last_episode_timestamp'],
        'last_memory_timestamp': row['last_memory_timestamp'],
        'last_task_change_timestamp': row['last_task_change_timestamp'],
    }
