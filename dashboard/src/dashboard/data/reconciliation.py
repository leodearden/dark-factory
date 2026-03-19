"""Async SQLite queries for reconciliation metrics.

Each function accepts a db_path, opens the database read-only via URI mode,
and returns structured data. Missing database files return empty defaults.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)


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
        logger.debug('get_recent_runs: DB unavailable at %s', db_path, exc_info=True)
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
        logger.debug('get_watermarks: DB unavailable at %s', db_path, exc_info=True)
        return {}

    if row is None:
        return {}

    return {
        'last_full_run_completed': row['last_full_run_completed'],
        'last_episode_timestamp': row['last_episode_timestamp'],
        'last_memory_timestamp': row['last_memory_timestamp'],
        'last_task_change_timestamp': row['last_task_change_timestamp'],
    }


_BUFFER_STATS_DEFAULT = {'buffered_count': 0, 'oldest_event_age_seconds': None}


async def get_buffer_stats(db_path: Path) -> dict:
    """Return event buffer statistics: count and oldest event age.

    Returns dict with buffered_count (int) and oldest_event_age_seconds (float|None).
    """
    try:
        async with aiosqlite.connect(f'file:{db_path}?mode=ro', uri=True) as db:
            async with db.execute(
                "SELECT COUNT(*) FROM event_buffer WHERE status = 'buffered'",
            ) as cursor:
                row = await cursor.fetchone()
                count = row[0] if row else 0

            async with db.execute(
                "SELECT MIN(timestamp) FROM event_buffer WHERE status = 'buffered'",
            ) as cursor:
                row = await cursor.fetchone()
                oldest_ts = row[0] if row else None
    except (FileNotFoundError, sqlite3.OperationalError):
        logger.debug('get_buffer_stats: DB unavailable at %s', db_path, exc_info=True)
        return dict(_BUFFER_STATS_DEFAULT)

    age = None
    if oldest_ts is not None:
        oldest_dt = datetime.fromisoformat(oldest_ts)
        if oldest_dt.tzinfo is None:
            oldest_dt = oldest_dt.replace(tzinfo=UTC)
        age = (datetime.now(UTC) - oldest_dt).total_seconds()

    return {'buffered_count': count, 'oldest_event_age_seconds': age}


async def get_burst_state(db_path: Path) -> list[dict]:
    """Return current burst state for all agents.

    Each dict contains: agent_id, state, last_write_at, burst_started_at.
    """
    try:
        async with aiosqlite.connect(f'file:{db_path}?mode=ro', uri=True) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                'SELECT agent_id, state, last_write_at, burst_started_at'
                ' FROM burst_state',
            ) as cursor:
                rows = await cursor.fetchall()
    except (FileNotFoundError, sqlite3.OperationalError):
        logger.debug('get_burst_state: DB unavailable at %s', db_path, exc_info=True)
        return []

    return [
        {
            'agent_id': row['agent_id'],
            'state': row['state'],
            'last_write_at': row['last_write_at'],
            'burst_started_at': row['burst_started_at'],
        }
        for row in rows
    ]


async def get_latest_verdict(db_path: Path) -> dict | None:
    """Return the most recent judge verdict, or None if none exist.

    Returns dict with: run_id, severity, action_taken, reviewed_at.
    """
    try:
        async with aiosqlite.connect(f'file:{db_path}?mode=ro', uri=True) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                'SELECT run_id, severity, action_taken, reviewed_at'
                ' FROM judge_verdicts ORDER BY reviewed_at DESC LIMIT 1',
            ) as cursor:
                row = await cursor.fetchone()
    except (FileNotFoundError, sqlite3.OperationalError):
        logger.debug('get_latest_verdict: DB unavailable at %s', db_path, exc_info=True)
        return None

    if row is None:
        return None

    return {
        'run_id': row['run_id'],
        'severity': row['severity'],
        'action_taken': row['action_taken'],
        'reviewed_at': row['reviewed_at'],
    }
