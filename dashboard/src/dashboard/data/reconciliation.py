"""Async SQLite queries for reconciliation metrics.

Each function accepts an optional ``aiosqlite.Connection`` (from the shared
:class:`DbPool`) and returns structured data.  A ``None`` connection is
treated as "database unavailable" and returns the declared default.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

import aiosqlite

from dashboard.data.db import with_db

logger = logging.getLogger(__name__)

# Default burst cooldown matches EventBuffer.burst_cooldown_seconds
_DEFAULT_BURST_COOLDOWN = 150


def _parse_utc(ts: str | None) -> datetime:
    """Parse an ISO timestamp string and ensure it has UTC timezone.

    Naive datetimes (no tzinfo) get UTC attached.  Aware datetimes are
    returned unchanged.  Invalid input propagates ValueError or TypeError
    from :func:`datetime.fromisoformat`.
    """
    dt = datetime.fromisoformat(ts)  # type: ignore[arg-type]
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


_ACTIVE_THRESHOLD_SECONDS = 3600  # 1 hour


def partition_burst_state(
    burst_state: list[dict],
    *,
    active_threshold_seconds: int = _ACTIVE_THRESHOLD_SECONDS,
) -> tuple[list[dict], list[dict]]:
    """Split burst agents into active and idle lists.

    Active means ``state != 'idle'`` **or** ``last_write_at`` within
    *active_threshold_seconds*.  Everything else is idle/stale.
    """
    now = datetime.now(UTC)
    active: list[dict] = []
    idle: list[dict] = []
    for agent in burst_state:
        if agent.get('state', 'idle') != 'idle':
            active.append(agent)
            continue
        # Idle agents with recent writes are still "active" for display
        try:
            last_write = _parse_utc(agent.get('last_write_at'))
            if (now - last_write).total_seconds() < active_threshold_seconds:
                active.append(agent)
                continue
        except (ValueError, TypeError):
            pass
        idle.append(agent)
    return active, idle


async def get_recent_runs(db: aiosqlite.Connection | None, *, limit: int = 50) -> list[dict]:
    """Return recent reconciliation runs ordered by started_at DESC.

    Each dict contains: id, project_id, run_type, trigger_reason, started_at,
    completed_at, events_processed, status, duration_seconds, journal_entry_count.
    """
    async def _query(db: aiosqlite.Connection) -> list[dict]:
        async with db.execute(
            'SELECT r.id, r.project_id, r.run_type, r.trigger_reason,'
            ' r.started_at, r.completed_at, r.events_processed, r.status,'
            ' (SELECT COUNT(*) FROM journal_entries je'
            '  WHERE je.run_id = r.id) AS journal_entry_count'
            ' FROM runs r ORDER BY r.started_at DESC LIMIT ?',
            (limit,),
        ) as cursor:
            rows = await cursor.fetchall()

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
                    'project_id': row['project_id'],
                    'run_type': row['run_type'],
                    'trigger_reason': row['trigger_reason'],
                    'started_at': row['started_at'],
                    'completed_at': row['completed_at'],
                    'events_processed': row['events_processed'],
                    'status': row['status'],
                    'duration_seconds': duration,
                    'journal_entry_count': row['journal_entry_count'],
                }
            )
        return results

    return await with_db(db, _query, [])


async def get_journal_entries(db: aiosqlite.Connection | None, run_id: str) -> list[dict]:
    """Return journal entries for a specific run, ordered by timestamp.

    Each dict contains: id, stage, timestamp, operation, target_system,
    before_state, after_state, reasoning, evidence.
    """
    async def _query(db: aiosqlite.Connection) -> list[dict]:
        async with db.execute(
            'SELECT id, stage, timestamp, operation, target_system,'
            ' before_state, after_state, reasoning, evidence'
            ' FROM journal_entries WHERE run_id = ? ORDER BY timestamp',
            (run_id,),
        ) as cursor:
            rows = await cursor.fetchall()

        results = []
        for row in rows:
            before_state = None
            after_state = None
            evidence = []
            try:
                if row['before_state']:
                    before_state = json.loads(row['before_state'])
            except (json.JSONDecodeError, TypeError):
                before_state = row['before_state']
            try:
                if row['after_state']:
                    after_state = json.loads(row['after_state'])
            except (json.JSONDecodeError, TypeError):
                after_state = row['after_state']
            try:
                if row['evidence']:
                    evidence = json.loads(row['evidence'])
            except (json.JSONDecodeError, TypeError):
                evidence = []

            results.append({
                'id': row['id'],
                'stage': row['stage'],
                'timestamp': row['timestamp'],
                'operation': row['operation'],
                'target_system': row['target_system'],
                'before_state': before_state,
                'after_state': after_state,
                'reasoning': row['reasoning'] or '',
                'evidence': evidence,
            })
        return results

    return await with_db(db, _query, [])


async def get_watermarks(db: aiosqlite.Connection | None) -> list[dict]:
    """Return watermark timestamps for all projects.

    Each dict contains: project_id, last_full_run_completed, last_episode_timestamp,
    last_memory_timestamp, last_task_change_timestamp.  Returns [] if none found.
    """
    async def _query(db: aiosqlite.Connection) -> list[dict]:
        async with db.execute(
            'SELECT project_id, last_full_run_completed, last_episode_timestamp,'
            ' last_memory_timestamp, last_task_change_timestamp'
            ' FROM watermarks ORDER BY project_id',
        ) as cursor:
            rows = await cursor.fetchall()
        return [
            {
                'project_id': row['project_id'],
                'last_full_run_completed': row['last_full_run_completed'],
                'last_episode_timestamp': row['last_episode_timestamp'],
                'last_memory_timestamp': row['last_memory_timestamp'],
                'last_task_change_timestamp': row['last_task_change_timestamp'],
            }
            for row in rows
        ]

    return await with_db(db, _query, [])


async def get_last_attempted_run(db: aiosqlite.Connection | None) -> dict[str, dict]:
    """Return the most recent run per project, regardless of status.

    Returns a dict keyed by project_id -> {id, status, started_at, completed_at}.
    """
    async def _query(db: aiosqlite.Connection) -> dict[str, dict]:
        async with db.execute(
            'SELECT r.id, r.project_id, r.status, r.started_at, r.completed_at'
            ' FROM runs r'
            ' INNER JOIN ('
            '   SELECT project_id, MAX(started_at) AS max_started'
            '   FROM runs GROUP BY project_id'
            ' ) latest ON r.project_id = latest.project_id'
            '   AND r.started_at = latest.max_started',
        ) as cursor:
            rows = await cursor.fetchall()
        return {
            row['project_id']: {
                'id': row['id'], 'status': row['status'],
                'started_at': row['started_at'], 'completed_at': row['completed_at'],
            }
            for row in rows
        }

    return await with_db(db, _query, {})


_BUFFER_STATS_DEFAULT = {'buffered_count': 0, 'oldest_event_age_seconds': None}


async def get_buffer_stats(db: aiosqlite.Connection | None) -> dict:
    """Return event buffer statistics: count and oldest event age.

    Returns dict with buffered_count (int) and oldest_event_age_seconds (float|None).
    """
    async def _query(db: aiosqlite.Connection) -> dict:
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

        age = None
        if oldest_ts is not None:
            oldest_dt = _parse_utc(oldest_ts)
            age = (datetime.now(UTC) - oldest_dt).total_seconds()

        return {'buffered_count': count, 'oldest_event_age_seconds': age}

    return await with_db(db, _query, dict(_BUFFER_STATS_DEFAULT))


async def get_burst_state(
    db: aiosqlite.Connection | None, *, burst_cooldown_seconds: int = _DEFAULT_BURST_COOLDOWN,
) -> list[dict]:
    """Return current burst state for all agents with cooldown applied.

    Agents whose ``last_write_at`` exceeds *burst_cooldown_seconds* are
    reported as ``idle`` regardless of the stored state — the EventBuffer
    only expires bursts during its own trigger checks, so the dashboard
    must compute effective state at read time.

    Each dict contains: agent_id, state, last_write_at, burst_started_at.
    """
    async def _query(db: aiosqlite.Connection) -> list[dict]:
        async with db.execute(
            'SELECT agent_id, state, last_write_at, burst_started_at'
            ' FROM burst_state'
            ' ORDER BY last_write_at DESC',
        ) as cursor:
            rows = await cursor.fetchall()

        now = datetime.now(UTC)
        results = []
        for row in rows:
            state = row['state']
            burst_started = row['burst_started_at']

            # Apply cooldown: if last write is older than cooldown, agent is idle
            if state == 'bursting':
                try:
                    last_write = _parse_utc(row['last_write_at'])
                    if (now - last_write).total_seconds() > burst_cooldown_seconds:
                        state = 'idle'
                        burst_started = None
                except (ValueError, TypeError):
                    state = 'idle'
                    burst_started = None

            results.append({
                'agent_id': row['agent_id'],
                'state': state,
                'last_write_at': row['last_write_at'],
                'burst_started_at': burst_started,
            })
        return results

    return await with_db(db, _query, [])


async def get_latest_verdict(db: aiosqlite.Connection | None) -> dict | None:
    """Return the most recent judge verdict, or None if none exist.

    Returns dict with: run_id, severity, action_taken, reviewed_at.
    """
    async def _query(db: aiosqlite.Connection) -> dict | None:
        async with db.execute(
            'SELECT run_id, severity, action_taken, reviewed_at'
            ' FROM judge_verdicts ORDER BY reviewed_at DESC LIMIT 1',
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return {
            'run_id': row['run_id'],
            'severity': row['severity'],
            'action_taken': row['action_taken'],
            'reviewed_at': row['reviewed_at'],
        }

    return await with_db(db, _query, None)
