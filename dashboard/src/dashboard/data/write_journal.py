"""Async SQLite queries for write journal metrics (memory operation graphs).

Queries the write_journal.db for time-series and breakdown data used by
the memory graphs section of the dashboard.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import UTC, datetime, timedelta

import aiosqlite

logger = logging.getLogger(__name__)


async def get_memory_timeseries(db: aiosqlite.Connection | None, *, hours: int = 24) -> dict:
    """Hourly read/write counts for the last *hours* hours.

    Returns ``{labels: ["HH:00", ...], reads: [int, ...], writes: [int, ...]}``.
    Labels cover every hour in the window, with zeros for gaps.
    """
    now = datetime.now(UTC)
    since = now - timedelta(hours=hours)

    # Pre-fill all hour buckets
    buckets: dict[str, dict[str, int]] = {}
    for i in range(hours):
        dt = since + timedelta(hours=i + 1)
        key = dt.strftime('%Y-%m-%dT%H:00')
        buckets[key] = {'read': 0, 'write': 0}

    if db is not None:
        try:
            async with db.execute(
                "SELECT strftime('%Y-%m-%dT%H:00', created_at) AS hour,"
                ' kind, COUNT(*) AS cnt'
                ' FROM write_ops WHERE created_at >= ?'
                ' GROUP BY hour, kind',
                (since.isoformat(),),
            ) as cursor:
                rows = await cursor.fetchall()

            for hour, kind, cnt in rows:
                if hour in buckets and kind in ('read', 'write'):
                    buckets[hour][kind] = cnt
        except sqlite3.OperationalError:
            logger.debug('get_memory_timeseries: query failed', exc_info=True)

    sorted_keys = sorted(buckets)
    return {
        'labels': [k[11:16] for k in sorted_keys],  # "HH:MM"
        'reads': [buckets[k]['read'] for k in sorted_keys],
        'writes': [buckets[k]['write'] for k in sorted_keys],
    }


async def get_operations_breakdown(db: aiosqlite.Connection | None, *, hours: int = 24) -> dict:
    """Operation type distribution for the last *hours* hours.

    Returns ``{labels: [str, ...], values: [int, ...]}``.
    """
    if db is None:
        return {'labels': [], 'values': []}

    since = (datetime.now(UTC) - timedelta(hours=hours)).isoformat()
    try:
        async with db.execute(
            'SELECT operation, COUNT(*) AS cnt FROM write_ops'
            ' WHERE created_at >= ? GROUP BY operation ORDER BY cnt DESC',
            (since,),
        ) as cursor:
            rows = await cursor.fetchall()
    except sqlite3.OperationalError:
        logger.debug('get_operations_breakdown: query failed', exc_info=True)
        return {'labels': [], 'values': []}

    return {
        'labels': [r[0] or 'unknown' for r in rows],
        'values': [r[1] for r in rows],
    }


async def get_agent_breakdown(db: aiosqlite.Connection | None, *, hours: int = 24) -> dict:
    """Agent distribution for the last *hours* hours.

    Returns ``{labels: [str, ...], values: [int, ...]}``.
    """
    if db is None:
        return {'labels': [], 'values': []}

    since = (datetime.now(UTC) - timedelta(hours=hours)).isoformat()
    try:
        async with db.execute(
            "SELECT COALESCE(agent_id, 'unknown') AS agent, COUNT(*) AS cnt"
            ' FROM write_ops WHERE created_at >= ?'
            ' GROUP BY agent ORDER BY cnt DESC',
            (since,),
        ) as cursor:
            rows = await cursor.fetchall()
    except sqlite3.OperationalError:
        logger.debug('get_agent_breakdown: query failed', exc_info=True)
        return {'labels': [], 'values': []}

    return {
        'labels': [r[0] for r in rows],
        'values': [r[1] for r in rows],
    }
