"""Pure helper functions for scheduler-state MCP tools.

Kept separate from server/tools.py so the logic is testable without
standing up the full MCP server.  Mirrors the pattern used by the
scheduler-override tools registered in 1259.

No orchestrator imports — these helpers read on-disk files written by
the orchestrator process (a separate process).  The only coupling is the
on-disk file format (JSON snapshot + SQLite runs.db schema).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

_log = logging.getLogger(__name__)


def _empty_skeleton() -> dict:
    """Build and return a fresh empty skeleton dict.

    Returns a new dict on every call so callers cannot accidentally mutate
    a shared module-level object and corrupt future calls.
    """
    return {
        'skip_counts': {},
        'parks': {},
        'park_stacks': {},
        'effective_priorities': {},
        'pin_queue': [],
        'overrides': {},
        'current_holders': {},
        'is_paused': False,
        'pause_reason': None,
        'snapshot_at': None,
    }


def read_scheduler_state(project_root: Path) -> dict:
    """Read and return the scheduler state snapshot from disk.

    Reads ``<project_root>/data/orchestrator/scheduler_state.json``.
    Returns the empty skeleton dict when the file is absent, unreadable,
    or contains invalid JSON.  A missing file is normal before the first
    orchestrator tick and is handled silently; any other error is logged
    as a warning before returning the empty skeleton.
    """
    path = project_root / 'data' / 'orchestrator' / 'scheduler_state.json'
    try:
        return json.loads(path.read_bytes())
    except FileNotFoundError:
        pass
    except (json.JSONDecodeError, OSError) as exc:
        _log.warning('read_scheduler_state: %s — returning empty skeleton', exc)
    return _empty_skeleton()


async def read_scheduler_events(
    project_root: Path,
    since: str | None,
    limit: int,
    event_types: list[str] | None,
) -> dict:
    """Read scheduler events from runs.db, newest-first.

    Opens ``<project_root>/data/orchestrator/runs.db`` in read-only URI mode
    via aiosqlite.  Returns ``{'events': [...], 'count': <int>}`` where each
    event is a dict with keys: id, timestamp, run_id, task_id, event_type, data.

    ``data`` is the JSON-parsed payload (dict), never a raw string.

    Returns ``{'events': [], 'count': 0}`` when the database is missing.

    **Path canonicalization**: the runs.db path is passed through
    ``Path.resolve()`` before the SQLite URI is constructed, mirroring the
    ``resolve()``-then-``as_uri()`` pattern used by ``DbPool.get`` in
    ``dashboard/src/dashboard/data/db.py``.  This ensures that symlinks in
    the path are expanded before the URI is built, so the connection always targets
    the real file.  **Symlink deployments beware**: if ``project_root`` is itself
    a symlink (or contains symlink components), the SQLite connection will target
    the resolved (canonical) path rather than the symlinked one.  This is
    intentional — it avoids URI-encoding ambiguity — but it is a silent behavioral
    change relative to a raw path open.
    """
    import aiosqlite

    db_path = (project_root / 'data' / 'orchestrator' / 'runs.db').resolve()
    if not db_path.exists():
        return {'events': [], 'count': 0}

    clauses: list[str] = []
    params: list = []

    if event_types:
        placeholders = ', '.join('?' for _ in event_types)
        clauses.append(f'event_type IN ({placeholders})')
        params.extend(event_types)

    if since is not None:
        clauses.append('timestamp >= ?')
        params.append(since)

    where = ('WHERE ' + ' AND '.join(clauses)) if clauses else ''
    # NOTE: When event_types is supplied the query planner may scan by
    # timestamp and filter by event_type (or vice versa) depending on
    # available indexes.  If runs.db grows large, consider adding a
    # composite index on (event_type, timestamp DESC) in the orchestrator's
    # schema migration and verify the plan with EXPLAIN QUERY PLAN.
    sql = (
        f'SELECT id, timestamp, run_id, task_id, event_type, data '
        f'FROM events {where} '
        f'ORDER BY timestamp DESC, id DESC '
        f'LIMIT ?'
    )
    params.append(limit)

    uri = f'{db_path.as_uri()}?mode=ro'
    async with aiosqlite.connect(uri, uri=True) as db:
        cursor = await db.execute(sql, params)
        rows = await cursor.fetchall()

    events = [
        {
            'id': r[0],
            'timestamp': r[1],
            'run_id': r[2],
            'task_id': r[3],
            'event_type': r[4],
            'data': json.loads(r[5] or '{}'),
        }
        for r in rows
    ]
    return {'events': events, 'count': len(events)}
