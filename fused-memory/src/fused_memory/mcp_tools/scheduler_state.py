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
from pathlib import Path

# ---------------------------------------------------------------------------
# Empty skeleton returned when the snapshot file is missing.
# Keys match Scheduler.get_state_snapshot() exactly.
# ---------------------------------------------------------------------------

_EMPTY_SKELETON: dict = {
    'skip_counts': {},
    'parks': {},
    'effective_priorities': {},
    'pin_queue': [],
    'overrides': {},
    'current_holders': {},
    'snapshot_at': None,
}


def read_scheduler_state(project_root: Path) -> dict:
    """Read and return the scheduler state snapshot from disk.

    Reads ``<project_root>/data/orchestrator/scheduler_state.json``.
    Returns the empty skeleton dict when the file is missing.
    Never raises — the file may be absent if the orchestrator hasn't
    ticked yet.
    """
    path = project_root / 'data' / 'orchestrator' / 'scheduler_state.json'
    try:
        return json.loads(path.read_bytes())
    except FileNotFoundError:
        return dict(_EMPTY_SKELETON)


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
    """
    import aiosqlite

    db_path = project_root / 'data' / 'orchestrator' / 'runs.db'
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
    sql = (
        f'SELECT id, timestamp, run_id, task_id, event_type, data '
        f'FROM events {where} '
        f'ORDER BY timestamp DESC, id DESC '
        f'LIMIT ?'
    )
    params.append(limit)

    uri = f'file:{db_path}?mode=ro'
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
