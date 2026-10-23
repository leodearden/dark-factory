"""Event-store reads that raise on any store failure instead of reading as "no rows".

EventStore's own reads are fire-safe: a locked, corrupt or schema-less runs.db
logs a warning and answers empty.  A gate that would fail OPEN on that empty
answer reads through here, so the failure reaches the gate's own fail-safe path.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from pathlib import Path

from orchestrator.event_store import EventType

BUSY_TIMEOUT_SECS: float = 5.0


class EventReadError(Exception):
    """The events DB could not answer a strict read."""


def task_ids_with_event(db_path: str | os.PathLike[str], event_type: EventType) -> frozenset[str]:
    """Every task id that recorded *event_type* in any run; raises :class:`EventReadError` when the store cannot say.

    Opened read-only through a ``file:`` URI, so a wrong path is an error and never a new, empty store.
    """
    uri = f'{Path(db_path).resolve().as_uri()}?mode=ro'
    try:
        with closing(sqlite3.connect(uri, uri=True, timeout=BUSY_TIMEOUT_SECS)) as conn:
            rows = conn.execute(
                'SELECT DISTINCT task_id FROM events WHERE event_type = ? AND task_id IS NOT NULL',
                (event_type.value,),
            ).fetchall()
    except sqlite3.Error as exc:
        raise EventReadError(f'cannot read {event_type.value} rows from {db_path}: {exc}') from exc
    return frozenset(str(task_id) for (task_id,) in rows)
