"""Synchronous durable write-ahead journal for reconciliation events.

:class:`EventJournal` is the durable-at-enqueue store backing
:class:`~fused_memory.reconciliation.event_queue.EventQueue`. Every event is
persisted here **synchronously** (a WAL + ``synchronous=FULL`` commit that
completes before the call returns) *before* it is handed to the in-memory
``asyncio.Queue``. This makes the in-memory queue a dispatch cache over a
durable row: a hard kill (``kill -9``) between enqueue and drain no longer
silently drops in-flight events — startup recovery re-enqueues every surviving
row (see ``EventQueue.recover``).

Why stdlib ``sqlite3`` and not ``aiosqlite``
--------------------------------------------
``EventQueue.enqueue()`` is a **synchronous** method (the WP-B hot-path
contract — its sole production caller, ``TaskInterceptor._journal``, invokes it
without ``await``, as do ~all existing tests). A synchronous method cannot
``await`` an aiosqlite coroutine, so a durable write that must complete *before*
``enqueue`` returns has to use synchronous ``sqlite3``. A WAL + FULL commit is
genuinely synchronous (~1-5 ms) and applies the *same* durability pragmas as
:func:`shared.async_sqlite_base.apply_full_durability_pragmas`.

Threading invariant
-------------------
The single ``sqlite3`` connection is only ever touched from the event-loop
thread: synchronous ``enqueue`` and the ``drainer``/``recover`` coroutines
(which call ``append``/``mark_processed``/``load_unprocessed`` without awaiting)
all run on that one thread, so no application-level locking is needed. The
operator-driven replay path (``EventQueue.replay_dead_letters``) offloads only
its blocking dead-letter *file* I/O to an ``asyncio.to_thread`` worker; the
re-enqueue — and therefore every ``append``/``mark_processed`` it triggers — is
marshalled back onto the loop thread (see
``EventQueue._resolve_replay_enqueue``), because the in-memory ``asyncio.Queue``
that ``enqueue`` mutates is not thread-safe. So the connection is never touched
off the loop thread. ``check_same_thread=False`` is retained as
defence-in-depth — Python's ``sqlite3`` serializes connection access internally
under its default thread-safe build — not because any current path requires it.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

from fused_memory.models.reconciliation import ReconciliationEvent

logger = logging.getLogger(__name__)

# Same durability triad as shared.async_sqlite_base.apply_full_durability_pragmas
# (WAL + synchronous=FULL + wal_autocheckpoint=100 + journal_size_limit=64 MiB),
# applied to a synchronous sqlite3 connection.  Kept in sync with that helper.
_JOURNAL_BUSY_TIMEOUT_MS = 5000
_JOURNAL_WAL_AUTOCHECKPOINT = 100
_JOURNAL_SIZE_LIMIT_BYTES = 67_108_864  # 64 MiB

_JOURNAL_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS event_journal (
    id TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    enqueued_at TEXT NOT NULL
);
"""


class EventJournal:
    """Durable write-ahead log of in-flight reconciliation events.

    A surviving row means "enqueued but not yet committed to the buffer nor
    dead-lettered". :meth:`append` persists a row durably; :meth:`mark_processed`
    deletes it once the drainer has handed the event off; :meth:`load_unprocessed`
    reconstructs every surviving row on startup.
    """

    def __init__(self, path: Path | str):
        self._path = Path(path)
        # Defensive: the data dir usually already exists (EventBuffer lives
        # alongside), but sqlite3.connect fails if the parent is missing.
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: see the module docstring's threading
        # invariant — the replay path may touch this connection from an
        # asyncio.to_thread worker.  sqlite3 serializes access internally.
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._closed = False
        self._apply_pragmas()
        self._conn.executescript(_JOURNAL_SCHEMA_SQL)
        self._conn.commit()

    def _apply_pragmas(self) -> None:
        """Apply the WAL + full-durability pragma set to the sync connection.

        Mirrors :func:`shared.async_sqlite_base.apply_full_durability_pragmas`:
        WAL journal mode, ``synchronous=FULL`` (per-commit fsync),
        ``wal_autocheckpoint=100``, and a 64 MiB ``journal_size_limit``, plus a
        ``busy_timeout``. The WAL result is verified (loud-over-silent) — the
        journal is always file-backed, so WAL must succeed.
        """
        cur = self._conn.execute('PRAGMA journal_mode=WAL')
        row = cur.fetchone()
        if row is None or str(row[0]).lower() != 'wal':
            got = row[0] if row is not None else None
            raise RuntimeError(
                f'EventJournal: failed to enable WAL journal mode (got {got!r}) '
                f'for {self._path}'
            )
        self._conn.execute(f'PRAGMA busy_timeout={_JOURNAL_BUSY_TIMEOUT_MS}')
        self._conn.execute('PRAGMA synchronous=FULL')
        self._conn.execute(f'PRAGMA wal_autocheckpoint={_JOURNAL_WAL_AUTOCHECKPOINT}')
        self._conn.execute(f'PRAGMA journal_size_limit={_JOURNAL_SIZE_LIMIT_BYTES}')

    def append(self, event: ReconciliationEvent) -> None:
        """Durably persist *event* (INSERT OR REPLACE + commit).

        The commit is synchronous (WAL + ``synchronous=FULL``) so the row is on
        disk before this returns — the durable-at-enqueue guarantee. INSERT OR
        REPLACE keyed on the event id (a UUID) makes a re-append of the same
        event idempotent.
        """
        payload = json.dumps(event.model_dump(mode='json'))
        enqueued_at = event.timestamp.isoformat()
        self._conn.execute(
            'INSERT OR REPLACE INTO event_journal (id, payload, enqueued_at) '
            'VALUES (?, ?, ?)',
            (event.id, payload, enqueued_at),
        )
        self._conn.commit()

    def mark_processed(self, event_id: str) -> None:
        """Delete the journal row for *event_id* (harmless no-op if absent).

        A DELETE — not an ``UPDATE processed=1`` flag — so the write-ahead log
        stays bounded to only in-flight events and needs no separate GC pass:
        the event's durable record already lives in ``event_buffer`` (on commit)
        or the dead-letter JSONL (on divert), so a processed row would be pure
        dead weight. The commit is synchronous, matching :meth:`append`.
        """
        self._conn.execute('DELETE FROM event_journal WHERE id = ?', (event_id,))
        self._conn.commit()

    def load_unprocessed(self) -> list[ReconciliationEvent]:
        """Reconstruct every surviving (unprocessed) row, oldest-first.

        "Unprocessed" == every row still present: :meth:`mark_processed` deletes
        rows once their event is committed to the buffer or dead-lettered, so the
        surviving set is exactly the in-flight events to redeliver on startup.
        """
        cur = self._conn.execute(
            'SELECT payload FROM event_journal ORDER BY enqueued_at, id'
        )
        events: list[ReconciliationEvent] = []
        for (payload,) in cur.fetchall():
            events.append(ReconciliationEvent.model_validate(json.loads(payload)))
        return events

    def close(self) -> None:
        """Close the underlying connection (idempotent)."""
        if self._closed:
            return
        self._closed = True
        self._conn.close()
