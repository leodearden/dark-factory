"""Ticket persistence store for two-phase add_task submit/resolve flow.

Tickets survive fused-memory restarts via SQLite (sibling DB to reconciliation.db).
On startup, any tickets left in 'pending' state from a prior run are marked as
'failed' with reason='server_restart'.
"""

from __future__ import annotations

import contextlib
import logging
import secrets
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import aiosqlite

# Crockford Base32 alphabet — omits I, L, O, U to reduce transcription errors.
_CROCKFORD = '0123456789ABCDEFGHJKMNPQRSTVWXYZ'


def _new_ticket_id() -> str:
    """Return a ``tkt_``-prefixed, lexicographically time-ordered ticket id.

    Composition: upper 6 bytes of ``time.time_ns()`` (big-endian, ~65 µs
    resolution) concatenated with 10 bytes of ``secrets.token_bytes``.
    The 16-byte payload is Crockford-base32 encoded into 26 characters.
    Total length: 30 (prefix) + 26 = 30 characters.
    """
    # Upper 6 bytes of the nanosecond timestamp give ~65 µs resolution and
    # sort correctly for hundreds of years without wrapping.
    ts = time.time_ns().to_bytes(8, 'big')[:6]
    rand = secrets.token_bytes(10)
    raw = ts + rand  # 16 bytes = 128 bits

    # Encode 128 bits into 26 Crockford-base32 chars (5 bits each, MSB first).
    n = int.from_bytes(raw, 'big')
    chars: list[str] = []
    for _ in range(26):
        chars.append(_CROCKFORD[n & 0x1F])
        n >>= 5
    return 'tkt_' + ''.join(reversed(chars))

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS tickets (
    ticket_id   TEXT PRIMARY KEY,
    project_id  TEXT NOT NULL,
    candidate_json TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    task_id     TEXT,
    reason      TEXT,
    result_json TEXT,
    created_at  TEXT NOT NULL,
    resolved_at TEXT,
    expires_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_tickets_project_status
    ON tickets (project_id, status);

CREATE INDEX IF NOT EXISTS ix_tickets_status_created
    ON tickets (status, created_at);
"""


class TicketStore:
    """SQLite-backed store for two-phase add_task tickets."""

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Open the SQLite connection and create the schema (idempotent)."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self._db_path))
        self._db.row_factory = aiosqlite.Row
        await self._db.execute('PRAGMA journal_mode=WAL')
        await self._db.execute('PRAGMA busy_timeout=5000')
        await self._db.execute('PRAGMA synchronous=NORMAL')
        await self._db.executescript(SCHEMA_SQL)
        await self._db.commit()
        logger.info('TicketStore initialized at %s', self._db_path)

    def _require_db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError('TicketStore not initialized — call initialize() first')
        return self._db

    @contextlib.asynccontextmanager
    async def _txn(self):
        """Explicit transaction: commit on success, rollback on any exception."""
        db = self._require_db()
        try:
            yield db
            await db.commit()
        except BaseException:
            with contextlib.suppress(Exception):
                await db.rollback()
            raise

    async def submit(
        self,
        project_id: str,
        candidate_json: str,
        ttl_seconds: int = 600,
    ) -> str:
        """Insert a new pending ticket and return its ticket_id."""
        ticket_id = _new_ticket_id()
        now = datetime.now(UTC)
        expires_at = now + timedelta(seconds=ttl_seconds)
        async with self._txn() as db:
            await db.execute(
                """
                INSERT INTO tickets
                    (ticket_id, project_id, candidate_json, status, created_at, expires_at)
                VALUES (?, ?, ?, 'pending', ?, ?)
                """,
                (ticket_id, project_id, candidate_json, now.isoformat(), expires_at.isoformat()),
            )
        return ticket_id

    async def mark_resolved(
        self,
        ticket_id: str,
        *,
        status: str,
        task_id: str | None = None,
        reason: str | None = None,
        result_json: str | None = None,
    ) -> bool:
        """Update the ticket to a terminal status.

        Only updates rows that are still ``pending``; a double-resolve attempt
        returns ``False`` without clobbering the existing terminal data.
        """
        now = datetime.now(UTC).isoformat()
        async with self._txn() as db:
            cursor = await db.execute(
                """
                UPDATE tickets
                SET status = ?, task_id = ?, reason = ?, result_json = ?, resolved_at = ?
                WHERE ticket_id = ? AND status = 'pending'
                """,
                (status, task_id, reason, result_json, now, ticket_id),
            )
            if cursor.rowcount == 0:
                logger.warning(
                    'mark_resolved: ticket %s not in pending state (double-resolve or unknown)',
                    ticket_id,
                )
                return False
        return True

    async def flush_pending_on_startup(self) -> int:
        """Mark all pending tickets as failed/server_restart.

        Called once at startup to clean up tickets left over from a previous
        server run.  Returns the number of rows updated.
        """
        now = datetime.now(UTC).isoformat()
        async with self._txn() as db:
            cursor = await db.execute(
                """
                UPDATE tickets
                SET status = 'failed', reason = 'server_restart', resolved_at = ?
                WHERE status = 'pending'
                """,
                (now,),
            )
        count = cursor.rowcount
        logger.info('flush_pending_on_startup: marked %d pending tickets as failed', count)
        return count

    async def get(self, ticket_id: str) -> dict | None:
        """Return the ticket row as a plain dict, or None if not found."""
        db = self._require_db()
        cursor = await db.execute(
            'SELECT * FROM tickets WHERE ticket_id = ?', (ticket_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return dict(row)

    async def close(self) -> None:
        """Close the underlying aiosqlite connection."""
        if self._db is not None:
            await self._db.close()
            self._db = None
