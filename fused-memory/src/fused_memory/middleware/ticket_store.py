"""Ticket persistence store for two-phase add_task submit/resolve flow.

Tickets survive fused-memory restarts via SQLite (sibling DB to reconciliation.db).
On startup, any tickets left in 'pending' state from a prior run are marked as
'failed' with reason='server_restart'.
"""

from __future__ import annotations

import contextlib
import logging
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite

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

    async def close(self) -> None:
        """Close the underlying aiosqlite connection."""
        if self._db is not None:
            await self._db.close()
            self._db = None
