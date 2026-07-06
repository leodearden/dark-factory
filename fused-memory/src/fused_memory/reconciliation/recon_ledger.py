"""ReconLedgerStore — SQLite control-plane ledger for recon markers,
suppressions, and cycle-summaries (task 2219, PRD
plans/recon-reliability-prd.md §8.1, stream W5 foundations phase).

FOUNDATIONS-FIRST: this module only adds the store; it does not delete any
Mem0-path code. Consumers ι/κ/λ switch reads to this ledger and delete the
Mem0 compensation chain later, under the write-both/read-new cutover (PRD
decision #6).

The store is CLOCK-INJECTED: callers supply ``created_at``/``expires_at`` on
:class:`ReconLedgerRecord`, and :meth:`ReconLedgerStore.gc` takes ``now``
explicitly. There is no ``datetime.now()`` call inside this module, which
keeps every store operation deterministic and mock-free to test.

The ``recon_ledger`` table lives inside the existing ``reconciliation.db``
file (shared with :class:`~fused_memory.reconciliation.journal.ReconciliationJournal`),
opened via a second ``aiosqlite`` connection — WAL mode + a busy timeout make
this safe, and ``CREATE TABLE IF NOT EXISTS`` only ever touches this store's
own table.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from pathlib import Path

import aiosqlite
from shared.async_sqlite_base import apply_full_durability_pragmas, connect_daemon

logger = logging.getLogger(__name__)

# Table creation runs first; index creation runs after so both are safe to
# re-run against an already-initialised database (CREATE ... IF NOT EXISTS).
TABLE_SQL = """
CREATE TABLE IF NOT EXISTS recon_ledger (
    project_id   TEXT NOT NULL,
    record_kind  TEXT NOT NULL,
    task_id      TEXT NOT NULL DEFAULT '',
    flag_type    TEXT NOT NULL DEFAULT '',
    run_id       TEXT NOT NULL DEFAULT '',
    payload_json TEXT NOT NULL,
    state        TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    expires_at   TEXT,
    PRIMARY KEY (project_id, record_kind, task_id, flag_type, run_id)
);
"""

INDEX_SQL = """
CREATE INDEX IF NOT EXISTS ix_recon_ledger_project_kind_state
    ON recon_ledger (project_id, record_kind, state);

CREATE INDEX IF NOT EXISTS ix_recon_ledger_project_expires
    ON recon_ledger (project_id, expires_at);
"""

SCHEMA_SQL = TABLE_SQL + INDEX_SQL


@dataclass(frozen=True)
class ReconLedgerRecord:
    """One row of the ``recon_ledger`` table.

    ``task_id``/``flag_type``/``run_id`` default to ``''`` (matching the
    schema's ``DEFAULT ''``) so callers that don't need the full five-part
    identity (e.g. a project-scoped ``cycle_summary`` row) can omit them.
    ``expires_at`` of ``None`` means the record never expires via TTL (it can
    still be GC'd via the terminal-task-referenced path in
    :meth:`ReconLedgerStore.gc`).
    """

    project_id: str
    record_kind: str
    payload_json: str
    state: str
    created_at: str
    task_id: str = ''
    flag_type: str = ''
    run_id: str = ''
    expires_at: str | None = None


class ReconLedgerStore:
    """SQLite-backed control-plane ledger for recon markers/suppressions/summaries."""

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Open the SQLite connection and create the schema.

        Idempotent at both the connection level and the schema level:

        * **Connection-level** — if ``self._db`` is already set (e.g. from a
          prior ``initialize()``), the existing connection is closed first via
          :meth:`close` (which also checkpoints the WAL and nulls ``self._db``)
          before a fresh one is opened. This prevents orphaning the aiosqlite
          worker thread, which would otherwise raise "Event loop is closed" on
          GC (ticket_store.py precedent, tasks 1560, 1562).

        * **Schema-level** — ``CREATE TABLE IF NOT EXISTS`` and
          ``CREATE INDEX IF NOT EXISTS`` make repeated calls safe on an
          already-initialised database.
        """
        if self._db is not None:
            await self.close()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await connect_daemon(str(self._db_path))
        self._db.row_factory = aiosqlite.Row
        await apply_full_durability_pragmas(self._db, busy_timeout_ms=5000)
        await self._db.executescript(SCHEMA_SQL)
        await self._db.commit()
        logger.info('ReconLedgerStore initialized at %s', self._db_path)

    def _require_db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError('ReconLedgerStore not initialized — call initialize() first')
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
        """Close the underlying aiosqlite connection.

        Runs a final ``wal_checkpoint(TRUNCATE)`` so the next open sees an
        empty WAL and the main DB is fully up to date. Best-effort —
        failures don't block the close.
        """
        if self._db is not None:
            with contextlib.suppress(Exception):
                await self._db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
            await self._db.close()
            self._db = None

    async def checkpoint(self) -> tuple[int, int, int]:
        """Run ``PRAGMA wal_checkpoint(TRUNCATE)`` and return ``(busy, log,
        checkpointed)``. Called by the periodic checkpoint loop in
        ``server/main.py``."""
        db = self._require_db()
        cursor = await db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        row = await cursor.fetchone()
        if row is None:
            return (-1, -1, -1)
        return int(row[0]), int(row[1]), int(row[2])
