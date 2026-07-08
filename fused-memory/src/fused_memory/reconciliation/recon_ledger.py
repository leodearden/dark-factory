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

# record_kind values that are per-task markers: gc() deletes these when their
# task_id has gone terminal. stage1_flag_suppression and cycle_summary are
# NOT per-task markers and are excluded — they expire only via expires_at (or
# live forever when expires_at is NULL).
MARKER_KINDS = ('stage1_flag_marker', 'stage2_persistence_marker', 'flag_for_stage2')


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


def _record_from_row(row: aiosqlite.Row) -> ReconLedgerRecord:
    """Map a ``recon_ledger`` row to a :class:`ReconLedgerRecord`."""
    return ReconLedgerRecord(
        project_id=row['project_id'],
        record_kind=row['record_kind'],
        task_id=row['task_id'],
        flag_type=row['flag_type'],
        run_id=row['run_id'],
        payload_json=row['payload_json'],
        state=row['state'],
        created_at=row['created_at'],
        expires_at=row['expires_at'],
    )


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

    async def upsert(self, record: ReconLedgerRecord) -> None:
        """Insert or update a ledger record.

        ``ON CONFLICT`` targets the full five-column primary key and
        overwrites ``payload_json``/``state``/``created_at``/``expires_at``
        with the new values (last-write-wins) — guarantees exactly one row
        per identity (journal.py's ``update_watermark`` precedent).
        """
        async with self._txn() as db:
            await db.execute(
                """
                INSERT INTO recon_ledger
                    (project_id, record_kind, task_id, flag_type, run_id,
                     payload_json, state, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(project_id, record_kind, task_id, flag_type, run_id) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    state = excluded.state,
                    created_at = excluded.created_at,
                    expires_at = excluded.expires_at
                """,
                (
                    record.project_id,
                    record.record_kind,
                    record.task_id,
                    record.flag_type,
                    record.run_id,
                    record.payload_json,
                    record.state,
                    record.created_at,
                    record.expires_at,
                ),
            )

    async def get_by_identity(
        self,
        project_id: str,
        record_kind: str,
        task_id: str = '',
        flag_type: str = '',
        run_id: str = '',
    ) -> ReconLedgerRecord | None:
        """Return the record matching the five-part identity, or None."""
        db = self._require_db()
        cursor = await db.execute(
            """
            SELECT * FROM recon_ledger
            WHERE project_id = ? AND record_kind = ? AND task_id = ?
              AND flag_type = ? AND run_id = ?
            """,
            (project_id, record_kind, task_id, flag_type, run_id),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return _record_from_row(row)

    async def list_suppressions(self, project_id: str) -> list[ReconLedgerRecord]:
        """Return the active ``stage1_flag_suppression`` rows for a project.

        Satisfied by the ``(project_id, record_kind, state)`` index.
        """
        db = self._require_db()
        cursor = await db.execute(
            """
            SELECT * FROM recon_ledger
            WHERE project_id = ? AND record_kind = 'stage1_flag_suppression' AND state = 'active'
            """,
            (project_id,),
        )
        rows = await cursor.fetchall()
        return [_record_from_row(row) for row in rows]

    async def is_suppressed(self, project_id: str, task_id: str, flag_type: str) -> bool:
        """Return True if ``task_id``/``flag_type`` is actively suppressed.

        ``flag_type = ''`` on a suppression row is a blanket/union suppression
        covering every flag_type for that task — matched here via
        ``flag_type IN (?, '')``.
        """
        db = self._require_db()
        cursor = await db.execute(
            """
            SELECT 1 FROM recon_ledger
            WHERE project_id = ? AND record_kind = 'stage1_flag_suppression' AND state = 'active'
              AND task_id = ? AND flag_type IN (?, '')
            LIMIT 1
            """,
            (project_id, task_id, flag_type),
        )
        row = await cursor.fetchone()
        return row is not None

    async def gc(
        self,
        project_id: str,
        now: str,
        terminal_task_ids: list[str] | tuple[str, ...],
    ) -> int:
        """Delete expired rows and terminal-task-referenced marker rows.

        A row is deleted when EITHER ``expires_at < now`` (NULL ``expires_at``
        is never-expire and is naturally excluded — SQL three-valued logic
        means ``NULL < ?`` evaluates to NULL, not true) OR its ``record_kind``
        is one of :data:`MARKER_KINDS` and its ``task_id`` is in
        ``terminal_task_ids``. Returns the number of rows deleted.

        When ``terminal_task_ids`` is empty, the terminal-referenced clause is
        omitted entirely and an expiry-only DELETE runs instead — an explicit
        guard rather than relying on an empty ``IN ()`` (which SQLite treats
        as always-false, but that's implicit engine behaviour, not a
        contract this store leans on).
        """
        async with self._txn() as db:
            if terminal_task_ids:
                marker_placeholders = ','.join('?' * len(MARKER_KINDS))
                terminal_placeholders = ','.join('?' * len(terminal_task_ids))
                cursor = await db.execute(
                    f"""
                    DELETE FROM recon_ledger
                    WHERE project_id = ? AND (
                        expires_at < ?
                        OR (record_kind IN ({marker_placeholders}) AND task_id IN ({terminal_placeholders}))
                    )
                    """,
                    (project_id, now, *MARKER_KINDS, *terminal_task_ids),
                )
            else:
                cursor = await db.execute(
                    """
                    DELETE FROM recon_ledger
                    WHERE project_id = ? AND expires_at < ?
                    """,
                    (project_id, now),
                )
        return cursor.rowcount

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
