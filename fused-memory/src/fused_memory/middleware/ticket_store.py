"""Ticket persistence store for two-phase add_task submit/resolve flow.

Tickets survive fused-memory restarts via SQLite (sibling DB to reconciliation.db).
On startup, any tickets left in 'pending' state from a prior run are marked as
'failed' with reason='server_restart'.
"""

from __future__ import annotations

import logging
import secrets
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import aiosqlite
from shared.async_sqlite_base import (
    AtomicConnection,
    CheckpointResult,
    apply_full_durability_pragmas,
    connect_daemon,
)


@dataclass(frozen=True)
class DedupHealth:
    """One project's curator verdicts over a window (see :meth:`TicketStore.dedup_health`)."""

    resolved: int
    combined: int
    median_resolve_seconds: float | None
    top_create_reasons: tuple[tuple[str, int], ...]


# Crockford Base32 alphabet — omits I, L, O, U to reduce transcription errors.
_CROCKFORD = '0123456789ABCDEFGHJKMNPQRSTVWXYZ'


def _new_ticket_id() -> str:
    """Return a ``tkt_``-prefixed, lexicographically time-ordered ticket id.

    Composition: upper 6 bytes of ``time.time_ns()`` (big-endian, ~65 µs
    resolution) concatenated with 10 bytes of ``secrets.token_bytes``.
    The 16-byte payload is Crockford-base32 encoded into 26 characters.
    Total length: 4 (prefix) + 26 = 30 characters.
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

# Table creation runs first; index creation runs after the in-place
# ``escalated_at`` migration so legacy DBs (without the column) don't trip
# the ``ix_tickets_status_escalated`` reference during schema bootstrap.
TABLE_SQL = """
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
    expires_at  TEXT NOT NULL,
    escalated_at TEXT
);
"""

INDEX_SQL = """
CREATE INDEX IF NOT EXISTS ix_tickets_project_status
    ON tickets (project_id, status);

CREATE INDEX IF NOT EXISTS ix_tickets_status_created
    ON tickets (status, created_at);

CREATE INDEX IF NOT EXISTS ix_tickets_status_escalated
    ON tickets (status, escalated_at);

CREATE INDEX IF NOT EXISTS ix_tickets_project_status_resolved
    ON tickets (project_id, status, resolved_at);
"""

# The window read behind :meth:`TicketStore.dedup_health`. It runs once a
# janitor tick per project, so ``ix_tickets_project_status_resolved`` keeps
# its cost bounded by the window rather than by the unpruned table.
DEDUP_HEALTH_SQL = """
SELECT status, reason, created_at, resolved_at FROM tickets
WHERE project_id = ? AND status IN ('created', 'combined')
  AND resolved_at IS NOT NULL AND resolved_at >= ?
"""

# Back-compat alias — third-party code (and old tests) imported SCHEMA_SQL
# directly. Keep the symbol exporting the combined script.
SCHEMA_SQL = TABLE_SQL + INDEX_SQL


class TicketStore:
    """SQLite-backed store for two-phase add_task tickets."""

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)
        self._access: AtomicConnection | None = None

    async def initialize(self) -> None:
        """Open the SQLite connection and create the schema.

        Idempotent at both the connection level and the schema level:

        * **Connection-level** — if the store is already open (e.g. from a
          prior ``initialize()`` or a reconnect-via-reinit pattern), the
          existing connection is closed first via :meth:`close` (which also
          checkpoints the WAL and nulls ``self._access``) before a fresh one is
          opened.  This prevents orphaning the aiosqlite worker thread, which
          would otherwise raise ``"Event loop is closed"`` on GC (tasks 1560,
          1562).

        * **Schema-level** — ``CREATE TABLE IF NOT EXISTS`` and
          ``CREATE INDEX IF NOT EXISTS`` make repeated calls safe on an
          already-initialised database.
        """
        if self._access is not None:
            # Idempotent / reconnect-safe: close (checkpoint WAL + close +
            # null out) any prior connection before reassigning.  Orphaning it
            # leaks the aiosqlite worker thread and raises "Event loop is
            # closed" on GC (tasks 1560, 1562).
            await self.close()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = await connect_daemon(str(self._db_path))
        conn.row_factory = aiosqlite.Row
        await apply_full_durability_pragmas(conn, busy_timeout_ms=5000)
        self._access = AtomicConnection(conn)
        # Tables first, then in-place migrate, then indexes — the
        # escalated_at index references the column added by the migration.
        async with self._access.write() as db:
            await db.executescript(TABLE_SQL)
        await self._migrate_add_escalated_at()
        async with self._access.write() as db:
            await db.executescript(INDEX_SQL)
        logger.info('TicketStore initialized at %s', self._db_path)

    async def _migrate_add_escalated_at(self) -> None:
        """Add the ``escalated_at`` column to existing DBs that pre-date it.

        ``CREATE TABLE IF NOT EXISTS`` is a no-op on existing tables, so the
        column needs an explicit ALTER for legacy DBs. Idempotent: probes
        ``PRAGMA table_info`` first.
        """
        access = self._require_access()
        cols = {row[1] for row in await access.read_all('PRAGMA table_info(tickets)')}
        if 'escalated_at' not in cols:
            async with access.write() as db:
                await db.execute('ALTER TABLE tickets ADD COLUMN escalated_at TEXT')
            logger.info('TicketStore: migrated tickets table — added escalated_at column')

    def _require_access(self) -> AtomicConnection:
        if self._access is None:
            raise RuntimeError('TicketStore not initialized — call initialize() first')
        return self._access

    async def submit(
        self,
        project_id: str,
        candidate_json: str,
    ) -> str:
        """Insert a new pending ticket and return its ticket_id.

        ``expires_at`` is written as a far-future placeholder. The wall-clock
        TTL janitor was retired in favour of a worker-liveness reaper (see
        :class:`TicketJanitor.tick`); the column stays NOT NULL for schema
        back-compat and the value is no longer load-bearing.
        """
        ticket_id = _new_ticket_id()
        now = datetime.now(UTC)
        # Advisory placeholder; reaper is worker-liveness based, not TTL.
        expires_at = now + timedelta(days=365)
        async with self._require_access().write() as db:
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
        async with self._require_access().write() as db:
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
        async with self._require_access().write() as db:
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
        row = await self._require_access().read_one(
            'SELECT * FROM tickets WHERE ticket_id = ?', (ticket_id,)
        )
        if row is None:
            return None
        return dict(row)

    async def list_projects_with_pending(self) -> list[str]:
        """Return distinct ``project_id``s that currently have pending tickets.

        Used by :class:`TicketJanitor` to drive the worker-liveness reaper —
        for each project with pending rows, the janitor asks an injected
        liveness probe whether the per-project worker is still alive; dead
        workers' rows get terminalised as ``failed/worker_dead``.
        """
        rows = await self._require_access().read_all(
            "SELECT DISTINCT project_id FROM tickets WHERE status = 'pending'",
        )
        return [row['project_id'] for row in rows]

    async def mark_pending_failed_for_project(
        self, project_id: str, *, reason: str,
    ) -> list[str]:
        """Bulk-terminalise every pending ticket in a project.

        Used by the worker-liveness reaper when the project's curator worker
        is gone. ``resolved_at`` is stamped so downstream consumers can
        attribute failure timing.

        Returns the list of ``ticket_id`` values that were pending (and are
        now terminalised as ``failed``).  The SELECT and the UPDATE run in
        ONE write unit, which holds the store's per-connection lock, so no
        other in-process writer can land between them: ``reaped_ids`` exactly
        matches the rows this call changed.
        """
        now = datetime.now(UTC).isoformat()
        async with self._require_access().write() as db:
            rows = await db.execute_fetchall(
                "SELECT ticket_id FROM tickets "
                "WHERE project_id = ? AND status = 'pending'",
                (project_id,),
            )
            reaped_ids = [row['ticket_id'] for row in rows]
            await db.execute(
                """
                UPDATE tickets
                SET status = 'failed', reason = ?, resolved_at = ?
                WHERE project_id = ? AND status = 'pending'
                """,
                (reason, now, project_id),
            )
        return reaped_ids

    async def list_tickets(
        self,
        project_id: str,
        *,
        status: str | None = None,
        since: datetime | None = None,
        limit: int = 500,
    ) -> list[dict]:
        """Return tickets for a project, newest-first, optionally filtered.

        ``status`` matches against the status column literal ('pending',
        'created', 'failed', 'combined', 'refused'). A ``refused`` row is
        terminal and has a NULL ``task_id`` — a deterministic guard rejected
        the candidate and no task was created. ``since`` filters by
        ``created_at``.
        Default window when ``since`` is None: last 7 days.
        """
        if since is None:
            since = datetime.now(UTC) - timedelta(days=7)
        sql_parts = [
            'SELECT * FROM tickets',
            'WHERE project_id = ? AND created_at >= ?',
        ]
        params: list = [project_id, since.isoformat()]
        if status is not None:
            sql_parts.append('AND status = ?')
            params.append(status)
        sql_parts.append('ORDER BY created_at DESC LIMIT ?')
        params.append(limit)
        rows = await self._require_access().read_all(' '.join(sql_parts), tuple(params))
        return [dict(r) for r in rows]

    async def dedup_health(self, project_id: str, *, since: datetime) -> DedupHealth:
        """Summarise ``created`` / ``combined`` verdicts resolved at or after *since*.

        Latency is raw wall clock (``resolved_at - created_at``). Pending rows
        and ``failed`` / ``refused`` / ``cancelled`` rows are not curator dedup
        verdicts and are excluded. A NULL create reason counts as
        ``'(unrecorded)'``.
        """
        rows = await self._require_access().read_all(
            DEDUP_HEALTH_SQL, (project_id, since.isoformat()),
        )
        latencies = [
            (
                datetime.fromisoformat(row['resolved_at'])
                - datetime.fromisoformat(row['created_at'])
            ).total_seconds()
            for row in rows
        ]
        create_reasons = Counter(
            row['reason'] or '(unrecorded)' for row in rows if row['status'] == 'created'
        )
        return DedupHealth(
            resolved=len(rows),
            combined=sum(1 for row in rows if row['status'] == 'combined'),
            median_resolve_seconds=statistics.median(latencies) if latencies else None,
            top_create_reasons=tuple(create_reasons.most_common(5)),
        )

    async def fetch_unescalated_failures(
        self,
        project_id: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Return failed tickets that the janitor has not yet reported.

        Selects ``status='failed'`` rows whose ``escalated_at`` is still NULL
        and whose ``reason`` is not ``idempotency_hit`` (idempotency-hits land
        as ``status='combined'`` so are already excluded by the status filter,
        but the explicit guard belts-and-braces against future renames). Rows
        are ordered by ``resolved_at`` so the oldest failure escalates first.

        ``status='refused'`` is DELIBERATELY excluded and must stay excluded.
        A refusal is a successful, intended outcome of an operator-authored
        policy (the cancelled-premise blocklist / recon premise registry) —
        not an error. Sweeping refusals into the janitor's failure path would
        page a steward every time the blocklist did its job, training
        operators to ignore the signal. Do not "fix" this omission — it is
        pinned by
        ``tests/test_ticket_store.py::test_fetch_unescalated_failures_never_returns_refused``,
        so broadening this predicate (e.g. to "all terminal non-created
        statuses") fails there rather than in production.

        Args:
            project_id: When set, restrict to a single project. Default None
                returns all projects (the janitor groups across projects).
            limit: Maximum rows per call (default 100). Aligns with
                ``curator.janitor.batch_limit``.
        """
        sql = (
            "SELECT * FROM tickets "
            "WHERE status = 'failed' "
            "  AND escalated_at IS NULL "
            "  AND (reason IS NULL OR reason != 'idempotency_hit')"
        )
        params: tuple = ()
        if project_id is not None:
            sql += " AND project_id = ?"
            params = (project_id,)
        sql += " ORDER BY resolved_at LIMIT ?"
        params = (*params, limit)
        rows = await self._require_access().read_all(sql, params)
        return [dict(r) for r in rows]

    async def mark_escalated(self, ticket_ids: list[str] | tuple[str, ...]) -> int:
        """Bulk-stamp ``escalated_at`` on the given tickets.

        Returns the number of rows updated. Caller is expected to pass an
        already-deduped sequence of ticket ids; SQLite handles parameter
        expansion via ``?, ?, ...`` placeholders.
        """
        if not ticket_ids:
            return 0
        now = datetime.now(UTC).isoformat()
        placeholders = ','.join('?' * len(ticket_ids))
        async with self._require_access().write() as db:
            cursor = await db.execute(
                f"UPDATE tickets SET escalated_at = ? "
                f"WHERE ticket_id IN ({placeholders})",
                (now, *ticket_ids),
            )
        return cursor.rowcount

    async def close(self) -> None:
        """Close the underlying aiosqlite connection.

        Runs a final ``wal_checkpoint(TRUNCATE)`` so the next open sees an
        empty WAL and the main DB is fully up to date. Best-effort —
        failures don't block the close.
        """
        if self._access is not None:
            await self._access.close()
            self._access = None

    async def checkpoint(self) -> CheckpointResult:
        return await AtomicConnection.checkpoint_or_unavailable(self._access)
