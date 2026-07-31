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
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import aiosqlite
from shared.async_sqlite_base import apply_full_durability_pragmas, connect_daemon

from fused_memory.reconciliation.standing_decision_constants import (
    EXPIRY_REASON_TTL,
    GROUNDS_ENUM,
    RECORD_KIND_ENTITY_STANDING_DECISION,
    STATE_ACTIVE,
    STATE_EXPIRED,
)

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
    entity_uuid  TEXT,
    PRIMARY KEY (project_id, record_kind, task_id, flag_type, run_id)
);
"""

INDEX_SQL = """
CREATE INDEX IF NOT EXISTS ix_recon_ledger_project_kind_state
    ON recon_ledger (project_id, record_kind, state);

CREATE INDEX IF NOT EXISTS ix_recon_ledger_project_expires
    ON recon_ledger (project_id, expires_at);

CREATE INDEX IF NOT EXISTS ix_recon_ledger_project_kind_entity
    ON recon_ledger (project_id, record_kind, entity_uuid);
"""

SCHEMA_SQL = TABLE_SQL + INDEX_SQL

# record_kind values that are per-task markers: gc() deletes these when their
# task_id has gone terminal. stage1_flag_suppression, cycle_summary and
# mem0_tombstone are NOT per-task markers and are excluded — they expire only
# via expires_at (or live forever when expires_at is NULL).
#
# mem0_tombstone (task 3041) is excluded for a stronger reason than the other
# two: its task_id column holds a Mem0 MEMORY UUID, not a Taskmaster task id.
# Both task-id-keyed paths in this store — gc()'s terminal-referenced DELETE
# arm and marker_task_ids(), whose result stages.task_knowledge_sync's
# _gc_recon_markers feeds straight back into that same gc() call — filter on
# `record_kind IN MARKER_KINDS`, so keeping the kind out of this tuple keeps
# memory uuids out of both. Adding it would mean an id collision between a
# memory uuid and a terminal task id could delete the audit trail of the very
# record an auditor is investigating.
MARKER_KINDS = ('stage1_flag_marker', 'stage2_persistence_marker', 'flag_for_stage2')

# record_kind for a recon-initiated Mem0 deletion tombstone (task 3041).
#
# Defined HERE rather than in reconciliation.mem0_tombstone (which owns the
# writer) because that module already imports ReconLedgerRecord from this one,
# so this is the only direction available without a cycle — and duplicating
# the literal across both would be exactly the lockstep duplication
# standing_decision_constants' INV-5 forbids. mem0_tombstone re-exports it, so
# `from ...mem0_tombstone import RECORD_KIND_MEM0_TOMBSTONE` still resolves.
RECORD_KIND_MEM0_TOMBSTONE = 'mem0_tombstone'

# Single-sourced INSERT ... ON CONFLICT used by BOTH upsert() and
# upsert_many(). Hoisted to module scope so the two write paths cannot drift
# apart in their conflict semantics — a batched write that resolved conflicts
# differently from the single-row one would be a silent correctness bug.
_UPSERT_SQL = """
    INSERT INTO recon_ledger
        (project_id, record_kind, task_id, flag_type, run_id,
         payload_json, state, created_at, expires_at, entity_uuid)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(project_id, record_kind, task_id, flag_type, run_id) DO UPDATE SET
        payload_json = excluded.payload_json,
        state = excluded.state,
        created_at = excluded.created_at,
        expires_at = excluded.expires_at,
        entity_uuid = excluded.entity_uuid
"""


@dataclass(frozen=True)
class ReconLedgerRecord:
    """One row of the ``recon_ledger`` table.

    ``task_id``/``flag_type``/``run_id`` default to ``''`` (matching the
    schema's ``DEFAULT ''``) so callers that don't need the full five-part
    identity (e.g. a project-scoped ``cycle_summary`` row) can omit them.
    ``expires_at`` of ``None`` means the record never expires via TTL (it can
    still be GC'd via the terminal-task-referenced path in
    :meth:`ReconLedgerStore.gc`).

    ``payload_json`` should encode a JSON *object* (dict) — callers that
    stamp acknowledgement metadata (:meth:`ReconLedgerStore.mark_addressed`)
    assume object semantics, though that method defensively tolerates other
    JSON shapes rather than raising.

    ``created_at``/``expires_at`` must be normalized UTC ISO-8601 strings in
    a single canonical, zero-padded format (e.g.
    ``'2026-07-01T00:00:00+00:00'``). :meth:`ReconLedgerStore.gc` compares
    ``expires_at`` against its ``now`` argument as plain SQLite TEXT, so
    ordering is lexicographic and only correct when every timestamp shares
    the same format/width/offset representation.
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
    entity_uuid: str | None = None


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
        entity_uuid=row['entity_uuid'],
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

        The schema is applied in three ordered phases — ``TABLE_SQL`` (create
        the table), the idempotent ``entity_uuid`` ADD COLUMN migration
        (task 2894 α), then ``INDEX_SQL`` — rather than a single
        ``executescript(SCHEMA_SQL)``. The migration MUST run before index
        creation because ``ix_recon_ledger_project_kind_entity`` references
        ``entity_uuid``: on a pre-migration DB the column does not exist until
        the ALTER adds it, so building the index first would raise. This
        mirrors the TABLE→INDEX split in ``middleware/ticket_store.py``.
        """
        if self._db is not None:
            await self.close()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await connect_daemon(str(self._db_path))
        self._db.row_factory = aiosqlite.Row
        await apply_full_durability_pragmas(self._db, busy_timeout_ms=5000)
        await self._db.executescript(TABLE_SQL)
        await self._db.commit()

        # Safe migration: add the nullable entity_uuid column to pre-existing
        # recon_ledger tables (task 2894 α, journal.py:214-249 ALTER-TABLE
        # house pattern). On a fresh DB the column is already created by
        # TABLE_SQL, so this ALTER fails ("duplicate column name") and is a
        # harmless rollback; on an old DB it adds the column. Runs BEFORE
        # INDEX_SQL so ix_recon_ledger_project_kind_entity can reference it.
        #
        # The catch is NARROWED to the expected "duplicate column name" case
        # (loud-over-silent): a genuine migration failure — a locked/corrupt DB,
        # a disk error — is re-raised here with its own message rather than
        # swallowed and left to re-surface one statement later as a confusing
        # "no such column: entity_uuid" from the INDEX_SQL that references it.
        try:
            await self._db.execute('ALTER TABLE recon_ledger ADD COLUMN entity_uuid TEXT')
            await self._db.commit()
        except Exception as exc:
            with contextlib.suppress(Exception):
                await self._db.rollback()
            if 'duplicate column name' not in str(exc).lower():
                raise  # Not the benign "column already exists" case — surface it.

        await self._db.executescript(INDEX_SQL)
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

    @staticmethod
    def _upsert_params(record: ReconLedgerRecord) -> tuple:
        """Bind parameters for :data:`_UPSERT_SQL`, in column order."""
        return (
            record.project_id,
            record.record_kind,
            record.task_id,
            record.flag_type,
            record.run_id,
            record.payload_json,
            record.state,
            record.created_at,
            record.expires_at,
            record.entity_uuid,
        )

    async def upsert(self, record: ReconLedgerRecord) -> None:
        """Insert or update a ledger record.

        ``ON CONFLICT`` targets the full five-column primary key and
        overwrites ``payload_json``/``state``/``created_at``/``expires_at``
        with the new values (last-write-wins) — guarantees exactly one row
        per identity (journal.py's ``update_watermark`` precedent).

        Use :meth:`upsert_many` when writing a batch: each call here is its own
        transaction, hence its own commit and its own fsync.
        """
        async with self._txn() as db:
            await db.execute(_UPSERT_SQL, self._upsert_params(record))

    async def upsert_many(self, records: Sequence[ReconLedgerRecord]) -> int:
        """Insert or update many records in ONE transaction (task 3041 amendment).

        Identical ``ON CONFLICT`` semantics to :meth:`upsert`, batched through
        ``executemany`` so N rows cost ONE commit instead of N. That matters
        because this store opens its connection with full-durability pragmas:
        every commit is an fsync, aiosqlite serializes all of them onto a
        single worker thread, and the connection is shared with the rest of
        the reconciliation cycle — so a per-row loop over a backlog sweep
        (e.g. a marker pool that grew during a trim outage) turns into N
        sequential fsyncs on the cycle's critical path.

        Rows within a batch are applied in the given order, so if two records
        share the same five-part identity the LAST one wins — the same
        last-write-wins rule a sequence of :meth:`upsert` calls would produce.

        An empty sequence is a no-op: no transaction is opened at all, so a
        caller with nothing to write pays nothing.

        Args:
            records: The rows to write.

        Returns:
            The number of records submitted (``len(records)``).
        """
        params = [self._upsert_params(record) for record in records]
        if not params:
            return 0
        async with self._txn() as db:
            await db.executemany(_UPSERT_SQL, params)
        return len(params)

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

    async def get_mem0_tombstone(
        self, project_id: str, memory_id: str
    ) -> ReconLedgerRecord | None:
        """Return the deletion tombstone for Mem0 record *memory_id*, or None.

        A tombstone (task 3041) records that a recon sweep deliberately deleted
        a Mem0 record: which sweep took it, which run it belonged to, and the
        victim's identifying metadata. It is written by
        :func:`~fused_memory.reconciliation.mem0_tombstone.record_mem0_deletion_tombstone`
        after every confirmed recon-initiated delete.

        This is a thin delegation to :meth:`get_by_identity` with
        ``flag_type``/``run_id`` left at their ``''`` defaults — matching how
        the writer stores the row — so callers don't hand-assemble the
        five-part identity. That default matters: an auditor arriving from a
        ``get_memory_by_id`` miss knows exactly one thing, the memory uuid, and
        this accessor is satisfiable from that alone.

        ``None`` means no tombstone exists, which covers both "the record never
        existed" and "it expired past
        :data:`~fused_memory.reconciliation.mem0_tombstone.MEM0_TOMBSTONE_TTL_DAYS`" — a
        tombstone proves deliberate deletion, but its absence does not prove
        the converse.
        """
        return await self.get_by_identity(
            project_id, RECORD_KIND_MEM0_TOMBSTONE, task_id=memory_id
        )

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

    async def marker_task_ids(self, project_id: str) -> set[str]:
        """Return the distinct non-empty ``task_id`` values across
        :data:`MARKER_KINDS` rows for *project_id*.

        Callers (:func:`~fused_memory.reconciliation.stages.task_knowledge_sync._gc_recon_markers`,
        task 2228 W5-κ amendment) use this to bound an otherwise
        project-lifetime-sized terminal-task-id list down to only the ids
        that could possibly match :meth:`gc`'s ``task_id IN (...)`` clause —
        a terminal id with no marker row in the ledger contributes nothing to
        that DELETE, so intersecting it away before the call keeps the
        bind-parameter count tied to current ledger occupancy (bounded by
        marker TTL and this GC pass itself) rather than to the project's
        total terminal-task count, which only grows over the project's
        lifetime and could otherwise approach SQLite's
        ``SQLITE_MAX_VARIABLE_NUMBER`` on a large, mature project.

        Uses ``SELECT DISTINCT`` scoped to ``project_id`` and
        ``record_kind IN MARKER_KINDS``; rows with ``task_id = ''`` (e.g.
        project-scoped ``cycle_summary`` records, which are not per-task
        markers) are excluded.
        """
        db = self._require_db()
        marker_placeholders = ','.join('?' * len(MARKER_KINDS))
        cursor = await db.execute(
            f"""
            SELECT DISTINCT task_id FROM recon_ledger
            WHERE project_id = ? AND record_kind IN ({marker_placeholders}) AND task_id != ''
            """,
            (project_id, *MARKER_KINDS),
        )
        rows = await cursor.fetchall()
        return {row['task_id'] for row in rows}

    async def gc(
        self,
        project_id: str,
        now: str,
        terminal_task_ids: list[str] | tuple[str, ...],
    ) -> int:
        """Delete expired/terminal-referenced marker rows, and TTL-flip expired
        ``entity_standing_decision`` rows.

        A row is deleted when EITHER ``expires_at < now`` (NULL ``expires_at``
        is never-expire and is naturally excluded — SQL three-valued logic
        means ``NULL < ?`` evaluates to NULL, not true) OR its ``record_kind``
        is one of :data:`MARKER_KINDS` and its ``task_id`` is in
        ``terminal_task_ids``.

        **entity_standing_decision rows are FLIPPED, not deleted, on expiry**
        (task 2894 α, INV-2): recurrence history must be preserved, so this
        kind is excluded from the ``expires_at < now`` DELETE arm in *both*
        branches, and instead — within the same transaction — every ACTIVE
        standing row past its ``expires_at`` is flipped to
        :data:`~fused_memory.reconciliation.standing_decision_constants.STATE_EXPIRED`
        with ``payload_json`` ``expiry_reason`` stamped
        :data:`~fused_memory.reconciliation.standing_decision_constants.EXPIRY_REASON_TTL`.
        The flip is gated on ``state = active`` and so is idempotent — an
        already-expired standing row is neither re-stamped nor deleted. The
        expiry_reason stamp reuses :meth:`mark_addressed`'s per-row
        read-modify-write (json.loads → defensive non-dict wrap → mutate →
        UPDATE) rather than a SQLite ``json_set()``, keeping the module's
        Python-json convention and avoiding a JSON1-extension dependency;
        standing decisions are rare, so the per-row loop cost is negligible.

        Returns ``deleted_count + flipped_count`` — the total number of rows
        this pass acted on (deleted plus TTL-flipped). The sole caller
        (``stages.task_knowledge_sync._gc_recon_markers``) uses the return
        only for logging, and gc passes over a project with no standing rows
        have ``flipped_count == 0``, so the sum is backward-compatible.

        **Multi-task (comma-joined) markers are intentionally NOT decomposed
        here** (task 2228 W5-κ, review finding robustness_regression): the
        terminal-referenced clause is an EXACT string match against the
        stored ``task_id`` column, so a marker written with a comma-joined
        multi-task ``task_id`` (e.g. ``'12,15'``, produced by
        ``flag_dedup.compute_flag_signature``'s ``cited_tasks`` fallback) can
        never equal a single id in ``terminal_task_ids`` — even when every
        cited task has gone terminal. Such markers are not lost; they are
        instead reaped later by the ``expires_at < now`` clause once their TTL
        elapses, rather than promptly on the referenced tasks' terminal
        transition (the retired ``_sweep_terminal_task_flag_markers`` split
        comma-joined ids and deleted when every component was terminal — this
        early-collection path is narrowed to single-task markers, the
        dominant/incident-driving shape, by this collapse). Fail-safe
        direction is preserved either way: uncertain/unmatched => keep, never
        delete on partial information.

        When ``terminal_task_ids`` is empty, the terminal-referenced clause is
        omitted entirely and an expiry-only DELETE runs instead — an explicit
        guard rather than relying on an empty ``IN ()`` (which SQLite treats
        as always-false, but that's implicit engine behaviour, not a
        contract this store leans on).

        ``now`` is compared against ``expires_at`` as plain SQLite TEXT
        (lexicographic string ordering), not a parsed timestamp — this is
        only correct when ``now`` and every stored ``expires_at`` are
        normalized UTC ISO-8601 strings in the same canonical, zero-padded
        format (see :class:`ReconLedgerRecord`). Mixed offset notations
        (``'Z'`` vs ``'+00:00'``) or inconsistent zero-padding would silently
        corrupt this comparison; callers are responsible for passing
        normalized timestamps.
        """
        async with self._txn() as db:
            if terminal_task_ids:
                marker_placeholders = ','.join('?' * len(MARKER_KINDS))
                terminal_placeholders = ','.join('?' * len(terminal_task_ids))
                cursor = await db.execute(
                    f"""
                    DELETE FROM recon_ledger
                    WHERE project_id = ? AND (
                        (expires_at < ? AND record_kind != ?)
                        OR (record_kind IN ({marker_placeholders}) AND task_id IN ({terminal_placeholders}))
                    )
                    """,
                    (project_id, now, RECORD_KIND_ENTITY_STANDING_DECISION, *MARKER_KINDS, *terminal_task_ids),
                )
            else:
                cursor = await db.execute(
                    """
                    DELETE FROM recon_ledger
                    WHERE project_id = ? AND expires_at < ? AND record_kind != ?
                    """,
                    (project_id, now, RECORD_KIND_ENTITY_STANDING_DECISION),
                )
            deleted_count = cursor.rowcount

            # TTL-flip: expired ACTIVE entity_standing_decision rows are kept
            # (excluded from the DELETE above) and flipped to state='expired'
            # with payload expiry_reason='ttl' (INV-2). Gated on state='active'
            # → idempotent; per-row read-modify-write mirrors mark_addressed.
            flip_cursor = await db.execute(
                """
                SELECT project_id, record_kind, task_id, flag_type, run_id, payload_json
                FROM recon_ledger
                WHERE project_id = ? AND record_kind = ? AND state = ?
                  AND expires_at IS NOT NULL AND expires_at < ?
                """,
                (project_id, RECORD_KIND_ENTITY_STANDING_DECISION, STATE_ACTIVE, now),
            )
            flip_rows = await flip_cursor.fetchall()
            flipped_count = 0
            for flip_row in flip_rows:
                payload = json.loads(flip_row['payload_json'])
                # Defensively tolerate a non-object payload (see mark_addressed).
                if not isinstance(payload, dict):
                    payload = {'_payload': payload}
                payload['expiry_reason'] = EXPIRY_REASON_TTL
                await db.execute(
                    """
                    UPDATE recon_ledger SET state = ?, payload_json = ?
                    WHERE project_id = ? AND record_kind = ? AND task_id = ?
                      AND flag_type = ? AND run_id = ?
                    """,
                    (
                        STATE_EXPIRED,
                        json.dumps(payload),
                        flip_row['project_id'],
                        flip_row['record_kind'],
                        flip_row['task_id'],
                        flip_row['flag_type'],
                        flip_row['run_id'],
                    ),
                )
                flipped_count += 1
        return deleted_count + flipped_count

    async def mark_addressed(
        self,
        project_id: str,
        record_kind: str,
        task_id: str = '',
        flag_type: str = '',
        run_id: str = '',
        *,
        addressed_by: str,
        addressed_run_id: str,
    ) -> None:
        """Flip a record's state to 'addressed' and stamp acknowledgement
        metadata into payload_json.

        The §8.1 schema has no ``addressed_by``/``addressed_run_id`` columns,
        so this metadata rides in the ``payload_json`` blob via a
        read-modify-write. A call against an identity with no matching row is
        a silent no-op — it must not raise and must not create a row (avoids
        resurrecting a GC'd marker).

        ``payload_json`` is expected to hold a JSON object (see
        :class:`ReconLedgerRecord`); a non-object payload (e.g. a JSON
        list/number/string) is wrapped as ``{'_payload': <value>}`` before the
        acknowledgement keys are stamped in, so this never raises — and never
        rolls back the transaction — on an unexpected payload shape.
        """
        async with self._txn() as db:
            cursor = await db.execute(
                """
                SELECT payload_json FROM recon_ledger
                WHERE project_id = ? AND record_kind = ? AND task_id = ?
                  AND flag_type = ? AND run_id = ?
                """,
                (project_id, record_kind, task_id, flag_type, run_id),
            )
            row = await cursor.fetchone()
            if row is None:
                return
            payload = json.loads(row['payload_json'])
            # Defensively tolerate a non-object payload_json (valid JSON that
            # isn't a dict — a list, number, or string) instead of letting
            # the item assignment below raise TypeError and roll back the
            # whole transaction. See ReconLedgerRecord's payload_json contract.
            if not isinstance(payload, dict):
                payload = {'_payload': payload}
            payload['addressed_by'] = addressed_by
            payload['addressed_run_id'] = addressed_run_id
            await db.execute(
                """
                UPDATE recon_ledger SET state = 'addressed', payload_json = ?
                WHERE project_id = ? AND record_kind = ? AND task_id = ?
                  AND flag_type = ? AND run_id = ?
                """,
                (json.dumps(payload), project_id, record_kind, task_id, flag_type, run_id),
            )

    async def upsert_entity_standing_decision(
        self,
        *,
        project_id: str,
        entity_uuid: str,
        grounds: str,
        decided_at: str,
        expires_at: str,
        edge_count_at_decision: int,
        evidence: object,
        state: str = STATE_ACTIVE,
        expiry_reason: str | None = None,
    ) -> None:
        """Write (last-write-wins) one ``entity_standing_decision`` ledger row.

        A standing decision records that a class of complaint about a specific
        entity — identified by ``grounds`` (a closed-enum value in
        :data:`~fused_memory.reconciliation.standing_decision_constants.GROUNDS_ENUM`)
        — has been investigated and dismissed, so future recon flags matching
        that (entity, grounds) can be filtered/annotated (γ/δ) rather than
        re-raised. The record kind is
        :data:`~fused_memory.reconciliation.standing_decision_constants.RECORD_KIND_ENTITY_STANDING_DECISION`.

        **PK-slot mapping** (PRD Open Question 3, decided in α): the ledger PK
        is fixed at ``(project_id, record_kind, task_id, flag_type, run_id)``,
        so standing rows encode their natural identity by reusing slots —
        ``task_id=''`` (entity-scoped, no task anchor), ``flag_type=grounds``
        (the closed-enum sub-classification → per-(entity, grounds)
        uniqueness), ``run_id=entity_uuid`` (per-entity uniqueness). The new
        first-class ``entity_uuid`` column mirrors ``run_id`` and is the
        indexed field Hook A/B/sweeps (γ/δ/ζ) query on. ``grounds`` is mirrored
        canonically into ``payload_json`` (it is a COMPARED field, never a
        WHERE-queried column). ``decided_at`` is stored as the row's
        ``created_at``; ``edge_count_at_decision``/``evidence`` (and an optional
        ``expiry_reason``) ride in ``payload_json``.

        **Loud-over-silent validation (INV-1):** ``entity_uuid`` must be a
        non-empty string, ``expires_at`` must not be ``None`` (the never-None
        expires_at invariant — α enforces it at the write boundary while the β
        writer computes ``decided_at + STANDING_DECISION_TTL_DAYS``), and
        ``grounds`` must be a member of ``GROUNDS_ENUM``. A violation raises
        ``ValueError`` naming the failing field rather than writing a malformed
        row.
        """
        if not entity_uuid:
            raise ValueError(
                'upsert_entity_standing_decision: entity_uuid must be a non-empty '
                f'string (got {entity_uuid!r})'
            )
        if expires_at is None:
            raise ValueError(
                'upsert_entity_standing_decision: expires_at must not be None '
                '(never-None-expires_at invariant; standing decisions carry a '
                'decided_at + STANDING_DECISION_TTL_DAYS expiry)'
            )
        if grounds not in GROUNDS_ENUM:
            raise ValueError(
                'upsert_entity_standing_decision: grounds must be a member of '
                f'GROUNDS_ENUM {sorted(GROUNDS_ENUM)} (got {grounds!r})'
            )
        payload: dict[str, object] = {
            'grounds': grounds,
            'decided_at': decided_at,
            'edge_count_at_decision': edge_count_at_decision,
            'evidence': evidence,
        }
        if expiry_reason:
            payload['expiry_reason'] = expiry_reason
        record = ReconLedgerRecord(
            project_id=project_id,
            record_kind=RECORD_KIND_ENTITY_STANDING_DECISION,
            task_id='',
            flag_type=grounds,
            run_id=entity_uuid,
            entity_uuid=entity_uuid,
            payload_json=json.dumps(payload),
            state=state,
            created_at=decided_at,
            expires_at=expires_at,
        )
        await self.upsert(record)

    async def list_entity_standing_decisions(
        self, project_id: str, *, state: str | None = None
    ) -> list[ReconLedgerRecord]:
        """Return the ``entity_standing_decision`` rows for a project.

        Scoped to ``record_kind = entity_standing_decision`` and *project_id*;
        optionally further filtered to a single ``state``. This is the ζ
        growth/merge sweep source and the round-trip read for α's own tests.
        """
        db = self._require_db()
        if state is None:
            cursor = await db.execute(
                """
                SELECT * FROM recon_ledger
                WHERE project_id = ? AND record_kind = ?
                """,
                (project_id, RECORD_KIND_ENTITY_STANDING_DECISION),
            )
        else:
            cursor = await db.execute(
                """
                SELECT * FROM recon_ledger
                WHERE project_id = ? AND record_kind = ? AND state = ?
                """,
                (project_id, RECORD_KIND_ENTITY_STANDING_DECISION, state),
            )
        rows = await cursor.fetchall()
        return [_record_from_row(row) for row in rows]

    async def get_active_entity_standing_decision(
        self, project_id: str, entity_uuid: str
    ) -> ReconLedgerRecord | None:
        """Return the single ACTIVE ``entity_standing_decision`` for
        ``(project_id, entity_uuid)``, or ``None``.

        The shared active-decision-by-uuid lookup Hook A (γ) and Hook B (δ)
        consume (INV-5). Keyed on the indexed ``entity_uuid`` column and
        gated on ``state = active`` — an expired/revoked row (or an unknown
        entity) yields ``None``.

        **At-most-one-active-grounds-per-entity assumption + fast-fail
        (loud-over-silent, INV-1):** an ``entity_uuid`` occupies the ``run_id``
        PK slot while ``grounds`` occupies ``flag_type``, so a single entity
        *can* carry several rows — one per grounds — and, once ``GROUNDS_ENUM``
        grows past its single seed value, more than one of them could be ACTIVE
        at the same time. This helper collapses to a single row, so silently
        returning an arbitrary one of several (an unordered ``LIMIT 1``) would
        make the γ/δ hooks non-deterministic. Rather than pick one under an
        unstated ordering, it fetches up to two active rows and raises
        ``ValueError`` if more than one exists, naming the entity and the
        conflicting grounds. This guard is latent — it cannot fire today
        (single-member ``GROUNDS_ENUM`` ⇒ at most one active row per entity) —
        and the design question it flags (whether this lookup should take an
        explicit ``grounds`` argument) MUST be resolved before a second grounds
        value is admitted to ``GROUNDS_ENUM``.
        """
        db = self._require_db()
        cursor = await db.execute(
            """
            SELECT * FROM recon_ledger
            WHERE project_id = ? AND record_kind = ? AND entity_uuid = ? AND state = ?
            ORDER BY flag_type
            LIMIT 2
            """,
            (project_id, RECORD_KIND_ENTITY_STANDING_DECISION, entity_uuid, STATE_ACTIVE),
        )
        # aiosqlite types fetchall() as Iterable[Row] (not Sized); materialize
        # to a list so len()/indexing below type-check (it's already a list at
        # runtime).
        rows = list(await cursor.fetchall())
        if not rows:
            return None
        if len(rows) > 1:
            raise ValueError(
                'get_active_entity_standing_decision: found more than one ACTIVE '
                f'entity_standing_decision for entity_uuid={entity_uuid!r} in '
                f'project {project_id!r} (grounds={[row["flag_type"] for row in rows]}). '
                'This lookup assumes at most one active grounds per entity; add an '
                'explicit grounds argument before growing GROUNDS_ENUM past one value.'
            )
        return _record_from_row(rows[0])

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
