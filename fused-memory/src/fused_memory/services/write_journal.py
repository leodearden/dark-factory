"""SQLite-backed write journal for durable auditing of all memory writes.

``OPERATOR_TELEMETRY_QUERY`` is the read-only per-write telemetry query, which
operators run through ``fused-memory/scripts/telemetry_query.py`` (OPERATIONS.md
§11). In it, a NULL ``backend_ops.duration_ms`` means the row predates the
column; a value is the measured backend-call time, excluding identity-lock and
queue wait. Its token columns are populated only on graphiti LLM-bearing
writes. They are attributed to their write exactly, but they count what the
upstream LLM client records, so tokens from failed or retried attempts are not
included.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import json
import logging
import time
import uuid as uuid_mod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType

import aiosqlite
from shared.async_sqlite_base import (
    AtomicConnection,
    CheckpointResult,
    apply_full_durability_pragmas,
    connect_daemon,
)

from fused_memory.backends.llm_token_usage import LlmTokenUsage
from fused_memory.services.read_rollup import (
    COUNT_READ_SQL,
    READ_OP_COUNTS_SCHEMA_SQL,
    keeps_read_row,
)

logger = logging.getLogger(__name__)

#: The clock ``prune_write_ops`` measures its wall-clock deadline against, named
#: at module scope so a test can substitute one without reaching into the stdlib
#: ``time`` module globally. Monotonic, so a clock adjustment mid-sweep cannot
#: turn a 30 s budget into an unbounded one.
_monotonic = time.monotonic

SCHEMA_SQL = """
-- The read rollup table read_op_counts (task 5405) is created beside these; its DDL lives in services/read_rollup.py.
CREATE TABLE IF NOT EXISTS write_ops (
    id TEXT PRIMARY KEY,
    causation_id TEXT,
    source TEXT,
    provenance TEXT DEFAULT 'original',
    operation TEXT,
    project_id TEXT,
    agent_id TEXT,
    session_id TEXT,
    kind TEXT NOT NULL DEFAULT 'write',
    params TEXT DEFAULT '{}',
    result_summary TEXT,
    success INTEGER DEFAULT 1,
    error TEXT,
    created_at TEXT NOT NULL,
    -- terminal_* (task 3582): the durable queue's TERMINAL outcome for this
    -- write, written back by DurableWriteQueue's on_terminal hook.
    --
    -- TWO DIFFERENT FACTS, both kept. `success` above means "this operation's
    -- write was ACCEPTED" — for a queued op it is stamped the instant
    -- durable_queue.enqueue() commits, which is genuinely useful for a caller
    -- that got a 200 back. These columns mean "the write LANDED". Before they
    -- existed, a row for a write with a 0% landing rate was byte-for-byte
    -- indistinguishable from a row for a write that landed. terminal_* was
    -- added as a SECOND fact rather than silently redefining `success` under
    -- readers who never asked.
    --
    -- `success` IS NOT PER-LEG (task 4322). "The enqueue was accepted" is
    -- exact only for a single-leg op. MemoryService.add_memory is the sole
    -- dual-leg one: it mints ONE write_op_id for a queued Graphiti leg AND a
    -- synchronous Mem0 leg, then journals a single row here with
    -- success = not (_graphiti_error or _mem0_error) — an AND, so either leg's
    -- failure zeroes it. Its reader reconciliation/stage_stats.py therefore
    -- gates on `success` only for single-leg writes, and reads per-leg
    -- evidence (result_summary's `memory_ids` for the Mem0 leg, `stores` for
    -- the Graphiti one) for THAT row. Scope this to the producer, not to the
    -- operation name: _execute_mem0_write (source='durable_queue') and
    -- _execute_mem0_classify_and_add (provenance='derived') also journal
    -- operation='add_memory' but are single-leg, so their `success` IS exact
    -- and stage_stats keeps gating on it. Read `success` as a whole-op
    -- summary, never as one leg's verdict.
    --
    -- terminal_status domain:
    --   NULL        no terminal outcome recorded — either still in flight, or
    --               an operation that never goes through the durable queue at
    --               all (search / delete_memory / task writes).
    --   'completed' the queue executed the write successfully.
    --   'dead'      the queue item EXHAUSTED ITS ATTEMPTS. Read this as "the
    --               queue gave up", NOT as "the backend write never happened":
    --               DurableWriteQueue._process_item runs the registered
    --               callback (dual_write_episode / refresh_entity_summaries)
    --               AFTER _execute_write has already returned, and a callback
    --               that keeps failing dead-letters an item whose Graphiti
    --               write DID land. Blind-replaying such an item duplicates
    --               the write. That case is separable: the queue prefixes
    --               terminal_error with POST_EXECUTE_DEAD_PREFIX
    --               ('post-execute failure (the backend write LANDED; ...)')
    --               when a backend write for the item had landed. Before
    --               replaying, read the queue row's structured `executed`
    --               rather than matching that prefix; its domain and the
    --               replay rule are in
    --               services/durable_queue.py::DurableWriteQueue.get_dead_items.
    --
    -- LAST-WRITE-WINS: replay_dead resets a dead item to pending, so a
    -- dead-letter that is later replayed and lands correctly re-stamps
    -- 'completed' and clears terminal_error. A sticky-dead rule would leave a
    -- permanently stale 'dead' on a write that did land.
    --
    -- NO WRITE-BACK RACE. Both queue producers INSERT their Layer-1 row AFTER
    -- the enqueue commits (MemoryService.add_episode in a `finally`;
    -- add_memory only after the synchronous Mem0 leg, a network/LLM round trip
    -- of unbounded width), so a worker can legitimately reach a terminal state
    -- BEFORE the row it wants to stamp exists. record_terminal_outcome
    -- therefore UPSERTs rather than UPDATEs: it creates the row carrying the
    -- terminal fact alone (operation NULL — stage_stats skips non-str
    -- operations, and the dashboard's GROUP BY tolerates the bucket), and
    -- log_write_op is itself an upsert whose DO UPDATE deliberately omits the
    -- terminal_* columns, so the producer's late row fills in AROUND an
    -- already-recorded outcome instead of clobbering it. There is no retry
    -- ladder and no bounded window to outrun — a time-boxed retry would drop
    -- the outcome exactly when the producer was slowest, which is exactly when
    -- an operator most needs it.
    --
    -- HISTORICAL ROWS STAY NULL. _migrate() deliberately does NOT backfill
    -- (unlike `kind`): a pre-change row's terminal outcome is genuinely
    -- unknown, and inventing one would repeat the original sin of an audit
    -- trail asserting more than it knows.
    --
    -- NO INDEX. The write-back is `WHERE id = ?`, a PRIMARY KEY seek. Adding
    -- an index here is not free: see the idx_wo_created deployment note below,
    -- where one measured ~47 s of one-time startup DDL on the live 16.6M-row
    -- journal against a ~120 s watchdog startup grace.
    --
    -- OPERATOR AUDIT QUERY, now a single-row read with no join:
    --   SELECT terminal_status, terminal_error FROM write_ops WHERE id = ?
    -- (WriteJournal.get_write_op). Keep the standing caveat that any
    -- JOIN-based query must go through write_op_id, NOT backend_ops.operation:
    -- _execute_graphiti_write journals the literal 'add_episode' for BOTH the
    -- add_episode and add_memory_graphiti paths, which is what reports ~8547
    -- spurious add_episode "successes" when that column is filtered directly.
    terminal_status TEXT,
    terminal_at TEXT,
    terminal_error TEXT
);
CREATE INDEX IF NOT EXISTS idx_wo_causation ON write_ops(causation_id);
CREATE INDEX IF NOT EXISTS idx_wo_project_time ON write_ops(project_id, created_at);
CREATE INDEX IF NOT EXISTS idx_wo_operation ON write_ops(operation);
-- idx_wo_created (task 3304): the dashboard range-filters `created_at` alone
-- in get_memory_ops
-- (dashboard/src/dashboard/data/write_journal.py — cited by function name, not
-- line number, so the reference survives edits to that file). Every other
-- write_ops index has created_at in SECOND position, so none of them can
-- range-seek that predicate — the queries full-scanned all 16.6M rows.
-- Lives here rather than in _migrate() because initialize() runs
-- executescript(SCHEMA_SQL) unconditionally on every start and the DDL is
-- IF NOT EXISTS: this block alone builds the index once on an existing DB and is
-- free thereafter. _migrate() is for ALTER TABLE column additions plus indexes
-- that depend on those new columns; duplicating the DDL there would be lock-step
-- duplication. Same shape as idx_bo_created on backend_ops below.
--
-- One-time build cost, measured 2026-07-31 on a page-for-page copy of the live
-- journal (16,635,866 rows / 7.07 GB): 47.1 s wall, +696,004,608 bytes on disk
-- (+9.9%, 7.069 -> 7.765 GB).
--
-- DEPLOYMENT — that 47.1 s is NOT 73 s of headroom under the ~120 s watchdog
-- grace; the grace covers the WHOLE start-to-listening path, not just this DDL.
-- initialize() is awaited at server/main.py:545, well before uvicorn accepts
-- traffic, and the baseline it adds to (FalkorDB + Qdrant cold-start handshake
-- plus event-buffer replay) "routinely runs 30-60s" per
-- scripts/fused-memory.service.template. So the first start after this lands is
-- ~77-107 s against STARTUP_GRACE_SECS = 120 in scripts/orchestrator-watchdog.py
-- — roughly 13-43 s of real margin. Blow it and fused_memory_liveness_pass()
-- sees port 8002 down past the grace and issues stop->start, SIGTERMing the
-- in-progress CREATE INDEX; SQLite rolls the build back and the next start pays
-- the full 47 s again — a restart loop that never converges, on the one restart
-- that is supposed to deliver the index. The same restart can also blow the
-- orchestrator's FM_RESTART_RETRY_WINDOW_SECS (120 s, sized against a "~15s
-- observed fm start") and any deterministic deploy hook left at
-- DeterministicRunner's default timeout_secs = 60.
-- PREFERRED, therefore: build it out-of-band with fused-memory STOPPED, before
-- the deploy restart --
--   sqlite3 write_journal.db 'CREATE INDEX IF NOT EXISTS idx_wo_created ON write_ops(created_at);'
-- -- after which initialize() finds it present and IF NOT EXISTS costs nothing.
-- The alternative is to widen STARTUP_GRACE_SECS (and the deploy hook's
-- timeout_secs) for that one restart. Either way, NEVER build it against a
-- SERVING instance: busy_timeout is 5000 ms and log_write_op swallows its
-- errors, so a multi-minute write-lock hold silently drops journal rows.
--
-- Steady state: created_at is monotonically increasing, so every insert is a
-- right-most B-tree append — write amplification on the hot log_write_op path is
-- negligible. The +9.9% is permanent and grows with the table, though:
-- prune_write_ops bounds the rows, and services/journal_growth_alarm.py watches the file.
CREATE INDEX IF NOT EXISTS idx_wo_created ON write_ops(created_at);

CREATE TABLE IF NOT EXISTS backend_ops (
    id TEXT PRIMARY KEY,
    write_op_id TEXT,
    causation_id TEXT,
    backend TEXT,
    operation TEXT,
    payload TEXT DEFAULT '{}',
    result_summary TEXT,
    success INTEGER DEFAULT 1,
    error TEXT,
    created_at TEXT NOT NULL,
    duration_ms REAL  -- backend-call milliseconds; NULL = not measured / predates the column
);
CREATE INDEX IF NOT EXISTS idx_bo_write_op ON backend_ops(write_op_id);
CREATE INDEX IF NOT EXISTS idx_bo_causation ON backend_ops(causation_id);
CREATE INDEX IF NOT EXISTS idx_bo_created ON backend_ops(created_at);

-- mem0_intents: write-ahead intent journal (task 2710). Grows by one row per
-- mem0 write (the happy path logs 'pending' then stamps 'completed'), so
-- terminal completed/failed rows are aged out by WriteJournal.prune_mem0_intents
-- (run at startup after recovery); 'dead' rows are preserved for manual replay.
CREATE TABLE IF NOT EXISTS mem0_intents (
    id TEXT PRIMARY KEY,
    write_op_id TEXT,
    causation_id TEXT,
    project_id TEXT,
    agent_id TEXT,
    session_id TEXT,
    category TEXT,
    payload_digest TEXT,
    content TEXT,
    metadata TEXT DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'pending',
    reason TEXT,
    created_at TEXT NOT NULL,
    resolved_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_mi_status ON mem0_intents(status);
CREATE INDEX IF NOT EXISTS idx_mi_write_op ON mem0_intents(write_op_id);

-- idempotent_ops: client-supplied idempotency keys for mutating task writes
-- (task 2712). client_op_id PRIMARY KEY IS the required unique index, so a
-- retry with the same key records once (INSERT OR IGNORE, first-write-wins)
-- and replays the recorded outcome. CREATE TABLE IF NOT EXISTS runs on every
-- initialize(), so existing DBs gain the table with no ALTER migration.
CREATE TABLE IF NOT EXISTS idempotent_ops (
    client_op_id TEXT PRIMARY KEY,
    operation TEXT NOT NULL,
    result TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- referent_findings: the durable half of leaf zeta's diagnosis (task 4984).
-- `_verify_episode_referents` returns its findings in-process and they are then
-- gone, so an episode whose edges were fully diagnosed with no path to repair
-- left nothing a later pass could act on.
--
-- A DIAGNOSIS LOG, NOT A WORK QUEUE. There is deliberately no status/claimed
-- column: a row records what was TRUE of the graph at one moment, not an
-- instruction, so the phase-5 replay pass must re-verify a row against the live
-- graph before acting on it. That is the fail-closed direction, and the same one
-- `ReferentFinding.resolvable` takes by defaulting to False. A status column now
-- would pre-empt phase 5's own read-path choice with speculative generality.
--
-- HOW A READER DRAINS IT, given no status column: WriteJournal.get_referent_findings
-- returns the OLDEST page and takes an `after_seq` cursor over the sqlite rowid,
-- which is the only total insertion order here — one episode's findings are
-- written in ONE batched commit and deliberately share a single `created_at`.
-- RETENTION is NOT yet wired: the startup sweep that runs prune_mem0_intents /
-- prune_idempotent_ops / prune_write_ops lives in server/main.py, outside task
-- 4984's locks, so growth is watched rather than bounded for now (the write path
-- is the ~0.2%-of-edges finding path, so the rate is low, not zero).
--
-- No ALTER migration: initialize() runs executescript(SCHEMA_SQL) unconditionally
-- and this DDL is IF NOT EXISTS, so a fresh AND an existing database both pick
-- the table up on the next start (same reasoning as the idx_wo_created block
-- above). _migrate() is for ALTER TABLE column additions.
CREATE TABLE IF NOT EXISTS referent_findings (
    id TEXT PRIMARY KEY,
    group_id TEXT,
    episode_uuid TEXT,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);
-- ONE index, serving the ONLY planned read's filter (`WHERE group_id = ?`); the
-- rowid ordering that read applies on top of it is free, since an index entry
-- already carries the rowid. The idx_wo_created note above measures what a new
-- index costs on a populated table — 47 s of startup DDL against a ~120 s
-- watchdog grace; this table starts EMPTY so that build is free today, and must
-- not be multiplied later without taking that measurement again.
CREATE INDEX IF NOT EXISTS idx_rf_group_time ON referent_findings(group_id, created_at);
"""


def _prune_batch_sql(horizon_rows: str) -> str:
    return (
        'DELETE FROM write_ops WHERE id IN ('
        'SELECT id FROM write_ops INDEXED BY idx_wo_kind_time '
        f'WHERE {horizon_rows} AND created_at >= ? AND created_at < ? '
        'ORDER BY created_at LIMIT ?) RETURNING created_at'
    )


#: One batch of ``WriteJournal.prune_write_ops`` per retention horizon, in the
#: order it drains them: delete the oldest ``LIMIT`` rows of the horizon created
#: in ``[resume_from, cutoff)`` and return their ``created_at``. Parameters:
#: ``(resume_from, cutoff, limit)``. Each range-seeks idx_wo_kind_time, and
#: INDEXED BY pins it, so a missing index errors instead of scanning.
WRITE_OPS_PRUNE_BATCH_SQL: Mapping[str, str] = MappingProxyType({
    'read': _prune_batch_sql("kind = 'read' AND operation <> 'search'"),
    'search': _prune_batch_sql("kind = 'read' AND operation = 'search'"),
    'write': _prune_batch_sql("kind = 'write'"),
})

#: Parameters: ``(since_iso_timestamp, limit)``.
OPERATOR_TELEMETRY_QUERY = """
SELECT
    bo.created_at AS created_at,
    wo.operation AS operation,
    wo.project_id AS project_id,
    bo.backend AS backend,
    bo.success AS success,
    bo.duration_ms AS duration_ms,
    CASE WHEN json_valid(bo.result_summary)
        THEN json_extract(bo.result_summary, '$.tokens.input_tokens') END AS input_tokens,
    CASE WHEN json_valid(bo.result_summary)
        THEN json_extract(bo.result_summary, '$.tokens.output_tokens') END AS output_tokens,
    CASE WHEN json_valid(bo.result_summary)
        THEN json_extract(bo.result_summary, '$.tokens.total_tokens') END AS total_tokens,
    CASE WHEN json_valid(bo.result_summary)
        THEN json_extract(bo.result_summary, '$.tokens.llm_calls') END AS llm_calls
FROM backend_ops AS bo
LEFT JOIN write_ops AS wo ON wo.id = bo.write_op_id
WHERE bo.created_at >= ?
ORDER BY bo.created_at DESC
LIMIT ?
"""


def _backend_result_summary(
    result_summary: dict | str | None, llm_tokens: LlmTokenUsage | None
) -> dict | str | None:
    """The persisted ``result_summary``: nested under ``result`` beside the
    ``tokens`` block ``OPERATOR_TELEMETRY_QUERY`` reads, when tokens were measured."""
    if llm_tokens is None:
        return result_summary
    if not isinstance(llm_tokens, LlmTokenUsage):
        raise TypeError(
            f'llm_tokens must be an LlmTokenUsage or None, got {type(llm_tokens).__name__}'
        )
    return {'result': result_summary, 'tokens': llm_tokens.as_journal_dict()}


@dataclass(frozen=True)
class JournalGrowthSample:
    """One growth measurement of the journal, taken by ``WriteJournal.growth_sample``.

    ``file_bytes`` is the on-disk footprint (``.db`` + ``-wal`` + ``-shm``);
    ``free_bytes`` is the part of it on the freelist, reusable by inserts and
    reclaimable only by VACUUM; ``rows_inserted`` counts ``write_ops`` inserts
    at or after ``since``.
    """

    file_bytes: int
    free_bytes: int
    rows_inserted: int
    since: datetime


_UPSERT_WRITE_OP_SQL = """INSERT INTO write_ops
   (id, causation_id, source, provenance, operation,
    project_id, agent_id, session_id, kind,
    params, result_summary, success, error, created_at)
   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
   ON CONFLICT(id) DO UPDATE SET
       causation_id = excluded.causation_id,
       source = excluded.source,
       provenance = excluded.provenance,
       operation = excluded.operation,
       project_id = excluded.project_id,
       agent_id = excluded.agent_id,
       session_id = excluded.session_id,
       kind = excluded.kind,
       params = excluded.params,
       result_summary = excluded.result_summary,
       success = excluded.success,
       error = excluded.error,
       created_at = excluded.created_at"""


async def _read_int(access: AtomicConnection, sql: str) -> int:
    """The single integer *sql* selects, or a RuntimeError naming *sql*."""
    row = await access.read_one(sql)
    if row is None or row[0] is None:
        raise RuntimeError(f'write_journal growth sample: {sql!r} returned no value')
    return row[0]


class WriteJournal:
    """Two-layer write journal backed by SQLite (WAL mode)."""

    def __init__(self, data_dir: Path | str):
        self.data_dir = Path(data_dir)
        self._access: AtomicConnection | None = None
        self._dropped: collections.Counter[str] = collections.Counter()
        self._pruned_below: dict[str, str] = {}

    @property
    def db_path(self) -> Path:
        return self.data_dir / 'write_journal.db'

    async def initialize(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        db_path = self.db_path
        conn = await connect_daemon(str(db_path))
        conn.row_factory = aiosqlite.Row
        await apply_full_durability_pragmas(conn, busy_timeout_ms=5000)
        self._access = AtomicConnection(conn)
        async with self._access.write() as db:
            await db.executescript(SCHEMA_SQL)
            await db.executescript(READ_OP_COUNTS_SCHEMA_SQL)
        await self._migrate()
        logger.info(f'Write journal initialized at {db_path}')

    async def close(self) -> None:
        if self._access is not None:
            await self._access.close()
            self._access = None

    async def checkpoint(self) -> CheckpointResult:
        return await AtomicConnection.checkpoint_or_unavailable(self._access)

    async def growth_sample(self, *, since: datetime) -> JournalGrowthSample:
        """Measure the journal's size and insert volume since ``since``.

        Every statement is O(1) or a single index seek, because the live
        ``write_ops`` table holds tens of millions of rows. The consumer is
        ``services/journal_growth_alarm.py``.

        ``rows_inserted`` relies on rowid order tracking ``created_at`` order, and
        counts INSERTS: rows the retention prune has since deleted still count.
        One known skew: ``log_write_op``'s upsert re-stamps ``created_at``, so the
        boundary can shift by the rows inserted during one write's enqueue-to-journal
        latency.
        """
        access = self._require_access()
        freelist_count = await _read_int(access, 'PRAGMA freelist_count')
        page_size = await _read_int(access, 'PRAGMA page_size')
        return JournalGrowthSample(
            file_bytes=await asyncio.to_thread(self._on_disk_bytes),
            free_bytes=freelist_count * page_size,
            rows_inserted=await self._rows_inserted_since(access, since),
            since=since,
        )

    @staticmethod
    async def _rows_inserted_since(access: AtomicConnection, since: datetime) -> int:
        # MIN and MAX in ONE select is a full index scan (1m28s on the live
        # journal); each alone is a seek. INDEXED BY makes a missing index an
        # error instead of a silent table scan.
        first_row = await access.read_one(
            'SELECT rowid FROM write_ops INDEXED BY idx_wo_created '
            'WHERE created_at >= ? ORDER BY created_at LIMIT 1',
            (since.isoformat(),),
        )
        if first_row is None:
            return 0
        newest_rowid = await _read_int(access, 'SELECT MAX(rowid) FROM write_ops')
        return newest_rowid - first_row[0] + 1

    def _on_disk_bytes(self) -> int:
        total = 0
        for suffix in ('', '-wal', '-shm'):
            with contextlib.suppress(FileNotFoundError):
                total += self.db_path.with_name(self.db_path.name + suffix).stat().st_size
        return total

    def _require_access(self) -> AtomicConnection:
        if self._access is None:
            raise RuntimeError('WriteJournal not initialized — call initialize() first')
        return self._access

    async def _migrate(self) -> None:
        """Add columns introduced after initial schema (idempotent)."""
        access = self._require_access()
        existing = {row[1] for row in await access.read_all('PRAGMA table_info(write_ops)')}
        existing_backend_ops = {
            row[1] for row in await access.read_all('PRAGMA table_info(backend_ops)')
        }

        async with access.write() as db:
            if 'session_id' not in existing:
                await db.execute('ALTER TABLE write_ops ADD COLUMN session_id TEXT')
                logger.info('Migration: added session_id column to write_ops')

            if 'kind' not in existing:
                await db.execute(
                    "ALTER TABLE write_ops ADD COLUMN kind TEXT NOT NULL DEFAULT 'write'"
                )
                await db.execute(
                    "UPDATE write_ops SET kind = 'read' WHERE operation = 'search'"
                )
                logger.info('Migration: added kind column to write_ops, backfilled reads')

            # terminal_* (task 3582): the durable queue's eventual outcome,
            # written back onto the write_op row. NULLable with no
            # `NOT NULL DEFAULT`, which keeps SQLite's ADD COLUMN O(1) on the
            # 16.6M-row live journal (no table rewrite, no startup stall — see
            # the idx_wo_created deployment note in SCHEMA_SQL above).
            # Deliberately NOT backfilled (unlike `kind`): a pre-change row's
            # terminal outcome is genuinely unknown and must stay NULL rather
            # than be guessed.
            if 'terminal_status' not in existing:
                await db.execute('ALTER TABLE write_ops ADD COLUMN terminal_status TEXT')
                logger.info('Migration: added terminal_status column to write_ops')

            if 'terminal_at' not in existing:
                await db.execute('ALTER TABLE write_ops ADD COLUMN terminal_at TEXT')
                logger.info('Migration: added terminal_at column to write_ops')

            if 'terminal_error' not in existing:
                await db.execute('ALTER TABLE write_ops ADD COLUMN terminal_error TEXT')
                logger.info('Migration: added terminal_error column to write_ops')

            # Same O(1), no-backfill stance as terminal_* above: a historical
            # row's duration is unknown, so it stays NULL.
            if 'duration_ms' not in existing_backend_ops:
                await db.execute('ALTER TABLE backend_ops ADD COLUMN duration_ms REAL')
                logger.info('Migration: added duration_ms column to backend_ops')

            # Indexes on new columns (safe after migration ensures columns exist)
            await db.execute(
                'CREATE INDEX IF NOT EXISTS idx_wo_kind_time ON write_ops(kind, created_at)'
            )
            await db.execute(
                'CREATE INDEX IF NOT EXISTS idx_wo_agent_time ON write_ops(agent_id, created_at)'
            )

    async def log_write_op(
        self,
        *,
        write_op_id: str,
        causation_id: str | None = None,
        source: str = 'mcp_tool',
        provenance: str = 'original',
        operation: str,
        project_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        kind: str = 'write',
        params: dict | None = None,
        result_summary: dict | str | None = None,
        success: bool = True,
        error: str | None = None,
    ) -> None:
        """Log a Layer 1 operation. Fire-and-forget — never raises.

        A successful read with no per-row consumer (``read_rollup.keeps_read_row``
        is False) is COUNTED into ``read_op_counts`` and never stored (task 5405).
        Each read therefore lands in exactly one of the two tables, which is what
        lets the dashboard add them.

        An UPSERT rather than a plain INSERT (task 3582): the durable queue's
        terminal write-back can legitimately reach this ``write_op_id`` before
        its producer gets here — ``add_memory`` enqueues the Graphiti leg and
        only journals this row after the synchronous Mem0 leg, an unbounded
        network/LLM round trip — in which case ``record_terminal_outcome`` has
        already created the row carrying the terminal fact alone. The
        ``DO UPDATE`` therefore fills in every Layer-1 column and DELIBERATELY
        OMITS ``terminal_status`` / ``terminal_at`` / ``terminal_error``, so the
        late producer completes that row instead of clobbering the outcome it
        was racing.

        ``agent_id`` READS DIFFERENTLY ON ONE ROW KIND. Interceptor rows
        normally carry the literal ``'task-interceptor'`` — the component that
        wrote the row. On a ``set_task_status`` DONE row it is instead the
        RESOLVED CALLER, and ``NULL`` when the caller supplied no identity
        (task 5241, PRD C5): that row is the audit surface for who recorded a
        task's completion, so naming the component there would assert a caller
        that is really just the writer. Such a row also carries
        ``params['done_provenance_kind']`` and lands for a REFUSED write too,
        with ``success=0`` and the refusal in ``error``.

        A ``search`` IS ALWAYS STORED AS ``kind='read'``, because the search
        retention horizon of :meth:`prune_write_ops` selects on that pair; any
        other ``kind`` passed with it is logged and overridden.
        """
        if operation == 'search' and kind != 'read':
            logger.warning(
                "write_op %s: operation='search' journalled with kind=%r; stored as "
                "kind='read', the kind the search retention horizon selects",
                write_op_id, kind,
            )
            kind = 'read'
        created_at = datetime.now(UTC).isoformat()
        counted = kind == 'read' and not keeps_read_row(operation=operation, success=success)
        try:
            async with self._require_access().write() as db:
                if counted:
                    await db.execute(
                        COUNT_READ_SQL, (created_at, project_id, agent_id, operation, 1)
                    )
                else:
                    await db.execute(
                        _UPSERT_WRITE_OP_SQL,
                        (
                            write_op_id,
                            causation_id,
                            source,
                            provenance,
                            operation,
                            project_id,
                            agent_id,
                            session_id,
                            kind,
                            json.dumps(params) if params else '{}',
                            json.dumps(result_summary) if isinstance(result_summary, dict) else result_summary,
                            1 if success else 0,
                            error,
                            created_at,
                        ),
                    )
        except Exception as e:
            self._dropped[operation] += 1
            logger.warning(f'Failed to log write_op: {e}')

    async def log_backend_op(
        self,
        *,
        write_op_id: str | None = None,
        causation_id: str | None = None,
        backend: str,
        operation: str,
        payload: dict | None = None,
        result_summary: dict | str | None = None,
        success: bool = True,
        error: str | None = None,
        duration_ms: float | None = None,
        llm_tokens: LlmTokenUsage | None = None,
    ) -> None:
        """Log a Layer 2 backend dispatch. Fire-and-forget — never raises."""
        try:
            summary = _backend_result_summary(result_summary, llm_tokens)
            async with self._require_access().write() as db:
                await db.execute(
                    """INSERT INTO backend_ops
                       (id, write_op_id, causation_id, backend, operation,
                        payload, result_summary, success, error, created_at,
                        duration_ms)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(uuid_mod.uuid4()),
                        write_op_id,
                        causation_id,
                        backend,
                        operation,
                        json.dumps(payload) if payload else '{}',
                        json.dumps(summary) if isinstance(summary, dict) else summary,
                        1 if success else 0,
                        error,
                        datetime.now(UTC).isoformat(),
                        duration_ms,
                    ),
                )
        except Exception as e:
            self._dropped[operation] += 1
            logger.warning(f'Failed to log backend_op: {e}')

    async def log_referent_finding(
        self, *, payload: dict, group_id: str, episode_uuid: str
    ) -> None:
        """Persist ONE referent finding. Fire-and-forget — never raises.

        The singular convenience over :meth:`log_referent_findings`, which owns
        the only INSERT. One commit either way, so a caller already holding an
        episode's findings should hand them over together rather than call this
        in a loop — see that method's BATCHED note for what the loop costs.
        """
        await self.log_referent_findings(
            [payload], group_id=group_id, episode_uuid=episode_uuid,
        )

    async def log_referent_findings(
        self, payloads: Sequence[dict], *, group_id: str, episode_uuid: str
    ) -> None:
        """Persist ONE EPISODE's referent findings, in ONE commit. Never raises.

        Modelled on :meth:`log_backend_op` and counting its losses on the SAME
        ``_dropped`` counter (INV-5) rather than growing a second, divergent
        one. The contract matters to the caller: it runs inside the per-group
        identity lock AFTER the episode write has already committed, so a
        journal fault must cost a diagnosis and never the write — and because
        the guard is here, no call site needs a try/except of its own.

        BATCHED, not per finding, because every commit here is a
        ``synchronous=FULL`` fsync (~1-5 ms, and up to the 5000 ms
        ``busy_timeout`` under contention) taken while that identity lock
        serializes same-group writes. An episode's finding count has NO
        ceiling — the verify pass's warn cap is documented as capping the log
        and nothing else — so a storm episode of 50-100 misattached ends would
        hold the lock for 50-500 ms of fsyncs one row at a time. One
        transaction makes that cost per EPISODE instead of per finding.

        A dropped batch counts EVERY row it lost, because
        :meth:`journal_drop_stats` reports rows LOST rather than calls failed.
        """
        if not payloads:
            return
        try:
            # ONE timestamp for the batch: these rows are one episode's
            # diagnosis, taken under one lock at one moment, and stamping them
            # microseconds apart would invent a precision the observation does
            # not have. `get_referent_findings` orders on the rowid precisely
            # so a shared `created_at` cannot scramble their order.
            created_at = datetime.now(UTC).isoformat()
            async with self._require_access().write() as db:
                await db.executemany(
                    """INSERT INTO referent_findings
                       (id, group_id, episode_uuid, payload, created_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    [
                        (
                            str(uuid_mod.uuid4()),
                            group_id,
                            episode_uuid,
                            json.dumps(payload),
                            created_at,
                        )
                        for payload in payloads
                    ],
                )
        except Exception as e:
            self._dropped['referent_finding'] += len(payloads)
            logger.warning(
                f'Failed to log {len(payloads)} referent_finding row(s): {e}'
            )

    async def get_referent_findings(
        self,
        *,
        group_id: str | None = None,
        after_seq: int | None = None,
        limit: int = 1000,
    ) -> list[dict]:
        """The OLDEST *limit* findings, in insertion order, ``payload`` decoded.

        OLDEST-FIRST AND TRUNCATED AT *limit* — say it plainly, because the
        table has no prune path and no status column to mark a row handled (it
        is a diagnosis log, not a work queue). A consumer of a table longer
        than one page that does not carry *after_seq* forward therefore re-reads
        the same oldest page forever and never reaches a newer diagnosis. Pass
        the ``seq`` of the last row handled to get the next page.

        ``seq`` is the row's sqlite ``rowid``: assigned at INSERT, so it is the
        ONE key that totally orders these rows the way they were written —
        including inside a batched commit, where every row deliberately shares
        one ``created_at``. It is also gap-tolerant as a cursor: a future
        retention sweep deleting the oldest rows cannot renumber the newer ones
        under a reader mid-page.

        Unlike :meth:`log_referent_findings` this one MAY raise: it is an
        operator/replay read, not a hot write path, and an unreadable journal is
        an answer the caller has to see rather than an empty list to act on.
        """
        clauses: list[str] = []
        params: list = []
        if group_id is not None:
            clauses.append('group_id = ?')
            params.append(group_id)
        if after_seq is not None:
            clauses.append('rowid > ?')
            params.append(after_seq)
        where = f'WHERE {" AND ".join(clauses)} ' if clauses else ''
        params.append(limit)
        rows = await self._require_access().read_all(
            f'SELECT rowid AS seq, * FROM referent_findings {where}'
            'ORDER BY rowid LIMIT ?',
            params,
        )
        return [{**dict(row), 'payload': json.loads(row['payload'])} for row in rows]

    def journal_drop_stats(self) -> dict:
        """Return ``{'dropped_total': int, 'by_operation': dict}`` — rows LOST.

        Both ``log_write_op`` and ``log_backend_op`` are fire-and-forget by
        design, which means a burst of lost rows is otherwise indistinguishable
        from a quiet period. That fail-soft became LOAD-BEARING the moment leaf
        eta (task 3213) began computing a metric from these rows: a dropped
        search row is a search the metric silently scores as "never asked",
        which is a wrong answer rather than a missing one.

        Process-local and reset by a restart, like every other in-process
        counter here — this exists to make a live burst VISIBLE, not to keep
        durable statistics. The returned dicts are copies, so a reader cannot
        mutate the journal's internal state.
        """
        return {
            'dropped_total': sum(self._dropped.values()),
            'by_operation': dict(self._dropped),
        }

    def record_journal_drop(self, operation: str) -> None:
        """Count a row lost by an OUT-OF-PACKAGE fallback on the same counter.

        ``server/tools.py::_log_read`` has its own outer ``except``; it fires
        rarely, because the two log methods above swallow their failures first,
        but when it does the row is just as lost. It feeds this counter rather
        than growing a second, divergent one (INV-5).
        """
        self._dropped[operation] += 1

    async def get_ops_by_causation(self, causation_id: str) -> list[dict]:
        """Return all write_ops and backend_ops for a causation_id."""
        access = self._require_access()
        write_ops = await access.read_all(
            'SELECT * FROM write_ops WHERE causation_id = ? ORDER BY created_at',
            (causation_id,),
        )
        backend_ops = await access.read_all(
            'SELECT * FROM backend_ops WHERE causation_id = ? ORDER BY created_at',
            (causation_id,),
        )
        results = [{'layer': 'write_op', **dict(row)} for row in write_ops]
        results += [{'layer': 'backend_op', **dict(row)} for row in backend_ops]

        results.sort(key=lambda r: r.get('created_at', ''))
        return results

    async def get_stored_ops_since(
        self, since: str, limit: int = 100, kind: str | None = None
    ) -> list[dict]:
        """Return the ``write_ops`` rows STORED since a timestamp, optionally by kind.

        Successful reads without a per-row consumer are counted into
        ``read_op_counts`` and never stored (``read_rollup.py``), so this is not
        an activity count: ``kind='read'`` yields searches and failed reads only.
        """
        if kind:
            sql = 'SELECT * FROM write_ops WHERE created_at >= ? AND kind = ? ORDER BY created_at LIMIT ?'
            params = (since, kind, limit)
        else:
            sql = 'SELECT * FROM write_ops WHERE created_at >= ? ORDER BY created_at LIMIT ?'
            params = (since, limit)
        return [dict(row) for row in await self._require_access().read_all(sql, params)]

    async def get_write_op(self, write_op_id: str) -> dict | None:
        """Return a single ``write_ops`` row by id, or ``None`` on a miss.

        The no-join audit read. ``success`` answers "was the write ACCEPTED"
        (the enqueue committed); ``terminal_status`` answers "did the write
        LAND" (the durable queue reached ``completed`` / ``dead``). Reading
        this row alone is now sufficient to tell those two facts apart — no
        ``backend_ops`` join required, which matters because
        ``backend_ops.operation`` is the literal ``'add_episode'`` for BOTH
        the ``add_episode`` and ``add_memory_graphiti`` paths.
        """
        row = await self._require_access().read_one(
            'SELECT * FROM write_ops WHERE id = ?',
            (write_op_id,),
        )
        return dict(row) if row is not None else None

    async def get_backend_ops_for_write_op(self, write_op_id: str) -> list[dict]:
        """Return all backend_ops linked to a write_op."""
        rows = await self._require_access().read_all(
            'SELECT * FROM backend_ops WHERE write_op_id = ? ORDER BY created_at',
            (write_op_id,),
        )
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # idempotent_ops: client-supplied idempotency keys for mutating task
    # writes (task 2712 / survey finding D5)
    # ------------------------------------------------------------------

    async def get_idempotent_result(self, client_op_id: str) -> dict | None:
        """Replay a previously-recorded result for ``client_op_id``.

        Returns the recorded outcome dict on a hit, or ``None`` on a miss.
        Fails OPEN: any read error logs a warning and returns ``None`` (so a
        rare DB hiccup lets the write proceed rather than wedging it) — the
        journal's never-block discipline.
        """
        try:
            row = await self._require_access().read_one(
                'SELECT result FROM idempotent_ops WHERE client_op_id = ?',
                (client_op_id,),
            )
            if row is None:
                return None
            return json.loads(row[0])
        except Exception as e:
            logger.warning(f'Failed to read idempotent_ops for {client_op_id}: {e}')
            return None

    async def record_idempotent_result(
        self, client_op_id: str, operation: str, result: dict
    ) -> None:
        """Record the outcome of a mutating write under ``client_op_id``.

        ``INSERT OR IGNORE`` on the ``client_op_id`` PRIMARY KEY makes the
        first write win: a duplicate key (a rare concurrent double) is a no-op,
        never a corruption or an overwrite. Fire-and-forget — logs on failure
        but never raises (mirrors ``log_write_op``).
        """
        try:
            async with self._require_access().write() as db:
                await db.execute(
                    """INSERT OR IGNORE INTO idempotent_ops
                       (client_op_id, operation, result, created_at)
                       VALUES (?, ?, ?, ?)""",
                    (
                        client_op_id,
                        operation,
                        json.dumps(result, default=str),
                        datetime.now(UTC).isoformat(),
                    ),
                )
        except Exception as e:
            logger.warning(
                f'Failed to record idempotent_ops for {client_op_id}: {e}'
            )

    async def prune_idempotent_ops(self, *, older_than_days: float = 7.0) -> int:
        """Age out old ``idempotent_ops`` rows past a retention window.

        One row accrues per mutating task write (update_task / set_task_status /
        add_dependency / remove_dependency), so — like ``mem0_intents`` — this
        table grows without bound if never swept. A client idempotency key only
        needs to outlive the ambiguous-transport-failure retry window it guards
        (seconds to minutes; a higher-level retry at most hours), so an
        aggressive TTL is safe: rows whose ``created_at`` is older than
        ``older_than_days`` are deleted and the count returned. Intended to run
        at startup alongside ``prune_mem0_intents``.

        Fire-and-forget — logs loudly on failure but never raises, so a prune
        hiccup cannot crash startup (mirrors ``prune_mem0_intents``).
        """
        try:
            cutoff = (datetime.now(UTC) - timedelta(days=older_than_days)).isoformat()
            deleted = 0
            async with self._require_access().write() as db:
                cursor = await db.execute(
                    'DELETE FROM idempotent_ops WHERE created_at < ?',
                    (cutoff,),
                )
                deleted = max(cursor.rowcount, 0)
            if deleted:
                logger.info(
                    'Pruned %d idempotent_ops rows older than %s days',
                    deleted,
                    older_than_days,
                )
            return deleted
        except Exception as e:
            logger.error(f'Failed to prune idempotent_ops: {e}')
            return 0

    # ------------------------------------------------------------------
    # mem0_intents: write-ahead intent journal for the dual-store add path
    # (task 2710 / survey finding D3)
    # ------------------------------------------------------------------

    async def log_mem0_intent(
        self,
        *,
        intent_id: str,
        write_op_id: str,
        causation_id: str | None = None,
        project_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        category: str | None = None,
        content: str,
        metadata: dict | None = None,
        payload_digest: str | None,
    ) -> None:
        """Write-ahead a Mem0 write intent.

        Committed synchronously (durable-before-await) BEFORE the risky
        Mem0 ``add()`` call — mirrors ``DurableWriteQueue.enqueue``'s
        commit-before-return so a crash mid-Mem0-write leaves a durable
        ``pending`` trace instead of a silently orphaned Graphiti twin.
        Stores the FULL content + metadata + scope so recovery can faithfully
        re-issue (``write_ops`` truncates content to 200 chars).

        Fire-and-forget: logs loudly (ERROR) on failure but never raises — a
        journaling hiccup must not fail an otherwise-good write.
        """
        try:
            async with self._require_access().write() as db:
                await db.execute(
                    """INSERT INTO mem0_intents
                       (id, write_op_id, causation_id, project_id, agent_id,
                        session_id, category, payload_digest, content, metadata,
                        status, reason, created_at, resolved_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', NULL, ?, NULL)""",
                    (
                        intent_id,
                        write_op_id,
                        causation_id,
                        project_id,
                        agent_id,
                        session_id,
                        category,
                        payload_digest,
                        content,
                        json.dumps(metadata) if metadata else '{}',
                        datetime.now(UTC).isoformat(),
                    ),
                )
        except Exception as e:
            logger.error(f'Failed to log mem0_intent {intent_id}: {e}')

    async def resolve_mem0_intent(
        self, intent_id: str, status: str, reason: str | None = None
    ) -> None:
        """Stamp a mem0_intent terminal (``completed``|``failed``|``dead``).

        Fire-and-forget — logs loudly on failure but never raises.
        """
        try:
            async with self._require_access().write() as db:
                await db.execute(
                    'UPDATE mem0_intents SET status = ?, reason = ?, resolved_at = ? '
                    'WHERE id = ?',
                    (status, reason, datetime.now(UTC).isoformat(), intent_id),
                )
        except Exception as e:
            logger.error(
                f'Failed to resolve mem0_intent {intent_id} -> {status}: {e}'
            )

    async def record_terminal_outcome(
        self,
        *,
        write_op_id: str,
        terminal_status: str,
        terminal_error: str | None = None,
    ) -> bool:
        """Stamp the durable queue's terminal outcome back onto a write_op.

        ``terminal_status`` is ``'completed'`` or ``'dead'``. Returns whether
        the outcome was durably recorded.

        ``success`` is deliberately LEFT UNTOUCHED: it means "this operation's
        write was accepted", and its reader
        (``reconciliation/stage_stats.py::derive_stage_stats``) keeps its
        current meaning. This method records a second, different fact — "the
        write landed" — rather than redefining the first one under readers who
        never asked. That reader gates on ``success`` for single-leg operations
        and on per-leg evidence for the dual-leg ``add_memory`` row, whose
        ``success`` is an AND across both legs (see the ``write_ops`` schema
        comment in ``SCHEMA_SQL``); this method's non-interference holds either
        way.

        All three terminal fields are always written, so a replayed dead-letter
        that later lands clears its stale ``terminal_error`` (last-write-wins).

        UPSERT, NOT UPDATE. Both queue producers journal their Layer-1 row
        AFTER the enqueue commits — ``add_episode`` in a ``finally``,
        ``add_memory`` only after the synchronous Mem0 leg, a network/LLM round
        trip of unbounded width — so a worker can reach a terminal state before
        the row exists. A plain ``UPDATE`` would match 0 rows and drop the
        outcome; a *time-boxed retry* would drop it precisely when the producer
        was slowest, which is exactly when an operator most needs it, while
        stalling the queue worker that awaits this call. Creating the row
        instead makes the record unconditional: it carries the terminal fact
        alone (``operation`` NULL — ``stage_stats`` skips non-``str``
        operations), and the producer's later :meth:`log_write_op` upsert fills
        in the Layer-1 columns without touching ``terminal_*``.

        ``ON CONFLICT(id)`` is a PRIMARY KEY seek and needs no new index —
        which matters, because building one on the live 16.6M-row journal costs
        ~47 s of startup DDL (see the ``idx_wo_created`` note in
        ``SCHEMA_SQL``).

        Fire-and-forget per the journal's never-block discipline: logs loudly
        and returns ``False`` on failure, never raises — a journaling hiccup
        must not propagate into the queue worker that calls it.
        """
        try:
            now = datetime.now(UTC).isoformat()
            async with self._require_access().write() as db:
                await db.execute(
                    """INSERT INTO write_ops
                       (id, source, kind, created_at,
                        terminal_status, terminal_at, terminal_error)
                       VALUES (?, 'durable_queue', 'write', ?, ?, ?, ?)
                       ON CONFLICT(id) DO UPDATE SET
                           terminal_status = excluded.terminal_status,
                           terminal_at = excluded.terminal_at,
                           terminal_error = excluded.terminal_error""",
                    (
                        write_op_id,
                        now,
                        terminal_status,
                        now,
                        terminal_error,
                    ),
                )
            return True
        except Exception as e:
            logger.error(
                f'Failed to record terminal outcome {terminal_status} '
                f'for write_op {write_op_id}: {e}'
            )
            return False

    async def get_incomplete_mem0_intents(self) -> list[dict]:
        """Return mem0_intents still ``pending``.

        A ``pending`` row is one whose process died between the intent write
        and its terminal stamp — the set the startup reconciler resolves.
        """
        rows = await self._require_access().read_all(
            "SELECT * FROM mem0_intents WHERE status = 'pending' ORDER BY created_at"
        )
        return [dict(row) for row in rows]

    async def get_mem0_intents(self, status: str | None = None) -> list[dict]:
        """Return mem0_intents, optionally filtered by status."""
        access = self._require_access()
        if status is not None:
            rows = await access.read_all(
                'SELECT * FROM mem0_intents WHERE status = ? ORDER BY created_at',
                (status,),
            )
        else:
            rows = await access.read_all('SELECT * FROM mem0_intents ORDER BY created_at')
        return [dict(row) for row in rows]

    async def prune_mem0_intents(
        self,
        *,
        older_than_days: float = 30.0,
        statuses: tuple[str, ...] = ('completed', 'failed'),
    ) -> int:
        """Age out terminal ``mem0_intents`` past a retention window.

        The happy path logs one ``pending`` intent per mem0 write and stamps
        it ``completed`` — so, unpruned, this table grows by a row for every
        successful write and never shrinks. This bounded-retention sweep
        deletes rows whose ``status`` is in ``statuses`` and whose
        ``resolved_at`` is older than ``older_than_days``, returning the count
        deleted. Intended to run once at startup (after
        ``MemoryService.recover_mem0_intents``) or from a periodic
        maintenance task.

        ``dead`` is deliberately EXCLUDED from the default ``statuses``: a
        dead-lettered intent carries the full content needed for manual replay
        and is an operator-visible signal, so it must not be silently aged
        out. Override ``statuses`` to include it explicitly.

        Fire-and-forget — logs loudly on failure but never raises, so a prune
        hiccup cannot crash startup.
        """
        if not statuses:
            return 0
        try:
            cutoff = (datetime.now(UTC) - timedelta(days=older_than_days)).isoformat()
            placeholders = ','.join('?' for _ in statuses)
            deleted = 0
            async with self._require_access().write() as db:
                cursor = await db.execute(
                    f'DELETE FROM mem0_intents '
                    f'WHERE status IN ({placeholders}) '
                    f'AND resolved_at IS NOT NULL AND resolved_at < ?',
                    (*statuses, cutoff),
                )
                deleted = max(cursor.rowcount, 0)
            if deleted:
                logger.info(
                    'Pruned %d terminal mem0_intents older than %s days '
                    '(statuses=%s; dead-letters preserved)',
                    deleted,
                    older_than_days,
                    statuses,
                )
            return deleted
        except Exception as e:
            logger.error(f'Failed to prune mem0_intents: {e}')
            return 0

    async def prune_write_ops(
        self,
        *,
        read_older_than_days: float,
        search_older_than_days: float,
        write_older_than_days: float,
        batch_size: int,
        max_rows: int,
        max_seconds: float,
    ) -> int:
        """Age out ``write_ops`` past three independent retention horizons.

        Same fire-and-forget contract as ``prune_mem0_intents`` — logs loudly,
        never raises, returns the count deleted, returns 0 when uninitialised.
        Runs at startup and every ``prune_interval_seconds``
        (``services/write_ops_retention.py``).

        EVERY BOUND IS REQUIRED, with no default here. The six numbers have a
        single home — ``config/schema.py::WriteJournalConfig`` — and the sole
        production caller, ``services/write_ops_retention.py::prune_write_ops_pass``,
        passes all six from it. A second copy as signature defaults would be a
        copy nothing reads, kept equal to the first only by a drift-guard test;
        requiring the arguments deletes the copy instead of policing it.

        THREE HORIZONS, because the rows are not interchangeable. Measured
        2026-09-11 over a never-pruned 157-day span: non-search READS are
        34,669,782 rows (97.9% of the table — ``get_task``, ``get_tasks``,
        ``get_statuses``, ``get_external_statuses``) and have no
        downstream consumer, so 30 days is generous incident-forensics headroom.
        SEARCH reads are the opposite in both directions: only 481,447 rows
        (1.36%, ~3,066/day), and they are the SOLE data source for leaf eta's
        write-after-miss metric (task 3213) plus leaf theta's retro corpus
        (task 3214), which evaluate against trailing baseline windows — so a
        short horizon would silently starve the metric, and a long one is nearly
        free (365 days is ~1.1M rows). WRITES are the durable audit trail joined
        by ``causation_id`` in ``reconciliation/stats_verifier.py`` and are 0.73%
        of volume, so 730 days costs essentially nothing. Since task 5405 a
        successful non-search read is counted into ``read_op_counts`` and never
        stored, so the 'read' horizon now governs only failed reads and history
        not yet moved by ``scripts/roll_up_journal_reads.py``.

        BATCHED AND DOUBLY BOUNDED, unlike this file's two sibling prunes, whose
        tables are small enough for a single unbounded DELETE. This one runs
        against a table measured at 35,428,715 rows / 16 GB, and the
        ``idx_wo_created`` note in ``SCHEMA_SQL`` above measures 47 s merely to
        build one index on a copy of it, against ``STARTUP_GRACE_SECS = 120`` in
        ``scripts/orchestrator-watchdog.py``. That same note states the failure
        mode: a long write-lock on a serving instance SILENTLY DROPS journal rows,
        because ``busy_timeout`` is 5000 ms and ``log_write_op`` swallows its own
        errors. A multi-minute startup DELETE would therefore corrupt the very
        telemetry it is pruning. Each batch is its own ``write()`` unit, so both
        the per-connection lock and SQLite's write lock are released between
        batches and serving writes interleave. That matters most for the
        periodic pass, which runs while serving: a journalled tool call waits on
        the connection lock for at most one batch, never a whole pass, and
        ``batch_size`` is sized for that wait in ``fused-memory/config/config.yaml``.
        The sweep stops on whichever bound comes first. ``max_rows`` is only a
        PROXY for the risk; ``max_seconds`` bounds it directly, because
        rows-per-second is not knowable in advance — it depends on page-cache
        warmth and on six index b-trees per delete. The deadline is checked
        BETWEEN batches, never mid-batch.

        EACH HORIZON RESUMES WHERE ITS LAST BATCH STOPPED. ``operation`` is not
        in idx_wo_kind_time, so the read horizon tests ``operation <> 'search'``
        with a table-row lookup per index entry, and its range holds every kept
        search row older than the oldest expired read. A batch started from the
        bottom walks all of them before deleting anything: measured 2026-10-12 on
        the live journal, 430,040 rows took 9.4 s cold and 0.4-0.6 s warm in one
        statement the deadline cannot interrupt, against 2 ms from a resume
        point. ``_pruned_below`` keeps, per horizon, the ``created_at`` below
        which every matching row is already deleted. It stays true because every
        ``write_ops`` writer stamps ``created_at`` with the current time, so no
        row lands behind it. It lives as long as the process, so only the first
        batch after a start pays the walk, and that runs before READY.

        WHAT NET PROGRESS REQUIRES, stated so the claim is checkable rather than
        hopeful: rows removed per day (passes per day x rows per pass) must
        exceed the non-search reads inserted per day. Below that the backlog
        never clears while the WARNING fires on every pass and reads as normal.
        The dated measurement of both sides, and the interval chosen from it,
        live in the ``write_journal`` block of ``fused-memory/config/config.yaml``
        (2026-10-11); the rows and rows/sec logged on every partial pass are the
        live answer.

        ``terminal_status='dead'`` rows are NOT exempted, unlike the dead-letters
        ``prune_mem0_intents`` deliberately preserves: a dead ``write_ops`` row
        carries no replay payload, only a record that an outcome was terminal, so
        age is the only thing that should decide its fate.

        THIS BOUNDS GROWTH; IT DOES NOT RECLAIM DISK. Measured on the live
        journal: ``PRAGMA auto_vacuum`` is 0 (NONE). With auto-vacuum off a
        DELETE returns pages to SQLite's freelist for REUSE by later inserts; it
        does not return them to the filesystem, so ``ls -lh`` reports the same
        size after a fully successful prune — and the freed pages are exactly
        what absorbs the new per-result search detail in place. Reclaiming the
        space needs an offline ``VACUUM``, a full-file rewrite requiring as much
        scratch again, which is categorically not a startup operation.
        """
        older_than_days = {
            'read': read_older_than_days,
            'search': search_older_than_days,
            'write': write_older_than_days,
        }
        try:
            deadline = _monotonic() + max_seconds
            total = 0
            by_horizon: dict[str, int] = {}
            halted = ''
            backlog: list[str] = []

            for label, batch_sql in WRITE_OPS_PRUNE_BATCH_SQL.items():
                if halted:
                    # Never attempted under this run's bounds — unknown, so
                    # reported as a backlog rather than assumed drained.
                    backlog.append(label)
                    continue
                cutoff = (
                    datetime.now(UTC) - timedelta(days=older_than_days[label])
                ).isoformat()
                while True:
                    limit = min(batch_size, max_rows - total)
                    if limit <= 0:
                        halted = 'row budget'
                        backlog.append(label)
                        break
                    resume_from = self._pruned_below.get(label, '')
                    async with self._require_access().write() as db:
                        deleted = list(await db.execute_fetchall(
                            batch_sql, (resume_from, cutoff, limit)
                        ))
                    removed = len(deleted)
                    total += removed
                    by_horizon[label] = by_horizon.get(label, 0) + removed
                    if removed < limit:
                        self._pruned_below[label] = cutoff
                        break  # a short batch means this horizon is drained
                    self._pruned_below[label] = max(row[0] for row in deleted)
                    if total >= max_rows:
                        halted = 'row budget'
                        backlog.append(label)
                        break
                    if _monotonic() >= deadline:
                        halted = 'deadline'
                        backlog.append(label)
                        break

            if total:
                logger.info(
                    'Pruned %d write_ops rows (read>%sd: %d, search>%sd: %d, '
                    'write>%sd: %d); frees pages for reuse, does not shrink the file',
                    total,
                    read_older_than_days, by_horizon.get('read', 0),
                    search_older_than_days, by_horizon.get('search', 0),
                    write_older_than_days, by_horizon.get('write', 0),
                )
            if halted:
                # The achieved rate is the ONLY thing that says whether the
                # passes keep up on this disk; see the net-progress rule above.
                elapsed = max(_monotonic() - (deadline - max_seconds), 1e-9)
                logger.warning(
                    'write_ops prune stopped early on the %s after %d rows in %.1fs '
                    '(%.0f rows/sec); horizons still holding a backlog: %s. Compare '
                    'that rate with the sizing basis in the write_journal block of '
                    'fused-memory/config/config.yaml. Since task 5405 a read backlog '
                    'is history not yet moved by scripts/roll_up_journal_reads.py.',
                    halted, total, elapsed, total / elapsed, ', '.join(backlog),
                )
            return total
        except Exception as e:
            logger.error(f'Failed to prune write_ops: {e}')
            return 0
