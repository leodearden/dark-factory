"""SQLite-backed event buffer with burst detection and quiescence triggers."""

import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, NamedTuple
from uuid import uuid4

import aiosqlite
from shared.async_sqlite_base import (
    AtomicConnection,
    CheckpointResult,
    apply_full_durability_pragmas,
    connect_daemon,
)
from shared.sqlite_column_migration import add_column_if_missing

from fused_memory.models.reconciliation import (
    EventSource,
    EventType,
    ReconciliationEvent,
)
from fused_memory.reconciliation.coalescing import (
    coalesce_key,
    coalesced_count,
    merge_coalesced_payload,
)
from fused_memory.reconciliation.throughput import utc_hour_bucket
from fused_memory.utils.sqlite_connector import Connector

logger = logging.getLogger(__name__)

# Defensive schema — EventBuffer creates its own tables even if journal hasn't run yet.
_BUFFER_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS event_buffer (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    event_source TEXT NOT NULL,
    agent_id TEXT,
    timestamp TEXT NOT NULL,
    payload TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'buffered',
    drained_by_run_id TEXT,
    coalesce_key TEXT
);
CREATE INDEX IF NOT EXISTS idx_eb_project_status ON event_buffer(project_id, status);
CREATE INDEX IF NOT EXISTS idx_eb_agent_timestamp ON event_buffer(agent_id, timestamp)
    WHERE agent_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS reconciliation_locks (
    project_id TEXT PRIMARY KEY,
    instance_id TEXT NOT NULL,
    acquired_at TEXT NOT NULL,
    heartbeat_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS burst_state (
    agent_id TEXT PRIMARY KEY,
    state TEXT NOT NULL DEFAULT 'idle',
    last_write_at TEXT NOT NULL,
    burst_started_at TEXT
);

CREATE TABLE IF NOT EXISTS deferred_writes (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    content TEXT NOT NULL,
    category TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    agent_id TEXT,
    created_at TEXT NOT NULL,
    claimed_at TEXT,
    attempt_count INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_dw_project_claimed ON deferred_writes(project_id, claimed_at);

-- Task 3049: durable hourly inflow record.  event_buffer rows are DELETEd by
-- cleanup_drained once they are drained and older than max_age_seconds
-- (default 3600s), so arrival history otherwise survives ~1 hour, not a week.
-- cleanup_drained rolls the rows it is about to delete into this aggregate in
-- the SAME transaction as the DELETE, which is what makes the count exact:
-- an event is either still live in event_buffer or already counted here,
-- never both and never neither.  Idempotent DDL, so existing DBs pick the
-- table up on their next initialize().
CREATE TABLE IF NOT EXISTS event_arrival_hourly (
    project_id TEXT NOT NULL,
    hour_bucket TEXT NOT NULL,
    event_type TEXT NOT NULL,
    event_count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (project_id, hour_bucket, event_type)
);

-- Task 4278 (recon lever 3): events push() folded into an older buffered row.
-- Counted in the same write unit as the fold, so the partition stays exact:
-- every pushed event is a live event_buffer row, OR in event_arrival_hourly,
-- OR in event_coalesced_hourly.
CREATE TABLE IF NOT EXISTS event_coalesced_hourly (
    project_id TEXT NOT NULL,
    hour_bucket TEXT NOT NULL,
    event_type TEXT NOT NULL,
    coalesced_count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (project_id, hour_bucket, event_type)
);
"""

# Maximum number of replay attempts before a deferred write is treated as a poison-pill
# and permanently dropped (logged at ERROR).
_MAX_DEFERRED_WRITE_ATTEMPTS: int = 5

# Rows fetched per fetchmany() call when draining cleanup_drained's
# DELETE ... RETURNING cursor.  Bounds peak memory at O(chunk) instead of
# O(rows deleted) — see cleanup_drained's docstring for the measurement.
_CLEANUP_FETCH_CHUNK: int = 10_000

# A folded row announces itself at WARNING each time its fold count reaches a
# multiple of this.  A looping task no longer builds backlog once its events
# fold, so the reconciliation_backlog alarm cannot see it; this line can.
_COALESCE_HOT_TASK_WARNING_INTERVAL: int = 100


class _PushOutcome(StrEnum):
    """What one push did to the buffer."""

    buffered = 'buffered'  # a new row
    coalesced = 'coalesced'  # folded into an older buffered row
    redelivered = 'redelivered'  # already stored, so nothing changed


class _PushResult(NamedTuple):
    outcome: _PushOutcome
    coalesced_count: int = 1


_UNFOLDED_PUSH_LOG_LINE = {
    _PushOutcome.buffered: 'reconciliation.event_buffered',
    _PushOutcome.redelivered: 'reconciliation.event_redelivered',
}


def _log_push(event: ReconciliationEvent, key: str | None, result: _PushResult) -> None:
    """One structured line per push; a WARNING when a folded row turns hot."""
    if result.outcome is not _PushOutcome.coalesced:
        logger.info(
            _UNFOLDED_PUSH_LOG_LINE[result.outcome],
            extra={
                'project_id': event.project_id,
                'event_type': event.type.value,
                'agent_id': event.agent_id,
            },
        )
        return
    fold_extra = {
        'project_id': event.project_id,
        'event_type': event.type.value,
        'coalesce_key': key,
        'coalesced_count': result.coalesced_count,
    }
    logger.info('reconciliation.event_coalesced', extra=fold_extra)
    if result.coalesced_count % _COALESCE_HOT_TASK_WARNING_INTERVAL == 0:
        logger.warning('reconciliation.coalesce_hot_task', extra=fold_extra)


class EventBuffer:
    """SQLite-backed event buffer with cross-instance visibility and burst detection.

    Replaces the per-instance in-memory buffer with a shared SQLite store so that
    multiple stdio MCP instances accumulate events into a single pool.  Burst
    detection tracks per-agent write rates and exposes a quiescence signal used
    by the conditional trigger (fires at 33% of the hard threshold when no agent
    is actively bursting and the durable queue is idle).
    """

    def __init__(
        self,
        db_path: Path | str | None = None,
        buffer_size_threshold: int = 10,
        max_staleness_seconds: int = 1800,
        conditional_trigger_ratio: float = 0.33,
        burst_window_seconds: float = 30.0,
        burst_cooldown_seconds: float = 150.0,
        stale_lock_seconds: float = 7200.0,
        queue_stats_fn: Callable[[], Any] | None = None,
        instance_id: str | None = None,
        *,
        connect: Connector = connect_daemon,
        coalesce_task_events: bool = True,
    ):
        """*connect* is how ``initialize`` opens the buffer's one connection.

        It is given the database path and defaults to ``connect_daemon``.
        *coalesce_task_events* switches push-time folding (see ``push``).
        """
        # _in_memory is derived from the caller's intent (db_path is None),
        # not from the resolved string ':memory:'.  This means explicitly
        # passing db_path=Path(':memory:') is treated as a file path and
        # would hit the helper's RuntimeError — that is intentional: callers
        # who want in-memory mode must pass None, not a string constant.
        self._in_memory = db_path is None
        self._db_path = str(db_path) if db_path else ':memory:'
        self.buffer_size_threshold = buffer_size_threshold
        self.max_staleness_seconds = max_staleness_seconds
        self.conditional_trigger_ratio = conditional_trigger_ratio
        self.burst_window_seconds = burst_window_seconds
        self.burst_cooldown_seconds = burst_cooldown_seconds
        self.stale_lock_seconds = stale_lock_seconds
        self._queue_stats_fn = queue_stats_fn
        self.instance_id = instance_id or str(uuid4())
        self._connect = connect
        self.coalesce_task_events = coalesce_task_events
        self._access: AtomicConnection | None = None
        self._manual_triggers: set[str] = set()

    async def initialize(self) -> None:
        """Open SQLite connection and ensure schema exists."""
        conn = await self._connect(self._db_path)
        conn.row_factory = aiosqlite.Row
        # WAL mode cannot be enabled on in-memory SQLite DBs — SQLite
        # returns 'memory' from PRAGMA journal_mode=WAL which would trigger
        # the helper's RuntimeError guard.  The in-memory path is used by
        # server/main.py:580 and server/tools.py:1598 as a Taskmaster-
        # disabled fallback.  We guard on _in_memory (set from db_path is
        # None at construction time) rather than comparing _db_path to the
        # literal ':memory:' string, so that the contract is tied to the
        # caller's intent rather than to a particular string representation.
        if not self._in_memory:
            await apply_full_durability_pragmas(conn, busy_timeout_ms=5000)
        self._access = AtomicConnection(conn)
        async with self._access.write() as db:
            await db.executescript(_BUFFER_SCHEMA_SQL)
        # Idempotent migration: add claimed_at column for pre-existing DBs.
        await self._migrate()
        logger.info(f'EventBuffer initialized (db={self._db_path}, instance={self.instance_id})')

    async def _migrate(self) -> None:
        """Add new columns to pre-existing DBs."""
        async with self._require_access().write() as db:
            # Add drained_by_run_id if missing — attributes a drained row to the run
            # that drained it (task 2711 / E7) so restore_drained() can be scoped to
            # a single run instead of unconditionally restoring the whole project.
            if await add_column_if_missing(db, 'event_buffer', 'drained_by_run_id', 'TEXT'):
                logger.info('EventBuffer: migrated event_buffer — added drained_by_run_id column')

            # Add coalesce_key if missing (task 4278).  Rows written before it
            # keep NULL and are never fold targets.  The index lives here, not
            # in _BUFFER_SCHEMA_SQL, because that script runs first and would
            # name a column a pre-existing table does not have yet.
            if 'coalesce_key' not in eb_columns:
                try:
                    await db.execute('ALTER TABLE event_buffer ADD COLUMN coalesce_key TEXT')
                    logger.info('EventBuffer: migrated event_buffer — added coalesce_key column')
                except Exception as exc:
                    if 'duplicate column name' not in str(exc).lower():
                        raise
                    logger.debug('EventBuffer: coalesce_key already exists (concurrent init)')
            await db.execute(
                'CREATE INDEX IF NOT EXISTS idx_eb_coalesce'
                ' ON event_buffer(project_id, coalesce_key, status)'
                ' WHERE coalesce_key IS NOT NULL'
            )

            # Drop the now-redundant single-column project index.  idx_dw_project_claimed
            # (project_id, claimed_at) is a strict superset; keeping both wastes write
            # IOPS and disk.  IF EXISTS makes this a no-op for fresh DBs that never
            # had idx_dw_project.
            await db.execute('DROP INDEX IF EXISTS idx_dw_project')

            if await add_column_if_missing(db, 'deferred_writes', 'claimed_at', 'TEXT'):
                await db.execute(
                    'CREATE INDEX IF NOT EXISTS idx_dw_project_claimed'
                    ' ON deferred_writes(project_id, claimed_at)'
                )
                logger.info('EventBuffer: migrated deferred_writes — added claimed_at column')

            # Add attempt_count if missing — tracks poison-pill detection across restarts.
            if await add_column_if_missing(
                db, 'deferred_writes', 'attempt_count', 'INTEGER NOT NULL DEFAULT 0'
            ):
                logger.info('EventBuffer: migrated deferred_writes — added attempt_count column')

    def _require_access(self) -> AtomicConnection:
        if self._access is None:
            raise RuntimeError('EventBuffer not initialized — call initialize() first')
        return self._access

    # ── Push ───────────────────────────────────────────────────────────

    async def push(self, event: ReconciliationEvent) -> None:
        """Insert event into the shared buffer, folding a repeat task event.

        Idempotent on the event id: ``ON CONFLICT(id) DO NOTHING`` makes the
        at-least-once redelivery of EventJournal startup recovery a no-op
        (task 2709), while any other constraint violation still raises.

        An event with a ``coalescing.coalesce_key`` folds into the OLDEST
        buffered row with the same key (task 4278).  That row keeps its id, so
        an id ``peek_buffered`` handed out still drains via ``drain_by_ids``,
        and its timestamp, which only ever moves earlier, so ``ORDER BY
        timestamp`` positions and the ``MIN(timestamp)`` staleness trigger
        hold.  The INSERT runs first: a unit whose first statement writes is
        serialised against other connections, which a unit that SELECTs first
        is not (``AtomicConnection.write``).  A fold landing between
        ``peek_buffered`` and ``drain_by_ids`` reaches the stage as the
        pre-fold payload; that is benign, because the target's task is already
        in that batch and ``_ctx_task_event`` reads the task's live state.
        """
        key = coalesce_key(event)
        async with self._require_access().write() as db:
            cursor = await db.execute(
                """INSERT INTO event_buffer
                   (id, project_id, event_type, event_source, agent_id, timestamp, payload,
                    status, coalesce_key)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 'buffered', ?)
                   ON CONFLICT(id) DO NOTHING""",
                (
                    event.id,
                    event.project_id,
                    event.type.value,
                    event.source.value,
                    event.agent_id,
                    event.timestamp.isoformat(),
                    json.dumps(event.payload),
                    key,
                ),
            )
            if cursor.rowcount == 0:
                result = _PushResult(_PushOutcome.redelivered)
            elif self.coalesce_task_events and key is not None:
                result = await self._fold_into_oldest(db, event, key)
            else:
                result = _PushResult(_PushOutcome.buffered)

        if event.agent_id is not None:
            await self._update_burst_state(event.agent_id, event.timestamp)

        _log_push(event, key, result)

    async def _fold_into_oldest(
        self, db: aiosqlite.Connection, event: ReconciliationEvent, key: str,
    ) -> _PushResult:
        """Fold the row ``push`` just inserted into the oldest buffered row sharing ``key``.

        Runs on the open write unit's connection.  The just-inserted row is
        uncommitted, so deleting it leaves no trace any reader could see.
        """
        # Oldest buffered row: stable id, MIN(timestamp) and ORDER BY position (see push).
        async with db.execute(
            """SELECT id, payload, timestamp FROM event_buffer
               WHERE project_id = ? AND coalesce_key = ? AND status = 'buffered'
                 AND id != ?
               ORDER BY timestamp ASC
               LIMIT 1""",
            (event.project_id, key, event.id),
        ) as cursor:
            target = await cursor.fetchone()
        if target is None:
            return _PushResult(_PushOutcome.buffered)

        await db.execute('DELETE FROM event_buffer WHERE id = ?', (event.id,))
        fold = merge_coalesced_payload(
            json.loads(target['payload']),
            event.payload,
            event_type=event.type,
            incoming_id=event.id,
            incoming_ts=event.timestamp,
            row_ts=datetime.fromisoformat(target['timestamp']),
        )
        if fold is None:
            return _PushResult(_PushOutcome.redelivered)

        await db.execute(
            'UPDATE event_buffer SET payload = ?, timestamp = ? WHERE id = ?',
            (json.dumps(fold.payload), fold.row_timestamp.isoformat(), target['id']),
        )
        await db.execute(
            """INSERT INTO event_coalesced_hourly
                   (project_id, hour_bucket, event_type, coalesced_count)
               VALUES (?, ?, ?, 1)
               ON CONFLICT(project_id, hour_bucket, event_type)
               DO UPDATE SET coalesced_count = coalesced_count + excluded.coalesced_count""",
            (
                event.project_id,
                utc_hour_bucket(fold.folded_timestamp.isoformat()),
                event.type.value,
            ),
        )
        return _PushResult(_PushOutcome.coalesced, coalesced_count(fold.payload))

    # ── Burst detection ────────────────────────────────────────────────

    async def _update_burst_state(self, agent_id: str, timestamp: datetime) -> None:
        """Track per-agent write bursts.  2+ writes within burst_window → bursting."""
        ts_iso = timestamp.isoformat()
        cutoff = (timestamp.timestamp() - self.burst_window_seconds)
        cutoff_iso = datetime.fromtimestamp(cutoff, tz=UTC).isoformat()

        row = await self._require_access().read_one(
            """SELECT COUNT(*) as cnt FROM event_buffer
               WHERE agent_id = ? AND timestamp >= ?""",
            (agent_id, cutoff_iso),
        )
        recent_count = row['cnt'] if row else 0

        async with self._require_access().write() as db:
            if recent_count >= 2:
                await db.execute(
                    """INSERT INTO burst_state (agent_id, state, last_write_at, burst_started_at)
                       VALUES (?, 'bursting', ?, ?)
                       ON CONFLICT(agent_id) DO UPDATE SET
                         state = 'bursting',
                         last_write_at = excluded.last_write_at,
                         burst_started_at = COALESCE(burst_state.burst_started_at, excluded.burst_started_at)""",
                    (agent_id, ts_iso, ts_iso),
                )
            else:
                await db.execute(
                    """INSERT INTO burst_state (agent_id, state, last_write_at)
                       VALUES (?, 'idle', ?)
                       ON CONFLICT(agent_id) DO UPDATE SET
                         last_write_at = excluded.last_write_at""",
                    (agent_id, ts_iso),
                )

    # ── Trigger logic ──────────────────────────────────────────────────

    async def should_trigger(self, project_id: str) -> tuple[bool, str]:
        """Three-tier trigger: hard thresholds, then conditional quiescence.

        Returns (should_trigger, reason).
        """
        # Check run lock
        if await self._is_run_locked(project_id):
            return False, ''

        # Manual trigger (placed after lock check so flag is preserved when busy)
        if self._consume_manual_trigger(project_id):
            return True, 'manual_trigger'

        # Count pending events + oldest timestamp
        row = await self._require_access().read_one(
            """SELECT COUNT(*) as cnt, MIN(timestamp) as oldest
               FROM event_buffer
               WHERE project_id = ? AND status = 'buffered'""",
            (project_id,),
        )

        count = row['cnt'] if row else 0
        oldest_str = row['oldest'] if row else None

        await self.expire_stale_bursts()

        if count == 0:
            return False, ''

        # Hard: count threshold
        if count >= self.buffer_size_threshold:
            return True, f'buffer_size:{count}'

        # Hard: staleness
        if oldest_str:
            oldest = datetime.fromisoformat(oldest_str)
            if oldest.tzinfo is None:
                oldest = oldest.replace(tzinfo=UTC)
            age = (datetime.now(UTC) - oldest).total_seconds()
            if age > self.max_staleness_seconds:
                return True, f'max_staleness:{oldest.isoformat()}'

        # Conditional: ratio threshold + quiescence
        conditional_threshold = int(self.buffer_size_threshold * self.conditional_trigger_ratio)
        if conditional_threshold > 0 and count >= conditional_threshold and await self._is_quiescent():
            return True, f'quiescent:{count}'

        return False, ''

    async def _is_run_locked(self, project_id: str) -> bool:
        """Check if another instance holds the reconciliation lock."""
        # Expire stale locks first
        cutoff = datetime.fromtimestamp(
            datetime.now(UTC).timestamp() - self.stale_lock_seconds,
            tz=UTC,
        ).isoformat()
        async with self._require_access().write() as db:
            await db.execute(
                'DELETE FROM reconciliation_locks WHERE heartbeat_at < ?',
                (cutoff,),
            )

        row = await self._require_access().read_one(
            'SELECT 1 FROM reconciliation_locks WHERE project_id = ?',
            (project_id,),
        )
        return row is not None

    async def get_lock_status(
        self, project_id: str
    ) -> tuple[str | None, float | None]:
        """Return (instance_id, heartbeat_age_seconds) for the live lock holder.

        Runs the same stale-lock sweep as _is_run_locked (DELETE rows with
        heartbeat_at older than stale_lock_seconds), then SELECTs instance_id
        and heartbeat_at for the surviving row.

        Returns:
            (instance_id, age_seconds) when a live lock row exists.
            (None, None) when no lock row exists or the row was swept.
        """
        cutoff = datetime.fromtimestamp(
            datetime.now(UTC).timestamp() - self.stale_lock_seconds,
            tz=UTC,
        ).isoformat()
        async with self._require_access().write() as db:
            await db.execute(
                'DELETE FROM reconciliation_locks WHERE heartbeat_at < ?',
                (cutoff,),
            )

        row = await self._require_access().read_one(
            'SELECT instance_id, heartbeat_at FROM reconciliation_locks WHERE project_id = ?',
            (project_id,),
        )
        if row is None:
            return (None, None)
        hb = datetime.fromisoformat(row['heartbeat_at'])
        if hb.tzinfo is None:
            hb = hb.replace(tzinfo=UTC)
        age = (datetime.now(UTC) - hb).total_seconds()
        return (row['instance_id'], age)

    async def get_lock_holder_instance_id(self, project_id: str) -> str | None:
        """Return the instance_id of the live lock holder for project_id, or None.

        Delegates to get_lock_status for the stale-lock sweep and SELECT.
        Used by the harness reaper to compare against a stale run's owning
        instance — see _recover_stale_runs.
        """
        return (await self.get_lock_status(project_id))[0]

    async def expire_stale_bursts(self) -> int:
        """Transition 'bursting' agents to 'idle' when cooldown has elapsed."""
        cooldown_cutoff = datetime.fromtimestamp(
            datetime.now(UTC).timestamp() - self.burst_cooldown_seconds,
            tz=UTC,
        ).isoformat()
        async with self._require_access().write() as db:
            cursor = await db.execute(
                "UPDATE burst_state SET state = 'idle', burst_started_at = NULL "
                "WHERE state = 'bursting' AND last_write_at < ?",
                (cooldown_cutoff,),
            )
            rowcount = cursor.rowcount
        return rowcount

    async def _is_quiescent(self) -> bool:
        """System is quiescent when no agent is bursting and durable queue is idle."""
        await self.expire_stale_bursts()

        # Any agents still bursting?
        bursting = await self._require_access().read_one(
            "SELECT 1 FROM burst_state WHERE state = 'bursting' LIMIT 1"
        )
        if bursting is not None:
            return False

        # Check durable queue
        if self._queue_stats_fn is not None:
            try:
                stats = await self._queue_stats_fn()
                counts = stats.get('counts', {})
                pending = counts.get('pending', 0)
                retry = counts.get('retry', 0)
                in_flight = counts.get('in_flight', 0)
                if (pending + retry + in_flight) > 0:
                    return False
            except Exception:
                # If we can't check the queue, assume not quiescent
                return False

        return True

    # ── Drain ──────────────────────────────────────────────────────────

    async def drain(
        self, project_id: str, run_id: str | None = None,
    ) -> list[ReconciliationEvent]:
        """Atomically drain buffered events for a project.

        When ``run_id`` is provided, the drained rows are stamped with
        ``drained_by_run_id = run_id`` so a later run-scoped ``restore_drained``
        can restore exactly this run's events without touching a concurrent
        run's drained rows (task 2711 / E7).
        """
        rows = await self._require_access().read_all(
            """SELECT * FROM event_buffer
               WHERE project_id = ? AND status = 'buffered'
               ORDER BY timestamp""",
            (project_id,),
        )

        if not rows:
            return []

        ids = [row['id'] for row in rows]
        placeholders = ','.join('?' for _ in ids)
        async with self._require_access().write() as db:
            await db.execute(
                f"UPDATE event_buffer SET status = 'drained', drained_by_run_id = ? "
                f"WHERE id IN ({placeholders})",
                [run_id, *ids],
            )

        events = []
        for row in rows:
            events.append(ReconciliationEvent(
                id=row['id'],
                project_id=row['project_id'],
                type=EventType(row['event_type']),
                source=EventSource(row['event_source']),
                agent_id=row['agent_id'],
                timestamp=datetime.fromisoformat(row['timestamp']),
                payload=json.loads(row['payload']),
            ))
        return events

    async def drain_oldest_chunk(
        self, project_id: str, limit: int,
        before: datetime | None = None,
    ) -> list[ReconciliationEvent]:
        """Drain the oldest `limit` buffered events for a project.

        Args:
            before: If set, only drain events with timestamp < this value.
                    Used by BacklogIterator to ignore events that arrived
                    after the backlog snapshot was taken.
        """
        if before is not None:
            query = """SELECT * FROM event_buffer
                       WHERE project_id = ? AND status = 'buffered'
                         AND timestamp < ?
                       ORDER BY timestamp ASC
                       LIMIT ?"""
            params: tuple = (project_id, before.isoformat(), limit)
        else:
            query = """SELECT * FROM event_buffer
                       WHERE project_id = ? AND status = 'buffered'
                       ORDER BY timestamp ASC
                       LIMIT ?"""
            params = (project_id, limit)
        rows = await self._require_access().read_all(query, params)

        if not rows:
            return []

        ids = [row['id'] for row in rows]
        placeholders = ','.join('?' for _ in ids)
        async with self._require_access().write() as db:
            await db.execute(
                f"UPDATE event_buffer SET status = 'drained' WHERE id IN ({placeholders})",
                ids,
            )

        events = []
        for row in rows:
            events.append(ReconciliationEvent(
                id=row['id'],
                project_id=row['project_id'],
                type=EventType(row['event_type']),
                source=EventSource(row['event_source']),
                agent_id=row['agent_id'],
                timestamp=datetime.fromisoformat(row['timestamp']),
                payload=json.loads(row['payload']),
            ))
        return events

    async def peek_buffered(
        self, project_id: str, limit: int,
        before: datetime | None = None,
    ) -> list[ReconciliationEvent]:
        """Read buffered events without marking them as drained.

        Same query as ``drain_oldest_chunk`` but read-only — used by the
        ContextAssembler to determine which events fit the token budget
        before committing to a drain.
        """
        if before is not None:
            query = """SELECT * FROM event_buffer
                       WHERE project_id = ? AND status = 'buffered'
                         AND timestamp < ?
                       ORDER BY timestamp ASC
                       LIMIT ?"""
            params: tuple = (project_id, before.isoformat(), limit)
        else:
            query = """SELECT * FROM event_buffer
                       WHERE project_id = ? AND status = 'buffered'
                       ORDER BY timestamp ASC
                       LIMIT ?"""
            params = (project_id, limit)
        rows = await self._require_access().read_all(query, params)

        return [
            ReconciliationEvent(
                id=row['id'],
                project_id=row['project_id'],
                type=EventType(row['event_type']),
                source=EventSource(row['event_source']),
                agent_id=row['agent_id'],
                timestamp=datetime.fromisoformat(row['timestamp']),
                payload=json.loads(row['payload']),
            )
            for row in rows
        ]

    async def get_drained_events(
        self, project_id: str, run_id: str,
    ) -> list[ReconciliationEvent]:
        """Read-only accessor for a run's drained events — no status mutation.

        Returns the events that ``drain(project_id, run_id=run_id)`` stamped
        ``status='drained', drained_by_run_id=run_id``, mapped to
        ``ReconciliationEvent`` exactly like ``drain``/``peek_buffered``, ordered
        by timestamp. Unlike ``restore_drained`` it does NOT flip the rows back
        to ``buffered`` — the resume path (task σ) feeds these to FRESH later
        stages while the events stay drained, so a resumed cycle never
        double-processes them.
        """
        rows = await self._require_access().read_all(
            """SELECT * FROM event_buffer
               WHERE project_id = ? AND status = 'drained' AND drained_by_run_id = ?
               ORDER BY timestamp""",
            (project_id, run_id),
        )

        return [
            ReconciliationEvent(
                id=row['id'],
                project_id=row['project_id'],
                type=EventType(row['event_type']),
                source=EventSource(row['event_source']),
                agent_id=row['agent_id'],
                timestamp=datetime.fromisoformat(row['timestamp']),
                payload=json.loads(row['payload']),
            )
            for row in rows
        ]

    async def drain_by_ids(
        self, project_id: str, ids: list[str], run_id: str,
    ) -> int:
        """Mark specific event IDs as drained by run *run_id*.

        Used after the ContextAssembler determines which events fit the
        token budget.  Returns the number of rows updated.  *run_id* is
        stamped as ``drained_by_run_id`` so a later run-scoped
        ``restore_drained`` can target exactly this chunk; it is required so
        this path can never leave an unattributed row (task 6453).
        """
        if not ids:
            return 0
        placeholders = ','.join('?' for _ in ids)
        async with self._require_access().write() as db:
            cursor = await db.execute(
                f"""UPDATE event_buffer SET status = 'drained', drained_by_run_id = ?
                    WHERE project_id = ? AND id IN ({placeholders})
                      AND status = 'buffered'""",
                [run_id, project_id, *ids],
            )
            rowcount = cursor.rowcount
        return rowcount

    async def restore_drained(
        self,
        project_id: str,
        run_id: str,
        *,
        include_unattributed: bool = False,
    ) -> int:
        """Restore one run's drained events to 'buffered' after it failed.

        Only rows stamped with ``drained_by_run_id = run_id`` are restored —
        this protects a concurrent live run's drained events from being
        clobbered when the reaper recovers a *different* orphaned run on the
        same project (task 2711 / E7), and an earlier completed run's consumed
        events from being re-processed (task 6453). There is deliberately no
        project-wide mode.

        ``include_unattributed`` additionally restores rows with
        ``drained_by_run_id IS NULL``. This closes the upgrade-boundary gap
        where events drained by a pre-task-2711 process carry no run
        attribution: since every production drain path stamps
        ``drained_by_run_id`` as it drains (``drain_by_ids`` requires a
        ``run_id``, and ``run_full_cycle``, ``drain``'s only production
        caller, always passes one; task 6453), a NULL row can only be left
        over from a process that no longer exists by the time a recovery pass
        runs, so it is provably safe to fold into a *specific* run's recovery
        without reintroducing the concurrent-run clobber the run scoping was
        added to prevent (task 2711 amendment, reviewer finding).
        Callers that need the exact-match-only contract (e.g. one caller's
        ``mark_drained_run_id`` attribution racing another's) must keep the
        default ``False``.
        """
        if include_unattributed:
            async with self._require_access().write() as db:
                cursor = await db.execute(
                    "UPDATE event_buffer SET status = 'buffered' "
                    "WHERE project_id = ? AND status = 'drained' "
                    "AND (drained_by_run_id = ? OR drained_by_run_id IS NULL)",
                    (project_id, run_id),
                )
                count = cursor.rowcount
        else:
            async with self._require_access().write() as db:
                cursor = await db.execute(
                    "UPDATE event_buffer SET status = 'buffered' "
                    "WHERE project_id = ? AND status = 'drained' AND drained_by_run_id = ?",
                    (project_id, run_id),
                )
                count = cursor.rowcount
        if count:
            logger.info(f'Restored {count} drained events to buffered for {project_id}')
        return count

    async def mark_drained_run_id(
        self, project_id: str, event_ids: list[str], run_id: str,
    ) -> int:
        """Attribute already-drained rows to the run about to process them.

        ``run_full_cycle``'s attribution for caller-supplied ``events``, so a
        later run-scoped ``restore_drained`` can restore exactly that run's
        events (task 2711 / E7). BacklogIterator already stamps each chunk's
        run_id at ``drain_by_ids`` (task 6453), which makes this an idempotent
        re-stamp on that path.

        Only rows currently ``status = 'drained'`` are touched; ids that don't
        exist, aren't currently drained, or are empty are silently ignored (no
        error). Returns the number of rows updated.
        """
        if not event_ids:
            return 0
        placeholders = ','.join('?' for _ in event_ids)
        async with self._require_access().write() as db:
            cursor = await db.execute(
                f"""UPDATE event_buffer SET drained_by_run_id = ?
                    WHERE project_id = ? AND status = 'drained'
                      AND id IN ({placeholders})""",
                [run_id, project_id, *event_ids],
            )
            rowcount = cursor.rowcount
        return rowcount

    async def mark_project_dead_letter(self, project_id: str) -> int:
        """Flip all 'buffered' rows for project_id to 'dead_letter'. Returns row count.

        Called from the reconciliation harness when UnknownProjectError is raised
        (task 1143, read-side strictness). Quarantining the rows stops
        get_active_projects() (which selects WHERE status='buffered') from returning
        the project_id, ending the management-loop respawn storm (2026-05-28 incident).

        Mirrors restore_drained: UPDATE-by-status in a single write unit.
        See task 1143 (read-side UnknownProjectError) and task 1549 (this write-side
        complement: quarantine so get_active_projects stops respawning the loop).

        NOTE — observability / lifecycle of 'dead_letter' rows in this table:
        These rows are terminal and orphaned.  There is currently no built-in
        cleanup or operator-facing read path for them in the event_buffer table
        (the MCP get_dead_letters/delete_dead_letters/replay_dead_letters tools
        operate on the separate event_queue JSONL file, not this SQLite table).
        The quarantined rows accumulate indefinitely but are harmless — they are
        excluded from all active-project queries.  To inspect them, query the
        event_buffer table directly (``SELECT * FROM event_buffer WHERE
        status='dead_letter'``).  A cleanup helper can be added as a follow-up.
        """
        async with self._require_access().write() as db:
            cursor = await db.execute(
                "UPDATE event_buffer SET status = 'dead_letter' "
                "WHERE project_id = ? AND status = 'buffered'",
                (project_id,),
            )
            count = cursor.rowcount
        if count:
            logger.info(
                f'mark_project_dead_letter: quarantined {count} buffered event(s) '
                f'for unknown project_id={project_id!r} (task 1143 / task 1549)'
            )
        return count

    async def count_buffered(self, project_id: str) -> int:
        """Return count of buffered events for a project."""
        row = await self._require_access().read_one(
            "SELECT COUNT(*) as cnt FROM event_buffer WHERE project_id = ? AND status = 'buffered'",
            (project_id,),
        )
        return row['cnt'] if row else 0

    # ── Run locking ────────────────────────────────────────────────────

    async def mark_run_active(self, project_id: str) -> bool:
        """Acquire cross-instance lock. Returns False if already held."""
        # Expire stale locks
        cutoff = datetime.fromtimestamp(
            datetime.now(UTC).timestamp() - self.stale_lock_seconds,
            tz=UTC,
        ).isoformat()
        async with self._require_access().write() as db:
            await db.execute(
                'DELETE FROM reconciliation_locks WHERE heartbeat_at < ?',
                (cutoff,),
            )

        now = datetime.now(UTC).isoformat()
        try:
            async with self._require_access().write() as db:
                await db.execute(
                    """INSERT INTO reconciliation_locks (project_id, instance_id, acquired_at, heartbeat_at)
                       VALUES (?, ?, ?, ?)""",
                    (project_id, self.instance_id, now, now),
                )
            return True
        except Exception:
            # PK constraint violation — another instance holds the lock.
            # The write unit already rolled back.
            return False

    async def mark_run_complete(
        self, project_id: str, instance_id: str | None = None,
    ) -> int:
        """Release the reconciliation lock for ``project_id``.

        When ``instance_id`` is provided, the DELETE is scoped to that lock
        holder, so a caller intending to release *its own* lock cannot
        accidentally evict a different live instance's row (regression guard
        for the 2026-05-28 cross-instance lock-theft incident — see
        ``plans/recon-stale-recovery-rca.md``).  When omitted, behaviour is
        unchanged: the row is removed regardless of holder.

        Returns the rowcount so callers can detect a no-op release.
        """
        async with self._require_access().write() as db:
            if instance_id is None:
                cursor = await db.execute(
                    'DELETE FROM reconciliation_locks WHERE project_id = ?',
                    (project_id,),
                )
            else:
                cursor = await db.execute(
                    'DELETE FROM reconciliation_locks '
                    'WHERE project_id = ? AND instance_id = ?',
                    (project_id, instance_id),
                )
            return cursor.rowcount

    async def heartbeat(self, project_id: str) -> None:
        """Update lock heartbeat to prevent stale-lock recovery."""
        now = datetime.now(UTC).isoformat()
        async with self._require_access().write() as db:
            await db.execute(
                'UPDATE reconciliation_locks SET heartbeat_at = ? WHERE project_id = ? AND instance_id = ?',
                (now, project_id, self.instance_id),
            )

    # ── Deferred writes (cycle fence) ─────────────────────────────────

    async def is_full_recon_active(self, project_id: str) -> bool:
        """Check if a full reconciliation cycle holds the lock for this project."""
        return await self._is_run_locked(project_id)

    async def defer_write(
        self,
        project_id: str,
        content: str,
        category: str,
        metadata: dict,
        agent_id: str | None = None,
    ) -> str:
        """Queue a memory write for replay after the current full cycle completes."""
        write_id = str(uuid4())
        async with self._require_access().write() as db:
            await db.execute(
                """INSERT INTO deferred_writes
                   (id, project_id, content, category, metadata, agent_id, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    write_id,
                    project_id,
                    content,
                    category,
                    json.dumps(metadata),
                    agent_id,
                    datetime.now(UTC).isoformat(),
                ),
            )
        logger.info(
            'reconciliation.write_deferred',
            extra={'project_id': project_id, 'category': category, 'write_id': write_id},
        )
        return write_id

    async def claim_deferred_writes(self, project_id: str) -> list[dict]:
        """Atomically claim all pending (unclaimed) deferred writes for a project.

        Returns list of ``{id, content, category, metadata, agent_id}`` dicts,
        ordered by created_at ASC.  Claimed rows have ``claimed_at`` set to now
        and are excluded from subsequent calls until released or deleted.

        **Concurrency note**: the SELECT + UPDATE are issued inside a single
        writer transaction, which is sufficient when there is at most one
        concurrent writer per project.  Two processes that both SELECT before
        either issues the UPDATE would see the same pending rows and both claim
        them, resulting in duplicate replay.  Correctness relies on the caller
        holding the per-project reconciliation lock so that only one process
        replays each project's writes at a time.
        """
        now_iso = datetime.now(UTC).isoformat()
        async with self._require_access().write() as db:
            # Fetch pending rows inside the transaction
            async with db.execute(
                """SELECT id, content, category, metadata, agent_id, created_at
                   FROM deferred_writes
                   WHERE project_id = ? AND claimed_at IS NULL
                   ORDER BY created_at""",
                (project_id,),
            ) as cursor:
                rows = await cursor.fetchall()

            if not rows:
                return []

            # Atomically mark them all as claimed
            ids = [row['id'] for row in rows]
            placeholders = ','.join('?' for _ in ids)
            await db.execute(
                f'UPDATE deferred_writes SET claimed_at = ? WHERE id IN ({placeholders})',
                [now_iso, *ids],
            )

        return [
            {
                'id': row['id'],
                'content': row['content'],
                'category': row['category'],
                'metadata': json.loads(row['metadata']),
                'agent_id': row['agent_id'],
            }
            for row in rows
        ]

    async def delete_deferred_write(self, write_id: str) -> None:
        """Delete a single deferred write by primary key (no-op if not found)."""
        async with self._require_access().write() as db:
            await db.execute('DELETE FROM deferred_writes WHERE id = ?', (write_id,))

    async def _debug_get_deferred_row(self, write_id: str) -> dict | None:
        """Debug accessor — returns the full deferred_writes row as a dict, or None.

        Intended for tests that need to inspect columns not exposed by
        ``claim_deferred_writes`` (e.g. ``attempt_count``, ``claimed_at``).
        ``metadata`` is parsed with ``json.loads`` to match the shape returned
        by ``claim_deferred_writes``, so callers can compare values from both
        APIs without shape mismatches.

        The leading underscore marks this as a test-support API, not part of
        EventBuffer's production interface.
        """
        row = await self._require_access().read_one(
            'SELECT id, project_id, content, category, metadata, agent_id,'
            '       created_at, claimed_at, attempt_count'
            ' FROM deferred_writes WHERE id = ?',
            (write_id,),
        )
        if row is None:
            return None
        return {
            'id': row['id'],
            'project_id': row['project_id'],
            'content': row['content'],
            'category': row['category'],
            'metadata': json.loads(row['metadata']),
            'agent_id': row['agent_id'],
            'created_at': row['created_at'],
            'claimed_at': row['claimed_at'],
            'attempt_count': row['attempt_count'],
        }

    async def release_stale_claims(self, max_age_seconds: float) -> int:
        """Reset claimed_at to NULL for rows claimed longer ago than max_age_seconds.

        The comparison is ``<=`` so that ``max_age_seconds=0.0`` is supported as
        "release everything currently claimed" (any row whose claimed_at is at
        most *now* qualifies).

        For each re-queued row ``attempt_count`` is incremented.  Rows that have
        already exhausted ``_MAX_DEFERRED_WRITE_ATTEMPTS`` are treated as
        poison-pills: they are deleted and logged at ERROR so they cannot cause an
        unbounded retry loop across restarts.

        Returns the number of rows re-queued (exhausted rows are not counted).
        """
        cutoff_iso = (datetime.now(UTC) - timedelta(seconds=max_age_seconds)).isoformat()
        async with self._require_access().write() as db:
            # Fetch all stale rows to split recoverable vs. exhausted.
            async with db.execute(
                'SELECT id, content, attempt_count FROM deferred_writes'
                ' WHERE claimed_at IS NOT NULL AND claimed_at <= ?',
                (cutoff_iso,),
            ) as cursor:
                stale = await cursor.fetchall()

            if not stale:
                return 0

            exhausted = [r for r in stale if r['attempt_count'] >= _MAX_DEFERRED_WRITE_ATTEMPTS]
            recoverable = [r for r in stale if r['attempt_count'] < _MAX_DEFERRED_WRITE_ATTEMPTS]

            if exhausted:
                for r in exhausted:
                    logger.error(
                        'EventBuffer: deferred write %s exhausted %d replay attempts'
                        ' — dropping permanently: %.120r',
                        r['id'],
                        _MAX_DEFERRED_WRITE_ATTEMPTS,
                        r['content'],
                    )
                placeholders = ','.join('?' * len(exhausted))
                await db.execute(
                    f'DELETE FROM deferred_writes WHERE id IN ({placeholders})',
                    [r['id'] for r in exhausted],
                )

            rowcount = 0
            if recoverable:
                placeholders = ','.join('?' * len(recoverable))
                cursor = await db.execute(
                    f'UPDATE deferred_writes'
                    f' SET claimed_at = NULL, attempt_count = attempt_count + 1'
                    f' WHERE id IN ({placeholders})',
                    [r['id'] for r in recoverable],
                )
                rowcount = cursor.rowcount

        if rowcount:
            logger.info(
                'release_stale_claims: re-queued %d deferred write(s) (cutoff=%s)',
                rowcount,
                cutoff_iso,
            )
        return rowcount

    # ── Queries ────────────────────────────────────────────────────────

    async def get_active_projects(self) -> list[str]:
        """Return project IDs that have buffered events."""
        rows = await self._require_access().read_all(
            "SELECT DISTINCT project_id FROM event_buffer WHERE status = 'buffered'"
        )
        return [row['project_id'] for row in rows]

    async def get_buffer_stats(self, project_id: str) -> dict:
        """Buffer size and oldest event age for a project."""
        row = await self._require_access().read_one(
            """SELECT COUNT(*) as cnt, MIN(timestamp) as oldest
               FROM event_buffer
               WHERE project_id = ? AND status = 'buffered'""",
            (project_id,),
        )

        count = row['cnt'] if row else 0
        oldest_str = row['oldest'] if row else None

        if count == 0 or not oldest_str:
            return {'size': count, 'oldest_event_age_seconds': None}

        oldest = datetime.fromisoformat(oldest_str)
        if oldest.tzinfo is None:
            oldest = oldest.replace(tzinfo=UTC)
        age = (datetime.now(UTC) - oldest).total_seconds()
        return {'size': count, 'oldest_event_age_seconds': round(age, 1)}

    # ── Maintenance ────────────────────────────────────────────────────

    async def cleanup_drained(self, max_age_seconds: float = 3600.0) -> int:
        """Delete drained events older than cutoff, skipping locked projects.

        Task 3049: before deleting, roll the very same rows up into
        ``event_arrival_hourly`` so the project's arrival history survives the
        DELETE.  This is the only DELETE of a COMMITTED ``event_buffer`` row in
        the codebase — the single chokepoint where inflow history is destroyed
        — so rolling up here captures 100% of events regardless of which drain
        path (``drain`` / ``drain_oldest_chunk`` / ``drain_by_ids``) marked
        them.  ``push``'s fold deletes only the row it inserted earlier in the
        same uncommitted unit, and counts that event in
        ``event_coalesced_hourly``.

        The rollup runs inside the SAME write unit as the DELETE, and takes its
        rows FROM that delete via ``DELETE ... RETURNING`` (SQLite >= 3.35,
        already relied on by ``services/durable_queue.py``).  That is the
        correctness property: each row is counted exactly once, because it is
        either still live in ``event_buffer`` (countable directly) or in the
        aggregate (countable there), never both and never neither.  Deriving
        the aggregated set from the deletion itself — rather than re-evaluating
        the same predicate in a separate SELECT — is what makes the two halves
        structurally incapable of disagreeing, and it halves the scan cost of a
        sweep that runs every 50s against a table that can hold six-figure row
        counts after a deep backlog drain.  In particular the lock exclusion is
        honoured by construction: a locked project's rows are not deleted, so
        they are not returned, so they are not rolled up — they simply roll up
        on the later sweep that does delete them.

        This "never both and never neither" guarantee, and the atomicity a
        mid-sweep fault rolls back to, both require that no *other* commit lands
        on the shared connection while this sweep is in flight.  That is now
        ENFORCED rather than assumed: ``AtomicConnection.write()`` holds the
        per-connection lock across the whole chunked fold, so the harness's
        concurrently running per-project tasks cannot interleave their own
        ``push``/``drain`` commits into it.  The enforcement matters most here,
        because draining the RETURNING cursor via chunked ``fetchmany`` (below)
        suspends once per chunk (~101 times at N=1,000,000) rather than once
        after a single ``fetchall()`` — a window the old unlocked transaction
        wrapper could only document, not close.

        A row whose timestamp cannot be parsed is still DELETED; it is only
        left out of the rollup, with a structured warning naming it.  Bucketing
        raises ``ValueError`` on a malformed timestamp and this transaction is
        the ONLY path that prunes the buffer, so letting one poison row abort
        it would wedge cleanup permanently (the caller logs a WARNING and
        retries every 50s forever) and grow ``event_buffer`` without bound.
        Losing one row from an inflow aggregate is a rounding error; losing the
        only pruning path is an outage.

        The RETURNING cursor is drained through a bounded
        ``fetchmany(_CLEANUP_FETCH_CHUNK)`` loop rather than a single
        ``fetchall()``, which keeps peak memory at O(chunk + distinct hour
        buckets) instead of O(rows deleted).  Measured on CPython 3.13 /
        aiosqlite 0.22.1 / SQLite 3.50.4 with N = 1,000,000 drained rows
        folding into 3,336 hour buckets: ``fetchall()`` (the prior shape)
        took 1 round-trip for a 281 MB tracemalloc peak and +505 MB max RSS
        over baseline; ``async for`` at aiosqlite's driver-default
        ``iter_chunk_size=64`` took 15,626 round-trips for a 1.0 MB peak and
        flat RSS; ``fetchmany(10_000)`` (this method) takes 101 round-trips
        for a 6.5 MB peak and flat RSS.  ``fetchmany(10_000)`` wins on
        memory — eliminating the 505 MB spike a single ``fetchall()`` holds
        under the writer lock — and on round-trips (155x fewer than
        ``async for``'s default), and is never slower than ``fetchall()``.
        It is NOT a speedup: run-to-run wall clock spread was 7.5-15.0s
        across all three shapes, swamping any between-shape difference,
        because per-row ``sqlite3.Row`` construction and the dict fold
        dominate, not driver round-trips.  10_000 is the knee of a
        chunk-size sweep at N=200,000 (1_000 -> 0.7 MB / 201 round-trips,
        10_000 -> 5.7 MB / 21, 50_000 -> 27.8 MB / 5): each step past it
        buys back progressively less memory per extra round-trip.  These
        numbers are host- and version-pinned — re-measure rather than trust
        them if the driver, SQLite version, or row shape changes materially.

        A connection-wide ``iter_chunk_size`` on ``connect_daemon`` plus
        ``async for`` was considered and rejected: it is a shared knob that
        would silently affect every ``async for row in cursor`` on that
        connection, present and future, and the bound would live hundreds of
        lines from the code whose comment explains it.

        Correction to the assumption this method was originally written
        under: abandoning a ``DELETE ... RETURNING`` cursor early does NOT
        leave the delete unmodified.  Measured on SQLite 3.50.4: consuming
        100 of 1,000 RETURNING rows, exiting the cursor, and committing
        still deleted all 1,000 rows — SQLite runs the DML to completion and
        stages RETURNING output before emitting the first row.  So the
        reason to consume the cursor to completion is NOT that the DELETE
        would otherwise be left half-done; it is that an early exit silently
        under-counts and under-rolls-up while the rows are already gone,
        permanently breaking the "never both and never neither" guarantee
        above with unrecoverable inflow loss.  This is a SQLite
        implementation detail, not a documented guarantee — re-verify it if
        the pinned version ever moves.

        Returns:
            The number of rows deleted — counted from the RETURNING output, so
            it stays exact regardless of how the driver reports ``rowcount``
            for a returning statement.
        """
        cutoff = datetime.fromtimestamp(
            datetime.now(UTC).timestamp() - max_age_seconds,
            tz=UTC,
        ).isoformat()
        counts: dict[tuple[str, str, str], int] = {}
        deleted_count = 0
        async with self._require_access().write() as db:
            async with db.execute(
                """DELETE FROM event_buffer
                   WHERE status = 'drained'
                     AND timestamp < ?
                     AND project_id NOT IN (
                         SELECT project_id FROM reconciliation_locks
                     )
                   RETURNING project_id, timestamp, event_type""",
                (cutoff,),
            ) as cursor:
                while True:
                    # list() bounds at _CLEANUP_FETCH_CHUNK: fetchmany() is
                    # typed Iterable[Row] like fetchall(), so len() needs a
                    # Sized (61ae0ee799).
                    chunk = list(await cursor.fetchmany(_CLEANUP_FETCH_CHUNK))
                    if not chunk:
                        break
                    deleted_count += len(chunk)
                    for row in chunk:
                        # Bucket by PARSING, never by a SQL comparison against a
                        # datetime('now') literal — see throughput's METHOD NOTE:
                        # event_buffer.timestamp carries an offset and is
                        # 'T'-separated, so string-comparing it against SQLite's
                        # space-separated render collapses a whole day into one bucket.
                        try:
                            bucket = utc_hour_bucket(row['timestamp'])
                        except ValueError as exc:
                            logger.warning(
                                'event_buffer.rollup_unparseable_timestamp',
                                extra={
                                    'project_id': row['project_id'],
                                    'event_type': row['event_type'],
                                    'timestamp': row['timestamp'],
                                    'error': str(exc),
                                },
                            )
                            continue
                        key = (row['project_id'], bucket, row['event_type'])
                        counts[key] = counts.get(key, 0) + 1

            if counts:
                await db.executemany(
                    """INSERT INTO event_arrival_hourly
                           (project_id, hour_bucket, event_type, event_count)
                       VALUES (?, ?, ?, ?)
                       ON CONFLICT(project_id, hour_bucket, event_type)
                       DO UPDATE SET event_count = event_count + excluded.event_count""",
                    [(pid, bucket, etype, n) for (pid, bucket, etype), n in counts.items()],
                )

        return deleted_count

    async def request_trigger(self, project_id: str) -> None:
        """Manually request a reconciliation trigger for a project.

        Sets a flag consumed by should_trigger(). If the buffer is empty,
        inserts a synthetic event so get_active_projects() returns the project.
        """
        self._manual_triggers.add(project_id)
        # Ensure the project shows up in get_active_projects()
        count = await self.count_buffered(project_id)
        if count == 0:
            await self.push(ReconciliationEvent(
                id=str(uuid4()),
                project_id=project_id,
                type=EventType.memory_added,
                source=EventSource.full_reconciliation,
                agent_id=None,
                timestamp=datetime.now(UTC),
                payload={'_synthetic': True, '_reason': 'manual_trigger'},
            ))
        logger.info(f'Manual reconciliation trigger requested for {project_id}')

    def _consume_manual_trigger(self, project_id: str) -> bool:
        """Return True and clear the flag if a manual trigger was requested."""
        if project_id in self._manual_triggers:
            self._manual_triggers.discard(project_id)
            return True
        return False

    async def close(self) -> None:
        """Close the database connection."""
        if self._access is not None:
            await self._access.close()
            self._access = None

    async def checkpoint(self) -> CheckpointResult:
        return await AtomicConnection.checkpoint_or_unavailable(self._access)
