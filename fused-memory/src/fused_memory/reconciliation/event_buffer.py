"""SQLite-backed event buffer with burst detection and quiescence triggers."""

import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import aiosqlite

from fused_memory.models.reconciliation import (
    EventSource,
    EventType,
    ReconciliationEvent,
)

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
    status TEXT NOT NULL DEFAULT 'buffered'
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
"""


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
    ):
        self._db_path = str(db_path) if db_path else ':memory:'
        self.buffer_size_threshold = buffer_size_threshold
        self.max_staleness_seconds = max_staleness_seconds
        self.conditional_trigger_ratio = conditional_trigger_ratio
        self.burst_window_seconds = burst_window_seconds
        self.burst_cooldown_seconds = burst_cooldown_seconds
        self.stale_lock_seconds = stale_lock_seconds
        self._queue_stats_fn = queue_stats_fn
        self.instance_id = instance_id or str(uuid4())
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Open SQLite connection and ensure schema exists."""
        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute('PRAGMA journal_mode=WAL')
        await self._db.execute('PRAGMA busy_timeout=5000')
        await self._db.executescript(_BUFFER_SCHEMA_SQL)
        await self._db.commit()
        logger.info(f'EventBuffer initialized (db={self._db_path}, instance={self.instance_id})')

    def _require_db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError('EventBuffer not initialized — call initialize() first')
        return self._db

    # ── Push ───────────────────────────────────────────────────────────

    async def push(self, event: ReconciliationEvent) -> None:
        """Insert event into the shared buffer."""
        db = self._require_db()
        await db.execute(
            """INSERT INTO event_buffer
               (id, project_id, event_type, event_source, agent_id, timestamp, payload, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'buffered')""",
            (
                event.id,
                event.project_id,
                event.type.value,
                event.source.value,
                event.agent_id,
                event.timestamp.isoformat(),
                json.dumps(event.payload),
            ),
        )
        await db.commit()

        if event.agent_id is not None:
            await self._update_burst_state(event.agent_id, event.timestamp)

        logger.info(
            'reconciliation.event_buffered',
            extra={
                'project_id': event.project_id,
                'event_type': event.type.value,
                'agent_id': event.agent_id,
            },
        )

    # ── Burst detection ────────────────────────────────────────────────

    async def _update_burst_state(self, agent_id: str, timestamp: datetime) -> None:
        """Track per-agent write bursts.  2+ writes within burst_window → bursting."""
        db = self._require_db()
        ts_iso = timestamp.isoformat()
        cutoff = (timestamp.timestamp() - self.burst_window_seconds)
        cutoff_iso = datetime.fromtimestamp(cutoff, tz=UTC).isoformat()

        async with db.execute(
            """SELECT COUNT(*) as cnt FROM event_buffer
               WHERE agent_id = ? AND timestamp >= ?""",
            (agent_id, cutoff_iso),
        ) as cursor:
            row = await cursor.fetchone()
            recent_count = row['cnt'] if row else 0

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
        await db.commit()

    # ── Trigger logic ──────────────────────────────────────────────────

    async def should_trigger(self, project_id: str) -> tuple[bool, str]:
        """Three-tier trigger: hard thresholds, then conditional quiescence.

        Returns (should_trigger, reason).
        """
        db = self._require_db()

        # Check run lock
        if await self._is_run_locked(project_id):
            return False, ''

        # Count pending events + oldest timestamp
        async with db.execute(
            """SELECT COUNT(*) as cnt, MIN(timestamp) as oldest
               FROM event_buffer
               WHERE project_id = ? AND status = 'buffered'""",
            (project_id,),
        ) as cursor:
            row = await cursor.fetchone()

        count = row['cnt'] if row else 0
        oldest_str = row['oldest'] if row else None

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
        db = self._require_db()
        # Expire stale locks first
        cutoff = datetime.fromtimestamp(
            datetime.now(UTC).timestamp() - self.stale_lock_seconds,
            tz=UTC,
        ).isoformat()
        await db.execute(
            'DELETE FROM reconciliation_locks WHERE heartbeat_at < ?',
            (cutoff,),
        )
        await db.commit()

        async with db.execute(
            'SELECT 1 FROM reconciliation_locks WHERE project_id = ?',
            (project_id,),
        ) as cursor:
            return await cursor.fetchone() is not None

    async def _is_quiescent(self) -> bool:
        """System is quiescent when no agent is bursting and durable queue is idle."""
        db = self._require_db()

        # Expire stale bursts (cooldown elapsed since last write)
        cooldown_cutoff = datetime.fromtimestamp(
            datetime.now(UTC).timestamp() - self.burst_cooldown_seconds,
            tz=UTC,
        ).isoformat()
        await db.execute(
            """UPDATE burst_state SET state = 'idle', burst_started_at = NULL
               WHERE state = 'bursting' AND last_write_at < ?""",
            (cooldown_cutoff,),
        )
        await db.commit()

        # Any agents still bursting?
        async with db.execute(
            "SELECT 1 FROM burst_state WHERE state = 'bursting' LIMIT 1"
        ) as cursor:
            if await cursor.fetchone() is not None:
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

    async def drain(self, project_id: str) -> list[ReconciliationEvent]:
        """Atomically drain buffered events for a project."""
        db = self._require_db()
        async with db.execute(
            """SELECT * FROM event_buffer
               WHERE project_id = ? AND status = 'buffered'
               ORDER BY timestamp""",
            (project_id,),
        ) as cursor:
            rows = await cursor.fetchall()

        if not rows:
            return []

        ids = [row['id'] for row in rows]
        placeholders = ','.join('?' for _ in ids)
        await db.execute(
            f"UPDATE event_buffer SET status = 'drained' WHERE id IN ({placeholders})",
            ids,
        )
        await db.commit()

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
    ) -> list[ReconciliationEvent]:
        """Drain the oldest `limit` buffered events for a project."""
        db = self._require_db()
        async with db.execute(
            """SELECT * FROM event_buffer
               WHERE project_id = ? AND status = 'buffered'
               ORDER BY timestamp ASC
               LIMIT ?""",
            (project_id, limit),
        ) as cursor:
            rows = await cursor.fetchall()

        if not rows:
            return []

        ids = [row['id'] for row in rows]
        placeholders = ','.join('?' for _ in ids)
        await db.execute(
            f"UPDATE event_buffer SET status = 'drained' WHERE id IN ({placeholders})",
            ids,
        )
        await db.commit()

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

    async def count_buffered(self, project_id: str) -> int:
        """Return count of buffered events for a project."""
        db = self._require_db()
        async with db.execute(
            "SELECT COUNT(*) as cnt FROM event_buffer WHERE project_id = ? AND status = 'buffered'",
            (project_id,),
        ) as cursor:
            row = await cursor.fetchone()
        return row['cnt'] if row else 0

    # ── Run locking ────────────────────────────────────────────────────

    async def mark_run_active(self, project_id: str) -> bool:
        """Acquire cross-instance lock. Returns False if already held."""
        db = self._require_db()
        # Expire stale locks
        cutoff = datetime.fromtimestamp(
            datetime.now(UTC).timestamp() - self.stale_lock_seconds,
            tz=UTC,
        ).isoformat()
        await db.execute(
            'DELETE FROM reconciliation_locks WHERE heartbeat_at < ?',
            (cutoff,),
        )
        await db.commit()

        now = datetime.now(UTC).isoformat()
        try:
            await db.execute(
                """INSERT INTO reconciliation_locks (project_id, instance_id, acquired_at, heartbeat_at)
                   VALUES (?, ?, ?, ?)""",
                (project_id, self.instance_id, now, now),
            )
            await db.commit()
            return True
        except Exception:
            # PK constraint violation — another instance holds the lock
            await db.rollback()
            return False

    async def mark_run_complete(self, project_id: str) -> None:
        """Release the reconciliation lock."""
        db = self._require_db()
        await db.execute(
            'DELETE FROM reconciliation_locks WHERE project_id = ?',
            (project_id,),
        )
        await db.commit()

    async def heartbeat(self, project_id: str) -> None:
        """Update lock heartbeat to prevent stale-lock recovery."""
        db = self._require_db()
        now = datetime.now(UTC).isoformat()
        await db.execute(
            'UPDATE reconciliation_locks SET heartbeat_at = ? WHERE project_id = ? AND instance_id = ?',
            (now, project_id, self.instance_id),
        )
        await db.commit()

    # ── Queries ────────────────────────────────────────────────────────

    async def get_active_projects(self) -> list[str]:
        """Return project IDs that have buffered events."""
        db = self._require_db()
        async with db.execute(
            "SELECT DISTINCT project_id FROM event_buffer WHERE status = 'buffered'"
        ) as cursor:
            rows = await cursor.fetchall()
        return [row['project_id'] for row in rows]

    async def get_buffer_stats(self, project_id: str) -> dict:
        """Buffer size and oldest event age for a project."""
        db = self._require_db()
        async with db.execute(
            """SELECT COUNT(*) as cnt, MIN(timestamp) as oldest
               FROM event_buffer
               WHERE project_id = ? AND status = 'buffered'""",
            (project_id,),
        ) as cursor:
            row = await cursor.fetchone()

        count = row['cnt'] if row else 0
        oldest_str = row['oldest'] if row else None

        if count == 0:
            return {'size': 0, 'oldest_event_age_seconds': None}

        oldest = datetime.fromisoformat(oldest_str)
        if oldest.tzinfo is None:
            oldest = oldest.replace(tzinfo=UTC)
        age = (datetime.now(UTC) - oldest).total_seconds()
        return {'size': count, 'oldest_event_age_seconds': round(age, 1)}

    # ── Maintenance ────────────────────────────────────────────────────

    async def cleanup_drained(self, max_age_seconds: float = 3600.0) -> int:
        """Delete drained events older than cutoff. Returns count deleted."""
        db = self._require_db()
        cutoff = datetime.fromtimestamp(
            datetime.now(UTC).timestamp() - max_age_seconds,
            tz=UTC,
        ).isoformat()
        cursor = await db.execute(
            "DELETE FROM event_buffer WHERE status = 'drained' AND timestamp < ?",
            (cutoff,),
        )
        await db.commit()
        return cursor.rowcount

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None
