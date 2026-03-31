"""SQLite-backed write journal for durable auditing of all memory writes."""

from __future__ import annotations

import json
import logging
import uuid as uuid_mod
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite

from shared.async_sqlite_base import AsyncSqliteBase

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
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
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_wo_causation ON write_ops(causation_id);
CREATE INDEX IF NOT EXISTS idx_wo_project_time ON write_ops(project_id, created_at);
CREATE INDEX IF NOT EXISTS idx_wo_operation ON write_ops(operation);

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
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_bo_write_op ON backend_ops(write_op_id);
CREATE INDEX IF NOT EXISTS idx_bo_causation ON backend_ops(causation_id);
CREATE INDEX IF NOT EXISTS idx_bo_created ON backend_ops(created_at);
"""


class WriteJournal(AsyncSqliteBase):
    """Two-layer write journal backed by SQLite (WAL mode)."""

    def __init__(self, data_dir: Path | str):
        super().__init__(Path(data_dir) / 'write_journal.db')

    @property
    def _schema(self) -> str:
        return SCHEMA_SQL

    async def open(self) -> None:
        await super().open()
        self._require_conn().row_factory = aiosqlite.Row
        await self._migrate()
        logger.info('Write journal initialized at %s', self.db_path)

    async def _migrate(self) -> None:
        """Add columns introduced after initial schema (idempotent)."""
        db = self._require_conn()
        async with db.execute('PRAGMA table_info(write_ops)') as cursor:
            existing = {row[1] for row in await cursor.fetchall()}

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

        # Indexes on new columns (safe after migration ensures columns exist)
        await db.execute(
            'CREATE INDEX IF NOT EXISTS idx_wo_kind_time ON write_ops(kind, created_at)'
        )
        await db.execute(
            'CREATE INDEX IF NOT EXISTS idx_wo_agent_time ON write_ops(agent_id, created_at)'
        )
        await db.commit()

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
        """Log a Layer 1 operation. Fire-and-forget — never raises."""
        try:
            db = self._require_conn()
            await db.execute(
                """INSERT INTO write_ops
                   (id, causation_id, source, provenance, operation,
                    project_id, agent_id, session_id, kind,
                    params, result_summary, success, error, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                    datetime.now(UTC).isoformat(),
                ),
            )
            await db.commit()
        except Exception as e:
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
    ) -> None:
        """Log a Layer 2 backend dispatch. Fire-and-forget — never raises."""
        try:
            db = self._require_conn()
            await db.execute(
                """INSERT INTO backend_ops
                   (id, write_op_id, causation_id, backend, operation,
                    payload, result_summary, success, error, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(uuid_mod.uuid4()),
                    write_op_id,
                    causation_id,
                    backend,
                    operation,
                    json.dumps(payload) if payload else '{}',
                    json.dumps(result_summary) if isinstance(result_summary, dict) else result_summary,
                    1 if success else 0,
                    error,
                    datetime.now(UTC).isoformat(),
                ),
            )
            await db.commit()
        except Exception as e:
            logger.warning(f'Failed to log backend_op: {e}')

    async def get_ops_by_causation(self, causation_id: str) -> list[dict]:
        """Return all write_ops and backend_ops for a causation_id."""
        db = self._require_conn()
        results: list[dict] = []

        async with db.execute(
            'SELECT * FROM write_ops WHERE causation_id = ? ORDER BY created_at',
            (causation_id,),
        ) as cursor:
            for row in await cursor.fetchall():
                results.append({'layer': 'write_op', **dict(row)})

        async with db.execute(
            'SELECT * FROM backend_ops WHERE causation_id = ? ORDER BY created_at',
            (causation_id,),
        ) as cursor:
            for row in await cursor.fetchall():
                results.append({'layer': 'backend_op', **dict(row)})

        results.sort(key=lambda r: r.get('created_at', ''))
        return results

    async def get_ops_since(
        self, since: str, limit: int = 100, kind: str | None = None
    ) -> list[dict]:
        """Return write_ops since a timestamp, optionally filtered by kind."""
        db = self._require_conn()
        if kind:
            sql = 'SELECT * FROM write_ops WHERE created_at >= ? AND kind = ? ORDER BY created_at LIMIT ?'
            params = (since, kind, limit)
        else:
            sql = 'SELECT * FROM write_ops WHERE created_at >= ? ORDER BY created_at LIMIT ?'
            params = (since, limit)
        async with db.execute(sql, params) as cursor:
            return [dict(row) for row in await cursor.fetchall()]

    async def get_backend_ops_for_write_op(self, write_op_id: str) -> list[dict]:
        """Return all backend_ops linked to a write_op."""
        db = self._require_conn()
        async with db.execute(
            'SELECT * FROM backend_ops WHERE write_op_id = ? ORDER BY created_at',
            (write_op_id,),
        ) as cursor:
            return [dict(row) for row in await cursor.fetchall()]

    async def get_usage_stats(
        self, since: str, project_id: str | None = None
    ) -> dict:
        """Aggregate read/write stats since a timestamp.

        Returns {reads, writes, by_operation, by_agent}.
        """
        db = self._require_conn()
        where = 'WHERE created_at >= ?'
        params: list = [since]
        if project_id:
            where += ' AND project_id = ?'
            params.append(project_id)

        # Totals by kind
        async with db.execute(
            f'SELECT kind, COUNT(*) FROM write_ops {where} GROUP BY kind', params
        ) as cursor:
            kind_counts = {row[0]: row[1] for row in await cursor.fetchall()}

        # By operation
        async with db.execute(
            f'SELECT operation, COUNT(*) FROM write_ops {where} GROUP BY operation',
            params,
        ) as cursor:
            by_operation = {row[0]: row[1] for row in await cursor.fetchall()}

        # By agent (read/write breakdown)
        async with db.execute(
            f'SELECT agent_id, kind, COUNT(*) FROM write_ops {where} GROUP BY agent_id, kind',
            params,
        ) as cursor:
            by_agent: dict[str, dict[str, int]] = {}
            for row in await cursor.fetchall():
                aid = row[0] or '_unknown'
                if aid not in by_agent:
                    by_agent[aid] = {'read': 0, 'write': 0}
                by_agent[aid][row[1]] = row[2]

        return {
            'reads': kind_counts.get('read', 0),
            'writes': kind_counts.get('write', 0),
            'by_operation': by_operation,
            'by_agent': by_agent,
        }

    async def get_session_ops(
        self, agent_id: str, since: str | None = None, limit: int = 100
    ) -> list[dict]:
        """Return ops for a specific agent, most recent first."""
        db = self._require_conn()
        if since:
            sql = (
                'SELECT * FROM write_ops WHERE agent_id = ? AND created_at >= ? '
                'ORDER BY created_at DESC LIMIT ?'
            )
            params = (agent_id, since, limit)
        else:
            sql = (
                'SELECT * FROM write_ops WHERE agent_id = ? '
                'ORDER BY created_at DESC LIMIT ?'
            )
            params = (agent_id, limit)
        async with db.execute(sql, params) as cursor:
            return [dict(row) for row in await cursor.fetchall()]
