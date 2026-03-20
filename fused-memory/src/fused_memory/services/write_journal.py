"""SQLite-backed write journal for durable auditing of all memory writes."""

from __future__ import annotations

import json
import logging
import uuid as uuid_mod
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite

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


class WriteJournal:
    """Two-layer write journal backed by SQLite (WAL mode)."""

    def __init__(self, data_dir: Path | str):
        self.data_dir = Path(data_dir)
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        db_path = self.data_dir / 'write_journal.db'
        self._db = await aiosqlite.connect(str(db_path))
        self._db.row_factory = aiosqlite.Row
        await self._db.execute('PRAGMA journal_mode=WAL')
        await self._db.executescript(SCHEMA_SQL)
        await self._db.commit()
        logger.info(f'Write journal initialized at {db_path}')

    async def close(self) -> None:
        if self._db:
            await self._db.close()

    def _require_db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError('WriteJournal not initialized — call initialize() first')
        return self._db

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
        params: dict | None = None,
        result_summary: dict | str | None = None,
        success: bool = True,
        error: str | None = None,
    ) -> None:
        """Log a Layer 1 write operation. Fire-and-forget — never raises."""
        try:
            db = self._require_db()
            await db.execute(
                """INSERT INTO write_ops
                   (id, causation_id, source, provenance, operation,
                    project_id, agent_id, params, result_summary, success, error, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    write_op_id,
                    causation_id,
                    source,
                    provenance,
                    operation,
                    project_id,
                    agent_id,
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
            db = self._require_db()
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
        db = self._require_db()
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

    async def get_ops_since(self, since: str, limit: int = 100) -> list[dict]:
        """Return write_ops since a timestamp."""
        db = self._require_db()
        async with db.execute(
            'SELECT * FROM write_ops WHERE created_at >= ? ORDER BY created_at LIMIT ?',
            (since, limit),
        ) as cursor:
            return [dict(row) for row in await cursor.fetchall()]

    async def get_backend_ops_for_write_op(self, write_op_id: str) -> list[dict]:
        """Return all backend_ops linked to a write_op."""
        db = self._require_db()
        async with db.execute(
            'SELECT * FROM backend_ops WHERE write_op_id = ? ORDER BY created_at',
            (write_op_id,),
        ) as cursor:
            return [dict(row) for row in await cursor.fetchall()]
