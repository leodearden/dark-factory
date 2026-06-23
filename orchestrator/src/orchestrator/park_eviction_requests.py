"""Park-eviction request persistence for the scheduler.

Operators enqueue a park-eviction request for a stranded owner via the
dashboard/MCP.  Each scheduler tick drains the table atomically and applies
the D4 live-owner guard before calling ModuleLockTable.force_clear.

Stored in a separate SQLite file (park_eviction_requests.db) isolated from
the overrides schema — mirroring the overrides.db/runs.db separate-file
pattern to avoid blast-radius from cross-schema changes.  See design
decision in plan.json §D3 for full rationale.

The fused-memory ε-layer writes raw SQL matching this 3-col contract; there
is no shared import — the hand-mirroring is intentional.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from shared.sqlite_sync_base import apply_full_durability_pragmas_sync

from orchestrator.config import OrchestratorConfig

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS park_eviction_requests (
    task_id       TEXT NOT NULL,
    project_root  TEXT NOT NULL,
    requested_at  TEXT NOT NULL
);
"""


class ParkEvictionRequestStore:
    """Synchronous SQLite store for operator park-eviction requests.

    Mirrors OverrideStore: connect-per-call, WAL journal mode,
    ``mkdir(parents=True)`` on init.  Thread-safe for the orchestrator's
    single-tick async consumer pattern.

    Usage pattern (drain → act → already deleted):
        task_ids = store.drain(project_root)
        for task_id in task_ids:
            # apply guard and force_clear; row already gone from the DB
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    @classmethod
    def from_config(cls, config: OrchestratorConfig) -> ParkEvictionRequestStore:
        """Build the canonical ParkEvictionRequestStore for *config*.

        Centralizes the DB-path derivation so Harness, CLI, and any future
        caller stay in lock-step.  See ``OrchestratorConfig.park_eviction_requests_db_path``.
        """
        return cls(config.park_eviction_requests_db_path)

    def _connect(self) -> sqlite3.Connection:
        """Open a connection with WAL pragmas applied."""
        conn = sqlite3.connect(str(self.db_path))
        apply_full_durability_pragmas_sync(conn, busy_timeout_ms=5000)
        return conn

    def _ensure_schema(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(_SCHEMA)
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    def enqueue(
        self,
        task_id: str,
        project_root: str,
        requested_at: str | None = None,
    ) -> None:
        """Insert one park-eviction request row.

        *requested_at* defaults to ``datetime.now(UTC).isoformat()`` when
        omitted, matching the OverrideStore created_at convention.
        """
        if requested_at is None:
            requested_at = datetime.now(UTC).isoformat()
        conn = self._connect()
        try:
            conn.execute(
                'INSERT INTO park_eviction_requests (task_id, project_root, requested_at) '
                'VALUES (?, ?, ?)',
                (task_id, project_root, requested_at),
            )
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Read/drain API
    # ------------------------------------------------------------------

    def drain(self, project_root: str) -> list[str]:
        """Atomically DELETE all rows for *project_root* and return their task_ids.

        One-shot: a second call returns [].  Mirrors the DELETE...RETURNING
        idiom from OverrideStore.clear_terminal (overrides.py:454-461).
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                'DELETE FROM park_eviction_requests WHERE project_root=? RETURNING task_id',
                (project_root,),
            ).fetchall()
            conn.commit()
            return [r[0] for r in rows]
        finally:
            conn.close()
