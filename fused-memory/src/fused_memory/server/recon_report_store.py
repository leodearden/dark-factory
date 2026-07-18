"""Sync SQLite write-through persistence for ReconReportState (task 2716).

Backs the in-process ``ReconReportState`` (server/recon_report.py) with a
durable table ``recon_report_state`` keyed by ``(run_id, stage)``, so a
mid-stage server restart does not lose previously-filed findings.

Synchronous by design: the ``ReconReportState`` mutators that must write
through (``start_report``/``add_finding``/``set_stat``/``inc_stat``/
``complete``/``delete_finding``) are themselves synchronous and have
non-MCP-wrapper synchronous callers (``server/recon_lifecycle_filer.py``,
``reconciliation/stages/base.py``) — an async store cannot be awaited from
inside them. Uses ``shared.sqlite_sync_base.apply_full_durability_pragmas_sync``,
the sanctioned synchronous counterpart to ``shared.async_sqlite_base``, on a
single persistent ``sqlite3.Connection``. All recon-report writes originate
on the single recon-report uvicorn event-loop thread, so a persistent
sync connection is safe; ``check_same_thread`` is left at its default
(True) so an accidental cross-thread call fails loudly instead of silently
corrupting state.

Lifecycle::

    store = ReconReportStore(path)
    store.open()
    try:
        store.upsert_entry(...)
    finally:
        store.close()
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from shared.sqlite_sync_base import apply_full_durability_pragmas_sync

__all__ = ['ReconReportStore']

# Schema without PRAGMA — pragmas are set once on the persistent connection
# by apply_full_durability_pragmas_sync (see open()).
_SCHEMA = """\
CREATE TABLE IF NOT EXISTS recon_report_state (
    run_id      TEXT    NOT NULL,
    stage       TEXT    NOT NULL,
    project_id  TEXT    NOT NULL,
    is_active   INTEGER NOT NULL DEFAULT 0,
    entry_json  TEXT    NOT NULL,
    updated_at  REAL    NOT NULL,
    PRIMARY KEY (run_id, stage)
);

CREATE INDEX IF NOT EXISTS idx_rrs_run
    ON recon_report_state (run_id);
"""


class ReconReportStore:
    """Persistent-connection sync SQLite writer for recon_report_state rows."""

    def __init__(self, db_path: Path, *, busy_timeout_ms: int = 30000) -> None:
        self.db_path = db_path
        self.busy_timeout_ms = busy_timeout_ms
        self._conn: sqlite3.Connection | None = None

    def open(self) -> None:
        """Open the persistent connection, apply durability pragmas, ensure schema.

        Raises:
            RuntimeError: if called while already open.
        """
        if self._conn is not None:
            raise RuntimeError(f'{type(self).__name__} already opened')
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        try:
            apply_full_durability_pragmas_sync(conn, busy_timeout_ms=self.busy_timeout_ms)
            conn.executescript(_SCHEMA)
            conn.commit()
        except BaseException:
            conn.close()
            raise
        self._conn = conn

    def close(self) -> None:
        """Close the connection. Idempotent — safe to call when already closed
        or when ``open()`` was never called."""
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None

    def _require_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError(f'{type(self).__name__} not opened')
        return self._conn

    def upsert_entry(
        self,
        *,
        run_id: str,
        stage: str,
        project_id: str,
        is_active: bool,
        entry_json: str,
        updated_at: float,
    ) -> None:
        """Insert or replace the row for ``(run_id, stage)``."""
        conn = self._require_conn()
        conn.execute(
            'INSERT INTO recon_report_state '
            '(run_id, stage, project_id, is_active, entry_json, updated_at) '
            'VALUES (?, ?, ?, ?, ?, ?) '
            'ON CONFLICT(run_id, stage) DO UPDATE SET '
            'project_id = excluded.project_id, '
            'is_active = excluded.is_active, '
            'entry_json = excluded.entry_json, '
            'updated_at = excluded.updated_at',
            (run_id, stage, project_id, int(is_active), entry_json, updated_at),
        )
        conn.commit()

    def delete_run(self, run_id: str) -> None:
        """Delete every row belonging to ``run_id`` (GC at run quiescence)."""
        conn = self._require_conn()
        conn.execute('DELETE FROM recon_report_state WHERE run_id = ?', (run_id,))
        conn.commit()

    def load_all(self) -> list[dict[str, Any]]:
        """Return every persisted row as a dict, for hydrate-on-boot."""
        conn = self._require_conn()
        cur = conn.execute(
            'SELECT run_id, stage, project_id, is_active, entry_json, updated_at '
            'FROM recon_report_state'
        )
        return [
            {
                'run_id': row[0],
                'stage': row[1],
                'project_id': row[2],
                'is_active': bool(row[3]),
                'entry_json': row[4],
                'updated_at': row[5],
            }
            for row in cur.fetchall()
        ]
