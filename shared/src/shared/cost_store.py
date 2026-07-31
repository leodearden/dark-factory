"""Async aiosqlite-backed store for per-invocation cost records and account events.

Uses a persistent connection opened via open()/close() or the async context manager::

    async with CostStore(path) as store:
        await store.save_invocation(...)
        await store.save_account_event(...)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from shared.async_sqlite_base import AsyncSqliteBase

__all__ = ['CostStore']

# Schema without PRAGMA — pragmas are set once on the persistent connection.
_SCHEMA = """\
CREATE TABLE IF NOT EXISTS invocations (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              TEXT NOT NULL,
    task_id             TEXT,
    project_id          TEXT NOT NULL,
    account_name        TEXT NOT NULL,
    model               TEXT NOT NULL,
    role                TEXT NOT NULL,
    cost_usd            REAL NOT NULL DEFAULT 0.0,
    input_tokens        INTEGER,
    output_tokens       INTEGER,
    cache_read_tokens   INTEGER,
    cache_create_tokens INTEGER,
    duration_ms         INTEGER NOT NULL DEFAULT 0,
    capped              INTEGER NOT NULL DEFAULT 0,
    started_at          TEXT NOT NULL,
    completed_at        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_inv_project
    ON invocations(project_id);

CREATE INDEX IF NOT EXISTS idx_inv_account
    ON invocations(account_name);

CREATE INDEX IF NOT EXISTS idx_inv_run
    ON invocations(run_id);

CREATE INDEX IF NOT EXISTS idx_inv_completed_at
    ON invocations(completed_at);

CREATE TABLE IF NOT EXISTS account_events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    account_name TEXT NOT NULL,
    event_type   TEXT NOT NULL,
    project_id   TEXT,
    run_id       TEXT,
    details      TEXT,
    created_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_acct_evt_account
    ON account_events(account_name);
"""


class CostStore(AsyncSqliteBase):
    """Persistent-connection SQLite writer for cost records and account events.

    Lifecycle::

        store = CostStore(path)
        await store.open()
        try:
            await store.save_invocation(...)
        finally:
            await store.close()

    Or via async context manager::

        async with CostStore(path) as store:
            await store.save_invocation(...)
    """

    def __init__(self, db_path: Path) -> None:
        super().__init__(db_path, busy_timeout_ms=30000)

    @property
    def _schema(self) -> str:
        return _SCHEMA

    # -- internal helpers -----------------------------------------------------

    async def _execute(self, sql: str, params: tuple[Any, ...]) -> None:
        """Execute a single statement and commit."""
        conn = self._require_conn()
        await conn.execute(sql, params)
        await conn.commit()

    # -- public write API -----------------------------------------------------

    async def save_invocation(
        self,
        *,
        run_id: str,
        task_id: str | None,
        project_id: str,
        account_name: str,
        model: str,
        role: str,
        cost_usd: float,
        input_tokens: int | None,
        output_tokens: int | None,
        cache_read_tokens: int | None,
        cache_create_tokens: int | None,
        duration_ms: int,
        capped: bool,
        started_at: str,
        completed_at: str,
    ) -> None:
        """Insert one row into the invocations table."""
        await self._execute(
            'INSERT INTO invocations '
            '(run_id, task_id, project_id, account_name, model, role, '
            ' cost_usd, input_tokens, output_tokens, cache_read_tokens, '
            ' cache_create_tokens, duration_ms, capped, started_at, completed_at) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (
                run_id,
                task_id,
                project_id,
                account_name,
                model,
                role,
                cost_usd,
                input_tokens,
                output_tokens,
                cache_read_tokens,
                cache_create_tokens,
                duration_ms,
                int(capped),
                started_at,
                completed_at,
            ),
        )

    # -- public read API ------------------------------------------------------

    async def cost_totals_in_window(
        self,
        start_iso: str,
        end_iso: str,
    ) -> tuple[float, float]:
        """Return ``(total_usd, watcher_usd)`` for invocations completed in ``[start_iso, end_iso]``.

        Uses a single SELECT with conditional aggregation so the invocations
        table is scanned exactly once.  The ``%watcher%`` LIKE pattern is
        hardcoded here — this method always returns the watcher split (total,
        watcher) rather than a generic aggregation.

        The window is **inclusive** at both ends (SQLite ``BETWEEN``).  For
        trailing-24h callers, pass ``end_iso = datetime.now(UTC).isoformat()``
        — any invocations whose ``completed_at`` is written after that snapshot
        are silently excluded, which is acceptable for a fail-open cost guard.

        Returns:
            (total_usd, watcher_usd): total cost across all roles, and the
            subset matching ``role LIKE '%watcher%'``.  Both values are 0.0
            when the window contains no rows.

        Raises:
            RuntimeError: if the store has not been opened (via ``_require_conn()``).
        """
        conn = self._require_conn()
        async with conn.execute(
            'SELECT '
            '  COALESCE(SUM(cost_usd), 0.0), '
            '  COALESCE(SUM(CASE WHEN role LIKE ? THEN cost_usd END), 0.0) '
            'FROM invocations '
            'WHERE completed_at BETWEEN ? AND ?',
            ('%watcher%', start_iso, end_iso),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return (0.0, 0.0)
        return (float(row[0]), float(row[1]))

    async def model_cost_in_window(
        self,
        model: str,
        start_iso: str,
        end_iso: str,
    ) -> float:
        """Return total ``cost_usd`` for ``model`` completed in ``[start_iso, end_iso]``.

        Uses a single SELECT so the invocations table is scanned exactly once.
        The window is **inclusive** at both ends (SQLite ``BETWEEN``), matching
        :meth:`cost_totals_in_window`.

        Returns:
            Total cost for ``model`` in the window; 0.0 when no rows match.

        Raises:
            RuntimeError: if the store has not been opened (via ``_require_conn()``).
        """
        conn = self._require_conn()
        async with conn.execute(
            'SELECT COALESCE(SUM(cost_usd), 0.0) '
            'FROM invocations '
            'WHERE model = ? AND completed_at BETWEEN ? AND ?',
            (model, start_iso, end_iso),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return 0.0
        return float(row[0])

    async def save_account_event(
        self,
        *,
        account_name: str,
        event_type: str,
        project_id: str | None,
        run_id: str | None,
        details: str | None,
        created_at: str,
    ) -> None:
        """Insert one row into the account_events table."""
        await self._execute(
            'INSERT INTO account_events '
            '(account_name, event_type, project_id, run_id, details, created_at) '
            'VALUES (?, ?, ?, ?, ?, ?)',
            (account_name, event_type, project_id, run_id, details, created_at),
        )

    async def save_api_error_event(
        self,
        *,
        account_name: str,
        project_id: str | None,
        run_id: str | None,
        details: str | None,
        created_at: str,
    ) -> None:
        """Insert one ``api_error`` account event (server-side 5xx forensics).

        Typed wrapper over :meth:`save_account_event` for
        ``plans/server-side-api-error-handling-prd.md`` contract C4 (task mu):
        ``ApiHealthGate`` emits one row per ServerError report so a trip is
        reconstructable from the row stream alone.

        The literal ``'api_error'`` event_type lives HERE and nowhere else —
        it is the discriminator the gate writes, the dashboard provider-health
        strip queries, and operators grep for.  Callers pass no ``event_type``
        so a typo cannot fork the row stream into two silently-disjoint types.

        ``details`` is a JSON blob (HTTP status, task_id, role, post-report
        state and window stats); it is opaque to this layer.
        """
        await self.save_account_event(
            account_name=account_name,
            event_type='api_error',
            project_id=project_id,
            run_id=run_id,
            details=details,
            created_at=created_at,
        )
