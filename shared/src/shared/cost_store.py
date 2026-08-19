"""Async aiosqlite-backed store for per-invocation cost records and account events.

Uses a persistent connection opened via open()/close() or the async context manager::

    async with CostStore(path) as store:
        await store.save_invocation(...)
        await store.save_account_event(...)
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from shared.async_sqlite_base import AsyncSqliteBase

__all__ = ['CostStore']

logger = logging.getLogger(__name__)

# The `account_events.event_type` discriminator for server-side 5xx forensics
# (plans/server-side-api-error-handling-prd.md C4).  Single-sourced here so a
# writer, a reader and an operator's grep all agree on one spelling; importable
# by out-of-package readers (the dashboard strip) that build their own SQL.
# Deliberately NOT in __all__: shared/tests/test_public_api.py pins
# cost_store.__all__ to exactly {'CostStore'}, and this is a module-level
# constant, not part of the shared package's re-exported surface.
API_ERROR_EVENT_TYPE = 'api_error'

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

CREATE INDEX IF NOT EXISTS idx_acct_evt_type_created
    ON account_events(event_type, created_at);

CREATE INDEX IF NOT EXISTS idx_acct_evt_created
    ON account_events(created_at);
"""
# Two account_events indexes on purpose.  The composite one serves the
# type-scoped window read (`event_type='api_error'`), but SQLite's
# leading-column rule makes it unusable when event_type is omitted — and the
# type-less branch of account_events_in_window is the shared reader for the
# pre-existing cap_hit/switch/resume rows.  Without a bare created_at index
# that branch degrades to a full scan of an append-only table that grows for
# the life of the deployment.


def _inclusive_end_bound(end_iso: str) -> str:
    """Widen a whole-second upper bound so it covers that entire second.

    ``created_at`` is compared as a STRING — SQLite ``BETWEEN`` over ISO-8601
    text — and ``datetime.isoformat()`` emits microseconds only when they are
    non-zero.  So a row stored at ``2026-07-31T23:59:59.500000+00:00`` sorts
    AFTER a bound of ``2026-07-31T23:59:59+00:00``: past the shared prefix,
    ``'.'`` (0x2E) beats ``'+'`` (0x2B).  A caller who floors their upper
    bound to the second therefore loses every sub-second row inside it, and on
    the forensics read path a dropped row is exactly the 5xx an operator went
    looking for.

    Widening a fraction-less bound to ``.999999`` makes "inclusive at both
    ends" true at the granularity the bound is actually spelled at.  A bound
    that already carries microseconds is returned untouched — it is exact and
    the caller meant it.

    The widen is applied only when ``end_iso`` round-trips through
    ``datetime.isoformat()`` unchanged, i.e. it is a canonical, fraction-less
    timestamp.  A date-only (``2026-07-31``) or ``Z``-suffixed bound is left
    alone rather than silently reinterpreted: those spellings do not compare
    correctly against stored ``isoformat()`` values in the first place, and
    inventing a different window would hide that from the caller.

    An unparseable bound is passed through verbatim AND logged at WARNING: a
    read path must not refuse the query over a bound it cannot normalise, but
    the caller's malformed input must not vanish silently either.
    """
    try:
        parsed = datetime.fromisoformat(end_iso)
    except ValueError:
        logger.warning(
            'account_events_in_window: upper bound %r is not ISO-8601 — using it verbatim; '
            'sub-second rows in the final second may be excluded',
            end_iso,
        )
        return end_iso
    if parsed.microsecond or parsed.isoformat() != end_iso:
        return end_iso
    return parsed.replace(microsecond=999999).isoformat()


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

        Sub-second caveat: ``BETWEEN`` compares ISO-8601 STRINGS, so a
        fraction-less ``end_iso`` (``...T23:59:59+00:00``) excludes rows
        stamped inside that second (``...T23:59:59.5+00:00``) — ``'.'`` sorts
        after ``'+'``.  Bounds taken from ``datetime.now(UTC).isoformat()``
        carry microseconds and are unaffected; floor-to-the-second bounds are
        not.  This method deliberately does NOT normalise (unlike
        :meth:`account_events_in_window`): it feeds the cost guard, whose
        behaviour under load is not worth perturbing for a sub-second edge.

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
        :meth:`cost_totals_in_window` — including its sub-second caveat: a
        fraction-less ``end_iso`` excludes rows stamped inside that second.

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

    async def account_events_in_window(
        self,
        start_iso: str,
        end_iso: str,
        *,
        event_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return account_events created in ``[start_iso, end_iso]``, oldest first.

        The CostStore read path for ``api_error`` forensics
        (``plans/server-side-api-error-handling-prd.md`` C4): an outage is
        reconstructed from this row stream, so ordering is by ``created_at``
        ascending rather than insertion order.

        Note on consumers, so nobody reads a claim this layer cannot make:
        today's only callers are this package's tests and any future
        CostStore-holding reader.  The dashboard reads ``account_events``
        through its own aiosqlite connections and raw SQL
        (``dashboard/data/costs.py``, ``dashboard/data/cap_history.py``) and
        has NOT been repointed here — consolidating those onto this method is
        out of this task's lock scope and filed as follow-up work.

        The window is **inclusive** at both ends (SQLite ``BETWEEN``), matching
        :meth:`cost_totals_in_window` and :meth:`model_cost_in_window`.  Unlike
        those two, the upper bound is normalised by :func:`_inclusive_end_bound`
        first, so a whole-second ``end_iso`` also covers the sub-second rows
        inside that second — see that function for why string comparison
        otherwise drops them.

        Args:
            start_iso: Inclusive lower bound on ``created_at``.  NOT normalised:
                a fraction-less lower bound already sorts before every
                sub-second row in its second, so it is inclusive as spelled.
            end_iso: Inclusive upper bound on ``created_at``, widened to the end
                of its second when it carries no fractional part.
            event_type: When given, restrict to that one type (e.g.
                ``'api_error'``).  When None, every type in the window is
                returned — the pre-existing ``cap_hit``/``switch``/``resume``
                rows share this reader rather than needing a second method.

        Returns:
            One dict per row, keyed by column name (``account_name``,
            ``event_type``, ``project_id``, ``run_id``, ``details``,
            ``created_at``); ``[]`` when the window contains no rows.
            ``details`` is returned as the raw stored string — decoding the
            JSON is the caller's business.

        Raises:
            RuntimeError: if the store has not been opened (via ``_require_conn()``).
                A closed store must not read as a quiet empty window: that would
                render a healthy dashboard strip during an outage.
        """
        conn = self._require_conn()
        sql = (
            'SELECT account_name, event_type, project_id, run_id, details, created_at '
            'FROM account_events '
            'WHERE created_at BETWEEN ? AND ?'
        )
        params: tuple[Any, ...] = (start_iso, _inclusive_end_bound(end_iso))
        if event_type is not None:
            sql += ' AND event_type = ?'
            params += (event_type,)
        sql += ' ORDER BY created_at ASC'
        async with conn.execute(sql, params) as cur:
            rows = await cur.fetchall()
        return [
            {
                'account_name': row[0],
                'event_type': row[1],
                'project_id': row[2],
                'run_id': row[3],
                'details': row[4],
                'created_at': row[5],
            }
            for row in rows
        ]

    # -- public write API: account events --------------------------------------

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

        The event_type is the module-level :data:`API_ERROR_EVENT_TYPE`
        constant — the single source for the discriminator the gate writes,
        the dashboard provider-health strip queries, and operators grep for.
        The constant (not this wrapper) is what out-of-package readers building
        their own SQL should import; the wrapper additionally spares gate-side
        callers from passing ``event_type`` at all, so a typo cannot fork the
        row stream into two silently-disjoint types.

        The sibling types (``cap_hit``/``switch``/``resume``) are still written
        as inline literals by ``UsageGate._write_cost_event``; giving them the
        same treatment means editing ``orchestrator/`` and is out of this
        task's lock scope.

        ``details`` is a JSON blob (HTTP status, task_id, role, post-report
        state and window stats); it is opaque to this layer.
        """
        await self.save_account_event(
            account_name=account_name,
            event_type=API_ERROR_EVENT_TYPE,
            project_id=project_id,
            run_id=run_id,
            details=details,
            created_at=created_at,
        )
