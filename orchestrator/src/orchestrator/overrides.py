"""Priority-override persistence for the scheduler.

Overrides (Boost / Pin / Reserve-Now) let users force a task to dispatch ahead
of scored candidates.  They are stored in a **separate** SQLite file at
``<project_root>/data/orchestrator/scheduler_overrides.db`` so that reify
cycles and ``set_task_status`` MCP writes cannot wipe them.  See the design
decision in the plan for the full rationale.

Why a separate SQLite file?
----------------------------
Task metadata (priority, files, status) lives in the Taskmaster/fused-memory
store, which is periodically rewritten by reconciliation cycles and
``set_task_status`` calls.  If we stored override rows alongside task state
they would be silently dropped every time the reconciler ran a full reify.

By keeping overrides in a dedicated file (mirroring the ``runs.db`` pattern
from ``run_store.py``) we make the "override survives task-status churn"
property *structural* rather than something the application layer has to
actively defend.  The scheduler loads both sources on each tick and merges
them at read-time.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from orchestrator.config import PRIORITY_RANK

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS overrides (
    project_root  TEXT NOT NULL,
    task_id       TEXT NOT NULL,
    boost_tier    TEXT,
    pinned        INTEGER NOT NULL DEFAULT 0,
    pin_order     INTEGER,
    reserve_now   INTEGER NOT NULL DEFAULT 0,
    ttl_until     TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    PRIMARY KEY (project_root, task_id)
);

CREATE INDEX IF NOT EXISTS idx_overrides_pinned
    ON overrides(project_root, pinned, pin_order);
"""


class PinOrderCollision(Exception):
    """Raised when two tasks would share the same ``pin_order`` value.

    Both task IDs appear in the message so callers (e.g. the 1229b MCP tools)
    can surface a useful error to the user without re-querying the store.
    """


@dataclass(frozen=True)
class OverrideRow:
    """Read-side projection of a single overrides row.

    Omits storage-only columns (``created_at`` / ``updated_at``) so consumers
    don't accidentally depend on bookkeeping internals.  Mirrors the
    ``RequeueRecord`` frozen-dataclass pattern from ``scheduler.py:254``.
    """

    boost_tier: str | None
    pinned: bool
    pin_order: int | None
    reserve_now: bool
    ttl_until: datetime | None


def _row_to_override(row: Any) -> OverrideRow:
    """Convert a sqlite3 Row to an OverrideRow."""
    boost_tier, pinned, pin_order, reserve_now, ttl_until = (
        row[0], row[1], row[2], row[3], row[4]
    )
    return OverrideRow(
        boost_tier=boost_tier,
        pinned=bool(pinned),
        pin_order=pin_order,
        reserve_now=bool(reserve_now),
        ttl_until=datetime.fromisoformat(ttl_until) if ttl_until else None,
    )


class OverrideStore:
    """Synchronous SQLite reader/writer for scheduler priority overrides.

    Mirrors the ``RunStore`` pattern from ``run_store.py``: connect-per-call,
    WAL journal mode, ``mkdir(parents=True)`` on init.  Thread-safe for the
    orchestrator's single-tick async consumer pattern.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # Debug hook: called between the MAX(pin_order) SELECT and the UPSERT
        # inside set_override.  Set to a callable in tests to inject an
        # artificial pause and make the auto-assign race condition observable.
        self._after_max_select: Callable[[], None] | None = None
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute('PRAGMA journal_mode=WAL')
            conn.executescript(_SCHEMA)
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    def set_override(
        self,
        project_root: str,
        task_id: str,
        *,
        boost_tier: str | None = None,
        pinned: bool | None = None,
        pin_order: int | None = None,
        reserve_now: bool | None = None,
        ttl_until: datetime | None = None,
    ) -> None:
        """Upsert an override row for ``(project_root, task_id)``.

        Only supplied (non-None) keyword arguments are written; unsupplied
        fields preserve existing values via ``COALESCE`` in the UPSERT.

        **Asymmetry note:** ``boost_tier=None`` and ``ttl_until=None`` both
        mean "unchanged" (not supplied), NOT "clear this field".  To null out
        ``boost_tier`` or ``ttl_until`` use :meth:`clear_override` with
        ``field='boost_tier'`` or ``field='ttl'`` respectively.  Passing
        ``pinned=False`` is an explicit write that zeroes ``pinned`` and
        ``pin_order``; by contrast ``pinned=None`` (omitted) means unchanged.
        Use :meth:`clear_override` as the canonical way to clear any field to
        avoid this asymmetry.
        """
        if boost_tier is not None and boost_tier not in PRIORITY_RANK:
            raise ValueError(
                f'boost_tier must be one of {sorted(PRIORITY_RANK)}; '
                f'got {boost_tier!r}'
            )

        # Guard: pin_order without pinned=True is an invariant violation.
        # The collision-check and auto-assignment logic only fire for pinned=1
        # rows; writing a non-zero pin_order with pinned=0 creates a dangling
        # value that the collision check will later silently ignore (it filters
        # on pinned=1), potentially producing duplicate pin_order values.
        if pin_order is not None and pinned is not True:
            raise ValueError(
                'pin_order may only be supplied together with pinned=True; '
                f'got pin_order={pin_order!r} with pinned={pinned!r}'
            )

        if ttl_until is not None and ttl_until.tzinfo is None:
            raise ValueError(
                'ttl_until must be timezone-aware; got a naive datetime'
            )

        now_iso = datetime.now(UTC).isoformat()
        # timeout=30: when BEGIN IMMEDIATE blocks (another writer holds the lock)
        # SQLite will retry for up to 30 s before raising OperationalError.  The
        # default 5 s is tight for MCP override tools that could drive concurrent
        # writes from multiple sessions.
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        try:
            # Acquire a write lock up-front (IMMEDIATE) so that the MAX(pin_order)
            # read and the subsequent UPSERT are serialized with respect to
            # concurrent set_override calls.  Without this, two concurrent
            # callers can both read MAX=N and both attempt to write N+1, causing
            # a PinOrderCollision or silent duplicate pin_order values.
            with conn:
                conn.execute('BEGIN IMMEDIATE')

                # Resolve auto-assigned pin_order when pinning without an explicit order.
                if pinned is True and pin_order is None:
                    # If this task is already pinned, preserve its existing pin_order
                    # so that re-pinning (e.g. set_override(pinned=True, boost_tier='high'))
                    # is idempotent with respect to queue position.  Without this
                    # guard the MAX query would include the task's OWN row and produce
                    # MAX+1, silently shifting an already-pinned task to the back.
                    already_pinned = conn.execute(
                        'SELECT pin_order FROM overrides '
                        'WHERE project_root=? AND task_id=? AND pinned=1',
                        (project_root, task_id),
                    ).fetchone()
                    if already_pinned is not None:
                        pin_order = already_pinned[0]  # preserve; no re-assignment
                    else:
                        row = conn.execute(
                            'SELECT COALESCE(MAX(pin_order), 0) + 1 '
                            'FROM overrides WHERE project_root=? AND pinned=1',
                            (project_root,),
                        ).fetchone()
                        pin_order = row[0]
                        # Debug hook: tests may inject a sleep here to expose
                        # the race between reading MAX and writing the UPSERT.
                        # Post-fix (BEGIN IMMEDIATE), the sleep holds the write
                        # lock so the other thread blocks at its BEGIN IMMEDIATE.
                        if self._after_max_select is not None:
                            self._after_max_select()

                # Collision check for explicit or auto-assigned pin_order.
                if pin_order is not None:
                    existing = conn.execute(
                        'SELECT task_id FROM overrides '
                        'WHERE project_root=? AND pinned=1 AND pin_order=? AND task_id != ?',
                        (project_root, pin_order, task_id),
                    ).fetchone()
                    if existing:
                        raise PinOrderCollision(
                            f'pin_order={pin_order} already in use by task '
                            f'{existing[0]}; cannot pin task {task_id}'
                        )

                # Convert Python types to SQLite-friendly values.
                pinned_int = int(pinned) if pinned is not None else None
                reserve_now_int = int(reserve_now) if reserve_now is not None else None
                ttl_iso = ttl_until.astimezone(UTC).isoformat() if ttl_until is not None else None

                conn.execute(
                    """
                    INSERT INTO overrides
                        (project_root, task_id, boost_tier, pinned, pin_order,
                         reserve_now, ttl_until, created_at, updated_at)
                    VALUES (?, ?, ?, COALESCE(?, 0), ?, COALESCE(?, 0), ?, ?, ?)
                    ON CONFLICT(project_root, task_id) DO UPDATE SET
                        boost_tier  = COALESCE(?, boost_tier),
                        pinned      = COALESCE(?, pinned),
                        pin_order   = CASE WHEN ?=0 THEN NULL
                                           ELSE COALESCE(?, pin_order) END,
                        reserve_now = COALESCE(?, reserve_now),
                        ttl_until   = COALESCE(?, ttl_until),
                        updated_at  = ?
                    """,
                    (
                        # INSERT values
                        project_root, task_id,
                        boost_tier, pinned_int, pin_order,
                        reserve_now_int, ttl_iso,
                        now_iso, now_iso,
                        # UPDATE SET values
                        boost_tier, pinned_int,
                        pinned_int, pin_order,   # CASE WHEN new_pinned=0 THEN NULL ELSE COALESCE(new_pin_order, existing_pin_order) END
                        reserve_now_int, ttl_iso,
                        now_iso,
                    ),
                )
        finally:
            conn.close()

    def clear_override(
        self,
        project_root: str,
        task_id: str,
        field: str | None = None,
    ) -> bool:
        """Clear an override row or a single field within it.

        When ``field`` is None, the entire row is deleted.  Otherwise one
        field is nulled/zeroed.  Valid field names: ``'boost_tier'``,
        ``'pinned'``, ``'reserve_now'``, ``'ttl'``.  Clearing ``'pinned'``
        also sets ``pin_order=NULL``.

        Returns True iff a row existed (and was modified).
        """
        _VALID_FIELDS = {'boost_tier', 'pinned', 'reserve_now', 'ttl'}
        now_iso = datetime.now(UTC).isoformat()
        conn = sqlite3.connect(str(self.db_path))
        try:
            if field is None:
                cur = conn.execute(
                    'DELETE FROM overrides WHERE project_root=? AND task_id=?',
                    (project_root, task_id),
                )
                conn.commit()
                return cur.rowcount > 0

            if field not in _VALID_FIELDS:
                raise ValueError(
                    f'field must be one of {sorted(_VALID_FIELDS)}; got {field!r}'
                )

            if field == 'boost_tier':
                sql = (
                    'UPDATE overrides SET boost_tier=NULL, updated_at=? '
                    'WHERE project_root=? AND task_id=?'
                )
            elif field == 'pinned':
                sql = (
                    'UPDATE overrides SET pinned=0, pin_order=NULL, updated_at=? '
                    'WHERE project_root=? AND task_id=?'
                )
            elif field == 'reserve_now':
                sql = (
                    'UPDATE overrides SET reserve_now=0, updated_at=? '
                    'WHERE project_root=? AND task_id=?'
                )
            else:  # 'ttl'
                sql = (
                    'UPDATE overrides SET ttl_until=NULL, updated_at=? '
                    'WHERE project_root=? AND task_id=?'
                )

            cur = conn.execute(sql, (now_iso, project_root, task_id))
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    def reorder_pin_queue(
        self,
        project_root: str,
        ordered_task_ids: list[str] | str,
    ) -> None:
        """Rewrite ``pin_order`` values to match the supplied ordering.

        ``ordered_task_ids`` may be a list or a comma-separated string.  The
        supplied set must exactly match the current pinned-task set; passing
        extra or missing IDs raises ``ValueError``.
        """
        if isinstance(ordered_task_ids, str):
            ordered_task_ids = [t.strip() for t in ordered_task_ids.split(',')]

        if len(ordered_task_ids) != len(set(ordered_task_ids)):
            raise ValueError(
                f'reorder_pin_queue: duplicate task ids in input: {ordered_task_ids!r}'
            )

        current_pairs = self.get_pin_queue(project_root)
        current_ids = {tid for tid, _ in current_pairs}
        supplied_ids = set(ordered_task_ids)

        if supplied_ids != current_ids:
            raise ValueError(
                f'reorder_pin_queue: ids do not match current pin queue. '
                f'supplied={sorted(supplied_ids)!r}, '
                f'expected={sorted(current_ids)!r}'
            )

        now_iso = datetime.now(UTC).isoformat()
        conn = sqlite3.connect(str(self.db_path))
        try:
            # Wrap in a single transaction so a mid-loop failure (interrupt,
            # OS error, SQLite lock contention) rolls back all updates atomically.
            # Without this, partial writes leave the pin queue in an inconsistent
            # state with gaps or duplicate pin_order values until reorder is retried.
            with conn:
                for idx, tid in enumerate(ordered_task_ids, start=1):
                    conn.execute(
                        'UPDATE overrides SET pin_order=?, updated_at=? '
                        'WHERE project_root=? AND task_id=?',
                        (idx, now_iso, project_root, tid),
                    )
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # GC sweeps
    # ------------------------------------------------------------------

    def clear_terminal(
        self,
        project_root: str,
        terminal_task_ids: set[str],
    ) -> list[str]:
        """Delete override rows for terminal (done/cancelled/etc.) tasks.

        Returns the sorted list of task IDs that were deleted.
        """
        if not terminal_task_ids:
            return []

        placeholders = ','.join('?' * len(terminal_task_ids))
        conn = sqlite3.connect(str(self.db_path))
        try:
            rows = conn.execute(
                f'DELETE FROM overrides '
                f'WHERE project_root=? AND task_id IN ({placeholders}) '
                f'RETURNING task_id',
                (project_root, *sorted(terminal_task_ids)),
            ).fetchall()
            conn.commit()
            return sorted(r[0] for r in rows)
        finally:
            conn.close()

    def clear_expired(
        self,
        project_root: str,
        now: datetime | None = None,
    ) -> list[str]:
        """Delete override rows whose TTL has expired.

        ``now`` defaults to the current UTC time.  Returns the sorted list of
        deleted task IDs.
        """
        if now is not None and now.tzinfo is None:
            raise ValueError(
                'now must be timezone-aware; got a naive datetime'
            )

        if now is None:
            now = datetime.now(UTC)

        conn = sqlite3.connect(str(self.db_path))
        try:
            rows = conn.execute(
                'DELETE FROM overrides '
                'WHERE project_root=? AND ttl_until IS NOT NULL AND ttl_until < ? '
                'RETURNING task_id',
                # Normalise to UTC so the ISO-string comparison is invariant
                # under caller-supplied tzinfo.  Stored TTLs are also UTC-
                # normalised (set_override guarantees this since step-6), so
                # the lexicographic comparison produces correct results.
                (project_root, now.astimezone(UTC).isoformat()),
            ).fetchall()
            conn.commit()
            return sorted(r[0] for r in rows)
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def get_overrides(self, project_root: str) -> dict[str, OverrideRow]:
        """Return all override rows for *project_root*, keyed by task_id."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            rows = conn.execute(
                'SELECT task_id, boost_tier, pinned, pin_order, reserve_now, ttl_until '
                'FROM overrides WHERE project_root=?',
                (project_root,),
            ).fetchall()
            return {
                row[0]: _row_to_override(row[1:])
                for row in rows
            }
        finally:
            conn.close()

    def get_pin_queue(
        self,
        project_root: str,
    ) -> list[tuple[str, OverrideRow]]:
        """Return pinned tasks in ascending ``pin_order``.

        Returns a list of ``(task_id, OverrideRow)`` tuples filtered to
        ``pinned=1`` and ordered by ``pin_order ASC``.
        """
        conn = sqlite3.connect(str(self.db_path))
        try:
            rows = conn.execute(
                'SELECT task_id, boost_tier, pinned, pin_order, reserve_now, ttl_until '
                'FROM overrides '
                'WHERE project_root=? AND pinned=1 '
                'ORDER BY pin_order ASC',
                (project_root,),
            ).fetchall()
            return [(row[0], _row_to_override(row[1:])) for row in rows]
        finally:
            conn.close()
