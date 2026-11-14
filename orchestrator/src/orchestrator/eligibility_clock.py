"""The starvation watchdog's durable eligibility clock.

The one rule: each task has at most one wall-clock anchor, ``eligible_since``,
stamped the first tick the task is a dispatch candidate.  The anchor is kept
across process restarts and across transient candidacy gaps (a cooldown, a
PSI hold, one tick off the candidate list), and is dropped only when the task
dispatches, leaves ``pending``, or goes terminal.

Because candidacy already requires satisfied dependencies, the age measured
from this anchor excludes any dependency wait before the task's first
candidacy.  A wait after it is counted: a still-pending task that loses
candidacy (a dependency reopens, an external-dependency hold) keeps its
anchor, as for any other candidacy gap.  Because the anchors are
reloaded at startup, a restart re-stamps nothing, so the backlog cannot be
stamped into one cohort that ages past the watchdog's floor together.

Anchors live in their own SQLite file (``scheduler_eligibility.db``), beside
``park_eviction_requests.db`` and apart from the overrides schema that
fused-memory hand-mirrors.  A store fault never stops the scheduler: the
in-memory map stays authoritative for the process and the fault is logged.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path

from shared.sqlite_sync_base import apply_full_durability_pragmas_sync

from orchestrator.config import OrchestratorConfig

logger = logging.getLogger(__name__)

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS eligibility_anchors (
    project_root    TEXT NOT NULL,
    task_id         TEXT NOT NULL,
    eligible_since  TEXT NOT NULL,
    PRIMARY KEY (project_root, task_id)
);
"""


class EligibilityClockStore:
    """Connect-per-call SQLite persistence for eligibility anchors."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        try:
            conn.executescript(_SCHEMA)
        finally:
            conn.close()

    @classmethod
    def from_config(cls, config: OrchestratorConfig) -> EligibilityClockStore:
        return cls(config.eligibility_clock_db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        apply_full_durability_pragmas_sync(conn, busy_timeout_ms=5000)
        return conn

    def load(self, project_root: str) -> dict[str, datetime]:
        conn = self._connect()
        try:
            rows = conn.execute(
                'SELECT task_id, eligible_since FROM eligibility_anchors '
                'WHERE project_root=?',
                (project_root,),
            ).fetchall()
        finally:
            conn.close()
        return {task_id: datetime.fromisoformat(since) for task_id, since in rows}

    def stamp(self, project_root: str, anchors: Mapping[str, datetime]) -> None:
        """Persist *anchors*; a task that already has one keeps it."""
        conn = self._connect()
        try:
            with conn:
                conn.executemany(
                    'INSERT OR IGNORE INTO eligibility_anchors '
                    '(project_root, task_id, eligible_since) VALUES (?, ?, ?)',
                    [
                        (project_root, task_id, since.astimezone(UTC).isoformat())
                        for task_id, since in anchors.items()
                    ],
                )
        finally:
            conn.close()

    def forget(self, project_root: str, task_ids: Iterable[str]) -> None:
        conn = self._connect()
        try:
            with conn:
                conn.executemany(
                    'DELETE FROM eligibility_anchors '
                    'WHERE project_root=? AND task_id=?',
                    [(project_root, task_id) for task_id in task_ids],
                )
        finally:
            conn.close()


class EligibilityClock:
    """In-memory anchors for one project, loaded once and written through.

    *store* None gives a memory-only clock (bare-Scheduler unit tests).
    """

    def __init__(self, store: EligibilityClockStore | None, project_root: str) -> None:
        self._store = store
        self._project_root = project_root
        self._anchors: dict[str, datetime] = self._load()

    def _load(self) -> dict[str, datetime]:
        if self._store is None:
            return {}
        try:
            return self._store.load(self._project_root)
        except sqlite3.Error:
            logger.warning(
                'eligibility clock: cannot load anchors from %s; starting empty',
                self._store.db_path, exc_info=True,
            )
            return {}

    def observe(self, task_ids: Iterable[str], now: datetime) -> None:
        """Stamp *now* for every candidate that has no anchor yet."""
        new = {tid: now for tid in task_ids if tid not in self._anchors}
        if not new:
            return
        self._anchors.update(new)
        if self._store is None:
            return
        try:
            self._store.stamp(self._project_root, new)
        except sqlite3.Error:
            logger.warning(
                'eligibility clock: cannot persist %d anchor(s); they hold in '
                'memory for this process only', len(new), exc_info=True,
            )

    def age_secs(self, task_id: str, now: datetime) -> float:
        anchor = self._anchors.get(task_id)
        if anchor is None:
            return 0.0
        return max(0.0, (now - anchor).total_seconds())

    def forget(self, task_ids: Iterable[str]) -> None:
        gone = [tid for tid in task_ids if self._anchors.pop(tid, None) is not None]
        if not gone or self._store is None:
            return
        try:
            self._store.forget(self._project_root, gone)
        except sqlite3.Error:
            logger.warning(
                'eligibility clock: cannot forget %d anchor(s) durably; a '
                'restart will reload them', len(gone), exc_info=True,
            )

    def task_ids(self) -> frozenset[str]:
        return frozenset(self._anchors)
