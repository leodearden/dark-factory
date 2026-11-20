"""The starvation watchdog's restart-durable per-project state.

Four facts survive a process restart, in their own tables of
``scheduler_eligibility.db`` beside the eligibility anchors:

- each task's lock-refusal count (a refusal is a scan that refused the task
  its locks while a lower-ranked candidate dispatched);
- the tasks whose per-task watchdog escalation was filed (announced);
- the cap cohort: pin-gate firings the auto-pin cap turned away, each with
  the gate that fired, waiting for a watchdog pin slot;
- what the auto-pin cap L1 last announced about that cohort (the baseline).

Losing them at a restart re-accrued the same refusals and re-announced the
same cohort about a day later; plans/scheduler-dispatch-scoring-and-lock-layer-prd.md
C4 "Cap L1" records why each one is kept.  A store fault never stops the
scheduler: the in-memory state stays authoritative for the process and the
fault is logged, as for the eligibility clock.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar

from shared.sqlite_sync_base import apply_full_durability_pragmas_sync

from orchestrator.starvation_watchdog import StarvationGate

logger = logging.getLogger(__name__)

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS lock_refusals (
    project_root  TEXT NOT NULL,
    task_id       TEXT NOT NULL,
    refusals      INTEGER NOT NULL,
    PRIMARY KEY (project_root, task_id)
);
CREATE TABLE IF NOT EXISTS watchdog_announced (
    project_root  TEXT NOT NULL,
    task_id       TEXT NOT NULL,
    PRIMARY KEY (project_root, task_id)
);
CREATE TABLE IF NOT EXISTS cap_waiting (
    project_root  TEXT NOT NULL,
    task_id       TEXT NOT NULL,
    gate          TEXT NOT NULL,
    PRIMARY KEY (project_root, task_id)
);
CREATE TABLE IF NOT EXISTS cap_announcement (
    project_root  TEXT PRIMARY KEY,
    cohort_size   INTEGER NOT NULL,
    modules       TEXT NOT NULL,
    subjects      TEXT NOT NULL
);
"""

_PER_TASK_TABLES = ('lock_refusals', 'watchdog_announced', 'cap_waiting')


@dataclass(frozen=True)
class CapBaseline:
    """What the auto-pin cap L1 last announced: cohort size, modules behind non-live holders, subjects."""

    EMPTY: ClassVar[CapBaseline]

    size: int
    modules: frozenset[str]
    subjects: frozenset[str]


CapBaseline.EMPTY = CapBaseline(size=0, modules=frozenset(), subjects=frozenset())


@dataclass(frozen=True)
class StoredWatchdogState:
    refusals: Mapping[str, int]
    announced: frozenset[str]
    waiting: Mapping[str, StarvationGate]
    baseline: CapBaseline


class StarvationStateStore:
    """Connect-per-call SQLite persistence for the watchdog state, keyed by project."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        try:
            conn.executescript(_SCHEMA)
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        apply_full_durability_pragmas_sync(conn, busy_timeout_ms=5000)
        return conn

    def _write(self, sql: str, rows: list[tuple[object, ...]]) -> None:
        if not rows:
            return
        conn = self._connect()
        try:
            with conn:
                conn.executemany(sql, rows)
        finally:
            conn.close()

    def load(self, project_root: str) -> StoredWatchdogState:
        key = (project_root,)
        conn = self._connect()
        try:
            refusals = conn.execute(
                'SELECT task_id, refusals FROM lock_refusals WHERE project_root=?', key,
            ).fetchall()
            announced = conn.execute(
                'SELECT task_id FROM watchdog_announced WHERE project_root=?', key,
            ).fetchall()
            waiting = conn.execute(
                'SELECT task_id, gate FROM cap_waiting WHERE project_root=?', key,
            ).fetchall()
            baseline = conn.execute(
                'SELECT cohort_size, modules, subjects FROM cap_announcement '
                'WHERE project_root=?', key,
            ).fetchone()
        finally:
            conn.close()
        return StoredWatchdogState(
            refusals=dict(refusals),
            announced=frozenset(tid for (tid,) in announced),
            waiting=_parse_waiting(waiting),
            baseline=CapBaseline.EMPTY if baseline is None else _parse_baseline(baseline),
        )

    def increment_refusals(self, project_root: str, task_ids: Iterable[str]) -> None:
        """Add one refusal per entry of *task_ids*, in one transaction: a task listed n times gains n."""
        self._write(
            'INSERT INTO lock_refusals (project_root, task_id, refusals) VALUES (?, ?, 1) '
            'ON CONFLICT (project_root, task_id) DO UPDATE SET refusals = refusals + 1',
            [(project_root, tid) for tid in task_ids],
        )

    def announce(self, project_root: str, task_id: str) -> None:
        self._write(
            'INSERT OR IGNORE INTO watchdog_announced (project_root, task_id) VALUES (?, ?)',
            [(project_root, task_id)],
        )

    def wait(self, project_root: str, task_id: str, gate: StarvationGate) -> None:
        self._write(
            'INSERT OR REPLACE INTO cap_waiting (project_root, task_id, gate) VALUES (?, ?, ?)',
            [(project_root, task_id, gate.value)],
        )

    def drop_waiting(self, project_root: str, task_ids: Iterable[str]) -> None:
        self._write(
            'DELETE FROM cap_waiting WHERE project_root=? AND task_id=?',
            [(project_root, tid) for tid in task_ids],
        )

    def forget(self, project_root: str, task_ids: Iterable[str]) -> None:
        """Drop every per-task fact of *task_ids*, in one transaction."""
        keys = [(project_root, tid) for tid in task_ids]
        if not keys:
            return
        conn = self._connect()
        try:
            with conn:
                for table in _PER_TASK_TABLES:
                    conn.executemany(
                        f'DELETE FROM {table} WHERE project_root=? AND task_id=?', keys,
                    )
        finally:
            conn.close()

    def record_baseline(self, project_root: str, baseline: CapBaseline) -> None:
        self._write(
            'INSERT OR REPLACE INTO cap_announcement '
            '(project_root, cohort_size, modules, subjects) VALUES (?, ?, ?, ?)',
            [(
                project_root,
                baseline.size,
                json.dumps(sorted(baseline.modules)),
                json.dumps(sorted(baseline.subjects)),
            )],
        )


def _parse_waiting(rows: Iterable[tuple[str, str]]) -> dict[str, StarvationGate]:
    waiting: dict[str, StarvationGate] = {}
    for tid, gate in rows:
        try:
            waiting[tid] = StarvationGate(gate)
        except ValueError:
            logger.warning(
                'starvation state: task %s waits under unknown gate %r; dropping it '
                'from the cap cohort', tid, gate,
            )
    return waiting


def _parse_baseline(row: tuple[int, str, str]) -> CapBaseline:
    size, modules, subjects = row
    try:
        return CapBaseline(
            size=size,
            modules=frozenset(json.loads(modules)),
            subjects=frozenset(json.loads(subjects)),
        )
    except (ValueError, TypeError):
        logger.warning(
            'starvation state: unreadable cap announcement %r; starting from an empty '
            'baseline', row, exc_info=True,
        )
        return CapBaseline.EMPTY


def _best_effort(what: str, write: Callable[[], None]) -> None:
    """Run one store write; a fault is logged and the in-memory change stands."""
    try:
        write()
    except sqlite3.Error:
        logger.warning(
            'starvation state: cannot persist %s; it holds in memory for this '
            'process only', what, exc_info=True,
        )


class WatchdogState:
    """One project's watchdog state, loaded once and written through.

    *store* None gives memory-only state (bare-Scheduler unit tests).
    """

    def __init__(self, store: StarvationStateStore | None, project_root: str) -> None:
        self._store = store
        self._project_root = project_root
        stored = self._load()
        self._refusals: Counter[str] = Counter(stored.refusals)
        self._announced: set[str] = set(stored.announced)
        self._waiting: dict[str, StarvationGate] = dict(stored.waiting)
        self._baseline = stored.baseline

    @classmethod
    def load_from(cls, db_path: Path | None, project_root: str) -> WatchdogState:
        """*project_root*'s state in *db_path*; memory-only when None or when the file cannot be opened."""
        if db_path is None:
            return cls(None, project_root)
        try:
            store = StarvationStateStore(db_path)
        except (sqlite3.Error, OSError):
            logger.warning(
                'starvation state: cannot open %s; it holds in memory for this process '
                'only', db_path, exc_info=True,
            )
            return cls(None, project_root)
        return cls(store, project_root)

    def _load(self) -> StoredWatchdogState:
        empty = StoredWatchdogState({}, frozenset(), {}, CapBaseline.EMPTY)
        if self._store is None:
            return empty
        try:
            return self._store.load(self._project_root)
        except sqlite3.Error:
            logger.warning(
                'starvation state: cannot load from %s; starting empty',
                self._store.db_path, exc_info=True,
            )
            return empty

    def _persist(self, what: str, write: Callable[[StarvationStateStore, str], None]) -> None:
        store = self._store
        if store is not None:
            _best_effort(what, lambda: write(store, self._project_root))

    def note_scan(self, refused: Iterable[str]) -> None:
        """Count one refusal for every task one scan refused, however often it is listed."""
        once = sorted(set(refused))
        if not once:
            return
        self._refusals.update(once)
        self._persist(f'{len(once)} refusal(s)', lambda s, r: s.increment_refusals(r, once))

    def refusals(self, task_id: str) -> int:
        return self._refusals.get(task_id, 0)

    def announced(self) -> frozenset[str]:
        return frozenset(self._announced)

    def announce(self, task_id: str) -> None:
        if task_id in self._announced:
            return
        self._announced.add(task_id)
        self._persist(f'the announcement of {task_id}', lambda s, r: s.announce(r, task_id))

    def waiting(self) -> Mapping[str, StarvationGate]:
        return MappingProxyType(dict(self._waiting))

    def wait(self, task_id: str, gate: StarvationGate) -> None:
        if self._waiting.get(task_id) is gate:
            return
        self._waiting[task_id] = gate
        self._persist(f'the cap wait of {task_id}', lambda s, r: s.wait(r, task_id, gate))

    def drop_waiting(self, task_ids: Iterable[str]) -> None:
        gone = [tid for tid in task_ids if self._waiting.pop(tid, None) is not None]
        if gone:
            self._persist(f'{len(gone)} cap wait drop(s)', lambda s, r: s.drop_waiting(r, gone))

    def forget(self, task_ids: Iterable[str]) -> None:
        """Drop every per-task fact of *task_ids*; the baseline is kept."""
        tracked = self.task_ids()
        gone = [tid for tid in task_ids if tid in tracked]
        if not gone:
            return
        for tid in gone:
            self._refusals.pop(tid, None)
            self._announced.discard(tid)
            self._waiting.pop(tid, None)
        self._persist(f'forgetting {len(gone)} task(s)', lambda s, r: s.forget(r, gone))

    def task_ids(self) -> frozenset[str]:
        return frozenset(self._refusals) | frozenset(self._announced) | frozenset(self._waiting)

    def baseline(self) -> CapBaseline:
        return self._baseline

    def record_baseline(self, baseline: CapBaseline) -> None:
        if baseline == self._baseline:
            return
        self._baseline = baseline
        self._persist('the cap baseline', lambda s, r: s.record_baseline(r, baseline))
