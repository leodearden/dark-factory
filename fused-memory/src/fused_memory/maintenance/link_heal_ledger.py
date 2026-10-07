"""The link-heal ledger and run lock (plans/write-triage-link-healing-prd.md H1, H2).

``link_heal.db`` is the one home of heal history and of the link adjudicator's
verdicts: an adjudication is stored nowhere else (D8). Every table keys on an
AUTOINCREMENT id, so a ``sqlite_sequence`` row proves something has written
here (INV-13). Detail and counts are stored as JSON.

:class:`RunLock` keeps runs one at a time: an exclusive ``flock`` plus a JSON
record of its holder. The kernel drops a crashed holder's ``flock``, which is
what reclaims its lock; the record it left is reported as ``reclaimed_from``.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
import sqlite3
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from fused_memory.maintenance.link_heal import (
    BasisSource,
    HealAction,
    LinkImage,
    PlannedAction,
    Verdict,
)

LEDGER_FILENAME = 'link_heal.db'
LOCK_FILENAME = 'link_heal.lock'

UNDO_ACTION = 'undo'


class RunSource(StrEnum):
    CORPUS = 'corpus'
    ADJUDICATOR = 'adjudicator'
    UNDO = 'undo'


class ActionState(StrEnum):
    PLANNED = 'planned'
    APPLIED = 'applied'
    SKIPPED_STALE = 'skipped_stale'
    SKIPPED_CAP = 'skipped_cap'
    FAILED = 'failed'
    UNDONE = 'undone'


PENDING_STATES = frozenset({ActionState.PLANNED, ActionState.SKIPPED_CAP})
_PENDING_VALUES = tuple(sorted(state.value for state in PENDING_STATES))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL UNIQUE,
    source TEXT NOT NULL,
    writes INTEGER NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    counts_json TEXT,
    plan_sha256 TEXT
);
CREATE TABLE IF NOT EXISTS adjudications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL,
    child_id TEXT NOT NULL,
    parent_id TEXT NOT NULL,
    child_sha256 TEXT NOT NULL,
    parent_sha256 TEXT NOT NULL,
    verdict TEXT NOT NULL,
    reason TEXT,
    model TEXT,
    run_id TEXT NOT NULL,
    at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS run_adjudications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    adjudication_id INTEGER NOT NULL REFERENCES adjudications (id)
);
CREATE UNIQUE INDEX IF NOT EXISTS run_adjudications_by_run
    ON run_adjudications (run_id, adjudication_id);
CREATE TABLE IF NOT EXISTS actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    child_id TEXT NOT NULL,
    action TEXT NOT NULL,
    pre_image_json TEXT NOT NULL,
    post_image_json TEXT NOT NULL,
    basis_source TEXT NOT NULL,
    basis_key TEXT,
    child_sha256 TEXT NOT NULL,
    parent_sha256 TEXT,
    state TEXT NOT NULL,
    detail TEXT,
    at TEXT NOT NULL,
    executed_run_id TEXT,
    undone_run_id TEXT
);
CREATE INDEX IF NOT EXISTS actions_by_link ON actions (project_id, child_id);
"""

_ACTION_COLUMNS = (
    'actions.id, actions.run_id, project_id, child_id, action, pre_image_json, '
    'post_image_json, basis_source, basis_key, child_sha256, parent_sha256, state, '
    'detail, executed_run_id, undone_run_id'
)
_HEALS_ONLY = f"basis_source != '{BasisSource.UNDO}'"
_RUN_PREFIX = re.compile(r'[0-9a-f]{8,32}')


def new_run_id() -> str:
    return uuid.uuid4().hex


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _image_json(image: LinkImage) -> str:
    return json.dumps(image.as_dict(), sort_keys=True)


def _image(text: str) -> LinkImage:
    return LinkImage.from_metadata(json.loads(text))


def _json_or_none(value: Mapping[str, Any] | None) -> str | None:
    return None if value is None else json.dumps(value, sort_keys=True)


def _loads_or_none(text: str | None) -> Any:
    return None if text is None else json.loads(text)


class UnknownRun(LookupError):
    """No run has that id or prefix."""


class AmbiguousRun(LookupError):
    """More than one run shares that prefix."""


@dataclass(frozen=True)
class RunRow:
    run_id: str
    source: RunSource
    writes: bool
    started_at: str
    finished_at: str | None
    counts: Mapping[str, Any] | None
    plan_sha256: str | None


@dataclass(frozen=True)
class ActionRow:
    """A planned heal and what became of it."""

    action_id: int
    run_id: str
    planned: PlannedAction
    state: ActionState
    detail: Mapping[str, Any] | None
    executed_run_id: str | None
    undone_run_id: str | None


@dataclass(frozen=True)
class AdjudicationRecord:
    """The adjudicator's verdict on one (child, parent) pair, at the text hashes it judged."""

    project_id: str
    child_id: str
    parent_id: str
    child_sha256: str
    parent_sha256: str
    verdict: Verdict
    reason: str
    model: str

    def __post_init__(self) -> None:
        if not isinstance(self.verdict, Verdict):
            raise TypeError(f'adjudication verdict {self.verdict!r} is not a Verdict')


@dataclass(frozen=True)
class AdjudicationRow:
    adjudication_id: int
    run_id: str
    at: str
    record: AdjudicationRecord


@dataclass(frozen=True)
class UndoStepRow:
    """One write an undo run made (or skipped) to take a heal back."""

    action_id: int
    undo_run_id: str
    original_action_id: int
    project_id: str
    child_id: str
    before: LinkImage
    after: LinkImage
    basis_source: BasisSource
    basis_key: str
    state: ActionState
    detail: Mapping[str, Any] | None


class LinkHealLedger:
    """``link_heal.db``: runs, adjudications and actions. One connection; a commit per operation."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def start_run(self, source: RunSource, *, writes: bool) -> str:
        run_id = new_run_id()
        with self._conn:
            self._conn.execute(
                'INSERT INTO runs (run_id, source, writes, started_at) VALUES (?, ?, ?, ?)',
                (run_id, source.value, int(writes), _now()),
            )
        return run_id

    def finish_run(
        self, run_id: str, *, counts: Mapping[str, Any], plan_sha256: str | None = None,
    ) -> None:
        with self._conn:
            self._conn.execute(
                'UPDATE runs SET finished_at = ?, counts_json = ?, plan_sha256 = ? '
                'WHERE run_id = ?',
                (_now(), json.dumps(counts, sort_keys=True), plan_sha256, run_id),
            )

    def writing_run_count(self) -> int:
        (count,) = self._conn.execute('SELECT COUNT(*) FROM runs WHERE writes = 1').fetchone()
        return count

    def recent_runs(self, limit: int) -> list[RunRow]:
        rows = self._conn.execute('SELECT * FROM runs ORDER BY id DESC LIMIT ?', (limit,))
        return [_run_row(row) for row in rows]

    def resolve_run(self, run_ref: str) -> RunRow:
        """The run with id *run_ref*, or the one run whose id starts with it."""
        if not _RUN_PREFIX.fullmatch(run_ref):
            raise UnknownRun(f'no link-heal run {run_ref!r}: expected 8-32 hex characters')
        rows = self._conn.execute(
            'SELECT * FROM runs WHERE run_id LIKE ? ORDER BY id', (f'{run_ref}%',),
        ).fetchall()
        if not rows:
            raise UnknownRun(f'no link-heal run {run_ref!r} in {self.db_path}')
        if len(rows) > 1:
            raise AmbiguousRun(
                f'{run_ref!r} matches {len(rows)} runs: '
                + ', '.join(row['run_id'] for row in rows),
            )
        return _run_row(rows[0])

    def add_planned(self, run_id: str, actions: Sequence[PlannedAction]) -> list[int]:
        with self._conn:
            return self._insert_all_planned(run_id, actions)

    @contextlib.contextmanager
    def publishing_planned(
        self, run_id: str, actions: Sequence[PlannedAction], source: RunSource,
    ) -> Iterator[list[ActionRow]]:
        """Add *actions* as *run_id*'s planned rows and yield *source*'s pending rows, theirs
        included. The rows commit when the body returns and are rolled back when it raises."""
        with self._conn:
            self._insert_all_planned(run_id, actions)
            yield self.pending_actions(source)

    def _insert_all_planned(self, run_id: str, actions: Sequence[PlannedAction]) -> list[int]:
        at = _now()
        return [self._insert_planned(run_id, planned, at) for planned in actions]

    def _insert_planned(self, run_id: str, planned: PlannedAction, at: str) -> int:
        cursor = self._conn.execute(
            'INSERT INTO actions (run_id, project_id, child_id, action, pre_image_json, '
            'post_image_json, basis_source, basis_key, child_sha256, parent_sha256, state, at) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (
                run_id, planned.project_id, planned.child_id, planned.action.value,
                _image_json(planned.pre_image), _image_json(planned.post_image),
                planned.basis_source.value, planned.basis_key, planned.child_sha256,
                planned.parent_sha256, ActionState.PLANNED.value, at,
            ),
        )
        return _lastrowid(cursor)

    def pending_actions(self, source: RunSource) -> list[ActionRow]:
        """Planned or cap-skipped heals planned by runs of *source*, oldest first."""
        return self._action_rows(
            f'SELECT {_ACTION_COLUMNS} FROM actions JOIN runs ON runs.run_id = actions.run_id '
            f'WHERE runs.source = ? AND state IN (?, ?) AND {_HEALS_ONLY} ORDER BY actions.id',
            (source.value, *_PENDING_VALUES),
        )

    def run_actions(self, run_id: str) -> list[ActionRow]:
        """The heals *run_id* planned, oldest first."""
        return self._action_rows(
            f'SELECT {_ACTION_COLUMNS} FROM actions WHERE run_id = ? AND {_HEALS_ONLY} '
            'ORDER BY id',
            (run_id,),
        )

    def applied_actions(self, executed_run_id: str) -> list[ActionRow]:
        return self._action_rows(
            f'SELECT {_ACTION_COLUMNS} FROM actions WHERE executed_run_id = ? AND state = ? '
            f'AND {_HEALS_ONLY} ORDER BY id',
            (executed_run_id, ActionState.APPLIED.value),
        )

    def set_outcome(
        self,
        action_id: int,
        state: ActionState,
        executed_run_id: str,
        detail: Mapping[str, Any] | None,
    ) -> None:
        with self._conn:
            self._conn.execute(
                'UPDATE actions SET state = ?, executed_run_id = ?, detail = ?, at = ? '
                'WHERE id = ?',
                (state.value, executed_run_id, _json_or_none(detail), _now(), action_id),
            )

    def mark_undone(self, action_id: int, undo_run_id: str) -> None:
        with self._conn:
            self._conn.execute(
                'UPDATE actions SET state = ?, undone_run_id = ?, at = ? WHERE id = ?',
                (ActionState.UNDONE.value, undo_run_id, _now(), action_id),
            )

    def find_pending(self, planned: PlannedAction) -> ActionRow | None:
        """A pending row planning exactly *planned* against the same texts."""
        rows = self._same_heal(planned, states=_PENDING_VALUES)
        matches = [row for row in rows if row.planned.pre_image == planned.pre_image]
        return matches[0] if matches else None

    def is_undo_suppressed(self, planned: PlannedAction) -> bool:
        """An undo took this heal back for this pair at these text hashes."""
        rows = self._same_heal(planned, states=[ActionState.UNDONE.value])
        return any(
            row.planned.pre_image.parent_id == planned.pre_image.parent_id for row in rows
        )

    def _same_heal(self, planned: PlannedAction, *, states: Sequence[str]) -> list[ActionRow]:
        placeholders = ', '.join('?' for _ in states)
        return self._action_rows(
            f'SELECT {_ACTION_COLUMNS} FROM actions WHERE project_id = ? AND child_id = ? '
            f'AND action = ? AND child_sha256 = ? AND parent_sha256 IS ? '
            f'AND state IN ({placeholders}) AND {_HEALS_ONLY} ORDER BY id',
            (
                planned.project_id, planned.child_id, planned.action.value,
                planned.child_sha256, planned.parent_sha256, *states,
            ),
        )

    def add_undo_step(
        self,
        undo_run_id: str,
        original: ActionRow,
        *,
        before: LinkImage,
        after: LinkImage,
        state: ActionState,
        detail: Mapping[str, Any] | None,
    ) -> int:
        heal = original.planned
        with self._conn:
            cursor = self._conn.execute(
                'INSERT INTO actions (run_id, project_id, child_id, action, pre_image_json, '
                'post_image_json, basis_source, basis_key, child_sha256, parent_sha256, '
                'state, detail, at, executed_run_id) '
                'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (
                    undo_run_id, heal.project_id, heal.child_id, UNDO_ACTION,
                    _image_json(before), _image_json(after), BasisSource.UNDO.value,
                    str(original.action_id), heal.child_sha256, heal.parent_sha256,
                    state.value, _json_or_none(detail), _now(), undo_run_id,
                ),
            )
        return _lastrowid(cursor)

    def undo_steps(self, undo_run_id: str) -> list[UndoStepRow]:
        rows = self._conn.execute(
            'SELECT * FROM actions WHERE run_id = ? AND basis_source = ? ORDER BY id',
            (undo_run_id, BasisSource.UNDO.value),
        )
        return [_undo_step_row(row) for row in rows]

    def last_applied_undo_step(self, action_id: int) -> UndoStepRow | None:
        """The newest undo step, of any undo run, that applied to heal *action_id*."""
        row = self._conn.execute(
            'SELECT * FROM actions WHERE basis_source = ? AND basis_key = ? AND state = ? '
            'ORDER BY id DESC LIMIT 1',
            (BasisSource.UNDO.value, str(action_id), ActionState.APPLIED.value),
        ).fetchone()
        return None if row is None else _undo_step_row(row)

    def add_adjudications(
        self, run_id: str, records: Sequence[AdjudicationRecord],
    ) -> list[int]:
        """Ledger *records* as made by *run_id*, which rests on each; their ids, in order."""
        at = _now()
        with self._conn:
            ids = [
                _lastrowid(self._conn.execute(
                    'INSERT INTO adjudications (project_id, child_id, parent_id, child_sha256, '
                    'parent_sha256, verdict, reason, model, run_id, at) '
                    'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                    (
                        record.project_id, record.child_id, record.parent_id,
                        record.child_sha256, record.parent_sha256, record.verdict.value,
                        record.reason, record.model, run_id, at,
                    ),
                ))
                for record in records
            ]
            self._rest_on(run_id, ids)
        return ids

    def reuse_adjudications(self, run_id: str, adjudication_ids: Iterable[int]) -> None:
        """Record that *run_id* rests on adjudications an earlier run made."""
        with self._conn:
            self._rest_on(run_id, adjudication_ids)

    def _rest_on(self, run_id: str, adjudication_ids: Iterable[int]) -> None:
        self._conn.executemany(
            'INSERT OR IGNORE INTO run_adjudications (run_id, adjudication_id) VALUES (?, ?)',
            [(run_id, adjudication_id) for adjudication_id in adjudication_ids],
        )

    def adjudication_at(
        self,
        project_id: str,
        child_id: str,
        parent_id: str,
        child_sha256: str,
        parent_sha256: str,
    ) -> AdjudicationRow | None:
        """The newest adjudication of this pair at exactly these text hashes."""
        row = self._conn.execute(
            'SELECT * FROM adjudications WHERE project_id = ? AND child_id = ? AND parent_id = ? '
            'AND child_sha256 = ? AND parent_sha256 = ? ORDER BY id DESC LIMIT 1',
            (project_id, child_id, parent_id, child_sha256, parent_sha256),
        ).fetchone()
        return None if row is None else _adjudication_row(row)

    def adjudication_verdicts(self, run_ids: Iterable[str]) -> list[Verdict]:
        """The verdict of every adjudication the runs *run_ids* rest on, made or reused, once each."""
        wanted = sorted(set(run_ids))
        placeholders = ', '.join('?' for _ in wanted)
        rows = self._conn.execute(
            'SELECT verdict FROM adjudications WHERE id IN (SELECT adjudication_id FROM '
            f'run_adjudications WHERE run_id IN ({placeholders})) ORDER BY id',
            wanted,
        )
        return [Verdict(verdict) for (verdict,) in rows]

    def _action_rows(self, sql: str, params: Sequence[Any]) -> list[ActionRow]:
        return [_action_row(row) for row in self._conn.execute(sql, params)]


def _lastrowid(cursor: sqlite3.Cursor) -> int:
    rowid = cursor.lastrowid
    if rowid is None:
        raise RuntimeError('sqlite reported no row id for an insert')
    return rowid


def _run_row(row: sqlite3.Row) -> RunRow:
    return RunRow(
        run_id=row['run_id'],
        source=RunSource(row['source']),
        writes=bool(row['writes']),
        started_at=row['started_at'],
        finished_at=row['finished_at'],
        counts=_loads_or_none(row['counts_json']),
        plan_sha256=row['plan_sha256'],
    )


def _action_row(row: sqlite3.Row) -> ActionRow:
    return ActionRow(
        action_id=row['id'],
        run_id=row['run_id'],
        planned=PlannedAction(
            project_id=row['project_id'],
            child_id=row['child_id'],
            action=HealAction(row['action']),
            pre_image=_image(row['pre_image_json']),
            post_image=_image(row['post_image_json']),
            basis_source=BasisSource(row['basis_source']),
            basis_key=row['basis_key'],
            child_sha256=row['child_sha256'],
            parent_sha256=row['parent_sha256'],
        ),
        state=ActionState(row['state']),
        detail=_loads_or_none(row['detail']),
        executed_run_id=row['executed_run_id'],
        undone_run_id=row['undone_run_id'],
    )


def _adjudication_row(row: sqlite3.Row) -> AdjudicationRow:
    return AdjudicationRow(
        adjudication_id=row['id'],
        run_id=row['run_id'],
        at=row['at'],
        record=AdjudicationRecord(
            project_id=row['project_id'],
            child_id=row['child_id'],
            parent_id=row['parent_id'],
            child_sha256=row['child_sha256'],
            parent_sha256=row['parent_sha256'],
            verdict=Verdict(row['verdict']),
            reason=row['reason'],
            model=row['model'],
        ),
    )


def _undo_step_row(row: sqlite3.Row) -> UndoStepRow:
    return UndoStepRow(
        action_id=row['id'],
        undo_run_id=row['run_id'],
        original_action_id=int(row['basis_key']),
        project_id=row['project_id'],
        child_id=row['child_id'],
        before=_image(row['pre_image_json']),
        after=_image(row['post_image_json']),
        basis_source=BasisSource(row['basis_source']),
        basis_key=row['basis_key'],
        state=ActionState(row['state']),
        detail=_loads_or_none(row['detail']),
    )


@dataclass(frozen=True)
class LockHolder:
    pid: int
    started_at: str


class RunLockHeld(RuntimeError):
    """Another link-heal run holds the lock."""

    def __init__(self, lock_path: Path, holder: LockHolder | None) -> None:
        self.lock_path = lock_path
        self.holder = holder
        named = (
            f'pid {holder.pid} since {holder.started_at}' if holder else 'an unrecorded holder'
        )
        super().__init__(f'link-heal run lock {lock_path} is held by {named}')


class RunLock:
    """An exclusive, crash-reclaimed lock on one ledger directory's runs."""

    def __init__(self, lock_path: Path) -> None:
        self.lock_path = lock_path
        self.reclaimed_from: LockHolder | None = None
        self._fd: int | None = None

    def __enter__(self) -> RunLock:
        fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            holder = _read_holder(fd)
            os.close(fd)
            raise RunLockHeld(self.lock_path, holder) from None
        self.reclaimed_from = _read_holder(fd)
        _write_holder(fd, {'pid': os.getpid(), 'started_at': _now()})
        self._fd = fd
        return self

    def __exit__(self, *exc_info: object) -> None:
        fd, self._fd = self._fd, None
        if fd is None:
            return
        with contextlib.suppress(OSError):
            os.ftruncate(fd, 0)
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _read_holder(fd: int) -> LockHolder | None:
    os.lseek(fd, 0, os.SEEK_SET)
    raw = os.read(fd, 4096)
    try:
        record = json.loads(raw)
        return LockHolder(pid=int(record['pid']), started_at=str(record['started_at']))
    except (ValueError, KeyError, TypeError):
        return None


def _write_holder(fd: int, record: Mapping[str, Any]) -> None:
    os.ftruncate(fd, 0)
    os.lseek(fd, 0, os.SEEK_SET)
    os.write(fd, json.dumps(record).encode('utf-8'))
