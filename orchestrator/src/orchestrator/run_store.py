"""SQLite persistence for orchestrator run results."""

from __future__ import annotations

import logging
import sqlite3
import uuid
from dataclasses import dataclass, fields, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, get_type_hints

from shared.sqlite_column_migration import add_column_if_missing_sync
from shared.sqlite_sync_base import CheckpointResult, apply_full_durability_pragmas_sync

from orchestrator.shadow_types import (
    CostBasis,
    PlanSource,
    ReferenceKind,
    ShadowCellState,
    ShadowShape,
    ShadowStormCause,
)
from orchestrator.workflow_types import WorkflowOutcome

if TYPE_CHECKING:
    from orchestrator.harness import HarnessReport, TaskReport

logger = logging.getLogger(__name__)

# One blank-character set for every "non-empty text" rule, in the record and in the DDL.
_BLANK_CHARS = ' \t\n\r'
_STRATUM_SEPARATOR = '×'
_SHADOW_CELL_DEFAULT_OWNER = 'shadow_coordinator'
_SHADOW_CELL_TIME_COLUMNS = (
    'opened_at', 'settle_deadline', 'running_at', 'run_ended_at', 'settled_at',
)


def _is_blank(text: str | None) -> bool:
    return text is None or not text.strip(_BLANK_CHARS)


def _utc_iso(moment: datetime) -> str:
    """The one stored text form of an instant, so stored times compare as text."""
    if moment.utcoffset() is None:
        raise ValueError(f'a stored instant must be timezone-aware (got naive {moment!r})')
    return moment.astimezone(UTC).isoformat()


@dataclass(frozen=True)
class ShadowCell:
    """One ``shadow_cells`` row (plans/live-shadow-eval-prd.md C1); field names are its columns."""

    cell_id: str
    project_id: str
    task_id: str
    shape: ShadowShape
    candidate: str
    incumbent: str
    stratum: tuple[str, str, str]
    base_sha: str
    state: ShadowCellState
    cost_basis: CostBasis
    opened_at: datetime
    settle_deadline: datetime
    parent_cell_id: str | None = None
    worktree_path: Path | None = None
    plan_source: PlanSource | None = None
    reason: str | None = None
    reference_kind: ReferenceKind | None = None
    reference_sha: str | None = None
    result_path: Path | None = None
    cost_usd: float = 0.0
    cost_credits: float = 0.0
    run_id: str | None = None
    running_at: datetime | None = None
    run_ended_at: datetime | None = None
    settled_at: datetime | None = None
    owner: str = _SHADOW_CELL_DEFAULT_OWNER

    def __post_init__(self) -> None:
        self._check_identity()
        self._check_timestamps()
        self._check_costs()
        self._check_stratum()
        if self.state.requires_reason and _is_blank(self.reason):
            self._refuse(f'state {self.state.value!r} requires a non-blank reason (got {self.reason!r})')

    def _refuse(self, invariant: str) -> None:
        raise ValueError(f'ShadowCell {self.cell_id!r}: {invariant}')

    def _check_identity(self) -> None:
        for name in ('cell_id', 'project_id', 'task_id', 'candidate', 'incumbent', 'base_sha', 'owner'):
            value = getattr(self, name)
            if _is_blank(value):
                self._refuse(f'{name} must be non-blank (got {value!r})')
        if self.parent_cell_id == self.cell_id:
            self._refuse(f'parent_cell_id must differ from cell_id (got {self.parent_cell_id!r})')

    def _check_timestamps(self) -> None:
        for name in _SHADOW_CELL_TIME_COLUMNS:
            value = getattr(self, name)
            if value is not None and value.utcoffset() is None:
                self._refuse(f'{name} must be timezone-aware (got naive {value!r})')
        if self.settle_deadline <= self.opened_at:
            self._refuse(
                f'settle_deadline must lie after opened_at '
                f'(got {self.settle_deadline.isoformat()} <= {self.opened_at.isoformat()})'
            )

    def _check_costs(self) -> None:
        for name in ('cost_usd', 'cost_credits'):
            value = getattr(self, name)
            if not value >= 0:
                self._refuse(f'{name} must be >= 0 (got {value!r})')
        if self.cost_basis is CostBasis.METERED and self.cost_credits != 0:
            self._refuse(
                f'a {CostBasis.METERED.value} arm meters dollars, never credits '
                f'(got cost_credits={self.cost_credits!r})'
            )

    def _check_stratum(self) -> None:
        parts = self.stratum
        if len(parts) != 3 or any(
            _is_blank(part) or _STRATUM_SEPARATOR in part for part in parts
        ):
            self._refuse(
                f'stratum must be three non-blank parts free of {_STRATUM_SEPARATOR!r} '
                f'(got {parts!r})'
            )


_REASON_REQUIRED_STATES = ', '.join(
    f"'{state.value}'" for state in ShadowCellState if state.requires_reason
)
_SQL_BLANK_CHARS = f"char({', '.join(str(ord(c)) for c in _BLANK_CHARS)})"

_SHADOW_CELLS_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS shadow_cells (
    cell_id         TEXT PRIMARY KEY,
    parent_cell_id  TEXT,
    project_id      TEXT NOT NULL,
    task_id         TEXT NOT NULL,
    shape           TEXT NOT NULL,
    candidate       TEXT NOT NULL,
    incumbent       TEXT NOT NULL,
    stratum         TEXT NOT NULL,
    base_sha        TEXT NOT NULL,
    worktree_path   TEXT,
    plan_source     TEXT,
    state           TEXT NOT NULL,
    reason          TEXT,
    reference_kind  TEXT,
    reference_sha   TEXT,
    result_path     TEXT,
    cost_usd        REAL NOT NULL DEFAULT 0,
    cost_credits    REAL NOT NULL DEFAULT 0,
    cost_basis      TEXT NOT NULL,
    run_id          TEXT,
    opened_at       TEXT NOT NULL,
    running_at      TEXT,
    run_ended_at    TEXT,
    settled_at      TEXT,
    settle_deadline TEXT NOT NULL,
    owner           TEXT NOT NULL DEFAULT '{_SHADOW_CELL_DEFAULT_OWNER}',
    CHECK (state NOT IN ({_REASON_REQUIRED_STATES})
           OR trim(coalesce(reason, ''), {_SQL_BLANK_CHARS}) <> ''),
    CHECK (cost_basis <> '{CostBasis.METERED.value}' OR cost_credits = 0),
    CHECK (cost_usd >= 0 AND cost_credits >= 0),
    CHECK (trim(owner, {_SQL_BLANK_CHARS}) <> ''),
    CHECK (parent_cell_id IS NULL OR parent_cell_id <> cell_id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_shadow_cells_one_per_triple
    ON shadow_cells(project_id, task_id, shape, candidate);

CREATE INDEX IF NOT EXISTS idx_shadow_cells_project_state
    ON shadow_cells(project_id, state);

CREATE TABLE IF NOT EXISTS shadow_storm (
    project_id  TEXT PRIMARY KEY,
    cause       TEXT,
    detail      TEXT,
    paused_at   TEXT,
    resumed_at  TEXT,
    CHECK ((cause IS NULL) = (paused_at IS NULL))
);
"""

_SHADOW_CELL_COLUMNS = tuple(field.name for field in fields(ShadowCell))
_SHADOW_CELL_ENUM_COLUMNS: dict[str, type[StrEnum]] = {
    'shape': ShadowShape,
    'state': ShadowCellState,
    'cost_basis': CostBasis,
    'plan_source': PlanSource,
    'reference_kind': ReferenceKind,
}
_SHADOW_CELL_PATH_COLUMNS = ('worktree_path', 'result_path')
_PARENT_SHARED_COLUMNS = ('project_id', 'task_id', 'base_sha')


def _shadow_cell_to_row(cell: ShadowCell) -> dict[str, Any]:
    row: dict[str, Any] = {name: getattr(cell, name) for name in _SHADOW_CELL_COLUMNS}
    for name in _SHADOW_CELL_ENUM_COLUMNS:
        row[name] = None if row[name] is None else row[name].value
    for name in _SHADOW_CELL_TIME_COLUMNS:
        row[name] = None if row[name] is None else _utc_iso(row[name])
    for name in _SHADOW_CELL_PATH_COLUMNS:
        row[name] = None if row[name] is None else str(row[name])
    row['stratum'] = _STRATUM_SEPARATOR.join(cell.stratum)
    return row


def _shadow_cell_from_row(row: sqlite3.Row) -> ShadowCell:
    values: dict[str, Any] = {name: row[name] for name in _SHADOW_CELL_COLUMNS}
    for name, enum_cls in _SHADOW_CELL_ENUM_COLUMNS.items():
        values[name] = _decode_enum(values['cell_id'], name, enum_cls, values[name])
    for name in _SHADOW_CELL_TIME_COLUMNS:
        values[name] = None if values[name] is None else datetime.fromisoformat(values[name])
    for name in _SHADOW_CELL_PATH_COLUMNS:
        values[name] = None if values[name] is None else Path(values[name])
    values['stratum'] = tuple(values['stratum'].split(_STRATUM_SEPARATOR))
    return ShadowCell(**values)


def _decode_enum(
    cell_id: str, column: str, enum_cls: type[StrEnum], raw: str | None,
) -> StrEnum | None:
    if raw is None:
        return None
    try:
        return enum_cls(raw)
    except ValueError:
        raise ValueError(
            f'shadow_cells row {cell_id!r}: unknown {column} value {raw!r}'
        ) from None


_INSERT_SHADOW_CELL_SQL = (
    f'INSERT INTO shadow_cells ({", ".join(_SHADOW_CELL_COLUMNS)}) '
    f'VALUES ({", ".join(f":{name}" for name in _SHADOW_CELL_COLUMNS)})'
)


def _require_valid_parent(conn: sqlite3.Connection, cell: ShadowCell) -> None:
    parent = conn.execute(
        f'SELECT {", ".join(_PARENT_SHARED_COLUMNS)} FROM shadow_cells WHERE cell_id = ?',
        (cell.parent_cell_id,),
    ).fetchone()
    if parent is None:
        raise ValueError(
            f'ShadowCell {cell.cell_id!r}: parent_cell_id {cell.parent_cell_id!r} '
            f'names no stored cell'
        )
    for column in _PARENT_SHARED_COLUMNS:
        if parent[column] != getattr(cell, column):
            raise ValueError(
                f'ShadowCell {cell.cell_id!r}: parent {cell.parent_cell_id!r} has '
                f'{column}={parent[column]!r} but the child has '
                f'{column}={getattr(cell, column)!r}; a child cell shares its '
                f"parent's project, task and base"
            )


@dataclass(frozen=True)
class ShadowCellQuery:
    """A ``find_shadow_cells`` filter: every unset field matches every cell of the project."""

    project_id: str
    states: frozenset[ShadowCellState] | None = None
    task_id: str | None = None
    shape: ShadowShape | None = None
    candidate: str | None = None
    run_id: str | None = None
    opened_since: datetime | None = None
    run_ended_since: datetime | None = None
    deadline_before: datetime | None = None
    newest_first: bool = False
    limit: int | None = None

    def __post_init__(self) -> None:
        if _is_blank(self.project_id):
            raise ValueError(f'ShadowCellQuery: project_id must be non-blank (got {self.project_id!r})')
        if self.states is not None and not self.states:
            raise ValueError('ShadowCellQuery: states must be None (any state) or non-empty')
        if self.limit is not None and self.limit < 1:
            raise ValueError(f'ShadowCellQuery: limit must be None or >= 1 (got {self.limit!r})')


# (query field, SQL condition) — each *_since is inclusive, deadline_before strict.
_SHADOW_CELL_QUERY_FILTERS = (
    ('task_id', 'task_id = ?'),
    ('shape', 'shape = ?'),
    ('candidate', 'candidate = ?'),
    ('run_id', 'run_id = ?'),
    ('opened_since', 'opened_at >= ?'),
    ('run_ended_since', 'run_ended_at >= ?'),
    ('deadline_before', 'settle_deadline < ?'),
)


def _query_param(value: str | datetime) -> str:
    return _utc_iso(value) if isinstance(value, datetime) else str(value)


def _find_shadow_cells_sql(query: ShadowCellQuery) -> tuple[str, list[Any]]:
    conditions = ['project_id = ?']
    params: list[Any] = [query.project_id]
    if query.states is not None:
        values = sorted(state.value for state in query.states)
        conditions.append(f'state IN ({", ".join("?" for _ in values)})')
        params.extend(values)
    for field_name, condition in _SHADOW_CELL_QUERY_FILTERS:
        value = getattr(query, field_name)
        if value is not None:
            conditions.append(condition)
            params.append(_query_param(value))
    direction = 'DESC' if query.newest_first else 'ASC'
    sql = (
        f'SELECT * FROM shadow_cells WHERE {" AND ".join(conditions)} '
        f'ORDER BY opened_at {direction}, cell_id {direction}'
    )
    if query.limit is not None:
        sql += ' LIMIT ?'
        params.append(query.limit)
    return sql, params


@dataclass(frozen=True)
class CellChange:
    """One requested transition of a stored cell; an unset field keeps the stored value.

    Only a ``settled`` change carries a reference, and it must be coherent: a
    ``landed`` reference names its sha, a ``none`` reference names none.
    """

    state: ShadowCellState
    at: datetime
    worktree_path: Path | None = None
    result_path: Path | None = None
    cost_usd: float | None = None
    cost_credits: float | None = None
    reason: str | None = None
    reference_kind: ReferenceKind | None = None
    reference_sha: str | None = None

    def __post_init__(self) -> None:
        if self.state is ShadowCellState.SETTLED:
            self._check_reference()
        elif self.reference_kind is not None or self.reference_sha is not None:
            self._refuse(
                f'reference fields belong only to a {ShadowCellState.SETTLED.value!r} change '
                f'(got reference_kind={self.reference_kind!r}, reference_sha={self.reference_sha!r})'
            )

    def _check_reference(self) -> None:
        if self.reference_kind is None:
            self._refuse('a settled change requires a reference_kind')
        if self.reference_kind is ReferenceKind.LANDED and _is_blank(self.reference_sha):
            self._refuse(f'a landed reference requires a non-blank reference_sha (got {self.reference_sha!r})')
        if self.reference_kind is ReferenceKind.NONE and self.reference_sha is not None:
            self._refuse(f'a none reference carries no reference_sha (got {self.reference_sha!r})')

    def _refuse(self, invariant: str) -> None:
        raise ValueError(f'CellChange to {self.state.value!r}: {invariant}')


_CELL_CHANGE_CARRIED_FIELDS = tuple(
    field.name for field in fields(CellChange) if field.name not in ('state', 'at')
)


def _successor_cell(cell: ShadowCell, change: CellChange) -> ShadowCell:
    """The cell after *change*; ``replace`` re-runs every ``ShadowCell`` invariant."""
    carried: dict[str, Any] = {
        name: getattr(change, name)
        for name in _CELL_CHANGE_CARRIED_FIELDS
        if getattr(change, name) is not None
    }
    if change.state is ShadowCellState.RUNNING:
        carried['running_at'] = change.at
    if change.state.stamps_run_ended_at:
        carried['run_ended_at'] = change.at
    if change.state.stamps_settled_at:
        carried['settled_at'] = change.at
    return replace(cell, state=change.state, **carried)


_ADVANCE_SHADOW_CELL_SQL = (
    'UPDATE shadow_cells SET '
    + ', '.join(f'{name} = :{name}' for name in _SHADOW_CELL_COLUMNS if name != 'cell_id')
    + ' WHERE cell_id = :cell_id AND state = :expected_state'
)


@dataclass(frozen=True)
class ShadowStormRecord:
    """A project's ``shadow_storm`` row: shadow evaluation is paused while ``cause`` is set."""

    project_id: str
    cause: ShadowStormCause | None
    detail: str | None
    paused_at: datetime | None
    resumed_at: datetime | None

    def __post_init__(self) -> None:
        if (self.cause is None) != (self.paused_at is None):
            raise ValueError(
                f'ShadowStormRecord {self.project_id!r}: cause and paused_at are set together '
                f'(got cause={self.cause!r}, paused_at={self.paused_at!r})'
            )

    @property
    def paused(self) -> bool:
        return self.cause is not None


def _optional_instant(raw: str | None) -> datetime | None:
    return None if raw is None else datetime.fromisoformat(raw)


def _read_shadow_storm(conn: sqlite3.Connection, project_id: str) -> ShadowStormRecord:
    row = conn.execute(
        'SELECT cause, detail, paused_at, resumed_at FROM shadow_storm WHERE project_id = ?',
        (project_id,),
    ).fetchone()
    if row is None:
        return ShadowStormRecord(project_id, None, None, None, None)
    cause, detail, paused_at, resumed_at = row
    return ShadowStormRecord(
        project_id=project_id,
        cause=None if cause is None else ShadowStormCause(cause),
        detail=detail,
        paused_at=_optional_instant(paused_at),
        resumed_at=_optional_instant(resumed_at),
    )


@dataclass(frozen=True)
class TaskResultRecord:
    """One typed ``task_results`` row.

    A NULL cost or counter reads 0, as in ``RunStore.get_task_cost``; NULL block
    fields mean the row predates task 3068.
    """

    run_id: str
    project_id: str
    task_id: str
    title: str | None
    outcome: WorkflowOutcome
    cost_usd: float
    steward_cost_usd: float
    steward_invocations: int
    duration_ms: int
    agent_invocations: int
    execute_iterations: int
    verify_attempts: int
    review_cycles: int
    completed_at: datetime | None
    block_reason: str | None
    block_phase: str | None


_TASK_RESULT_COLUMNS = tuple(field.name for field in fields(TaskResultRecord))
_TASK_RESULT_NUMERIC_TYPES = {
    name: hint for name, hint in get_type_hints(TaskResultRecord).items() if hint in (int, float)
}


def _task_result_from_row(row: sqlite3.Row) -> TaskResultRecord:
    values: dict[str, Any] = {name: row[name] for name in _TASK_RESULT_COLUMNS}
    try:
        values['outcome'] = WorkflowOutcome(values['outcome'])
    except ValueError:
        raise ValueError(
            f'task_results row for task {values["task_id"]!r} (run {values["run_id"]!r}): '
            f'unknown outcome {values["outcome"]!r}'
        ) from None
    for name, numeric_type in _TASK_RESULT_NUMERIC_TYPES.items():
        values[name] = numeric_type(values[name] or 0)
    values['completed_at'] = (
        datetime.fromisoformat(values['completed_at']) if values['completed_at'] else None
    )
    return TaskResultRecord(**values)


_SCHEMA = """\
CREATE TABLE IF NOT EXISTS runs (
    run_id         TEXT PRIMARY KEY,
    project_id     TEXT NOT NULL,
    prd_path       TEXT,
    started_at     TEXT NOT NULL,
    completed_at   TEXT,
    total_tasks    INTEGER DEFAULT 0,
    completed      INTEGER DEFAULT 0,
    blocked        INTEGER DEFAULT 0,
    escalated      INTEGER DEFAULT 0,
    total_cost_usd REAL DEFAULT 0.0,
    paused_for_cap INTEGER DEFAULT 0,
    cap_pause_secs REAL DEFAULT 0.0,
    -- Task 6565: the commit the process's verify wire path was imported from
    -- (orchestrator.deployed_sha.DeployedSha.sha); NULL when unreadable or the
    -- row predates 6565.  Migrated by _migrate_runs_code_sha.
    code_sha       TEXT
);

CREATE TABLE IF NOT EXISTS task_results (
    run_id              TEXT NOT NULL REFERENCES runs(run_id),
    task_id             TEXT NOT NULL,
    project_id          TEXT NOT NULL,
    title               TEXT,
    outcome             TEXT NOT NULL,
    cost_usd            REAL DEFAULT 0.0,
    duration_ms         INTEGER DEFAULT 0,
    agent_invocations   INTEGER DEFAULT 0,
    execute_iterations  INTEGER DEFAULT 0,
    verify_attempts     INTEGER DEFAULT 0,
    review_cycles       INTEGER DEFAULT 0,
    steward_cost_usd    REAL DEFAULT 0.0,
    steward_invocations INTEGER DEFAULT 0,
    completed_at        TEXT,
    -- Task 3068 (origin incident: reify esc-5556-1).  The classified block
    -- reason + PRE-block working phase, so a requeue loop leaves a durable,
    -- GROUP-BY-able WHY here and not only in a rotated journald log.
    --
    -- These MUST stay LAST, after completed_at: SQLite's ALTER TABLE ADD
    -- COLUMN can only append, so declaring them anywhere else would give a
    -- freshly-created runs.db a different physical column ORDER than one
    -- migrated by _migrate_task_results_block_context.  Both writers name
    -- their columns explicitly, but an operator forensic `SELECT *` — exactly
    -- the use case this task exists to serve — is positional.
    --
    -- block_detail is deliberately NOT persisted: unbounded raw agent/verify
    -- output in an operationally-queried store (see task 3068 decisions).
    block_reason        TEXT,
    block_phase         TEXT,
    PRIMARY KEY (run_id, task_id)
);

CREATE INDEX IF NOT EXISTS idx_task_results_project
    ON task_results(project_id, completed_at);

CREATE TABLE IF NOT EXISTS scheduler_state (
    project_id     TEXT PRIMARY KEY,
    pause_reason   TEXT NOT NULL,
    pause_at       TEXT NOT NULL,
    set_by_run_id  TEXT NOT NULL,
    -- Task 4559: the live EWA value at pause time, recorded for EVERY pause
    -- class as operator evidence.  Nullable on purpose: NULL means "row
    -- predates task 4559", which the restart predicate treats as unknowable
    -- and therefore restores the halt blind.
    ewa_value      REAL
);

-- Task 5812: the last COLD main-tip sweep that reached a verdict.
CREATE TABLE IF NOT EXISTS main_sweep_cold_control (
    project_id  TEXT PRIMARY KEY,
    verdict_at  TEXT NOT NULL,
    swept_sha   TEXT NOT NULL
);
""" + _SHADOW_CELLS_SCHEMA


class RunStore:
    """Synchronous SQLite writer for orchestrator run results."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        """Open a connection to the store's DB with the full durability pragma triad applied."""
        conn = sqlite3.connect(str(self.db_path))
        apply_full_durability_pragmas_sync(conn, busy_timeout_ms=5000)
        return conn

    def checkpoint(self) -> CheckpointResult:
        """Run ``PRAGMA wal_checkpoint(TRUNCATE)`` and return the result.

        Returns:
            A :class:`CheckpointResult` named-tuple ``(busy, log, checkpointed)``.
        """
        conn = self._connect()
        try:
            row = conn.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()
        finally:
            conn.close()
        if row is None:
            raise RuntimeError('PRAGMA wal_checkpoint returned no rows')
        return CheckpointResult(int(row[0]), int(row[1]), int(row[2]))

    def _ensure_schema(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(_SCHEMA)
            # Task 3068: CREATE TABLE IF NOT EXISTS is a no-op on an existing
            # table, so the _SCHEMA additions above reach FRESH DBs only.  Every
            # already-deployed runs.db needs the explicit ALTER.
            self._migrate_task_results_block_context(conn)
            self._migrate_scheduler_state_ewa_value(conn)
            self._migrate_runs_code_sha(conn)
            self._migrate_shadow_cells_run_ended_at(conn)
        finally:
            conn.close()

    def _migrate_task_results_block_context(self, conn: sqlite3.Connection) -> None:
        """Add ``block_reason``/``block_phase`` to a pre-3068 ``task_results``.

        Additive-only, with no ``PRAGMA user_version`` ladder (``RunStore`` has
        never had one), per the rule in
        ``shared/src/shared/sqlite_column_migration.py``.

        Idempotent, and a no-op on a fresh DB — ``_SCHEMA`` already declares
        both columns there, so the probe finds them present.

        Existing rows read NULL for both columns, which is exactly right: NULL
        means "row predates task 3068", while ``''`` (what both writers now
        always store) means "clean exit, no block".

        FAILURE BEHAVIOUR — note what this actually is, not what would be
        tidy.  A genuinely unrecoverable ``ALTER`` propagates out of
        ``__init__``, but the sole production caller (``Harness.run``) wraps
        ``RunStore(db_path)`` in ``except Exception: logger.error(...)`` and
        continues, so a raise here does NOT halt the run — it drops
        ``task_results`` persistence for the whole run.  That is strictly worse
        than writing NULLs, which is why tolerating a concurrent construction
        that already added the column matters here.
        """
        added = [
            column
            for column in ('block_reason', 'block_phase')
            if add_column_if_missing_sync(conn, 'task_results', column, 'TEXT')
        ]
        if added:
            conn.commit()
            logger.info(
                'RunStore: migrated task_results — added %s', ', '.join(added)
            )

    def _migrate_scheduler_state_ewa_value(self, conn: sqlite3.Connection) -> None:
        """Add ``ewa_value`` to a pre-4559 ``scheduler_state``.

        Same shape as :meth:`_migrate_task_results_block_context`, deliberately:
        additive-only, no ``PRAGMA user_version`` ladder.

        Idempotent, and a no-op on a fresh DB — ``_SCHEMA`` already declares the
        column there, so the probe finds it present.

        An existing row reads NULL, which is exactly right: NULL means "row
        predates task 4559", and ``Harness._load_persisted_scheduler_pause``
        treats an unknowable predicate as a reason to RE-ASSERT the halt, never
        to release it.

        FAILURE BEHAVIOUR — the same asymmetry the sibling migration documents.
        A raise here propagates out of ``__init__``; the sole production caller
        (``Harness.run``) wraps ``RunStore(db_path)`` in
        ``except Exception: logger.error(...)`` and continues, so it would drop
        persistence for the whole run rather than halt it.  That is strictly
        worse than a NULL column.
        """
        if add_column_if_missing_sync(conn, 'scheduler_state', 'ewa_value', 'REAL'):
            conn.commit()
            logger.info('RunStore: migrated scheduler_state — added ewa_value')

    def _migrate_runs_code_sha(self, conn: sqlite3.Connection) -> None:
        """Add ``code_sha`` to a pre-6565 ``runs``.

        Same shape as :meth:`_migrate_scheduler_state_ewa_value`, deliberately:
        additive-only, the column added LAST.  An existing row reads NULL,
        meaning "the run predates task 6565".
        """
        if add_column_if_missing_sync(conn, 'runs', 'code_sha', 'TEXT'):
            conn.commit()
            logger.info('RunStore: migrated runs — added code_sha')

    def _migrate_shadow_cells_run_ended_at(self, conn: sqlite3.Connection) -> None:
        """Add ``run_ended_at`` to a pre-5387 ``shadow_cells``.

        Same shape as :meth:`_migrate_runs_code_sha`. An existing row reads
        NULL, meaning "the run had not ended, or ended before task 5387".
        """
        if add_column_if_missing_sync(conn, 'shadow_cells', 'run_ended_at', 'TEXT'):
            conn.commit()
            logger.info('RunStore: migrated shadow_cells — added run_ended_at')

    def start_run(
        self,
        run_id: str,
        project_id: str,
        started_at: str,
        prd_path: str | None = None,
        *,
        code_sha: str | None = None,
    ) -> None:
        """Insert an initial runs row at orchestrator startup.

        *code_sha* is the commit the process's wire path was imported from
        (task 6565), ``None`` when it could not be read.  The row is updated
        with final aggregates via :meth:`finish_run` when the run completes.
        """
        conn = self._connect()
        try:
            conn.execute(
                'INSERT OR IGNORE INTO runs '
                '(run_id, project_id, prd_path, started_at, code_sha) '
                'VALUES (?, ?, ?, ?, ?)',
                (run_id, project_id, prd_path, started_at, code_sha),
            )
            conn.commit()
        finally:
            conn.close()
        logger.info(f'Started run {run_id} for project {project_id}')

    def save_task_result(
        self,
        run_id: str,
        task_report: TaskReport,
        project_id: str,
    ) -> None:
        """Persist a single TaskReport row immediately on task completion."""
        conn = self._connect()
        try:
            conn.execute(
                'INSERT OR REPLACE INTO task_results '
                '(run_id, task_id, project_id, title, outcome, '
                ' cost_usd, duration_ms, agent_invocations, '
                ' execute_iterations, verify_attempts, review_cycles, '
                ' steward_cost_usd, steward_invocations, completed_at, '
                ' block_reason, block_phase) '
                'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (
                    run_id,
                    task_report.task_id,
                    project_id,
                    task_report.title,
                    task_report.outcome.value,
                    task_report.cost_usd,
                    task_report.duration_ms,
                    task_report.agent_invocations,
                    task_report.execute_iterations,
                    task_report.verify_attempts,
                    task_report.review_cycles,
                    task_report.steward_cost_usd,
                    task_report.steward_invocations,
                    task_report.completed_at,
                    # Task 3068 — '' on clean exits, never NULL, so a NULL here
                    # unambiguously means "row predates 3068".
                    task_report.block_reason,
                    task_report.block_phase,
                ),
            )
            conn.commit()
        finally:
            conn.close()
        logger.debug(
            f'Persisted task result {task_report.task_id} '
            f'({task_report.outcome.value}) for run {run_id}'
        )

    def get_task_cost(self, run_id: str, task_id: str) -> float:
        """Return latest persisted ``cost_usd + steward_cost_usd`` for a task.

        Only the most recent ``save_task_result`` row is retained per
        ``(run_id, task_id)`` — prior requeue attempts are overwritten.
        Returns ``0.0`` when no row exists.
        """
        conn = self._connect()
        try:
            cur = conn.execute(
                'SELECT cost_usd, steward_cost_usd FROM task_results '
                'WHERE run_id = ? AND task_id = ?',
                (run_id, task_id),
            )
            row = cur.fetchone()
        finally:
            conn.close()
        if row is None:
            return 0.0
        cost_usd, steward_cost_usd = row
        return float(cost_usd or 0.0) + float(steward_cost_usd or 0.0)

    def get_task_outcome(self, project_id: str, task_id: str) -> TaskResultRecord | None:
        """Production's latest recorded outcome for *task_id*, across every run."""
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                f'SELECT {", ".join(_TASK_RESULT_COLUMNS)} FROM task_results '
                'WHERE project_id = ? AND task_id = ? '
                'ORDER BY completed_at DESC, rowid DESC LIMIT 1',
                (project_id, task_id),
            ).fetchone()
        finally:
            conn.close()
        return None if row is None else _task_result_from_row(row)

    def finish_run(
        self,
        run_id: str,
        report: HarnessReport,
    ) -> None:
        """Update the runs row with final aggregates at shutdown."""
        conn = self._connect()
        try:
            conn.execute(
                'UPDATE runs SET '
                'completed_at = ?, total_tasks = ?, completed = ?, '
                'blocked = ?, escalated = ?, total_cost_usd = ?, '
                'paused_for_cap = ?, cap_pause_secs = ? '
                'WHERE run_id = ?',
                (
                    report.completed_at,
                    report.total_tasks,
                    report.completed,
                    report.blocked,
                    report.escalated,
                    report.total_cost_usd,
                    int(report.paused_for_cap),
                    report.cap_pause_duration_secs,
                    run_id,
                ),
            )
            conn.commit()
        finally:
            conn.close()
        logger.info(
            f'Finished run {run_id}: {len(report.task_reports)} task results'
        )

    def save_run(
        self,
        report: HarnessReport,
        project_id: str,
        prd_path: str | None = None,
    ) -> str:
        """Persist a HarnessReport and its TaskReports to SQLite.

        Returns the generated run_id.
        """
        run_id = f'run-{uuid.uuid4().hex[:12]}'
        conn = self._connect()
        try:
            conn.execute(
                'INSERT INTO runs '
                '(run_id, project_id, prd_path, started_at, completed_at, '
                ' total_tasks, completed, blocked, escalated, '
                ' total_cost_usd, paused_for_cap, cap_pause_secs) '
                'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (
                    run_id,
                    project_id,
                    prd_path,
                    report.started_at,
                    report.completed_at,
                    report.total_tasks,
                    report.completed,
                    report.blocked,
                    report.escalated,
                    report.total_cost_usd,
                    int(report.paused_for_cap),
                    report.cap_pause_duration_secs,
                ),
            )
            for tr in report.task_reports:
                conn.execute(
                    'INSERT INTO task_results '
                    '(run_id, task_id, project_id, title, outcome, '
                    ' cost_usd, duration_ms, agent_invocations, '
                    ' execute_iterations, verify_attempts, review_cycles, '
                    ' steward_cost_usd, steward_invocations, completed_at, '
                    ' block_reason, block_phase) '
                    'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                    (
                        run_id,
                        tr.task_id,
                        project_id,
                        tr.title,
                        tr.outcome.value,
                        tr.cost_usd,
                        tr.duration_ms,
                        tr.agent_invocations,
                        tr.execute_iterations,
                        tr.verify_attempts,
                        tr.review_cycles,
                        tr.steward_cost_usd,
                        tr.steward_invocations,
                        tr.completed_at,
                        # Task 3068 — the batch path must stay in lockstep with
                        # save_task_result above, or it silently writes NULLs.
                        tr.block_reason,
                        tr.block_phase,
                    ),
                )
            conn.commit()
        finally:
            conn.close()

        logger.info(
            f'Persisted run {run_id}: {len(report.task_reports)} task results '
            f'for project {project_id}'
        )
        return run_id

    # ------------------------------------------------------------------ #
    # Scheduler park-and-stop pause persistence (task 1322)               #
    # ------------------------------------------------------------------ #

    def save_scheduler_pause(
        self,
        project_id: str,
        reason: str,
        pause_at_iso: str,
        set_by_run_id: str,
        ewa_value: float | None = None,
    ) -> None:
        """Persist (or replace) the scheduler pause state for *project_id*.

        Uses INSERT OR REPLACE so repeated saves are idempotent — the most
        recent call wins, matching UPSERT semantics.

        *ewa_value* (task 4559) is the live EWA at pause time, recorded for
        EVERY pause class as operator evidence rather than as a trip flag — the
        pause CLASS is carried by *reason*.  Keyword-defaulted to ``None`` so
        every existing call site and ``MagicMock(spec=RunStore)`` spy stays
        valid; ``None`` stores NULL, which reads back as "predates task 4559".
        """
        conn = self._connect()
        try:
            conn.execute(
                'INSERT OR REPLACE INTO scheduler_state '
                '(project_id, pause_reason, pause_at, set_by_run_id, ewa_value) '
                'VALUES (?, ?, ?, ?, ?)',
                (project_id, reason, pause_at_iso, set_by_run_id, ewa_value),
            )
            conn.commit()
        finally:
            conn.close()

    def refresh_scheduler_pause_ewa(self, project_id: str, ewa_value: float) -> bool:
        """Update ONLY ``ewa_value``, and only on a held ``ewa_trip_`` pause row.

        Returns True if such a row existed and was updated.

        Task 4559 amendment.  ``save_scheduler_pause`` writes a trip-TIME
        snapshot, and it is the only writer, so once a halt is asserted the
        stored scalar freezes at the tripping value even as the live EWA decays
        across subsequent digest steps.  That froze two things solid: the
        restart-time predicate re-check in
        ``Harness._load_persisted_scheduler_pause`` could essentially never
        fire (a stored value is by construction >= the threshold at write
        time), and a restart re-seeded live ``_ewa_value`` with the stale high
        number, discarding whatever decay the drain had achieved.  On the 8h
        fleet-redeploy cadence a paused unit therefore reset its healing
        progress on every restart.

        This is deliberately an UPDATE, never an upsert: with no pause row
        there is nothing to refresh, and inserting one would fabricate a halt
        out of a routine digest step.  ``reason``, ``pause_at`` and
        ``set_by_run_id`` are left untouched so the row keeps recording WHEN
        and WHY the halt was asserted while ``ewa_value`` tracks the live
        statistic.  The bool return lets the caller distinguish "refreshed"
        from "no ``ewa_trip_`` row to refresh" without a second read.

        The ``ewa_trip_`` scope is enforced HERE, in the WHERE clause, not only
        at the call site.  ``_maybe_write_digest`` step (13b) does pre-filter on
        ``scheduler.pause_reason``, but that is the IN-MEMORY reason and the two
        can diverge: ``Scheduler.pause`` is first-wins in memory while
        ``save_scheduler_pause`` is INSERT OR REPLACE, i.e. last-wins on disk.
        A held ``ewa_trip_`` halt plus a later ``pause_scheduler('park-stop: …')``
        — an in-memory no-op that still rewrites the row — would otherwise let
        every subsequent digest step overwrite that park-stop row's trip-time
        ``ewa_value`` with the decaying EWA, destroying the forensic evidence
        ``pause_scheduler`` promises is never overwritten for non-EWA classes.
        For ``ewa_trip_`` rows alone the stored scalar IS the live statistic;
        for every other class it is evidence about that pause.
        """
        conn = self._connect()
        try:
            # substr(...) rather than LIKE 'ewa_trip_%': in SQL LIKE, `_` is a
            # single-character WILDCARD, so that pattern would also match
            # reasons like 'ewaXtripY...'.  substr is an exact prefix test and
            # mirrors the caller's Python `startswith('ewa_trip_')` byte for
            # byte.  len('ewa_trip_') == 9.
            cursor = conn.execute(
                'UPDATE scheduler_state SET ewa_value = ? '
                "WHERE project_id = ? AND substr(pause_reason, 1, 9) = 'ewa_trip_'",
                (ewa_value, project_id),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def load_scheduler_pause(self, project_id: str) -> dict | None:
        """Return the persisted pause record for *project_id*, or ``None``.

        Returns a dict with keys ``reason``, ``pause_at``, ``set_by_run_id``
        and ``ewa_value`` (task 4559; ``None`` for a row written before that
        migration).
        """
        conn = self._connect()
        try:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                'SELECT pause_reason, pause_at, set_by_run_id, ewa_value '
                'FROM scheduler_state WHERE project_id = ?',
                (project_id,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        return {
            'reason': row['pause_reason'],
            'pause_at': row['pause_at'],
            'set_by_run_id': row['set_by_run_id'],
            'ewa_value': row['ewa_value'],
        }

    def clear_scheduler_pause(self, project_id: str) -> None:
        """Remove any persisted pause state for *project_id*."""
        conn = self._connect()
        try:
            conn.execute(
                'DELETE FROM scheduler_state WHERE project_id = ?',
                (project_id,),
            )
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    # Main-tip sweep cold-control cadence (task 5812)                     #
    # ------------------------------------------------------------------ #

    def save_main_sweep_cold_verdict(
        self, project_id: str, *, verdict_at: datetime, swept_sha: str,
    ) -> None:
        """Persist (last wins) when *project_id*'s last COLD sweep reached a verdict.

        Stored here so the cold-control cadence survives an orchestrator restart;
        the consumer is ``orchestrator/main_tip_sweep_cadence.py::MainSweepColdControl``.
        """
        conn = self._connect()
        try:
            conn.execute(
                'INSERT OR REPLACE INTO main_sweep_cold_control '
                '(project_id, verdict_at, swept_sha) VALUES (?, ?, ?)',
                (project_id, verdict_at.isoformat(), swept_sha),
            )
            conn.commit()
        finally:
            conn.close()

    def load_main_sweep_last_cold_verdict(self, project_id: str) -> datetime | None:
        """The aware-UTC time of *project_id*'s last cold sweep verdict, or ``None``."""
        conn = self._connect()
        try:
            row = conn.execute(
                'SELECT verdict_at FROM main_sweep_cold_control WHERE project_id = ?',
                (project_id,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        verdict_at = datetime.fromisoformat(row[0])
        if verdict_at.tzinfo is None:
            return verdict_at.replace(tzinfo=UTC)
        return verdict_at

    # ------------------------------------------------------------------ #
    # Shadow-eval cells (plans/live-shadow-eval-prd.md C1, task 5382)    #
    # ------------------------------------------------------------------ #

    def insert_shadow_cell(self, cell: ShadowCell) -> None:
        """Insert a new cell; an existing ``cell_id`` raises ``sqlite3.IntegrityError``.

        A ``parent_cell_id`` must name a stored cell of the same project, task
        and base, else ``ValueError`` and nothing is written.
        """
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                conn.execute('BEGIN IMMEDIATE')
                if cell.parent_cell_id is not None:
                    _require_valid_parent(conn, cell)
                conn.execute(_INSERT_SHADOW_CELL_SQL, _shadow_cell_to_row(cell))
        finally:
            conn.close()

    def get_shadow_cell(self, cell_id: str) -> ShadowCell | None:
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                'SELECT * FROM shadow_cells WHERE cell_id = ?', (cell_id,)
            ).fetchone()
        finally:
            conn.close()
        return None if row is None else _shadow_cell_from_row(row)

    def find_shadow_cells(self, query: ShadowCellQuery) -> list[ShadowCell]:
        """This project's cells matching *query*, by ``opened_at`` (oldest first unless asked)."""
        sql, params = _find_shadow_cells_sql(query)
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(sql, params).fetchall()
        finally:
            conn.close()
        return [_shadow_cell_from_row(row) for row in rows]

    def advance_shadow_cell(
        self, cell_id: str, *, expected: ShadowCellState, change: CellChange,
    ) -> ShadowCell | None:
        """Move a cell from *expected* to ``change.state``: the one transition choke point.

        Returns the stored successor, or ``None`` with nothing written when the
        stored state is no longer *expected* (a lost compare-and-set). An
        unknown cell raises ``KeyError``; a transition the successor table
        forbids, or a successor breaking a row invariant, raises ``ValueError``.
        Entering ``running`` stamps ``running_at``; entering a state that
        ``stamps_run_ended_at`` stamps ``run_ended_at``; entering a state that
        ``stamps_settled_at`` stamps ``settled_at``, and nothing else does.
        """
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                conn.execute('BEGIN IMMEDIATE')
                row = conn.execute(
                    'SELECT * FROM shadow_cells WHERE cell_id = ?', (cell_id,)
                ).fetchone()
                if row is None:
                    raise KeyError(f'shadow cell {cell_id!r} is not stored')
                if not expected.may_become(change.state):
                    raise ValueError(
                        f'shadow cell {cell_id!r}: {expected.value!r} may not become '
                        f'{change.state.value!r}'
                    )
                stored = _shadow_cell_from_row(row)
                if stored.state is not expected:
                    return None
                successor = _successor_cell(stored, change)
                conn.execute(
                    _ADVANCE_SHADOW_CELL_SQL,
                    {**_shadow_cell_to_row(successor), 'expected_state': expected.value},
                )
        finally:
            conn.close()
        return successor

    # ------------------------------------------------------------------ #
    # Shadow-eval storm pause (plans/live-shadow-eval-prd.md C2, task 5387) #
    # ------------------------------------------------------------------ #

    def get_shadow_storm(self, project_id: str) -> ShadowStormRecord:
        """*project_id*'s storm record; a never-paused project reads an all-``None`` record."""
        conn = self._connect()
        try:
            return _read_shadow_storm(conn, project_id)
        finally:
            conn.close()

    def pause_shadow_storm(
        self, project_id: str, cause: ShadowStormCause, detail: str, at: datetime,
    ) -> bool:
        """Pause *project_id*; True only for the call that created the pause.

        While a pause stands a further call changes nothing, so the first
        cause is kept and only its creator files the escalation.
        """
        conn = self._connect()
        try:
            with conn:
                conn.execute('BEGIN IMMEDIATE')
                if _read_shadow_storm(conn, project_id).paused:
                    return False
                conn.execute(
                    'INSERT INTO shadow_storm (project_id, cause, detail, paused_at) '
                    'VALUES (?, ?, ?, ?) ON CONFLICT(project_id) DO UPDATE SET '
                    'cause = excluded.cause, detail = excluded.detail, '
                    'paused_at = excluded.paused_at',
                    (project_id, cause.value, detail, _utc_iso(at)),
                )
        finally:
            conn.close()
        return True

    def resume_shadow_storm(self, project_id: str, at: datetime) -> bool:
        """Lift *project_id*'s pause and record *at* as ``resumed_at``; True if one stood."""
        conn = self._connect()
        try:
            with conn:
                conn.execute('BEGIN IMMEDIATE')
                if not _read_shadow_storm(conn, project_id).paused:
                    return False
                conn.execute(
                    'UPDATE shadow_storm SET cause = NULL, detail = NULL, paused_at = NULL, '
                    'resumed_at = ? WHERE project_id = ?',
                    (_utc_iso(at), project_id),
                )
        finally:
            conn.close()
        return True
