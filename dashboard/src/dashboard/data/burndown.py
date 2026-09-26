"""Burndown snapshot collection, downsampling, and chart queries.

Periodically writes one row per project into a SQLite ``snapshots`` table.
The background collector (``loops._burndown_loop``) writes via a dedicated
writable connection; route handlers read via DbPool (read-only).

THE SOURCE is the task snapshot unit,
``dashboard/src/dashboard/data/task_snapshot.py::acquire_snapshot`` — the same
cached unit ``/api/v2/dashboard/tasks`` serves — never a whole-tree read. The
unit's census gives the nine ``TaskStatus`` member columns; its rows give the
``in_progress_live`` / ``in_progress_stranded`` split and ``in_progress_rows``
(task 5591). ``concurrency_cap`` is ``max_concurrent_tasks`` as it stood AT
SNAPSHOT TIME, ``NULL`` when unknown (task 3543).

A row is written only from a unit whose two halves are both FRESH at the
collector's instant; see :func:`collect_snapshot`.
"""

from __future__ import annotations

import asyncio
import contextlib
import enum
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import Any

import aiosqlite
import httpx
from shared.task_statuses import TaskStatus

from dashboard.config import DashboardConfig
from dashboard.data.census import TaskCensus, TaskView
from dashboard.data.datum import DatumState
from dashboard.data.orchestrator import (
    _read_project_root_from_config,
    _resolve_project_root,
    find_running_orchestrators,
    read_max_concurrent_tasks,
)
from dashboard.data.task_snapshot import TaskSnapshot, acquire_snapshot, as_served
from dashboard.data.utils import resolve_now

logger = logging.getLogger(__name__)

# Whole-operation backstop for ONE root's snapshot acquisition, enforced by
# collect_snapshot's Phase-2 gather. Expiry is a GAP row naming this budget,
# never a skipped root.
#
# A BACKSTOP around a bounded, total-by-contract unit, and still worth its
# place: task 4884 showed an unbounded sampler await parks every project's
# history silently, and the sampler's invariant is one row per root per tick.
#
# FLOOR: the unit's own structural worst case, so the backstop never pre-empts
# a unit still inside its own bounds. Both halves run concurrently, each under
# wait_for(task_snapshot.PER_CALL_TIMEOUT) (4.4 s), after the unit cache's
# bounded lock wait (mcp_fanout._LOCK_ACQUIRE_TIMEOUT_SECONDS, 15 s) that
# precedes a bypass: ~19.4 s.
# CEILING: half of loops._SAMPLE_INTERVAL_SECONDS (600), so one collector
# cycle is finished before the next begins.
#
# 60.0 is ~3x the floor. Pinned by TestCollectSnapshotPerRootBudget in
# dashboard/tests/test_burndown_data.py.
_SNAPSHOT_PER_ROOT_BUDGET = 60.0

BURNDOWN_SCHEMA = """\
CREATE TABLE IF NOT EXISTS snapshots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id  TEXT    NOT NULL,
    ts          TEXT    NOT NULL,
    pending     INTEGER NOT NULL DEFAULT 0,
    in_progress INTEGER NOT NULL DEFAULT 0,
    blocked     INTEGER NOT NULL DEFAULT 0,
    deferred    INTEGER NOT NULL DEFAULT 0,
    cancelled   INTEGER NOT NULL DEFAULT 0,
    done        INTEGER NOT NULL DEFAULT 0,
    -- The in-progress rows' live/stranded split (task 3543). A partition, so
    -- it never adds to the member counts: of in_progress on a row written
    -- before task 5591, of in_progress_rows (below) on one written since.
    in_progress_live     INTEGER NOT NULL DEFAULT 0,
    in_progress_stranded INTEGER NOT NULL DEFAULT 0,
    -- max_concurrent_tasks in force AT SNAPSHOT TIME. Nullable on purpose:
    -- NULL means "cap unknown", which is not a breach. Storing 0 instead would
    -- read as a cap of zero and alarm on every snapshot. The cap is restart-
    -- only (red-tier), but a burndown window spans restarts and the cap also
    -- varies BETWEEN projects, so the only honest denominator for a historical
    -- row is the cap that was in force at that instant.
    concurrency_cap      INTEGER,
    -- Task 5591 (δ1). NULL in any of these six = not recorded when the row
    -- was written (a row from before this migration); never a measured zero.
    -- review/merge_deferred/infra_hold complete the nine TaskStatus members.
    -- in_progress_rows is the ROWS' in-progress count, which
    -- in_progress_live + in_progress_stranded partitions; the census's
    -- in_progress may differ from it by intra-unit skew.
    -- state is 'value' or 'gap'; NULL on a pre-migration row means a measured
    -- row, because before this change only a successful read wrote a row.
    -- A gap row carries no measurement: its count columns are NULL, or hold
    -- their DEFAULT physically where legacy NOT NULL forbids NULL, so every
    -- reader selects measured rows only. reason says why a gap is a gap.
    review               INTEGER,
    merge_deferred       INTEGER,
    infra_hold           INTEGER,
    in_progress_rows     INTEGER,
    state                TEXT,
    reason               TEXT
);
CREATE INDEX IF NOT EXISTS idx_snapshots_project_ts ON snapshots(project_id, ts);
"""

# Columns added after the original 6-zone schema shipped, in the order
# ensure_snapshot_columns adds them. ``CREATE TABLE IF NOT EXISTS`` is a no-op
# against an existing DB, so these reach a live burndown.db only via the
# migration.
_ADDED_SNAPSHOT_COLUMNS: tuple[tuple[str, str], ...] = (
    ('in_progress_live', 'INTEGER NOT NULL DEFAULT 0'),
    ('in_progress_stranded', 'INTEGER NOT NULL DEFAULT 0'),
    ('concurrency_cap', 'INTEGER'),
    ('review', 'INTEGER'),
    ('merge_deferred', 'INTEGER'),
    ('infra_hold', 'INTEGER'),
    ('in_progress_rows', 'INTEGER'),
    ('state', 'TEXT'),
    ('reason', 'TEXT'),
)



class SnapshotState(enum.StrEnum):
    """What one ``snapshots`` row records: a measurement, or its absence.

    NULL on a row written before task 5591 means ``VALUE``: until then only a
    successful read wrote a row at all.
    """

    VALUE = 'value'
    GAP = 'gap'


MEMBER_COLUMNS: Mapping[TaskStatus, str] = MappingProxyType(
    {member: member.value.replace('-', '_') for member in TaskStatus}
)
"""The ``snapshots`` column holding each ``TaskStatus`` member's census count.

ONE naming rule over the closed enum, in ``TaskStatus`` order, so the columns
follow the vocabulary rather than restating it. A tenth member maps to a
column that does not exist, and its INSERT fails loudly instead of the count
being absorbed into another member.
"""


async def ensure_snapshot_columns(conn: aiosqlite.Connection) -> None:
    """Bring an existing ``snapshots`` table up to the current column set.

    Additive only: probes ``PRAGMA table_info`` and issues one
    ``ALTER TABLE ... ADD COLUMN`` per missing column.  Idempotent, never drops
    or rewrites anything, so pre-existing rows keep their data and take the
    column defaults: ``0`` for the split (task 3543), and ``NULL`` for the cap
    and for every column task 5591 added — an honest "not recorded", since
    none of them was measured when the row was written.

    Mandatory, not cosmetic: :data:`BURNDOWN_SCHEMA` is applied with
    ``CREATE TABLE IF NOT EXISTS``, which is a no-op against every already
    deployed burndown.db.  Without this, the collector's INSERT, which names
    every current column, would fail on the first collection cycle after
    deploy.  Call it at store open, before the collector runs.

    Mirrors fused-memory's ``_migrate_v1_to_v2`` probe-then-add idiom.  The
    caller commits.
    """
    cursor = await conn.execute('PRAGMA table_info(snapshots)')
    existing = {row[1] for row in await cursor.fetchall()}
    await cursor.close()
    for column, ddl in _ADDED_SNAPSHOT_COLUMNS:
        if column in existing:
            continue
        await conn.execute(f'ALTER TABLE snapshots ADD COLUMN {column} {ddl}')
        logger.info('burndown: added snapshots.%s (%s)', column, ddl)


@dataclass(frozen=True, slots=True)
class _Measurement:
    """What a wholly fresh snapshot unit measured, narrowed to present values."""

    census: TaskCensus
    in_progress_live: int
    in_progress_stranded: int


def _measurement(served: TaskSnapshot) -> _Measurement | None:
    """*served*'s measurement, or ``None`` unless BOTH halves are ``FRESH``.

    A ``STALE`` half is a last-good value from an EARLIER instant, so writing
    it at this cycle's ``ts`` would record an old census as current. A fresh
    census beside unmeasured rows is not a measurement either: the legacy
    split columns are NOT NULL and cannot say "unknown".
    """
    if served.census.state is not DatumState.FRESH or served.rows.state is not DatumState.FRESH:
        return None
    census = served.census.value
    live, stranded = served.in_progress_live, served.in_progress_stranded
    if census is None or live is None or stranded is None:
        return None
    return _Measurement(census, live, stranded)


@dataclass(frozen=True, slots=True)
class _SnapshotRow:
    """One ``snapshots`` row as the collector writes it.

    *measured* maps column to value, in insert order.
    """

    project_id: str
    ts: str
    state: SnapshotState
    reason: str | None
    measured: Mapping[str, int | None]


def _value_row(
    project_id: str, ts: str, measurement: _Measurement, cap: int | None,
) -> _SnapshotRow:
    """The row recording *measurement*: nine members from the census, split from the rows."""
    counts = measurement.census.counts
    live, stranded = measurement.in_progress_live, measurement.in_progress_stranded
    return _SnapshotRow(
        project_id=project_id,
        ts=ts,
        state=SnapshotState.VALUE,
        reason=None,
        measured=MappingProxyType({
            **{column: counts[member] for member, column in MEMBER_COLUMNS.items()},
            'in_progress_live': live,
            'in_progress_stranded': stranded,
            # The unit's own partition of the rows, whole. Not a recount of
            # statuses here, so live + stranded == in_progress_rows by
            # construction.
            'in_progress_rows': live + stranded,
            # NULL, never 0, when the cap is unknown: a stored 0 would read as
            # a cap of zero and alarm on every row.
            'concurrency_cap': cap,
        }),
    )


def _gap_row(project_id: str, ts: str, reason: str) -> _SnapshotRow:
    """The row recording that *project_id* was NOT measured at *ts*, and why.

    It writes no count column at all: the nullable ones stay NULL, and the
    legacy NOT NULL ones take their DEFAULT, which carries no meaning.
    """
    return _SnapshotRow(
        project_id=project_id,
        ts=ts,
        state=SnapshotState.GAP,
        reason=reason,
        measured=MappingProxyType({}),
    )


def _unmeasured_reason(served: TaskSnapshot) -> str:
    """Each non-fresh half's own reason, verbatim, labelled with its half.

    Routing never reads this text: the state of each half decides value vs
    gap, and this only records why for the operator.
    """
    reasons = [
        f'{half}: {datum.reason}'
        for half, datum in (('census', served.census), ('rows', served.rows))
        if datum.state is not DatumState.FRESH
    ]
    return '; '.join(reasons) or 'the snapshot unit carried no measurement'


def _acquisition_failure_reason(exc: BaseException) -> str:
    """Why a root's acquisition produced no unit at all."""
    if isinstance(exc, TimeoutError):
        return (
            f'snapshot acquisition exceeded the {_SNAPSHOT_PER_ROOT_BUDGET}s '
            'per-root budget'
        )
    return f'snapshot acquisition raised {type(exc).__name__}: {exc}'


def _row_for_root(
    root: str, now_dt: datetime, result: TaskSnapshot | BaseException,
) -> _SnapshotRow:
    """The one row *root* owes this cycle: a value row, or a gap row saying why not.

    A raise (or the backstop expiring) is a bug signal from a total-by-contract
    unit, so it logs at WARNING. A non-fresh unit is routine — an offline root
    is a DEBUG-level record — because the gap row itself is the durable one.
    """
    ts = now_dt.isoformat()
    if isinstance(result, BaseException):
        logger.warning('Snapshot acquisition failed for %s', root, exc_info=result)
        return _gap_row(root, ts, _acquisition_failure_reason(result))
    served = as_served(result, now_dt)
    measurement = _measurement(served)
    if measurement is None:
        reason = _unmeasured_reason(served)
        logger.debug('Snapshot unit for %s is not wholly fresh: %s', root, reason)
        return _gap_row(root, ts, reason)
    # One cap read per ROOT: it touches the filesystem, and the value is a
    # per-project scalar stored on the row because max_concurrent_tasks varies
    # across restarts and across projects — see BURNDOWN_SCHEMA.
    cap = read_max_concurrent_tasks(root)
    # The `running` sub-view is the one number an operator compares against
    # max_concurrent_tasks (PRD decision 3).
    running = measurement.census.sub_views[TaskView.RUNNING]
    if cap is not None and running > cap:
        logger.warning(
            'Concurrency cap breached for %s: %d in-progress vs cap %d (%d stranded)',
            root, running, cap, measurement.in_progress_stranded,
        )
    return _value_row(root, ts, measurement, cap)


async def _insert_row(conn: aiosqlite.Connection, row: _SnapshotRow) -> None:
    """INSERT *row* by explicit column names, binding ``project_id`` first."""
    columns: dict[str, object] = {
        'project_id': row.project_id,
        'ts': row.ts,
        **row.measured,
        'state': row.state.value,
        'reason': row.reason,
    }
    await conn.execute(
        f'INSERT INTO snapshots ({", ".join(columns)}) '
        f'VALUES ({", ".join("?" for _ in columns)})',
        tuple(columns.values()),
    )


async def collect_snapshot(
    conn: aiosqlite.Connection,
    config: DashboardConfig,
    client: httpx.AsyncClient,
) -> None:
    """Discover projects and insert exactly one snapshot row per project per cycle.

    Each root is read through the task snapshot unit
    (``task_snapshot.acquire_snapshot``), and the unit is judged as served at
    this cycle's instant. A root whose census and rows are BOTH fresh gets a
    value row: the census gives the nine member columns, the rows give the
    live/stranded split. Every other root gets a gap row whose ``reason``
    says why — a non-fresh half (logged at DEBUG), or an acquisition that
    raised or overran ``_SNAPSHOT_PER_ROOT_BUDGET`` (logged at WARNING).

    Partial-failure semantics:

    - **Main project committed first**: the main project is always
      roots_to_snapshot[0], so its per-project INSERT + commit fires before
      any extra project is attempted.  A DB failure on an extra cannot roll
      back a row that is already committed.

    - **Orchestrator discovery is best-effort**: the block that calls
      find_running_orchestrators and iterates the results is wrapped in
      try/except.  If it raises, a WARNING is logged and the function
      degrades gracefully — the main project and known_project_roots are
      still snapshotted.

    - **Extra inserts are per-project / isolated**: each project in Phase 3
      gets its own INSERT + commit wrapped in try/except.  A disk-full or
      constraint error on one project is logged as a WARNING and the loop
      continues; other projects are unaffected.

    - **Explicit rollback on unexpected error** (defensive guard): if an
      exception escapes all inner guards (e.g., a future code change
      introduces a DB write before Phase 3), conn.rollback() is called
      before re-raising.  This is purely defensive — the realistic failure
      modes (per-project INSERT errors) are already handled by Phase 3's
      per-project try/except.  The guard ensures the long-lived writer
      connection in app.py is left in a clean state even for unexpected
      regressions.
    """
    try:
        now_dt = datetime.now(UTC)  # clock-exempt: single-capture writer
        # One instant for the whole cycle: the ``ts`` written to every row, the
        # instant a unit is measured at, and the one it is served at must be
        # the SAME capture, or a row could claim a measurement its own
        # timestamp contradicts.
        # config.project_root is already resolved by DashboardConfig.__post_init__
        resolved_root = str(config.project_root)

        # Phase 1 — Discovery (sequential, in-memory):
        # Build the ordered list of project_root strings to snapshot.
        # Main project is always first; seen_roots dedup is preserved exactly.
        # project_root is already canonical (symlink-resolved) so a symlinked
        # project_root deduplicates correctly against orchestrator /
        # known_project_roots entries that surface the real path.
        roots_to_snapshot: list[str] = []
        seen_roots: set[str] = {resolved_root}
        roots_to_snapshot.append(resolved_root)

        try:
            orchestrators = await asyncio.to_thread(find_running_orchestrators)
        except Exception:
            logger.warning(
                'Orchestrator discovery failed; skipping orchestrator-discovered extras',
                exc_info=True,
            )
        else:
            for proc in orchestrators:
                try:
                    if proc.get('prd'):
                        project_root = _resolve_project_root(proc['prd'], config.project_root)
                    elif proc.get('config_path'):
                        resolved = _read_project_root_from_config(proc['config_path'])
                        if resolved is None:
                            continue
                        project_root = resolved
                    else:
                        continue
                    # project_root is already resolved by _resolve_project_root / _read_project_root_from_config
                    root_str = str(project_root)
                    if root_str in seen_roots:
                        continue
                    seen_roots.add(root_str)
                    roots_to_snapshot.append(root_str)
                except OSError:
                    logger.warning(
                        'OSError while resolving orchestrator project root; skipping',
                        exc_info=True,
                    )
                except Exception:
                    logger.warning(
                        'Orchestrator entry processing failed; skipping entry',
                        exc_info=True,
                    )

        for known_root in config.known_project_roots:
            # known_root is already resolved by DashboardConfig.__post_init__.
            root_str = str(known_root)
            if root_str in seen_roots:
                continue
            seen_roots.add(root_str)
            roots_to_snapshot.append(root_str)

        # Phase 2 — Parallel read, one snapshot unit per root.
        # return_exceptions=True isolates a root whose acquisition raises, so a
        # single bad root can't sink the whole cycle (task 519).
        #
        # WHOLE-OPERATION BACKSTOP per root (task 4884 / #4424), see
        # _SNAPSHOT_PER_ROOT_BUDGET. Expiry becomes that root's own
        # TimeoutError result, which Phase 3 records as a gap row.
        all_results = await asyncio.gather(
            *(
                asyncio.wait_for(
                    acquire_snapshot(client, config, root, now=now_dt),
                    timeout=_SNAPSHOT_PER_ROOT_BUDGET,
                )
                for root in roots_to_snapshot
            ),
            return_exceptions=True,
        )

        # Phase 3 — Sequential insert (per-project commit):
        # aiosqlite serialises writes on a single connection; keep inserts sequential.
        # Each project gets its own INSERT + commit so a DB failure on one project
        # cannot roll back rows that were already committed for earlier projects.
        for root_str, result in zip(roots_to_snapshot, all_results, strict=True):
            try:
                await _insert_row(conn, _row_for_root(root_str, now_dt, result))
                await conn.commit()
            except Exception:
                logger.warning('Failed to insert snapshot for %s', root_str, exc_info=True)
                with contextlib.suppress(Exception):
                    await conn.rollback()
    except Exception:
        with contextlib.suppress(Exception):
            await conn.rollback()
        raise


async def _snapshot_columns(db: aiosqlite.Connection) -> frozenset[str]:
    """The columns this DB's ``snapshots`` table actually has.

    Probed, never assumed: ``_burndown_dbs`` opens OTHER projects'
    burndown.db files read-only and nothing migrates those, so a query naming
    a newer column would raise 'no such column' there.
    """
    async with db.execute('PRAGMA table_info(snapshots)') as cur:
        return frozenset(row[1] for row in await cur.fetchall())


@dataclass(frozen=True, slots=True)
class _Predicate:
    """A SQL boolean expression and the parameters it binds."""

    sql: str
    params: tuple[str, ...]


def _measured_rows(columns: frozenset[str]) -> _Predicate:
    """THE predicate for "this row carries a measurement".

    A POSITIVE selection of ``state = 'value'`` plus a NULL state (a row from
    before task 5591, when only a successful read wrote one), so a state this
    code does not know is never counted as measured. A table with no ``state``
    column is un-migrated, and every row in it is measured by construction.
    """
    if 'state' not in columns:
        return _Predicate('TRUE', ())
    return _Predicate('(state IS NULL OR state = ?)', (SnapshotState.VALUE.value,))


async def downsample(conn: aiosqlite.Connection) -> None:
    """Compact old snapshots: hourly after 7 days, expire after 90 days.

    Each old (project, hour) keeps ONE row: its latest measured row when it
    has one, else its latest gap, so a gap never displaces a measurement.
    """
    now = datetime.now(UTC)  # clock-exempt: single-capture writer
    cutoff_7d = (now - timedelta(days=7)).isoformat()
    cutoff_90d = (now - timedelta(days=90)).isoformat()
    measured = _measured_rows(await _snapshot_columns(conn))

    # Phase 1: For rows older than 7 days, keep one per (project_id, hour).
    await conn.execute(
        f"""
        DELETE FROM snapshots
        WHERE ts < ?
          AND id NOT IN (
              SELECT id FROM (
                  SELECT id, ROW_NUMBER() OVER (
                      PARTITION BY project_id, strftime('%Y-%m-%dT%H', ts)
                      ORDER BY CASE WHEN {measured.sql} THEN 0 ELSE 1 END, ts DESC
                  ) AS rn
                  FROM snapshots
                  WHERE ts < ?
              )
              WHERE rn = 1
          )
        """,
        (cutoff_7d, *measured.params, cutoff_7d),
    )

    # Phase 2: Delete everything older than 90 days.
    await conn.execute('DELETE FROM snapshots WHERE ts < ?', (cutoff_90d,))

    await conn.commit()


# ---------------------------------------------------------------------------
# Read-side queries (used by route handlers via DbPool read-only connections)
# ---------------------------------------------------------------------------

# The per-row keys of a burndown series, in order, beside ``labels``. The first
# six are the display zones; the split partitions in_progress; concurrency_cap
# is a per-snapshot scalar (nullable); the last four are task 5591's and read
# None on any row written before them.
_ZONES = ('done', 'cancelled', 'blocked', 'deferred', 'in_progress', 'pending')
_SPLIT = ('in_progress_live', 'in_progress_stranded')
_NULLABLE_SERIES_KEYS = (
    'concurrency_cap', 'review', 'merge_deferred', 'infra_hold', 'in_progress_rows',
)
_SERIES_KEYS = (*_ZONES, *_SPLIT, *_NULLABLE_SERIES_KEYS)


async def aggregate_burndown_projects(
    dbs: list[aiosqlite.Connection | None],
) -> list[str]:
    """Return distinct project IDs across *all* burndown DBs, sorted.

    Calls :func:`get_burndown_projects` for each DB in *dbs* concurrently via
    ``asyncio.gather``, then unions the results and returns a sorted list.
    ``None`` entries are tolerated (``get_burndown_projects`` returns ``[]``
    for ``None``).
    """
    if not dbs:
        return []
    per_db: list[list[str]] = list(await asyncio.gather(
        *(get_burndown_projects(db) for db in dbs)
    ))
    seen: set[str] = set()
    for project_list in per_db:
        seen.update(project_list)
    return sorted(seen)


async def aggregate_burndown_series(
    dbs: list[aiosqlite.Connection | None],
    project_id: str,
    *,
    days: int = 7,
    now: datetime | None = None,
) -> dict:
    """Return merged time-series data for *project_id* across all burndown DBs.

    Calls :func:`get_burndown_series` for each DB in *dbs* concurrently, then
    merges by timestamp label using a last-writer-wins strategy (later DBs in
    the list overwrite earlier ones for the same timestamp).  The result is
    sorted by timestamp and returned in the same dict-of-lists shape as
    :func:`get_burndown_series`.

    *now* is resolved once (via :func:`dashboard.data.utils.resolve_now`) and
    threaded identically to every per-DB call so all queries share a single
    cutoff, closing the per-DB clock-skew race.

    Returns the empty-series default ``{labels: [], done: [], ...}`` when no
    rows are found across any DB.
    """
    # Every series key merges last-writer-wins with the zones — a snapshot is
    # one consistent observation, so its columns travel together.
    empty: dict = {'labels': [], **{k: [] for k in _SERIES_KEYS}}

    if not dbs:
        return empty

    now = resolve_now(now)
    per_db: list[dict] = list(await asyncio.gather(
        *(get_burndown_series(db, project_id, days=days, now=now) for db in dbs)
    ))

    # Why last-writer-wins and not sum-of-counts (as used in performance.py /
    # costs.py)?  Burndown values are *snapshots* of task state at an instant,
    # not counts of independent events.  Summing two snapshots recorded at the
    # same timestamp would double-count every task state.  In the expected
    # deployment a single collector writes all projects to the main DB, so two
    # distinct burndown DBs should never share a (project_id, timestamp) pair.
    # If they do (misconfigured overlapping roots), the later DB's values win
    # and a warning is emitted so the misconfiguration is visible in logs.
    merged: dict[str, dict[str, int]] = {}
    collisions = 0
    first_colliding: list[str] = []
    for series in per_db:
        for i, label in enumerate(series['labels']):
            if label in merged:
                collisions += 1
                if len(first_colliding) < 3:
                    first_colliding.append(label)
            merged[label] = {k: series[k][i] for k in _SERIES_KEYS}
    if collisions:
        logger.warning(
            'aggregate_burndown_series: %d timestamp collisions for project %r '
            '— last-writer-wins applied; check for overlapping project roots '
            '(first: %s)',
            collisions,
            project_id,
            ', '.join(first_colliding),
        )

    if not merged:
        return empty

    sorted_labels = sorted(merged)
    result: dict = {'labels': sorted_labels}
    for k in _SERIES_KEYS:
        result[k] = [merged[label][k] for label in sorted_labels]
    return result


async def get_burndown_projects(db: aiosqlite.Connection | None) -> list[str]:
    """Return distinct project IDs that have at least one MEASURED snapshot row.

    A project whose only rows are gaps stays off this list: how a gap renders
    is for the shaper to decide (task 5592), not for this read to imply.
    """
    if db is None:
        return []
    try:
        measured = _measured_rows(await _snapshot_columns(db))
        async with db.execute(
            f'SELECT DISTINCT project_id FROM snapshots WHERE {measured.sql} '
            'ORDER BY project_id',
            measured.params,
        ) as cur:
            rows = await cur.fetchall()
        return [row[0] for row in rows]
    except Exception:
        logger.warning('Error fetching burndown projects', exc_info=True)
        return []


_VELOCITY_FLOOR = 0.1  # tasks/day; prevents forecast from going to infinity


def compute_forecast_confidence(series: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``{forecast_low, forecast_high}`` days from a burndown series.

    Reads ``done`` and ``pending`` over the series' ``labels`` (each label
    is a snapshot timestamp).  Computes two velocities — one over the most
    recent 7 days of data, one over the full series — and returns the
    forecast clearance window.

    Returns ``{forecast_low: None, forecast_high: None}`` when the series
    has fewer than 7 distinct days of history (no synthesis on sparse
    history; the dashboard renders ``—``).
    """
    none = {'forecast_low': None, 'forecast_high': None}
    labels = series.get('labels') or []
    done = series.get('done') or []
    pending = series.get('pending') or []
    if not labels or not done or not pending:
        return none
    if len(labels) != len(done) or len(labels) != len(pending):
        return none

    # Distinct day count from ISO labels (date prefix).
    day_keys: list[str] = []
    seen: set[str] = set()
    for lbl in labels:
        day = (lbl[:10] if isinstance(lbl, str) and len(lbl) >= 10 else None)
        if day and day not in seen:
            seen.add(day)
            day_keys.append(day)
    if len(day_keys) < 7:
        return none

    last_pending = pending[-1] or 0
    if last_pending <= 0:
        return {'forecast_low': 0, 'forecast_high': 0}

    last_done = done[-1] or 0

    def _velocity_over(window_days: int) -> float:
        cutoff = day_keys[-window_days]
        # First label whose day >= cutoff.
        first_idx = 0
        for i, lbl in enumerate(labels):
            if isinstance(lbl, str) and len(lbl) >= 10 and lbl[:10] >= cutoff:
                first_idx = i
                break
        first_done = done[first_idx] or 0
        delta_done = max(0, last_done - first_done)
        # Duration in days, with a minimum of 1.0 to avoid div-by-zero on
        # narrow windows.
        return max(_VELOCITY_FLOOR, delta_done / max(1.0, window_days))

    v_recent = _velocity_over(7)
    v_lifetime = _velocity_over(len(day_keys))

    f_recent = last_pending / v_recent
    f_lifetime = last_pending / v_lifetime
    return {
        'forecast_low': round(min(f_recent, f_lifetime), 1),
        'forecast_high': round(max(f_recent, f_lifetime), 1),
    }


def distinct_iso_days(labels: list[Any]) -> int:
    """Count distinct ISO-date prefixes in *labels* (min 1 to avoid div-by-zero)."""
    seen: set[str] = set()
    for lbl in labels:
        day = (lbl[:10] if isinstance(lbl, str) and len(lbl) >= 10 else None)
        if day:
            seen.add(day)
    return max(1, len(seen))


def compute_parity_alarm(series: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``{parity_alarm, parity_cap, parity_peak, parity_breach_count}``.

    Answers one question about a burndown series: did in-progress work ever
    exceed the orchestrator's own concurrency cap?  That is the E12 defect —
    the cap was breached and every surface reported ordinary throughput.

    **Each snapshot is judged against the cap stored on THAT snapshot**, never
    against one "current" cap applied across history.  ``max_concurrent_tasks``
    is restart-only (red-tier), but a burndown window spans restarts, so it is
    still TIME-VARYING across the window: re-deriving a single cap would forgive
    a real past breach after a cap raise, and invent a fictional one after a cut.

    Semantics:

    * ``parity_breach_count`` — snapshots where ``in_progress > cap``.  The cap
      is INCLUSIVE, so running exactly at capacity is healthy, not an alarm.
    * ``parity_alarm`` — ``breach_count > 0``.
    * ``parity_peak`` / ``parity_cap`` — a MATCHED PAIR read off ONE snapshot,
      never assembled from two.  The cap can change mid-window (a restart), so
      a peak from one snapshot beside a cap from another misstates the severity
      of the very breach it is describing (50-over-40 rendered as 50-over-24),
      and in a healthy window it manufactures a breach that never happened
      (peak 33 / cap 24 printed next to ``parity_alarm: False``).

      The snapshot chosen is:

      - when the series breaches — the breaching snapshot with the WIDEST
        margin ``count - cap`` (ties broken toward the larger count), i.e. the
        worst moment and the cap that was actually in force at it;
      - otherwise — the snapshot with the highest ``in_progress`` (ties broken
        toward the tighter cap), i.e. the literal peak of a healthy window and
        the cap it was actually measured against.

      Capless snapshots are excluded from both: a peak means nothing without
      the cap it is compared to, and mixing them in would show a number the
      displayed cap cannot explain.

    A series with no cap anywhere is UNKNOWN, not "not breaching": the alarm
    stays False but ``parity_cap``/``parity_peak`` are ``None`` so a caller can
    tell the two apart, and the collapse is logged rather than left silent.

    Pure: no clock, no I/O, no mutation of *series*.  Ragged or non-numeric
    input degrades to the quiet result rather than raising inside a route.
    """
    in_progress = list(series.get('in_progress') or [])
    caps = list(series.get('concurrency_cap') or [])

    breaches = 0
    comparable = 0
    # Both candidates are (count, cap) pairs read off a SINGLE snapshot, so
    # whichever one is published cannot misdescribe the other half.
    worst_breach: tuple[int, int] | None = None   # widest margin among breaches
    highest: tuple[int, int] | None = None        # highest count overall
    for count, cap in zip(in_progress, caps, strict=False):
        if not isinstance(count, int) or isinstance(count, bool):
            continue
        if not isinstance(cap, int) or isinstance(cap, bool):
            continue
        comparable += 1
        if highest is None or (count, -cap) > (highest[0], -highest[1]):
            highest = (count, cap)
        if count > cap:
            breaches += 1
            margin = count - cap
            if worst_breach is None or (margin, count) > (
                worst_breach[0] - worst_breach[1], worst_breach[0]
            ):
                worst_breach = (count, cap)

    chosen = worst_breach if worst_breach is not None else highest
    peak, cap_at_peak = chosen if chosen is not None else (None, None)

    if not comparable:
        logger.debug(
            'compute_parity_alarm: no snapshot carries a concurrency cap over '
            '%d label(s); reporting unknown rather than "not breaching"',
            len(in_progress),
        )

    return {
        'parity_alarm': breaches > 0,
        'parity_cap': cap_at_peak,
        'parity_peak': peak,
        'parity_breach_count': breaches,
    }


def compute_window_completion(series: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``{completed, velocity, window_days}`` from a burndown series.

    ``completed`` is ``max(0, done[-1] - done[0])``, i.e. the net increase in
    the done-count over the window.  Negative deltas (tasks reopened) are
    clamped to 0.

    ``velocity`` is ``completed / distinct_day_count`` where distinct days are
    derived from ISO date prefixes of *labels* (same convention as
    :func:`compute_forecast_confidence`).  Returns 0.0 if the series has fewer
    than 2 snapshots.

    ``window_days`` is the distinct-day count used as the denominator (0 when
    the series has no usable delta, i.e. on empty / mismatched / single-
    snapshot input).

    Returns ``{completed: 0, velocity: 0.0, window_days: 0}`` on empty /
    mismatched / single-snapshot input.
    """
    zero: dict[str, Any] = {'completed': 0, 'velocity': 0.0, 'window_days': 0}
    labels = list(series.get('labels') or [])
    done = list(series.get('done') or [])
    if not labels or not done:
        return zero
    if len(labels) != len(done):
        return zero
    if len(labels) < 2:
        return zero  # single snapshot → no meaningful delta

    completed = max(0, (done[-1] or 0) - (done[0] or 0))
    day_count = distinct_iso_days(labels)
    velocity = completed / day_count
    return {'completed': completed, 'velocity': velocity, 'window_days': day_count}


def aggregate_window_completion(
    per_project: Mapping[str, Mapping[str, Any]],
    sorted_labels: list[Any],
) -> dict[str, Any]:
    """Return ``{completed, velocity, window_days}`` for the aggregate burndown.

    Uses the **sum-of-per-project-deltas** approach: each project contributes
    ``p['completed']`` (already clamped by :func:`compute_window_completion`),
    and the results are summed.  This is robust to projects entering mid-window
    — computing a delta on the aggregate ``done`` series would over-count tasks
    from projects whose first snapshot appears after the window start.

    ``window_days`` is the distinct-day count across all ``sorted_labels``
    (the union of all project label sets), used as the aggregate velocity
    denominator.
    """
    completed = sum(p.get('completed', 0) for p in per_project.values())
    window_days = distinct_iso_days(sorted_labels)
    velocity = completed / window_days
    return {'completed': completed, 'velocity': velocity, 'window_days': window_days}


async def get_burndown_series(
    db: aiosqlite.Connection | None,
    project_id: str,
    *,
    days: int = 7,
    now: datetime | None = None,
) -> dict:
    """Return time-series data for a project's burndown chart.

    Returns ``{labels: [...]}`` plus one list per :data:`_SERIES_KEYS` key,
    over the MEASURED rows only: a gap row's count columns carry no
    measurement and never reach a reader.

    *now* is the reference timestamp for the window cutoff; when ``None``
    (the default) it is resolved via :func:`dashboard.data.utils.resolve_now`.
    """
    empty: dict = {'labels': [], **{key: [] for key in _SERIES_KEYS}}
    if db is None:
        return empty
    since = (resolve_now(now) - timedelta(days=days)).isoformat()
    try:
        # Which of the later columns this DB actually has: a hardcoded widened
        # SELECT would raise 'no such column' on an un-migrated peer DB, hit the
        # guard below, and silently blank that project's entire chart — losing
        # the six zones it DOES have to report columns it does not.
        available = await _snapshot_columns(db)
        has_split = set(_SPLIT) <= available
        columns = [
            'ts', *_ZONES,
            *(_SPLIT if has_split else ()),
            *(key for key in _NULLABLE_SERIES_KEYS if key in available),
        ]
        measured = _measured_rows(available)
        async with db.execute(
            f'SELECT {", ".join(columns)} FROM snapshots '
            f'WHERE project_id = ? AND ts >= ? AND {measured.sql} ORDER BY ts',
            (project_id, since, *measured.params),
        ) as cur:
            rows = await cur.fetchall()
    except Exception:
        logger.warning('Error fetching burndown series', exc_info=True)
        return empty

    result: dict = {key: [] for key in empty}
    for row in rows:
        values = dict(zip(columns, row, strict=True))
        result['labels'].append(values['ts'])
        for key in _ZONES:
            result[key].append(values[key])
        if has_split:
            for key in _SPLIT:
                result[key].append(values[key])
        else:
            # Un-migrated DB: the split is unknown.  All-live keeps the
            # conservation invariant (live + stranded == in_progress) true and
            # errs toward under-reporting strands, never over-reporting.
            result['in_progress_live'].append(values['in_progress'])
            result['in_progress_stranded'].append(0)
        # A missing column is UNKNOWN, i.e. None — never 0.  For the cap, a 0
        # would read as a cap of zero and alarm on every row; for the task
        # 5591 members, as a measured zero nobody counted.
        for key in _NULLABLE_SERIES_KEYS:
            result[key].append(values.get(key))

    return result
