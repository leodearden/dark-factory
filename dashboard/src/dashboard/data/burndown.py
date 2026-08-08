"""Burndown snapshot collection, downsampling, and chart queries.

Periodically snapshots task status counts per project into a SQLite table.
The background collector (in app.py lifespan) writes via a dedicated writable
connection; route handlers read via DbPool (read-only).

Each snapshot row carries three things beyond the six display zones
(task 3543 / PRD ι, spec S8/E12):

* ``in_progress_live`` / ``in_progress_stranded`` — a partition of the
  ``in_progress`` zone, so a window full of strands is legible as such
  instead of reading as healthy throughput.
* ``concurrency_cap`` — ``max_concurrent_tasks`` as it stood AT SNAPSHOT
  TIME, ``NULL`` when unknown.

The collector's single source is ``fetch_tasks`` (MCP ``get_tasks``): the
compact ``fetch_statuses`` map carries no claimant columns, so the split
cannot be derived from it at all.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

import aiosqlite
import httpx

from dashboard.config import DashboardConfig
from dashboard.data.orchestrator import (
    _read_project_root_from_config,
    _resolve_project_root,
    find_running_orchestrators,
    read_max_concurrent_tasks,
)
from dashboard.data.tasks import fetch_tasks, task_is_stranded
from dashboard.data.utils import resolve_now

logger = logging.getLogger(__name__)

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
    -- The in_progress zone's live/stranded split (task 3543). Partitions
    -- in_progress; it does not add to the six zones.
    in_progress_live     INTEGER NOT NULL DEFAULT 0,
    in_progress_stranded INTEGER NOT NULL DEFAULT 0,
    -- max_concurrent_tasks in force AT SNAPSHOT TIME. Nullable on purpose:
    -- NULL means "cap unknown", which is not a breach. Storing 0 instead would
    -- read as a cap of zero and alarm on every snapshot. The cap is green-tier
    -- hot-reloadable, so the only honest denominator for a historical row is
    -- the cap that was in force at that instant.
    concurrency_cap      INTEGER
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
)

# Maps raw task statuses to the 6 display zones.
_STATUS_MAP: dict[str, str] = {
    'pending': 'pending',
    'in-progress': 'in_progress',
    'review': 'in_progress',
    'blocked': 'blocked',
    'deferred': 'deferred',
    'cancelled': 'cancelled',
    'done': 'done',
}

_ZONE_KEYS = ('pending', 'in_progress', 'blocked', 'deferred', 'cancelled', 'done')

# The in_progress zone's live/stranded split (task 3543 / PRD ι, spec S8).
# First-class snapshot columns, but NOT display zones: they partition
# ``in_progress`` rather than sitting alongside it, so summing _ZONE_KEYS still
# yields the task total.
_SPLIT_KEYS = ('in_progress_live', 'in_progress_stranded')

_INSERT_SNAPSHOT_SQL = (
    'INSERT INTO snapshots (project_id, ts, pending, in_progress, blocked, deferred, cancelled, done, '
    'in_progress_live, in_progress_stranded, concurrency_cap) '
    'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'
)


async def ensure_snapshot_columns(conn: aiosqlite.Connection) -> None:
    """Bring an existing ``snapshots`` table up to the current column set.

    Additive only: probes ``PRAGMA table_info`` and issues one
    ``ALTER TABLE ... ADD COLUMN`` per missing column.  Idempotent, never drops
    or rewrites anything, so pre-existing rows keep their data and take the
    column defaults (``0`` for the split, ``NULL`` for the cap — an honest
    "unknown", since no cap was recorded at the time).

    Mandatory, not cosmetic: :data:`BURNDOWN_SCHEMA` is applied with
    ``CREATE TABLE IF NOT EXISTS``, which is a no-op against every already
    deployed burndown.db.  Without this, the widened
    :data:`_INSERT_SNAPSHOT_SQL` would fail on the first collection cycle after
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


def _count_statuses(statuses: dict[int, str | None]) -> dict[str, int]:
    """Count statuses (id → status) by mapped display zone.

    Superseded by :func:`_count_zones` for the collector, which needs the
    claimant columns that the ``{id: status}`` map does not carry.  Retained
    for callers that only have statuses to hand.
    """
    counts: dict[str, int] = {k: 0 for k in _ZONE_KEYS}
    for raw in statuses.values():
        zone = _STATUS_MAP.get(raw or 'pending', 'pending')
        counts[zone] += 1
    return counts


def _count_zones(tasks: list[Mapping[str, Any]], now: datetime) -> dict[str, int]:
    """Count shaped task rows by display zone, splitting ``in_progress``.

    Returns the six :data:`_ZONE_KEYS` counts plus :data:`_SPLIT_KEYS`
    (``in_progress_live`` / ``in_progress_stranded``), which partition the
    ``in_progress`` zone rather than adding to it.

    Takes shaped task rows (as ``dashboard.data.tasks._shape_task`` emits) and
    NOT the ``{id: status}`` map, because the split needs the claimant columns
    that only ``get_tasks`` carries.

    *now* is the reference instant for the strand verdict, supplied by the
    caller (the collector captures one instant per cycle) — this function never
    reads the clock.

    **The split is a subtraction, deliberately.** ``in_progress_stranded`` is
    counted directly via :func:`dashboard.data.tasks.task_is_stranded`, and
    ``in_progress_live`` is then ``zone_total - stranded``.  Never derive
    ``live`` independently (e.g. via ``has_live_claimant``, which carries
    neither the status gate nor the ``metadata.infra_hold`` carve-out): only
    the subtraction makes the conservation invariant
    ``live + stranded == in_progress`` true by construction, and that invariant
    is what makes the stacked chart and the parity alarm auditable.

    A consequence worth naming: ``_STATUS_MAP`` folds BOTH 'in-progress' and
    'review' into the ``in_progress`` zone, but ``is_stranded`` hard-gates on
    ``status == 'in-progress'`` and can never fire for a 'review' row.  A
    claimant-less review row therefore lands in ``live``.  That under-reports
    strands and never over-reports them, which is the correct direction to err
    for a surface that raises alarms.
    """
    counts: dict[str, int] = {k: 0 for k in (*_ZONE_KEYS, *_SPLIT_KEYS)}
    for task in tasks:
        zone = _STATUS_MAP.get(task.get('status') or 'pending', 'pending')
        counts[zone] += 1
        if task_is_stranded(task, now=now):
            counts['in_progress_stranded'] += 1
    counts['in_progress_live'] = counts['in_progress'] - counts['in_progress_stranded']
    return counts


async def collect_snapshot(
    conn: aiosqlite.Connection,
    config: DashboardConfig,
    client: httpx.AsyncClient,
) -> None:
    """Discover projects and insert one snapshot row per project.

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
        # One instant for the whole cycle: the ``ts`` written to every row and
        # the reference the strand verdicts are judged against must be the SAME
        # capture, or a row could claim a strand count that its own timestamp
        # contradicts.
        now = now_dt.isoformat()
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

        # Phase 2 — Parallel read:
        # Each fetch_tasks call hits fused-memory MCP independently;
        # return_exceptions=True isolates per-project network failures so a
        # single offline server can't sink the whole cycle.
        #
        # SOURCE: fetch_tasks (MCP get_tasks), NOT the ~95%-smaller
        # fetch_statuses (get_statuses).  get_statuses returns a bare
        # {id: status} map with no claimant columns, so the live/stranded
        # split is physically underivable from it.  The collector runs once
        # per _SAMPLE_INTERVAL_SECONDS (600s), so the larger payload is a
        # ten-minute cost, not a per-render one — and it collapses the
        # collector onto ONE source instead of two that can disagree.
        all_results = await asyncio.gather(
            *(fetch_tasks(client, config, root) for root in roots_to_snapshot),
            return_exceptions=True,
        )

        # Phase 3 — Sequential insert (per-project commit):
        # aiosqlite serialises writes on a single connection; keep inserts sequential.
        # Each project gets its own INSERT + commit so a DB failure on one project
        # cannot roll back rows that were already committed for earlier projects.
        for root_str, result in zip(roots_to_snapshot, all_results, strict=True):
            if isinstance(result, BaseException):
                logger.warning('Failed to fetch tasks for %s', root_str, exc_info=result)
                continue
            # The offline marker is a dict, a healthy result is a list — check
            # the marker FIRST so a known outage is reported as an outage
            # rather than as a malformed result.
            if isinstance(result, dict) and result.get('offline'):
                logger.debug('fetch_tasks offline for %s: %s', root_str, result.get('error'))
                continue
            if not isinstance(result, list):
                logger.warning('Unexpected fetch_tasks result for %s: %r', root_str, type(result))
                continue
            tasks = [t for t in result if isinstance(t, Mapping)]
            try:
                counts = _count_zones(tasks, now_dt)
                # One cap read per ROOT (not per task): it touches the
                # filesystem, and the value is a per-project scalar.  Stored on
                # the row because max_concurrent_tasks is hot-reloadable — see
                # BURNDOWN_SCHEMA's concurrency_cap comment.
                cap = read_max_concurrent_tasks(root_str)
                if cap is not None and counts['in_progress'] > cap:
                    logger.warning(
                        'Concurrency cap breached for %s: %d in-progress vs cap %d '
                        '(%d stranded)',
                        root_str,
                        counts['in_progress'],
                        cap,
                        counts['in_progress_stranded'],
                    )
                await conn.execute(
                    _INSERT_SNAPSHOT_SQL,
                    (
                        root_str,
                        now,
                        counts['pending'],
                        counts['in_progress'],
                        counts['blocked'],
                        counts['deferred'],
                        counts['cancelled'],
                        counts['done'],
                        counts['in_progress_live'],
                        counts['in_progress_stranded'],
                        # NULL, never 0, when the cap is unknown: a stored 0
                        # would read as a cap of zero and alarm on every row.
                        cap,
                    ),
                )
                await conn.commit()
            except Exception:
                logger.warning('Failed to insert snapshot for %s', root_str, exc_info=True)
                with contextlib.suppress(Exception):
                    await conn.rollback()
    except Exception:
        with contextlib.suppress(Exception):
            await conn.rollback()
        raise


async def downsample(conn: aiosqlite.Connection) -> None:
    """Compact old snapshots: hourly after 7 days, expire after 90 days."""
    now = datetime.now(UTC)  # clock-exempt: single-capture writer
    cutoff_7d = (now - timedelta(days=7)).isoformat()
    cutoff_90d = (now - timedelta(days=90)).isoformat()

    # Phase 1: For rows older than 7 days, keep only the last per (project_id, hour).
    await conn.execute(
        """
        DELETE FROM snapshots
        WHERE ts < ?
          AND id NOT IN (
              SELECT id FROM (
                  SELECT id, ROW_NUMBER() OVER (
                      PARTITION BY project_id, strftime('%%Y-%%m-%%dT%%H', ts)
                      ORDER BY ts DESC
                  ) AS rn
                  FROM snapshots
                  WHERE ts < ?
              )
              WHERE rn = 1
          )
        """,
        (cutoff_7d, cutoff_7d),
    )

    # Phase 2: Delete everything older than 90 days.
    await conn.execute('DELETE FROM snapshots WHERE ts < ?', (cutoff_90d,))

    await conn.commit()


# ---------------------------------------------------------------------------
# Read-side queries (used by route handlers via DbPool read-only connections)
# ---------------------------------------------------------------------------


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
    _keys = (
        'done', 'cancelled', 'blocked', 'deferred', 'in_progress', 'pending',
        # The split partitions in_progress; concurrency_cap is a per-snapshot
        # scalar (nullable).  Both merge last-writer-wins with the zones — a
        # snapshot is one consistent observation, so its columns travel together.
        'in_progress_live', 'in_progress_stranded', 'concurrency_cap',
    )
    empty: dict = {'labels': [], **{k: [] for k in _keys}}

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
            merged[label] = {k: series[k][i] for k in _keys}
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
    for k in _keys:
        result[k] = [merged[label][k] for label in sorted_labels]
    return result


async def get_burndown_projects(db: aiosqlite.Connection | None) -> list[str]:
    """Return distinct project IDs that have snapshot data."""
    if db is None:
        return []
    try:
        async with db.execute('SELECT DISTINCT project_id FROM snapshots ORDER BY project_id') as cur:
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
    is green-tier hot-reloadable: re-deriving a single cap would forgive a real
    past breach after a cap raise, and invent a fictional one after a cap cut.

    Semantics:

    * ``parity_breach_count`` — snapshots where ``in_progress > cap``.  The cap
      is INCLUSIVE, so running exactly at capacity is healthy, not an alarm.
    * ``parity_alarm`` — ``breach_count > 0``.
    * ``parity_cap`` — the most recent KNOWN cap in the series, for display
      next to the peak.  ``None`` when no snapshot carries one.
    * ``parity_peak`` — the highest ``in_progress`` among snapshots that carry
      a cap.  Capless snapshots are excluded: a peak means nothing without the
      cap it is measured against, and mixing them in would show a number the
      displayed cap cannot explain.

    A series with no cap anywhere is UNKNOWN, not "not breaching": the alarm
    stays False but ``parity_cap``/``parity_peak`` are ``None`` so a caller can
    tell the two apart, and the collapse is logged rather than left silent.

    Pure: no clock, no I/O, no mutation of *series*.  Ragged or non-numeric
    input degrades to the quiet result rather than raising inside a route.
    """
    in_progress = list(series.get('in_progress') or [])
    caps = list(series.get('concurrency_cap') or [])

    peak: int | None = None
    latest_cap: int | None = None
    breaches = 0
    comparable = 0
    for count, cap in zip(in_progress, caps, strict=False):
        if not isinstance(count, int) or isinstance(count, bool):
            continue
        if not isinstance(cap, int) or isinstance(cap, bool):
            continue
        comparable += 1
        latest_cap = cap
        if peak is None or count > peak:
            peak = count
        if count > cap:
            breaches += 1

    if not comparable:
        logger.debug(
            'compute_parity_alarm: no snapshot carries a concurrency cap over '
            '%d label(s); reporting unknown rather than "not breaching"',
            len(in_progress),
        )

    return {
        'parity_alarm': breaches > 0,
        'parity_cap': latest_cap,
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

    Returns ``{labels: [...], done: [...], cancelled: [...], blocked: [...],
    deferred: [...], in_progress: [...], pending: [...]}``.

    *now* is the reference timestamp for the window cutoff; when ``None``
    (the default) it is resolved via :func:`dashboard.data.utils.resolve_now`.
    """
    empty: dict = {
        'labels': [],
        'done': [],
        'cancelled': [],
        'blocked': [],
        'deferred': [],
        'in_progress': [],
        'pending': [],
        'in_progress_live': [],
        'in_progress_stranded': [],
        'concurrency_cap': [],
    }
    if db is None:
        return empty
    since = (resolve_now(now) - timedelta(days=days)).isoformat()
    try:
        # Which of the post-3543 columns this DB actually has.  ``_burndown_dbs``
        # opens OTHER projects' burndown.db files read-only and nothing migrates
        # those, so a hardcoded widened SELECT would raise 'no such column'
        # there, hit the guard below, and silently blank that project's entire
        # chart — losing the six zones it DOES have to report three it does not.
        async with db.execute('PRAGMA table_info(snapshots)') as cur:
            available = {row[1] for row in await cur.fetchall()}
        has_split = {'in_progress_live', 'in_progress_stranded'} <= available
        has_cap = 'concurrency_cap' in available

        columns = ['ts', 'done', 'cancelled', 'blocked', 'deferred', 'in_progress', 'pending']
        if has_split:
            columns += ['in_progress_live', 'in_progress_stranded']
        if has_cap:
            columns.append('concurrency_cap')
        async with db.execute(
            f'SELECT {", ".join(columns)} '
            'FROM snapshots WHERE project_id = ? AND ts >= ? ORDER BY ts',
            (project_id, since),
        ) as cur:
            rows = await cur.fetchall()
    except Exception:
        logger.warning('Error fetching burndown series', exc_info=True)
        return empty

    result: dict = {key: [] for key in empty}
    for row in rows:
        result['labels'].append(row[0])
        result['done'].append(row[1])
        result['cancelled'].append(row[2])
        result['blocked'].append(row[3])
        result['deferred'].append(row[4])
        result['in_progress'].append(row[5])
        result['pending'].append(row[6])
        if has_split:
            result['in_progress_live'].append(row[7])
            result['in_progress_stranded'].append(row[8])
        else:
            # Un-migrated DB: the split is unknown.  All-live keeps the
            # conservation invariant (live + stranded == in_progress) true and
            # errs toward under-reporting strands, never over-reporting.
            result['in_progress_live'].append(row[5])
            result['in_progress_stranded'].append(0)
        # A missing cap column is UNKNOWN, i.e. NULL — never 0, which would
        # read as a cap of zero and alarm on every row.
        result['concurrency_cap'].append(row[-1] if has_cap else None)

    return result
