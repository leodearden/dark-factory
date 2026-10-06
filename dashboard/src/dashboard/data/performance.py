"""Async queries for orchestrator performance metrics.

Reads from data/orchestrator/runs.db (task results) and
data/escalations/ (escalation JSON files) to produce per-project
performance statistics.
"""

from __future__ import annotations

import asyncio
import copy
import enum
import json
import logging
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, TypeVar

import aiosqlite
from escalation.queue import iter_all_escalation_paths
from shared.timestamps import parse_timestamp_or_warn

from dashboard.data.datum import Datum, DatumState
from dashboard.data.db import with_db
from dashboard.data.stats_utils import percentile
from dashboard.data.utils import resolve_now, safe_gather_result

logger = logging.getLogger(__name__)

_Tally = TypeVar('_Tally')


def _load_escalations(escalations_dir: Path) -> list[dict]:
    """Load all escalation JSON files from the queue root and archive subtree.

    Uses :func:`escalation.queue.iter_all_escalation_paths` to perform a
    two-tier scan: queue root first, then ``archive/YYYY-MM-DD/`` subdirs.
    Deduplication by filename stem is handled by the helper (root wins on
    id collisions).  A missing or non-directory *escalations_dir* yields an
    empty list without raising.
    """
    results: list[dict] = []
    for path in iter_all_escalation_paths(escalations_dir):
        try:
            results.append(json.loads(path.read_text()))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning('Failed to load escalation %s: %s', path, exc)
            continue
    return results


# ---------------------------------------------------------------------------
# Time-window helper
# ---------------------------------------------------------------------------


def _cutoff(days: int, *, now: datetime | None = None) -> str:
    """Return ISO-format cutoff datetime for the given look-back window.

    Local copy of :func:`dashboard.data.costs._cutoff` — kept independent
    rather than imported so this module has no cross-module dependency on
    another data module's private helper. The card families and the
    hour-bucketed sparklines both bind this one cutoff, so a card and its
    sparkline count the same rows of the same wall-clock window.

    The ISO-with-offset return value is load-bearing, not cosmetic: it is
    compared against ``task_results.completed_at``, which orchestrator
    writes via ``datetime.now(UTC).isoformat()`` (``run_store.py::save_run``).
    A SQLite-side ``datetime('now', ...)`` call renders SPACE-separated with
    no offset, so a lexical TEXT comparison against it short-circuits at
    index 10 on ``'T'`` (0x54) vs ``' '`` (0x20) and silently degrades to
    DATE granularity — over-including up to a full extra day (task 4624).
    """
    return (resolve_now(now) - timedelta(days=days)).isoformat()


# A cancel ends an attempt from outside (shutdown drain, operator cancel, takeover); the resumed attempt
# records its own row, so no reading counts one. Values of orchestrator/src/orchestrator/workflow_types.py::WorkflowOutcome.
_NOT_A_CANCEL = "outcome NOT IN ('cancelled', 'soft-cancelled')"


async def _latest_completions(db: aiosqlite.Connection) -> dict[str, str]:
    """Return ``{project_id: MAX(completed_at)}`` for every project with a
    recorded completion, a cancel not counting (see ``_NOT_A_CANCEL``) —
    the projects a card family tallies by default."""
    rows = await db.execute_fetchall(
        'SELECT project_id, MAX(completed_at) '
        '  FROM task_results '
        " WHERE completed_at IS NOT NULL AND completed_at != '' "
        f'   AND {_NOT_A_CANCEL} '
        ' GROUP BY project_id',
    )
    return {row[0]: row[1] for row in rows}


async def _window_rows_by_project(
    db: aiosqlite.Connection,
    sql: str,
    since: str,
    projects: Collection[str] | None,
) -> dict[str, list[tuple]]:
    """Each listed project's in-window rows of *sql*, without their leading ``project_id``.

    *sql* selects ``project_id`` first and binds two parameters: the listed
    projects as a JSON array for ``json_each``, then *since*. One query per
    call, and ``project_id IN (json_each)`` keeps ``idx_task_results_project``
    usable. Every listed project gets an entry, so a project with nothing in
    the window reads ``[]`` — its empty tally, never an absence. *projects*
    defaults to every project with a recorded completion in *db*.
    """
    listed = list(projects if projects is not None else await _latest_completions(db))
    by_project: dict[str, list[tuple]] = {project_id: [] for project_id in listed}
    for row in await db.execute_fetchall(sql, (json.dumps(listed), since)):
        by_project[row[0]].append(tuple(row)[1:])
    return by_project


# ---------------------------------------------------------------------------
# 1. Completion paths
# ---------------------------------------------------------------------------


async def get_completion_paths(
    db: aiosqlite.Connection | None,
    escalations_dir: Path,
    *,
    days: int = 7,
    now: datetime | None = None,
    projects: Collection[str] | None = None,
) -> dict[str, list[dict]]:
    """Per-project completion path breakdown.

    Returns {project_id: [{path: str, count: int, pct: float}, ...]} for each
    of *projects* (default: every project with a recorded completion).
    Paths: one-pass, multi-pass, via-steward, via-interactive, blocked; a
    cancel outcome is no path and is not counted (see ``_NOT_A_CANCEL``).
    """
    escalations = _load_escalations(escalations_dir)

    # Build set of task_ids that had level-1 escalations resolved
    interactive_task_ids: set[str] = set()
    for esc in escalations:
        if esc.get('level') == 1 and esc.get('status') in ('resolved', 'dismissed'):
            interactive_task_ids.add(str(esc.get('task_id', '')))

    since = _cutoff(days, now=now)

    async def _query(db: aiosqlite.Connection) -> dict[str, list[dict]]:
        result: dict[str, list[dict]] = {}
        rows_by_project = await _window_rows_by_project(
            db,
            'SELECT project_id, task_id, outcome, review_cycles, '
            '       steward_invocations '
            '  FROM task_results '
            ' WHERE project_id IN (SELECT value FROM json_each(?)) AND completed_at >= ? '
            f'  AND {_NOT_A_CANCEL} ',
            since,
            projects,
        )
        for project_id, rows in rows_by_project.items():
            counts: dict[str, int] = {
                'one-pass': 0,
                'multi-pass': 0,
                'via-steward': 0,
                'via-interactive': 0,
                'blocked': 0,
            }
            for row in rows:
                task_id, outcome, review_cycles, steward_inv = row
                task_id = str(task_id)

                if outcome != 'done':
                    counts['blocked'] += 1
                elif task_id in interactive_task_ids:
                    counts['via-interactive'] += 1
                elif steward_inv and steward_inv > 0:
                    counts['via-steward'] += 1
                elif review_cycles and review_cycles > 0:
                    counts['multi-pass'] += 1
                else:
                    counts['one-pass'] += 1

            total = sum(counts.values()) or 1
            result[project_id] = [
                {
                    'path': path,
                    'count': count,
                    'pct': round(count / total * 100, 1),
                }
                for path, count in counts.items()
                if count > 0
            ]

        return result

    return await with_db(db, _query, {})


# ===========================================================================
# Multi-DB aggregation
# ===========================================================================


def _per_db_listings(
    projects_by_db: Sequence[Collection[str]] | None, db_count: int,
) -> Sequence[Collection[str] | None]:
    """Each DB's ``projects=`` argument; ``None`` lets a DB tally all it holds."""
    return projects_by_db if projects_by_db is not None else [None] * db_count


def _drop_partial_tallies(
    results: Sequence[dict[str, _Tally]],
    projects_by_db: Sequence[Collection[str]] | None,
) -> list[dict[str, _Tally]]:
    """The one merge rule for a family's per-DB *results*: drop, from every
    result, each project some DB was listed for but returned no tally for.

    A healthy per-DB call tallies every project it is listed for (an idle one
    reads its empty tally), while a call that failed open under ``with_db``
    returns ``{}``. So after this a merged reading lists a project only when
    every DB holding it tallied it, and a partial tally is never served as the
    whole project's. With *projects_by_db* ``None`` the results are unchanged.
    """
    if projects_by_db is None:
        return list(results)
    partial = {
        project_id
        for result, listed in zip(results, projects_by_db, strict=True)
        for project_id in listed
        if project_id not in result
    }
    return [
        {project_id: tally for project_id, tally in result.items() if project_id not in partial}
        for result in results
    ]


async def aggregate_completion_paths(
    dbs: list[aiosqlite.Connection | None],
    escalations_dirs: list[Path],
    *,
    days: int = 7,
    now: datetime | None = None,
    projects_by_db: Sequence[Collection[str]] | None = None,
) -> dict[str, list[dict]]:
    """Merge :func:`get_completion_paths` results from multiple databases.

    ``dbs`` and ``escalations_dirs`` must have the same length — they are
    zipped with ``strict=True`` so a mismatch raises immediately.  Each
    element of ``escalations_dirs`` must be the escalation directory
    corresponding to the project root whose runs.db is ``dbs[i]``.
    ``projects_by_db[i]`` lists the projects ``dbs[i]`` tallies (default:
    every project it has a completion for); see :func:`_drop_partial_tallies`.

    Merging rule: for a given project_id the *count* for each completion
    path is summed across DBs; ``pct`` is recomputed from the new totals.

    Shape contract: every project_id that appears in any per-DB
    :func:`get_completion_paths` result kept by
    :func:`_drop_partial_tallies` is a key in the returned dict.  Its
    value list contains only paths whose merged count is > 0 (the
    ``if count > 0`` guard on the list comprehension), so the list may be
    empty if every per-DB result for that project had an empty list — the
    case of an idle project, one with completions ever but none in the
    window.

    Note: :func:`aggregate_escalation_rates` includes projects with
    ``total_tasks == 0`` (returning rates of 0.0).  The difference is
    intentional — an escalation-rate row for a zero-task project can still
    carry ``human_attention`` bucket values, whereas a completion-path list
    with no non-zero entries has nothing meaningful to render.
    """
    if not dbs:
        return {}

    now = resolve_now(now)
    listings = _per_db_listings(projects_by_db, len(dbs))
    results = _drop_partial_tallies(
        await asyncio.gather(
            *(
                get_completion_paths(db, edir, days=days, now=now, projects=projects)
                for db, edir, projects in zip(dbs, escalations_dirs, listings, strict=True)
            )
        ),
        projects_by_db,
    )

    # Merge: sum counts per project_id per path
    merged: dict[str, dict[str, int]] = {}
    for result in results:
        for pid, paths in result.items():
            if pid not in merged:
                merged[pid] = {}
            for entry in paths:
                path = entry['path']
                merged[pid][path] = merged[pid].get(path, 0) + entry['count']

    # Recompute pct from merged totals
    final: dict[str, list[dict]] = {}
    for pid, path_counts in merged.items():
        total = sum(path_counts.values()) or 1
        final[pid] = [
            {
                'path': path,
                'count': count,
                'pct': round(count / total * 100, 1),
            }
            for path, count in path_counts.items()
            if count > 0
        ]
    return final


async def aggregate_escalation_rates(
    dbs: list[aiosqlite.Connection | None],
    escalations_dirs: list[Path],
    *,
    days: int = 7,
    now: datetime | None = None,
    projects_by_db: Sequence[Collection[str]] | None = None,
) -> dict[str, dict]:
    """Merge :func:`get_escalation_rates` results from multiple databases.

    ``dbs`` and ``escalations_dirs`` must have the same length.
    ``projects_by_db[i]`` lists the projects ``dbs[i]`` tallies (default:
    every project it has a completion for); see :func:`_drop_partial_tallies`.
    For each project_id: total_tasks, steward_count, interactive_count, and
    each human_attention bucket are summed across DBs.  steward_rate and
    interactive_rate are recomputed from the merged totals (not averaged).
    """
    if not dbs:
        return {}

    now = resolve_now(now)
    listings = _per_db_listings(projects_by_db, len(dbs))
    results = _drop_partial_tallies(
        await asyncio.gather(
            *(
                get_escalation_rates(db, edir, days=days, now=now, projects=projects)
                for db, edir, projects in zip(dbs, escalations_dirs, listings, strict=True)
            )
        ),
        projects_by_db,
    )

    merged: dict[str, dict] = {}
    for result in results:
        for pid, info in result.items():
            if pid not in merged:
                merged[pid] = {
                    'total_tasks': 0,
                    'steward_count': 0,
                    'interactive_count': 0,
                    'human_attention': {'zero': 0, 'minimal': 0, 'significant': 0},
                }
            m = merged[pid]
            m['total_tasks'] += info['total_tasks']
            m['steward_count'] += info['steward_count']
            m['interactive_count'] += info['interactive_count']
            for bucket in ('zero', 'minimal', 'significant'):
                m['human_attention'][bucket] += info['human_attention'][bucket]

    # Recompute rates from merged totals
    for _pid, m in merged.items():
        total = m['total_tasks']
        m['steward_rate'] = round(m['steward_count'] / total * 100, 1) if total else 0.0
        m['interactive_rate'] = round(m['interactive_count'] / total * 100, 1) if total else 0.0
    return merged


async def aggregate_loop_histograms(
    dbs: list[aiosqlite.Connection | None],
    *,
    days: int = 7,
    now: datetime | None = None,
    projects_by_db: Sequence[Collection[str]] | None = None,
) -> dict[str, dict]:
    """Merge :func:`get_loop_histograms` results from multiple databases.

    ``projects_by_db[i]`` lists the projects ``dbs[i]`` tallies (default:
    every project it has a completion for); see :func:`_drop_partial_tallies`.
    For each project_id: merge outer.values and inner.values by label key
    across DBs.  In the canonical case all DBs return the same label lists
    (4 bins for outer, 6 for inner), so the merge is equivalent to the
    previous element-wise sum.

    If a DB returns a different label list for a key, a ``WARNING`` is logged
    once per (project_id, key) identifying the mismatched lists.  The merge
    still proceeds: values for known labels are summed, and any new labels
    from the incoming result are appended at the end (preserving the
    accumulator's original label order).
    """
    if not dbs:
        return {}

    now = resolve_now(now)
    listings = _per_db_listings(projects_by_db, len(dbs))
    results = _drop_partial_tallies(
        await asyncio.gather(
            *(
                get_loop_histograms(db, days=days, now=now, projects=projects)
                for db, projects in zip(dbs, listings, strict=True)
            ),
        ),
        projects_by_db,
    )

    merged: dict[str, dict] = {}
    for result in results:
        for pid, info in result.items():
            if pid not in merged:
                merged[pid] = copy.deepcopy(info)
            else:
                m = merged[pid]
                for key in ('outer', 'inner'):
                    if info[key]['labels'] == m[key]['labels']:
                        # Fast path: canonical case — label lists match, element-wise sum.
                        for i, val in enumerate(info[key]['values']):
                            m[key]['values'][i] += val
                    else:
                        # Mismatch branch: merge by label dict and warn once per (pid, key).
                        logger.warning(
                            'aggregate_loop_histograms: label list mismatch'
                            ' for project %r key %r'
                            ' (base=%r, incoming=%r)',
                            pid,
                            key,
                            m[key]['labels'],
                            info[key]['labels'],
                        )
                        label_map = dict(zip(m[key]['labels'], m[key]['values'], strict=True))
                        for lbl, val in zip(info[key]['labels'], info[key]['values'], strict=True):
                            label_map[lbl] = label_map.get(lbl, 0) + val
                        known = list(m[key]['labels'])
                        for lbl in info[key]['labels']:
                            if lbl not in known:
                                known.append(lbl)
                        m[key]['labels'] = known
                        m[key]['values'] = [label_map[lbl] for lbl in known]
    return merged


async def _durations_by_project(
    db: aiosqlite.Connection | None,
    *,
    days: int = 7,
    now: datetime | None = None,
    projects: Collection[str] | None = None,
) -> dict[str, list[int]]:
    """Return raw ``duration_ms`` lists per project_id (internal helper).

    Unlike :func:`get_time_centiles`, this function preserves the raw sample
    list so that :func:`aggregate_time_centiles` can concatenate samples from
    multiple DBs before computing percentiles on the unified distribution.
    """

    since = _cutoff(days, now=now)

    async def _query(db: aiosqlite.Connection) -> dict[str, list[int]]:
        result: dict[str, list[int]] = {}
        rows_by_project = await _window_rows_by_project(
            db,
            'SELECT project_id, duration_ms FROM task_results '
            ' WHERE project_id IN (SELECT value FROM json_each(?)) AND completed_at >= ? '
            "   AND outcome = 'done' "
            ' ORDER BY duration_ms ',
            since,
            projects,
        )
        for project_id, rows in rows_by_project.items():
            result[project_id] = [row[0] for row in rows if row[0] is not None and row[0] > 0]

        return result

    return await with_db(db, _query, {})


async def aggregate_time_centiles(
    dbs: list[aiosqlite.Connection | None],
    *,
    days: int = 7,
    now: datetime | None = None,
    projects_by_db: Sequence[Collection[str]] | None = None,
) -> dict[str, dict]:
    """Merge time-centile data from multiple databases.

    Calls :func:`_durations_by_project` per DB to collect raw duration
    samples, concatenates them per project_id, then computes p50/p75/p90/p95
    from the unified sample distribution so percentiles are exact (not
    averages of per-DB percentiles).  ``count`` is the total number of tasks
    across all DBs. ``projects_by_db[i]`` lists the projects ``dbs[i]``
    tallies (default: every project it has a completion for); see
    :func:`_drop_partial_tallies`.
    """
    if not dbs:
        return {}

    now = resolve_now(now)
    listings = _per_db_listings(projects_by_db, len(dbs))
    results = _drop_partial_tallies(
        await asyncio.gather(
            *(
                _durations_by_project(db, days=days, now=now, projects=projects)
                for db, projects in zip(dbs, listings, strict=True)
            ),
        ),
        projects_by_db,
    )

    # Merge: concatenate raw duration lists per project_id
    merged: dict[str, list[int]] = {}
    for result in results:
        for pid, durations in result.items():
            if pid not in merged:
                merged[pid] = []
            merged[pid].extend(durations)

    # Compute percentiles from the merged (concatenated) sample
    final: dict[str, dict] = {}
    for pid, durations in merged.items():
        if not durations:
            final[pid] = {'p50': 0, 'p75': 0, 'p90': 0, 'p95': 0, 'count': 0}
            continue
        durations.sort()
        final[pid] = {
            'p50': round(percentile(durations, 50)),
            'p75': round(percentile(durations, 75)),
            'p90': round(percentile(durations, 90)),
            'p95': round(percentile(durations, 95)),
            'count': len(durations),
        }
    return final


# ---------------------------------------------------------------------------
# 2. Escalation rates
# ---------------------------------------------------------------------------


async def get_escalation_rates(
    db: aiosqlite.Connection | None,
    escalations_dir: Path,
    *,
    days: int = 7,
    now: datetime | None = None,
    projects: Collection[str] | None = None,
) -> dict[str, dict]:
    """Per-project escalation rates and human attention breakdown.

    Returns {project_id: {total_tasks, steward_count, interactive_count,
                          steward_rate, interactive_rate,
                          human_attention: {zero, minimal, significant}}}
    for each of *projects* (default: every project with a recorded completion).
    """
    escalations = _load_escalations(escalations_dir)

    # Index escalations by task_id
    esc_by_task: dict[str, list[dict]] = defaultdict(list)
    for esc in escalations:
        tid = str(esc.get('task_id', ''))
        if tid:
            esc_by_task[tid].append(esc)

    since = _cutoff(days, now=now)

    async def _query(db: aiosqlite.Connection) -> dict[str, dict]:
        result: dict[str, dict] = {}
        rows_by_project = await _window_rows_by_project(
            db,
            'SELECT project_id, task_id, steward_invocations '
            '  FROM task_results '
            ' WHERE project_id IN (SELECT value FROM json_each(?)) AND completed_at >= ? '
            f'  AND {_NOT_A_CANCEL} ',
            since,
            projects,
        )
        for project_id, rows in rows_by_project.items():
            total = len(rows)
            steward_count = 0
            interactive_count = 0
            attention = {'zero': 0, 'minimal': 0, 'significant': 0}

            for row in rows:
                task_id = str(row[0])
                steward_inv = row[1] or 0

                if steward_inv > 0:
                    steward_count += 1

                task_escs = esc_by_task.get(task_id, [])
                has_interactive = any(
                    e.get('level') == 1 and e.get('status') in ('resolved', 'dismissed')
                    for e in task_escs
                )
                if has_interactive:
                    interactive_count += 1

                    # Classify human effort from resolution_turns
                    max_turns = max(
                        (e.get('resolution_turns') or 0) for e in task_escs if e.get('level') == 1
                    )
                    if max_turns == 0:
                        attention['zero'] += 1
                    elif max_turns <= 2:
                        attention['minimal'] += 1
                    else:
                        attention['significant'] += 1

            result[project_id] = {
                'total_tasks': total,
                'steward_count': steward_count,
                'interactive_count': interactive_count,
                'steward_rate': round(steward_count / total * 100, 1) if total else 0.0,
                'interactive_rate': round(interactive_count / total * 100, 1) if total else 0.0,
                'human_attention': attention,
            }

        return result

    return await with_db(db, _query, {})


# ---------------------------------------------------------------------------
# 3. Loop histograms
# ---------------------------------------------------------------------------


async def get_loop_histograms(
    db: aiosqlite.Connection | None,
    *,
    days: int = 7,
    now: datetime | None = None,
    projects: Collection[str] | None = None,
) -> dict[str, dict]:
    """Per-project loop cycle distributions.

    Returns {project_id: {
        outer: {labels: [str], values: [int]},
        inner: {labels: [str], values: [int]},
    }} for each of *projects* (default: every project with a recorded completion).
    Outer = review_cycles (0,1,2,3+). Inner = verify_attempts (0,1,2,3,4,5+).
    Filtered to outcome=done tasks only.
    """

    since = _cutoff(days, now=now)

    async def _query(db: aiosqlite.Connection) -> dict[str, dict]:
        result: dict[str, dict] = {}
        rows_by_project = await _window_rows_by_project(
            db,
            'SELECT project_id, review_cycles, verify_attempts '
            '  FROM task_results '
            ' WHERE project_id IN (SELECT value FROM json_each(?)) AND completed_at >= ? '
            "   AND outcome = 'done' ",
            since,
            projects,
        )
        for project_id, rows in rows_by_project.items():
            # Outer loop: review cycles (0, 1, 2, 3+)
            outer_bins = [0, 0, 0, 0]  # indices 0-3
            outer_labels = ['0', '1', '2', '3+']

            # Inner loop: verify attempts (0, 1, 2, 3, 4, 5+)
            inner_bins = [0, 0, 0, 0, 0, 0]  # indices 0-5
            inner_labels = ['0', '1', '2', '3', '4', '5+']

            for row in rows:
                rc = row[0] or 0
                va = row[1] or 0

                outer_bins[min(rc, 3)] += 1
                inner_bins[min(va, 5)] += 1

            result[project_id] = {
                'outer': {'labels': outer_labels, 'values': outer_bins},
                'inner': {'labels': inner_labels, 'values': inner_bins},
            }

        return result

    return await with_db(db, _query, {})


# ---------------------------------------------------------------------------
# 4. Time-to-completion centiles
# ---------------------------------------------------------------------------


async def get_time_centiles(
    db: aiosqlite.Connection | None,
    *,
    days: int = 7,
    now: datetime | None = None,
) -> dict[str, dict]:
    """Per-project time-to-completion percentiles.

    Returns {project_id: {p50, p75, p90, p95, count}} in milliseconds.
    Filtered to outcome=done tasks only.

    Delegates raw-duration collection to :func:`_durations_by_project` so that
    the SQL lives in a single place (shared with
    :func:`aggregate_time_centiles`).
    """
    durations_by_pid = await _durations_by_project(db, days=days, now=now)

    result: dict[str, dict] = {}
    for project_id, durations in durations_by_pid.items():
        if not durations:
            result[project_id] = {
                'p50': 0,
                'p75': 0,
                'p90': 0,
                'p95': 0,
                'count': 0,
            }
            continue

        result[project_id] = {
            'p50': round(percentile(durations, 50)),
            'p75': round(percentile(durations, 75)),
            'p90': round(percentile(durations, 90)),
            'p95': round(percentile(durations, 95)),
            'count': len(durations),
        }

    return result


# ---------------------------------------------------------------------------
# 5. Per-hour history aggregators (sparks + per-project ttc trend)
# ---------------------------------------------------------------------------
#
# These walk task_results bucketed by hour (strftime('%Y-%m-%d %H', ...)) so
# the dashboard can render real time-series sparks without a new snapshot
# table.  /api/v2/dashboard/performance is hit every 3s, so the per-DB
# query is wrapped in a tiny self-invalidating cache keyed by
# (project_id, days, max(completed_at)) — bucketing only changes when a new
# counted (non-cancel) task_results row arrives, so the key is deterministic.

_HISTORY_CACHE: dict[tuple, dict] = {}
_HISTORY_CACHE_MAX = 64


async def _project_max_completed(
    db: aiosqlite.Connection,
    project_id: str,
) -> str:
    """Return the most recent completed_at for *project_id* (empty string if none),
    a cancel not counting (see ``_NOT_A_CANCEL``), as ``_hour_bucketed_history`` counts none."""
    async with db.execute(
        f'SELECT MAX(completed_at) FROM task_results WHERE project_id = ? AND {_NOT_A_CANCEL}',
        (project_id,),
    ) as cur:
        row = await cur.fetchone()
    return (row[0] if row and row[0] else '') or ''


async def _hour_bucketed_history(
    db: aiosqlite.Connection,
    project_id: str,
    *,
    days: int,
    now: datetime | None = None,
) -> dict[str, list]:
    """Return per-hour rows for *project_id* over the trailing *days* window.

    Each row carries: bucket label, p50/p95 of duration_ms (done tasks),
    count of done tasks, count of one-pass (review_cycles=0 done) tasks,
    count of escalated (steward_invocations>0) tasks, and total tasks.
    Caller derives ratios.

    Args:
        now: Reference timestamp forwarded to :func:`_cutoff`. None (the
            default) resolves to the current UTC clock.
    """
    # The cutoff is bound as a TEXT parameter (via _cutoff) rather than
    # computed SQL-side via datetime('now', ...): SQLite's datetime()
    # renders SPACE-separated with no UTC offset, which — compared
    # lexically against the ISO-with-offset `completed_at` column —
    # short-circuits at index 10 and silently degrades to DATE granularity,
    # over-including up to a full extra day (task 4624).
    # Binding the cutoff instead keeps idx_task_results_project (project_id
    # + completed_at) usable as a covering index for the WHERE clause —
    # re-confirmed via EXPLAIN QUERY PLAN and pinned by
    # TestHourBucketedHistoryWindowBoundary::test_binds_cutoff_as_parameter_and_keeps_covering_index.
    # strftime appears only in the SELECT/ORDER BY, not the WHERE clause, so
    # it does not defeat the index either.
    rows = await db.execute_fetchall(
        f"""
        SELECT strftime('%Y-%m-%dT%H:00', completed_at) AS bucket,
               duration_ms,
               outcome,
               review_cycles,
               steward_invocations
          FROM task_results
         WHERE project_id = ?
           AND completed_at >= ?
           AND completed_at IS NOT NULL
           AND completed_at != ''
           AND {_NOT_A_CANCEL}
         ORDER BY bucket
        """,
        (project_id, _cutoff(days, now=now)),
    )
    buckets: dict[str, dict] = {}
    for row in rows:
        bucket = row[0]
        duration = row[1]
        outcome = row[2]
        review_cycles = row[3] or 0
        steward = row[4] or 0
        b = buckets.setdefault(
            bucket,
            {'durations': [], 'total': 0, 'one_pass_done': 0, 'escalated': 0},
        )
        b['total'] += 1
        if outcome == 'done':
            if duration is not None and duration > 0:
                b['durations'].append(duration)
            if review_cycles == 0:
                b['one_pass_done'] += 1
        if steward > 0:
            b['escalated'] += 1

    labels: list[str] = []
    p50s: list[float] = []
    p95s: list[float] = []
    one_pass_pcts: list[float] = []
    escalation_pcts: list[float] = []
    for bucket in sorted(buckets):
        info = buckets[bucket]
        total = info['total']
        durations = sorted(info['durations'])
        labels.append(bucket)
        p50s.append(round(percentile(durations, 50)) if durations else 0)
        p95s.append(round(percentile(durations, 95)) if durations else 0)
        one_pass_pcts.append(
            round(info['one_pass_done'] / total * 100, 1) if total else 0.0
        )
        escalation_pcts.append(
            round(info['escalated'] / total * 100, 1) if total else 0.0
        )
    return {
        'labels': labels,
        'p50': p50s,
        'p95': p95s,
        'one_pass': one_pass_pcts,
        'escalation': escalation_pcts,
    }


async def _per_db_history(
    db: aiosqlite.Connection | None,
    project_id: str,
    *,
    days: int,
    now: datetime | None = None,
) -> dict[str, list]:
    """Cached wrapper for ``_hour_bucketed_history`` keyed by max(completed_at).

    The bucket layout only changes when a new counted task_results row arrives, so
    the cache is deterministic and self-invalidating. LRU-trim at
    ``_HISTORY_CACHE_MAX`` keeps memory bounded across many projects.

    The cache key deliberately excludes ``now``/the derived cutoff: this
    endpoint is polled every 3s with ``now=None``, so keying on the cutoff
    would make every request miss and defeat the cache's purpose. This is
    not a new staleness risk — the previous SQL-side cutoff
    (``datetime('now', ...)``) already moved between calls while the key
    stayed fixed, so omitting ``now`` here preserves that existing
    behaviour exactly (see design decision, task 4624). Callers that vary
    ``now`` across calls on the same ``db`` (i.e. tests) must clear
    ``_HISTORY_CACHE`` explicitly.
    """
    if db is None:
        return {'labels': [], 'p50': [], 'p95': [], 'one_pass': [], 'escalation': []}
    max_ts = await _project_max_completed(db, project_id)
    key = (id(db), project_id, days, max_ts)
    cached = _HISTORY_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        result = await _hour_bucketed_history(db, project_id, days=days, now=now)
    except Exception:
        logger.debug('per-db history failed', exc_info=True)
        return {'labels': [], 'p50': [], 'p95': [], 'one_pass': [], 'escalation': []}
    if len(_HISTORY_CACHE) >= _HISTORY_CACHE_MAX:
        # Drop the oldest insertion (dicts preserve insertion order).
        _HISTORY_CACHE.pop(next(iter(_HISTORY_CACHE)))
    _HISTORY_CACHE[key] = result
    return result


def _merge_history(per_db: list[dict]) -> dict:
    """Concatenate per-DB hour buckets, summing duplicate-hour samples.

    Multiple DBs writing for the same project_id is rare, but if it
    happens we merge by recomputing percentiles from concatenated raw
    durations. Since the helper has already lost raw durations after
    percentile-collapse, we approximate by averaging p50/p95 weighted by
    bucket presence and summing one-pass / escalation pct via simple mean.
    In the realistic single-DB case, this is a pass-through.
    """
    if not per_db:
        return {'labels': [], 'p50': [], 'p95': [], 'one_pass': [], 'escalation': []}
    if len(per_db) == 1:
        return per_db[0]
    by_bucket: dict[str, dict[str, list]] = {}
    for series in per_db:
        for i, bucket in enumerate(series['labels']):
            agg = by_bucket.setdefault(bucket, {'p50': [], 'p95': [], 'one_pass': [], 'escalation': []})
            for key in agg:
                agg[key].append(series[key][i])
    sorted_labels = sorted(by_bucket)
    out: dict = {'labels': sorted_labels}
    for key in ('p50', 'p95', 'one_pass', 'escalation'):
        out[key] = [
            round(sum(by_bucket[lbl][key]) / len(by_bucket[lbl][key]), 1)
            for lbl in sorted_labels
        ]
    return out


async def aggregate_performance_history(
    dbs: list[aiosqlite.Connection | None],
    *,
    days: int = 7,
    now: datetime | None = None,
) -> dict[str, dict]:
    """Return per-project bucketed history for ttc, one-pass, escalation.

    Result shape::

        {
          project_id: {
            time_centiles_history: {labels, p50, p95},
            one_pass_history:      {labels, values},
            escalation_history:    {labels, values},
          }
        }

    Args:
        now: Reference timestamp forwarded to :func:`_cutoff`. Resolved
            ONCE here (via :func:`resolve_now`) and threaded through both
            the project-discovery query and every :func:`_per_db_history`
            call, so the two legs share a single cutoff instant instead of
            each reading the clock independently and risking a straddled
            boundary. None (the default) resolves to the current UTC
            clock, matching the existing ``app.py`` call site.
    """
    if not dbs:
        return {}
    effective_now = resolve_now(now)
    since = _cutoff(days, now=effective_now)
    # Discover project IDs across all DBs.
    pid_sets: list[set[str]] = []
    for db in dbs:
        if db is None:
            continue
        try:
            rows = await db.execute_fetchall(
                'SELECT DISTINCT project_id FROM task_results '
                f'WHERE completed_at >= ? AND {_NOT_A_CANCEL}',
                (since,),
            )
            pid_sets.append({r[0] for r in rows if r[0]})
        except Exception:
            logger.debug('project_id discovery failed', exc_info=True)
    pids: set[str] = set()
    for s in pid_sets:
        pids |= s
    if not pids:
        return {}

    out: dict[str, dict] = {}
    for pid in pids:
        per_db = [
            await _per_db_history(db, pid, days=days, now=effective_now) for db in dbs
        ]
        merged = _merge_history(per_db)
        out[pid] = {
            'time_centiles_history': {
                'labels': merged['labels'],
                'p50': merged['p50'],
                'p95': merged['p95'],
            },
            'one_pass_history': {
                'labels': merged['labels'],
                'values': merged['one_pass'],
            },
            'escalation_history': {
                'labels': merged['labels'],
                'values': merged['escalation'],
            },
        }
    return out


# ---------------------------------------------------------------------------
# 6. The per-project cards Datum
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PerformanceCards:
    """One project's card block: the four card families' tally for one window."""

    paths: list[dict]
    escalation: dict
    hist_outer: dict
    hist_inner: dict
    ttc: dict

    def to_wire(self) -> dict[str, object]:
        return {
            'paths': copy.deepcopy(self.paths),
            'escalation': copy.deepcopy(self.escalation),
            'hist_outer': copy.deepcopy(self.hist_outer),
            'hist_inner': copy.deepcopy(self.hist_inner),
            'ttc': copy.deepcopy(self.ttc),
        }


def _window_bound_seconds(days: int) -> int:
    """The cards Datum's freshness bound: the served window's own length."""
    return int(timedelta(days=days).total_seconds())


def _cards_provenance(
    latest: datetime | None, served_at: datetime, days: int,
) -> tuple[datetime, DatumState, str | None]:
    """``(as_of, state, reason)`` for a project whose latest completion is *latest*.

    Fresh exactly when the latest completion lies inside ``[served_at - days,
    served_at]``, i.e. when the window's tally counts at least one completion.
    Reasons name the window and the instant, never a *served_at*-derived
    duration, so they do not change from one poll to the next.
    """
    if latest is None:
        return served_at, DatumState.STALE, 'latest completion time of this project could not be read'
    latest_iso = latest.astimezone(UTC).isoformat()
    age_seconds = (served_at - latest).total_seconds()
    if age_seconds < 0:
        return latest, DatumState.STALE, (
            f'last completion {latest_iso} is after the serving instant (clock skew)'
        )
    if age_seconds > _window_bound_seconds(days):
        return latest, DatumState.STALE, (
            f'no completions in the {days}d window; last completion {latest_iso}'
        )
    return latest, DatumState.FRESH, None


def _latest_instants(per_db: Sequence[Mapping[str, str]]) -> dict[str, datetime | None]:
    """``{project_id: latest completion across the per-DB discovery maps}``;
    ``None`` where none parses."""
    latest: dict[str, datetime | None] = {}
    for raw_by_project in per_db:
        for project_id, raw in raw_by_project.items():
            instant, parsed = parse_timestamp_or_warn(raw, context='performance.latest_completion')
            known = latest.get(project_id)
            if parsed and (known is None or instant > known):
                latest[project_id] = instant
            else:
                latest.setdefault(project_id, None)
    return latest


class _CardFamily(enum.Enum):
    """The four families a project's card block is built from, by display name."""

    PATHS = 'completion paths'
    ESCALATION = 'escalation rates'
    HISTOGRAMS = 'loop histograms'
    TTC = 'time centiles'


async def _read_card_families(
    dbs: list[aiosqlite.Connection | None],
    escalations_dirs: list[Path],
    *,
    days: int,
    now: datetime,
    projects_by_db: Sequence[frozenset[str]],
) -> dict[_CardFamily, Mapping[str, Any]]:
    """Each card family's ``{project_id: tally}``, each DB tallying the
    projects *projects_by_db* lists for it.

    Gathered with ``return_exceptions=True``: a family that raises reads as
    ``{}``, blanking only its own tallies. A listed project a family did not
    tally is logged here and served UNKNOWN by :func:`_cards_datum`.
    """
    reads = {
        _CardFamily.PATHS: aggregate_completion_paths(
            dbs, escalations_dirs, days=days, now=now, projects_by_db=projects_by_db,
        ),
        _CardFamily.ESCALATION: aggregate_escalation_rates(
            dbs, escalations_dirs, days=days, now=now, projects_by_db=projects_by_db,
        ),
        _CardFamily.HISTOGRAMS: aggregate_loop_histograms(
            dbs, days=days, now=now, projects_by_db=projects_by_db,
        ),
        _CardFamily.TTC: aggregate_time_centiles(
            dbs, days=days, now=now, projects_by_db=projects_by_db,
        ),
    }
    projects = frozenset[str]().union(*projects_by_db)
    results = await asyncio.gather(*reads.values(), return_exceptions=True)
    readings: dict[_CardFamily, Mapping[str, Any]] = {}
    for family, result in zip(reads, results, strict=True):
        reading: Mapping[str, Any] = safe_gather_result(result, {}, f'perf/cards/{family.value}')
        untallied = projects - reading.keys()
        if untallied:
            logger.warning('performance cards: no %s tally for %s', family.value, sorted(untallied))
        readings[family] = reading
    return readings


def _cards_datum(
    project_id: str,
    readings: Mapping[_CardFamily, Mapping[str, Any]],
    latest: datetime | None,
    served_at: datetime,
    days: int,
) -> Datum[PerformanceCards]:
    """*project_id*'s cards Datum: UNKNOWN, naming the families, when any
    family has no tally for it; otherwise its window tally, aged by *latest*."""
    bound = _window_bound_seconds(days)
    unread = [family.value for family, reading in readings.items() if project_id not in reading]
    if unread:
        return Datum(
            value=None,
            as_of=None,
            state=DatumState.UNKNOWN,
            reason=f'the {" and ".join(unread)} of this project could not be read',
            freshness_bound_seconds=bound,
        )
    histograms = readings[_CardFamily.HISTOGRAMS][project_id]
    as_of, state, reason = _cards_provenance(latest, served_at, days)
    return Datum(
        value=PerformanceCards(
            paths=readings[_CardFamily.PATHS][project_id],
            escalation=readings[_CardFamily.ESCALATION][project_id],
            hist_outer=histograms['outer'],
            hist_inner=histograms['inner'],
            ttc=readings[_CardFamily.TTC][project_id],
        ),
        as_of=as_of,
        state=state,
        reason=reason,
        freshness_bound_seconds=bound,
    )


async def aggregate_performance_cards(
    dbs: list[aiosqlite.Connection | None],
    escalations_dirs: list[Path],
    *,
    days: int = 7,
    now: datetime | None = None,
) -> dict[str, Datum[PerformanceCards]]:
    """Each project's card block for the window ``[now - days, now]``, as one Datum.

    The projects are discovered once per DB, and each DB tallies only the
    projects it holds (a runs.db whose own discovery query fails lists none;
    ``with_db`` logs the failure). A project is served a value only when
    every DB holding it tallied every family; otherwise it is UNKNOWN, and
    the reason names the families. The value is the project's window tally;
    ``as_of`` is its newest contributing event, its latest completion. So
    "fresh" means the last completion lies inside the window, and an idle
    project (completions ever, none in the window) is stale by the
    envelope's own bound.
    """
    served_at = resolve_now(now)
    per_db = await asyncio.gather(*(with_db(db, _latest_completions, {}) for db in dbs))
    latest = _latest_instants(per_db)
    if not latest:
        return {}
    readings = await _read_card_families(
        dbs, escalations_dirs, days=days, now=served_at,
        projects_by_db=[frozenset(found) for found in per_db],
    )
    return {
        project_id: _cards_datum(project_id, readings, latest[project_id], served_at, days)
        for project_id in sorted(latest)
    }
