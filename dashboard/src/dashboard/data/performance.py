"""Async queries for orchestrator performance metrics.

Reads from data/orchestrator/runs.db (task results) and
data/escalations/ (escalation JSON files) to produce per-project
performance statistics.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
from collections import defaultdict
from pathlib import Path

import aiosqlite
from escalation.queue import iter_all_escalation_paths

from dashboard.data.db import with_db
from dashboard.data.stats_utils import percentile

logger = logging.getLogger(__name__)


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
        except (json.JSONDecodeError, OSError):
            continue
    return results


# ---------------------------------------------------------------------------
# Time-window helper
# ---------------------------------------------------------------------------

_WINDOW_SQL = """\
SELECT project_id,
       datetime(MAX(completed_at), '-{days} days') AS cutoff
  FROM task_results
 WHERE completed_at IS NOT NULL AND completed_at != ''
 GROUP BY project_id
"""


async def _project_cutoffs(
    db: aiosqlite.Connection,
    days: int,
) -> dict[str, str]:
    """Return {project_id: cutoff_iso} based on most recent completed_at."""
    # aiosqlite doesn't support f-string in execute safely, use replace
    sql = _WINDOW_SQL.replace('{days}', str(int(days)))
    rows = await db.execute_fetchall(sql)
    return {row[0]: row[1] for row in rows}


# ---------------------------------------------------------------------------
# 1. Completion paths
# ---------------------------------------------------------------------------


async def get_completion_paths(
    db: aiosqlite.Connection | None,
    escalations_dir: Path,
    *,
    days: int = 7,
) -> dict[str, list[dict]]:
    """Per-project completion path breakdown.

    Returns {project_id: [{path: str, count: int, pct: float}, ...]}.
    Paths: one-pass, multi-pass, via-steward, via-interactive, blocked.
    """
    escalations = _load_escalations(escalations_dir)

    # Build set of task_ids that had level-1 escalations resolved
    interactive_task_ids: set[str] = set()
    for esc in escalations:
        if esc.get('level') == 1 and esc.get('status') in ('resolved', 'dismissed'):
            interactive_task_ids.add(str(esc.get('task_id', '')))

    async def _query(db: aiosqlite.Connection) -> dict[str, list[dict]]:
        cutoffs = await _project_cutoffs(db, days)
        if not cutoffs:
            return {}

        result: dict[str, list[dict]] = {}
        for project_id, cutoff in cutoffs.items():
            rows = await db.execute_fetchall(
                'SELECT task_id, outcome, review_cycles, '
                '       steward_invocations '
                '  FROM task_results '
                ' WHERE project_id = ? AND completed_at >= ? ',
                (project_id, cutoff),
            )

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


async def aggregate_completion_paths(
    dbs: list[aiosqlite.Connection | None],
    escalations_dirs: list[Path],
    *,
    days: int = 7,
) -> dict[str, list[dict]]:
    """Merge :func:`get_completion_paths` results from multiple databases.

    ``dbs`` and ``escalations_dirs`` must have the same length — they are
    zipped with ``strict=True`` so a mismatch raises immediately.  Each
    element of ``escalations_dirs`` must be the escalation directory
    corresponding to the project root whose runs.db is ``dbs[i]``.

    Merging rule: for a given project_id the *count* for each completion
    path is summed across DBs; ``pct`` is recomputed from the new totals.

    Shape contract: every project_id that appears in any per-DB
    :func:`get_completion_paths` result is a key in the returned dict.  Its
    value list contains only paths whose merged count is > 0 (the
    ``if count > 0`` guard on the list comprehension), so the list may be
    empty if every per-DB result for that project had an empty list.  In
    practice this does not occur because :func:`get_completion_paths` only
    yields a project_id when at least one ``task_results`` row exists for it,
    which always classifies into at least one non-zero path.

    Note: :func:`aggregate_escalation_rates` includes projects with
    ``total_tasks == 0`` (returning rates of 0.0).  The difference is
    intentional — an escalation-rate row for a zero-task project can still
    carry ``human_attention`` bucket values, whereas a completion-path list
    with no non-zero entries has nothing meaningful to render.
    """
    if not dbs:
        return {}

    results = await asyncio.gather(
        *(
            get_completion_paths(db, edir, days=days)
            for db, edir in zip(dbs, escalations_dirs, strict=True)
        )
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
) -> dict[str, dict]:
    """Merge :func:`get_escalation_rates` results from multiple databases.

    ``dbs`` and ``escalations_dirs`` must have the same length.
    For each project_id: total_tasks, steward_count, interactive_count, and
    each human_attention bucket are summed across DBs.  steward_rate and
    interactive_rate are recomputed from the merged totals (not averaged).
    """
    if not dbs:
        return {}

    results = await asyncio.gather(
        *(
            get_escalation_rates(db, edir, days=days)
            for db, edir in zip(dbs, escalations_dirs, strict=True)
        )
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
) -> dict[str, dict]:
    """Merge :func:`get_loop_histograms` results from multiple databases.

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

    results = await asyncio.gather(*(get_loop_histograms(db, days=days) for db in dbs))

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
) -> dict[str, list[int]]:
    """Return raw ``duration_ms`` lists per project_id (internal helper).

    Unlike :func:`get_time_centiles`, this function preserves the raw sample
    list so that :func:`aggregate_time_centiles` can concatenate samples from
    multiple DBs before computing percentiles on the unified distribution.
    """

    async def _query(db: aiosqlite.Connection) -> dict[str, list[int]]:
        cutoffs = await _project_cutoffs(db, days)
        if not cutoffs:
            return {}

        result: dict[str, list[int]] = {}
        for project_id, cutoff in cutoffs.items():
            rows = await db.execute_fetchall(
                'SELECT duration_ms FROM task_results '
                " WHERE project_id = ? AND completed_at >= ? AND outcome = 'done' "
                ' ORDER BY duration_ms ',
                (project_id, cutoff),
            )
            result[project_id] = [row[0] for row in rows if row[0] is not None and row[0] > 0]

        return result

    return await with_db(db, _query, {})


async def aggregate_time_centiles(
    dbs: list[aiosqlite.Connection | None],
    *,
    days: int = 7,
) -> dict[str, dict]:
    """Merge time-centile data from multiple databases.

    Calls :func:`_durations_by_project` per DB to collect raw duration
    samples, concatenates them per project_id, then computes p50/p75/p90/p95
    from the unified sample distribution so percentiles are exact (not
    averages of per-DB percentiles).  ``count`` is the total number of tasks
    across all DBs.
    """
    if not dbs:
        return {}

    results = await asyncio.gather(*(_durations_by_project(db, days=days) for db in dbs))

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
) -> dict[str, dict]:
    """Per-project escalation rates and human attention breakdown.

    Returns {project_id: {total_tasks, steward_count, interactive_count,
                          steward_rate, interactive_rate,
                          human_attention: {zero, minimal, significant}}}.
    """
    escalations = _load_escalations(escalations_dir)

    # Index escalations by task_id
    esc_by_task: dict[str, list[dict]] = defaultdict(list)
    for esc in escalations:
        tid = str(esc.get('task_id', ''))
        if tid:
            esc_by_task[tid].append(esc)

    async def _query(db: aiosqlite.Connection) -> dict[str, dict]:
        cutoffs = await _project_cutoffs(db, days)
        if not cutoffs:
            return {}

        result: dict[str, dict] = {}
        for project_id, cutoff in cutoffs.items():
            rows = list(
                await db.execute_fetchall(
                    'SELECT task_id, steward_invocations '
                    '  FROM task_results '
                    ' WHERE project_id = ? AND completed_at >= ? ',
                    (project_id, cutoff),
                )
            )

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
) -> dict[str, dict]:
    """Per-project loop cycle distributions.

    Returns {project_id: {
        outer: {labels: [str], values: [int]},
        inner: {labels: [str], values: [int]},
    }}.
    Outer = review_cycles (0,1,2,3+). Inner = verify_attempts (0,1,2,3,4,5+).
    Filtered to outcome=done tasks only.
    """

    async def _query(db: aiosqlite.Connection) -> dict[str, dict]:
        cutoffs = await _project_cutoffs(db, days)
        if not cutoffs:
            return {}

        result: dict[str, dict] = {}
        for project_id, cutoff in cutoffs.items():
            rows = await db.execute_fetchall(
                'SELECT review_cycles, verify_attempts '
                '  FROM task_results '
                " WHERE project_id = ? AND completed_at >= ? AND outcome = 'done' ",
                (project_id, cutoff),
            )

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
) -> dict[str, dict]:
    """Per-project time-to-completion percentiles.

    Returns {project_id: {p50, p75, p90, p95, count}} in milliseconds.
    Filtered to outcome=done tasks only.

    Delegates raw-duration collection to :func:`_durations_by_project` so that
    the SQL lives in a single place (shared with
    :func:`aggregate_time_centiles`).
    """
    durations_by_pid = await _durations_by_project(db, days=days)

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
