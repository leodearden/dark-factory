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

from dashboard.data.db import with_db
from dashboard.data.stats_utils import percentile

logger = logging.getLogger(__name__)


def _load_escalations(escalations_dir: Path) -> list[dict]:
    """Load all escalation JSON files from the directory."""
    if not escalations_dir.is_dir():
        return []
    results = []
    for path in escalations_dir.glob('esc-*.json'):
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
    db: aiosqlite.Connection, days: int,
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
    """
    if not dbs:
        return {}

    results = await asyncio.gather(
        *(get_completion_paths(db, edir, days=days)
          for db, edir in zip(dbs, escalations_dirs, strict=True))
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
        *(get_escalation_rates(db, edir, days=days)
          for db, edir in zip(dbs, escalations_dirs, strict=True))
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
    for pid, m in merged.items():
        total = m['total_tasks']
        m['steward_rate'] = round(m['steward_count'] / total * 100, 1) if total else 0.0
        m['interactive_rate'] = (
            round(m['interactive_count'] / total * 100, 1) if total else 0.0
        )
    return merged


async def aggregate_loop_histograms(
    dbs: list[aiosqlite.Connection | None],
    *,
    days: int = 7,
) -> dict[str, dict]:
    """Merge :func:`get_loop_histograms` results from multiple databases.

    For each project_id: element-wise sum the outer.values and inner.values
    arrays across DBs (labels are always the canonical fixed lists).
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
                    for i, val in enumerate(info[key]['values']):
                        m[key]['values'][i] += val
    return merged


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
            rows = list(await db.execute_fetchall(
                'SELECT task_id, steward_invocations '
                '  FROM task_results '
                ' WHERE project_id = ? AND completed_at >= ? ',
                (project_id, cutoff),
            ))

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
                    e.get('level') == 1
                    and e.get('status') in ('resolved', 'dismissed')
                    for e in task_escs
                )
                if has_interactive:
                    interactive_count += 1

                    # Classify human effort from resolution_turns
                    max_turns = max(
                        (e.get('resolution_turns') or 0)
                        for e in task_escs
                        if e.get('level') == 1
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
    """

    async def _query(db: aiosqlite.Connection) -> dict[str, dict]:
        cutoffs = await _project_cutoffs(db, days)
        if not cutoffs:
            return {}

        result: dict[str, dict] = {}
        for project_id, cutoff in cutoffs.items():
            rows = await db.execute_fetchall(
                'SELECT duration_ms FROM task_results '
                " WHERE project_id = ? AND completed_at >= ? AND outcome = 'done' "
                ' ORDER BY duration_ms ',
                (project_id, cutoff),
            )

            durations = [row[0] for row in rows if row[0] is not None and row[0] > 0]

            if not durations:
                result[project_id] = {
                    'p50': 0, 'p75': 0, 'p90': 0, 'p95': 0, 'count': 0,
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

    return await with_db(db, _query, {})
