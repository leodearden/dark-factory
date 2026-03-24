"""Async queries for orchestrator performance metrics.

Reads from data/orchestrator/runs.db (task results) and
data/escalations/ (escalation JSON files) to produce per-project
performance statistics.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from collections import defaultdict
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TypeVar

import aiosqlite

logger = logging.getLogger(__name__)

_T = TypeVar('_T')


async def _with_readonly_db(
    db_path: Path,
    fn: Callable[[aiosqlite.Connection], Awaitable[_T]],
    default: _T,
    *,
    caller: str,
) -> _T:
    """Open db_path read-only, run fn(db), and return the result."""
    try:
        async with aiosqlite.connect(f'file:{db_path}?mode=ro', uri=True) as db:
            return await fn(db)
    except (FileNotFoundError, sqlite3.OperationalError):
        logger.debug('%s: DB unavailable at %s', caller, db_path, exc_info=True)
        return default


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


def _percentile(sorted_values: list[float], p: float) -> float:
    """Compute the p-th percentile from a sorted list."""
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)


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
    db_path: Path,
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

    return await _with_readonly_db(db_path, _query, {}, caller='get_completion_paths')


# ---------------------------------------------------------------------------
# 2. Escalation rates
# ---------------------------------------------------------------------------

async def get_escalation_rates(
    db_path: Path,
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

    return await _with_readonly_db(db_path, _query, {}, caller='get_escalation_rates')


# ---------------------------------------------------------------------------
# 3. Loop histograms
# ---------------------------------------------------------------------------

async def get_loop_histograms(
    db_path: Path,
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

    return await _with_readonly_db(db_path, _query, {}, caller='get_loop_histograms')


# ---------------------------------------------------------------------------
# 4. Time-to-completion centiles
# ---------------------------------------------------------------------------

async def get_time_centiles(
    db_path: Path,
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
                'p50': round(_percentile(durations, 50)),
                'p75': round(_percentile(durations, 75)),
                'p90': round(_percentile(durations, 90)),
                'p95': round(_percentile(durations, 95)),
                'count': len(durations),
            }

        return result

    return await _with_readonly_db(db_path, _query, {}, caller='get_time_centiles')
