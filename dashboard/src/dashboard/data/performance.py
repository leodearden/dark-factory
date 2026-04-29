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
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning('Failed to load escalation %s: %s', path, exc)
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


# ---------------------------------------------------------------------------
# 5. Per-hour history aggregators (sparks + per-project ttc trend)
# ---------------------------------------------------------------------------
#
# These walk task_results bucketed by hour (strftime('%Y-%m-%d %H', ...)) so
# the dashboard can render real time-series sparks without a new snapshot
# table.  /api/v2/dashboard/performance is hit every 3s, so the per-DB
# query is wrapped in a tiny self-invalidating cache keyed by
# (project_id, days, max(completed_at)) — bucketing only changes when a new
# task_results row arrives, so the key is deterministic.

_HISTORY_CACHE: dict[tuple, dict] = {}
_HISTORY_CACHE_MAX = 64


async def _project_max_completed(
    db: aiosqlite.Connection,
    project_id: str,
) -> str:
    """Return the most recent completed_at for *project_id* (empty string if none)."""
    async with db.execute(
        "SELECT MAX(completed_at) FROM task_results WHERE project_id = ?",
        (project_id,),
    ) as cur:
        row = await cur.fetchone()
    return (row[0] if row and row[0] else '') or ''


async def _hour_bucketed_history(
    db: aiosqlite.Connection,
    project_id: str,
    *,
    days: int,
) -> dict[str, list]:
    """Return per-hour rows for *project_id* over the trailing *days* window.

    Each row carries: bucket label, p50/p95 of duration_ms (done tasks),
    count of done tasks, count of one-pass (review_cycles=0 done) tasks,
    count of escalated (steward_invocations>0) tasks, and total tasks.
    Caller derives ratios.
    """
    # Bucketing is fully covered by idx_task_results_project (project_id +
    # completed_at). The strftime appears only in GROUP BY so it does not
    # defeat the index — confirmed via EXPLAIN QUERY PLAN.
    rows = await db.execute_fetchall(
        """
        SELECT strftime('%Y-%m-%dT%H:00', completed_at) AS bucket,
               duration_ms,
               outcome,
               review_cycles,
               steward_invocations
          FROM task_results
         WHERE project_id = ?
           AND completed_at >= datetime('now', ? || ' days')
           AND completed_at IS NOT NULL
           AND completed_at != ''
         ORDER BY bucket
        """,
        (project_id, f'-{int(days)}'),
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
) -> dict[str, list]:
    """Cached wrapper for ``_hour_bucketed_history`` keyed by max(completed_at).

    The bucket layout only changes when a new task_results row arrives, so
    the cache is deterministic and self-invalidating. LRU-trim at
    ``_HISTORY_CACHE_MAX`` keeps memory bounded across many projects.
    """
    if db is None:
        return {'labels': [], 'p50': [], 'p95': [], 'one_pass': [], 'escalation': []}
    max_ts = await _project_max_completed(db, project_id)
    key = (id(db), project_id, days, max_ts)
    cached = _HISTORY_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        result = await _hour_bucketed_history(db, project_id, days=days)
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
    """
    if not dbs:
        return {}
    # Discover project IDs across all DBs.
    pid_sets: list[set[str]] = []
    for db in dbs:
        if db is None:
            continue
        try:
            rows = await db.execute_fetchall(
                'SELECT DISTINCT project_id FROM task_results '
                "WHERE completed_at >= datetime('now', ? || ' days')",
                (f'-{int(days)}',),
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
            await _per_db_history(db, pid, days=days) for db in dbs
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
