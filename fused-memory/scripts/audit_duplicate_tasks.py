#!/usr/bin/env python3
"""One-shot audit: find and cancel duplicate / near-duplicate task titles among
pending and in-progress tasks with IDs ≥ 1000.

Motivation: Taskmaster accumulates duplicate tasks over time — either from
concurrent planners that create near-identical subtasks or from re-planning
runs that forget to check whether a task already exists. This script finds
exact and near-duplicate titles, picks a survivor (highest priority; ties
broken by lowest task ID), and cancels the losers.

Safety carve-outs:
  - Only *pending* losers are auto-cancelled; *in-progress* losers are
    reported under ``needs_human_review`` so they can be handled after any
    in-flight agent work is stopped.
  - Only *exact* duplicate groups are auto-cancelled; near-duplicates are
    reported under ``near_duplicates`` for human confirmation.
  - Dependency references to cancelled tasks are remapped to the survivor.

Usage
-----
  # Dry run (default): print JSON report, change nothing.
  python scripts/audit_duplicate_tasks.py --project-root /path/to/project

  # Commit the cancellations + dependency remaps.
  python scripts/audit_duplicate_tasks.py --project-root /path/to/project --apply

  # Tune near-duplicate threshold (default 0.90).
  python scripts/audit_duplicate_tasks.py --project-root /path/to/project \\
      --threshold 0.85

  # Restrict scan to a specific Taskmaster tag.
  python scripts/audit_duplicate_tasks.py --project-root /path/to/project \\
      --tag master
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import logging
import sys
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger('audit_duplicate_tasks')

# Priority rank: higher number = higher priority. Absent / unknown → 0.
_PRIORITY_RANK: dict[str, int] = {
    'critical': 4,
    'high': 3,
    'medium': 2,
    'low': 1,
}


def _id_as_int(task: dict, fallback: int = 0) -> int:
    """Return the task's numeric ID as int, or *fallback* for non-numeric IDs.

    Handles dotted subtask IDs (e.g. ``'1.2'``), empty strings, and other
    non-integer shapes safely — returns *fallback* rather than raising
    ``ValueError``.  Uses ``str.isdecimal()`` to accept only pure ASCII
    decimal strings.
    """
    raw = str(task.get('id', ''))
    return int(raw) if raw.isdecimal() else fallback


def _sort_groups_deterministically(groups: list[list[dict]]) -> list[list[dict]]:
    """Sort *groups* in place for deterministic output and return them.

    Members within each group are sorted by numeric task ID (using
    ``_id_as_int`` with ``fallback=0`` so dotted subtask IDs like ``'1.2'``
    do not raise ``ValueError``).  The list of groups is then sorted by the
    minimum ID in each group (``_id_as_int(g[0])`` after the inner sort).

    Called by both :func:`find_exact_duplicate_groups` and
    :func:`find_near_duplicate_groups` so the two functions stay in sync
    if the ordering convention ever changes.
    """
    for g in groups:
        g.sort(key=lambda t: _id_as_int(t))
    groups.sort(key=lambda g: _id_as_int(g[0]))
    return groups


# ---------------------------------------------------------------------------
# Pure-function core (no I/O — fully testable without a live Taskmaster)
# ---------------------------------------------------------------------------

def find_exact_duplicate_groups(tasks: list[dict]) -> list[list[dict]]:
    """Group tasks by normalised title; return groups with ≥ 2 members.

    Normalisation: ``(task.get('title') or '').strip().lower()``.
    No status filtering — the caller is responsible for pre-filtering.

    Returns:
        List of groups (each a list of ≥ 2 tasks) whose normalised titles are
        identical.  Groups are sorted by the minimum task ID within the group
        so output is deterministic.
    """
    by_title: dict[str, list[dict]] = {}
    for task in tasks:
        key = (task.get('title') or '').strip().lower()
        by_title.setdefault(key, []).append(task)
    result = [group for group in by_title.values() if len(group) >= 2]
    return _sort_groups_deterministically(result)


def find_near_duplicate_groups(
    tasks: list[dict],
    threshold: float = 0.90,
    exclude_ids: set[str] | None = None,
) -> list[list[dict]]:
    """Find groups of tasks with near-duplicate titles using SequenceMatcher.

    Args:
        tasks: Task dicts (no status filtering performed here).
        threshold: Minimum ``difflib.SequenceMatcher.ratio()`` to flag a pair.
        exclude_ids: Task IDs to skip (typically tasks already in exact groups).

    Returns:
        List of groups (each a list of ≥ 2 tasks) formed by transitive closure
        of all pairs whose similarity ≥ threshold.  Groups are sorted by the
        minimum task ID within the group so output is deterministic.

    Complexity:
        O(n²) pairs × O(L) per ``SequenceMatcher.ratio()`` call (L = title
        length).  Acceptable for typical Taskmaster task counts (≤ hundreds);
        not suitable for thousands of tasks without a cheap pre-filter (e.g.
        token-set Jaccard) to skip obviously dissimilar pairs.
    """
    _excluded = exclude_ids or set()
    candidates = [t for t in tasks if str(t.get('id', '')) not in _excluded]
    n = len(candidates)
    if n < 2:
        return []

    # Union-find (path-compressed) over candidate indices.
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(n):
        ti = (candidates[i].get('title') or '').lower().strip()
        for j in range(i + 1, n):
            tj = (candidates[j].get('title') or '').lower().strip()
            ratio = difflib.SequenceMatcher(None, ti, tj).ratio()
            if ratio >= threshold:
                union(i, j)

    # Materialise groups: collect task lists per root, drop singletons.
    groups: dict[int, list[dict]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(candidates[i])

    result = [g for g in groups.values() if len(g) >= 2]
    return _sort_groups_deterministically(result)


def pick_survivor(group: list[dict]) -> tuple[dict, list[dict]]:
    """Pick the survivor from a duplicate group.

    Survivor = highest priority; ties broken by lowest int(task['id'])
    (earliest created).  Raises ``ValueError`` for groups with < 2 tasks.

    Priority ranking: critical(4) > high(3) > medium(2) > low(1) > absent(0).
    """
    if len(group) < 2:
        raise ValueError(f'pick_survivor requires a group of ≥ 2 tasks, got {len(group)}')

    def _rank(t: dict) -> tuple[int, int]:
        priority_score = _PRIORITY_RANK.get((t.get('priority') or '').lower(), 0)
        # Negate ID so that lower ID wins ties (max key → lowest ID).
        # Use sys.maxsize fallback so non-numeric IDs (e.g. '1.2') rank last
        # in tie-breaking rather than raising ValueError.
        return (priority_score, -_id_as_int(t, fallback=sys.maxsize))

    survivor = max(group, key=_rank)
    losers = [t for t in group if t is not survivor]
    return survivor, losers


@dataclass
class DependencyUpdate:
    """A planned dependency remap: remove a cancelled dep, optionally add survivor."""
    dependent_id: str
    remove_dep: str
    add_dep: str | None = None


def compute_dependency_updates(
    survivor_id: str,
    cancelled_ids: set[str],
    all_tasks: list[dict],
) -> list[DependencyUpdate]:
    """Compute dependency remaps needed when cancelled_ids are cancelled.

    For every task whose dependencies include a cancelled ID:
    - Emit a DependencyUpdate(dependent_id, remove_dep=cancelled_id, add_dep=survivor_id)
      unless survivor_id is already in that task's dependencies.
    - If the task depends on multiple cancelled IDs, emit one remove per
      cancelled dep, but only one add for the survivor.

    Tasks whose own ID is in cancelled_ids are skipped (they are being
    cancelled; their outgoing dependencies don't need hygiene).
    """
    updates: list[DependencyUpdate] = []
    for task in all_tasks:
        tid = str(task.get('id', ''))
        if tid in cancelled_ids:
            continue  # Task itself is being cancelled — skip.

        raw_deps = task.get('dependencies') or []
        # Normalise dependency IDs to str.
        dep_strs = {str(d) for d in raw_deps}
        overlap = dep_strs & cancelled_ids
        if not overlap:
            continue

        # Determine whether to add the survivor.
        # We add it at most once, and only if it isn't already a dependency.
        survivor_already_present = survivor_id in dep_strs
        add_once = not survivor_already_present

        for cancelled_id in sorted(overlap):  # sorted for determinism
            add_dep: str | None = None
            if add_once:
                add_dep = survivor_id
                add_once = False  # Only add survivor once per dependent task.
            updates.append(DependencyUpdate(
                dependent_id=tid,
                remove_dep=cancelled_id,
                add_dep=add_dep,
            ))
    return updates


# ---------------------------------------------------------------------------
# Orchestrator: build_audit_plan
# ---------------------------------------------------------------------------

def build_audit_plan(
    tasks: list[dict],
    threshold: float = 0.90,
    min_id: int = 1000,
) -> dict[str, Any]:
    """Orchestrate filtering → grouping → survivor selection → dep planning.

    Args:
        tasks: Raw task list (all statuses, all IDs).
        threshold: Near-duplicate similarity threshold.
        min_id: Minimum task ID to include in the audit.

    Returns:
        Plan dict with keys:
          candidates_total, exact_groups, near_duplicate_groups,
          auto_cancel, needs_human_review, dependency_updates.
    """
    # 1. Filter to active tasks within ID range.
    # _id_as_int handles non-numeric / dotted subtask IDs by returning 0,
    # which is always < min_id (≥ 1000 by default), so they are safely skipped.
    active = [
        t for t in tasks
        if t.get('status') in {'pending', 'in-progress'}
        and _id_as_int(t) >= min_id
    ]

    # 2. Exact-duplicate groups.
    exact_groups = find_exact_duplicate_groups(active)

    # Collect IDs already in exact groups to exclude from near-dup search.
    exact_ids: set[str] = {str(t['id']) for g in exact_groups for t in g}

    # 3. Near-duplicate groups (excluding exact matches).
    near_groups = find_near_duplicate_groups(active, threshold=threshold, exclude_ids=exact_ids)

    # 4. For each exact group: pick survivor, split losers by status.
    #    Cache (survivor, losers) per group so step 5 can reuse without a
    #    second pick_survivor call.
    auto_cancel: list[str] = []           # pending losers → safe to cancel
    needs_human_review: list[dict] = []   # in-progress losers → human needed
    exact_groups_report: list[dict] = []
    group_decisions: list[tuple[dict, list[dict]]] = []  # cached per-group results

    for group in exact_groups:
        survivor, losers = pick_survivor(group)
        group_decisions.append((survivor, losers))

        pending_losers = [t for t in losers if t.get('status') == 'pending']
        inprogress_losers = [t for t in losers if t.get('status') == 'in-progress']

        auto_cancel.extend(str(t['id']) for t in pending_losers)
        needs_human_review.extend(
            {'id': str(t['id']), 'title': t.get('title'), 'reason': 'in_progress_duplicate'}
            for t in inprogress_losers
        )
        exact_groups_report.append({
            'survivor_id': str(survivor['id']),
            'survivor_title': survivor.get('title'),
            'auto_cancel': [str(t['id']) for t in pending_losers],
            'needs_human_review': [str(t['id']) for t in inprogress_losers],
        })

    # 5. Dependency updates only for auto-cancelled IDs.
    #    Reuse the cached (survivor, losers) tuples — pick_survivor not called again.
    auto_cancel_set = set(auto_cancel)
    dep_updates = []
    for survivor, losers in group_decisions:
        cancelled_in_group = {str(t['id']) for t in losers if str(t['id']) in auto_cancel_set}
        if cancelled_in_group:
            updates = compute_dependency_updates(
                survivor_id=str(survivor['id']),
                cancelled_ids=cancelled_in_group,
                all_tasks=tasks,
            )
            dep_updates.extend(updates)

    # 6. Near-duplicate groups report (no auto-cancel — human review only).
    near_groups_report = [
        {
            'tasks': [
                {'id': str(t['id']), 'title': t.get('title'), 'status': t.get('status')}
                for t in group
            ],
        }
        for group in near_groups
    ]

    return {
        'candidates_total': len(active),
        'exact_groups': exact_groups_report,
        'near_duplicate_groups': near_groups_report,
        'auto_cancel': auto_cancel,
        'needs_human_review': needs_human_review,
        'dependency_updates': [
            {
                'dependent_id': u.dependent_id,
                'remove_dep': u.remove_dep,
                'add_dep': u.add_dep,
            }
            for u in dep_updates
        ],
    }


# ---------------------------------------------------------------------------
# Apply layer
# ---------------------------------------------------------------------------

async def apply_changes(
    backend: Any,
    project_root: str,
    plan: dict[str, Any],
    tag: str | None = None,
) -> dict[str, int]:
    """Apply the audit plan: cancel losers then remap dependencies.

    Cancellations are performed first; dependency updates follow.  Each
    operation is wrapped in try/except so partial progress is preserved even
    when individual calls fail.

    Returns:
        Dict with operation counts: ``cancelled``, ``cancel_errors``,
        ``dep_updates_applied``, ``dep_update_errors``.  The caller should
        check these to determine whether to exit non-zero.
    """
    cancelled = 0
    cancel_errors = 0
    dep_updates_applied = 0
    dep_update_errors = 0

    for task_id in plan.get('cancellations', []):
        try:
            await backend.set_task_status(task_id, 'cancelled', project_root, tag)
            logger.info('Cancelled task %s', task_id)
            cancelled += 1
        except Exception as exc:
            logger.error('Failed to cancel task %s: %s', task_id, exc)
            cancel_errors += 1

    for upd in plan.get('dependency_updates', []):
        dep_id = upd['dependent_id']
        remove_dep = upd['remove_dep']
        add_dep = upd.get('add_dep')
        op_ok = True
        try:
            await backend.remove_dependency(dep_id, remove_dep, project_root, tag)
            logger.info('Removed dep %s→%s', dep_id, remove_dep)
        except Exception as exc:
            logger.error('Failed to remove dep %s→%s: %s', dep_id, remove_dep, exc)
            op_ok = False
            dep_update_errors += 1
        if add_dep is not None:
            try:
                await backend.add_dependency(dep_id, add_dep, project_root, tag)
                logger.info('Added dep %s→%s', dep_id, add_dep)
            except Exception as exc:
                logger.error('Failed to add dep %s→%s: %s', dep_id, add_dep, exc)
                op_ok = False
                dep_update_errors += 1
        if op_ok:
            dep_updates_applied += 1

    return {
        'cancelled': cancelled,
        'cancel_errors': cancel_errors,
        'dep_updates_applied': dep_updates_applied,
        'dep_update_errors': dep_update_errors,
    }


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

async def _run(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
    )

    import os  # noqa: PLC0415

    from fused_memory.backends.taskmaster_client import TaskmasterBackend  # noqa: PLC0415
    from fused_memory.config.schema import FusedMemoryConfig  # noqa: PLC0415

    if args.config:
        os.environ['CONFIG_PATH'] = str(args.config)

    config = FusedMemoryConfig()
    if config.taskmaster is None:
        logger.error('Taskmaster backend not configured in fused-memory config')
        return 1

    backend = TaskmasterBackend(config.taskmaster)
    await backend.initialize()
    try:
        raw = await backend.get_tasks(args.project_root, args.tag)
        # Taskmaster response shape (verified 2026-04-25):
        #   {'data': {'tasks': [...], 'filter': ..., 'stats': ...},
        #    'version': {...}, 'tag': 'master'}
        # Unwrap defensively to handle future shape changes.
        tasks: list[dict] = []
        if isinstance(raw, dict):
            data = raw.get('data') or raw.get('tasks') or raw
            if isinstance(data, list):
                tasks = data
            elif isinstance(data, dict):
                tasks = data.get('tasks') or []
        elif isinstance(raw, list):
            tasks = raw

        logger.info('Fetched %d task(s) from Taskmaster', len(tasks))

        # Warn when the response had content but no tasks could be extracted —
        # silently producing an empty plan would look like "no duplicates found"
        # when the real issue is an unexpected Taskmaster response shape.
        if not tasks and raw:
            shape_hint: Any = (
                list(raw.keys()) if isinstance(raw, dict) else type(raw).__name__
            )
            logger.warning(
                'get_tasks returned a non-empty response but 0 tasks were '
                'extracted — possible response-shape mismatch.  '
                'Top-level keys/type: %s',
                shape_hint,
            )

        plan = build_audit_plan(tasks, threshold=args.threshold, min_id=args.min_id)
        print(json.dumps(plan, indent=2, default=str))

        if not args.apply:
            logger.info('Dry run — nothing was modified. Use --apply to commit.')
            return 0

        # Re-shape plan to match apply_changes expectation.
        apply_plan = {
            'cancellations': plan['auto_cancel'],
            'dependency_updates': plan['dependency_updates'],
        }
        result = await apply_changes(backend, args.project_root, apply_plan, args.tag)
        total_errors = result['cancel_errors'] + result['dep_update_errors']
        logger.info(
            'Applied: cancelled %d/%d task(s), %d/%d dep update(s); %d error(s)',
            result['cancelled'],
            len(plan['auto_cancel']),
            result['dep_updates_applied'],
            len(plan['dependency_updates']),
            total_errors,
        )
        return 1 if total_errors > 0 else 0
    finally:
        await backend.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--project-root', required=True,
        help='Absolute filesystem path to the project (required by Taskmaster)',
    )
    parser.add_argument(
        '--apply', action='store_true',
        help='Commit cancellations and dependency remaps (default: dry-run, report only)',
    )
    parser.add_argument(
        '--threshold', type=float, default=0.90,
        help='Near-duplicate similarity threshold (default: 0.90)',
    )
    parser.add_argument(
        '--tag', default=None,
        help='Taskmaster tag to scan (default: None → master)',
    )
    parser.add_argument(
        '--min-id', type=int, default=1000, dest='min_id',
        help='Minimum task ID to include in the audit (default: 1000)',
    )
    parser.add_argument(
        '--config', default=None,
        help='Path to fused-memory config file (sets CONFIG_PATH env var)',
    )
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == '__main__':
    sys.exit(main())
