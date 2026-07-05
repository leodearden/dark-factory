#!/usr/bin/env python3
"""One-shot sweep: detect and optionally delete stage1_flag_marker records in Mem0/Qdrant
that are missing the ``kind='stage1_flag_marker'`` metadata key (task-1659 orphans) or
that lack a usable ``task_id`` (task-2108 orphans).

Background
----------
Prior to task-1659, ``flag_dedup._write_and_confirm_marker`` wrote markers with
``metadata.source='stage1_flag_marker'`` but omitted ``metadata.kind``.  Dual-filter
queries keyed on *both* source and kind silently under-count those markers.  Fix (1) in
task-1659 adds kind to every new write; this script is Fix (2): a one-time sweep to
remove the 6 pre-existing orphans so the counts converge immediately.

Deletion vs backfill
--------------------
Orphan markers are deleted (not updated in place) for two reasons:
 1. Mem0/Qdrant exposes ``delete_memory`` but no payload-update primitive on this path.
 2. stage1_flag_markers are self-healing: a deleted marker is rewritten with both keys
    on the next MISS cycle (at most one extra re-escalation, within the existing
    best-effort-replacement tolerance).

Taskless markers (task 2108)
-----------------------------
In addition to the missing-``kind`` orphans above, this sweep also purges
stage1_flag_marker records that carry a valid ``kind`` but lack a usable
``task_id`` (missing key, ``None``, or ``''``) — see ``find_taskless_markers``.
This is safe for the same self-healing reason as above, plus one more:
``mem0_dedup.find_prior_memories`` filters candidate priors by
``str(meta.get('task_id', '')) != task_id_str`` (see
``fused_memory.reconciliation.flag_dedup``), so a marker without a task_id can
*never* be returned as a dedup prior. It cannot collapse a repeat flag,
cannot suppress re-escalation, and Stage 2 never sweeps it either — it is
pure dead weight from the moment it is written. Deleting it therefore loses
zero dedup capability; if the underlying flag recurs, the next MISS cycle
writes a fresh marker with both ``kind`` and ``task_id`` set.

Enumeration strategy
--------------------
Markers are enumerated via ``get_memories_by_metadata(filters={'source':MARKER_SOURCE})``,
which performs a deterministic Qdrant payload-filter scroll — NOT semantic search.
Semantic top-N silently drops low-similarity markers (the documented failure mode in
``_query_stage2_flags``), making it unsuitable for exhaustive enumeration.

Usage
-----
  # Dry run (default): print JSON report, touch nothing.
  python scripts/sweep_orphan_flag_markers.py

  # Commit the deletions.
  python scripts/sweep_orphan_flag_markers.py --apply

  # Override the target project (default: dark_factory).
  python scripts/sweep_orphan_flag_markers.py --apply --project-id my_project
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from typing import Any

# ---------------------------------------------------------------------------
# Module-level constants
# Cross-reference: payload contract defined in
#   fused_memory.reconciliation.flag_dedup._write_and_confirm_marker
#   (task-1659 adds kind to metadata dict alongside source).
# ---------------------------------------------------------------------------

MARKER_SOURCE: str = 'stage1_flag_marker'
MARKER_KIND: str = 'stage1_flag_marker'

logger = logging.getLogger('sweep_orphan_flag_markers')


# ---------------------------------------------------------------------------
# Pure core
# ---------------------------------------------------------------------------

def find_orphan_markers(members: list[dict]) -> list[dict]:
    """Return members whose metadata lacks ``kind == MARKER_KIND``.

    Args:
        members: List of scroll-shaped dicts ``{'id', 'created_at', 'metadata'}``,
            as returned by ``MemoryService.get_memories_by_metadata``.

    Returns:
        Subset of *members* for which ``metadata.get('kind') != MARKER_KIND``.
        Order is preserved.  An all-valid input returns ``[]``.
    """
    return [
        m for m in members
        if (m.get('metadata') or {}).get('kind') != MARKER_KIND
    ]


def find_taskless_markers(members: list[dict]) -> list[dict]:
    """Return members whose metadata lacks a usable ``task_id`` (task 2108).

    Mirrors the ``task_id is None or task_id == ''`` emptiness guard used by
    ``flag_dedup.filter_suppressed`` — a marker is taskless when its
    ``task_id`` is missing, ``None``, or ``''``. Numeric and ``fp:``-hash
    task_ids are valid and are never considered taskless, regardless of
    ``kind``.

    Args:
        members: List of scroll-shaped dicts ``{'id', 'created_at', 'metadata'}``,
            as returned by ``MemoryService.get_memories_by_metadata``.

    Returns:
        Subset of *members* for which ``metadata.get('task_id')`` is missing,
        ``None``, or ``''``. Order is preserved. An all-valid input returns ``[]``.
    """
    return [
        m for m in members
        if (m.get('metadata') or {}).get('task_id') in (None, '')
    ]


# ---------------------------------------------------------------------------
# Async delete
# ---------------------------------------------------------------------------

async def delete_orphan_markers(
    memory_service: Any,
    project_id: str,
    orphans: list[dict],
    *,
    causation_id: str | None = None,
) -> dict:
    """Delete orphan markers from Mem0 (best-effort, never raises).

    Mirrors the posture of ``_enforce_stage2_summary_pool_cap``:
    asyncio.gather with return_exceptions=True, per-item WARNING on failure,
    count only successes.

    Args:
        memory_service: Live (or mock) MemoryService instance.
        project_id: Project scope passed to each delete_memory call.
        orphans: List of orphan member dicts (output of find_orphan_markers).
        causation_id: Optional causation id forwarded to each delete_memory call.

    Returns:
        ``{'deleted': int, 'failed': [ids]}``
    """
    if not orphans:
        return {'deleted': 0, 'failed': []}

    async def _delete_one(orphan: dict):
        return await memory_service.delete_memory(
            memory_id=orphan['id'],
            store='mem0',
            project_id=project_id,
            causation_id=causation_id,
            _source='sweep_orphan_flag_markers',
        )

    results = await asyncio.gather(
        *(_delete_one(o) for o in orphans),
        return_exceptions=True,
    )

    deleted = 0
    failed: list[str] = []
    for orphan, result in zip(orphans, results, strict=False):
        if isinstance(result, BaseException):
            logger.warning(
                'sweep_orphan_flag_markers: failed to delete memory %s: %s',
                orphan['id'], result,
            )
            failed.append(orphan['id'])
        else:
            deleted += 1

    return {'deleted': deleted, 'failed': failed}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def run(args: Any, memory_service: Any) -> dict:
    """Enumerate orphan markers and optionally delete them.

    Args:
        args: argparse.Namespace (or SimpleNamespace) with at least:
            - apply (bool): commit deletions if True, dry-run otherwise
            - project_id (str): project to sweep
        memory_service: Live (or mock) MemoryService instance.

    Returns:
        JSON-serialisable report dict:
            - dry_run (bool)
            - before (dict with total_source and total_with_kind counts)
            - orphan_count (int): size of the deduplicated union of
              find_orphan_markers + find_taskless_markers — the actual
              number of records deleted (or that would be deleted).
            - orphan_ids (list[str])
            - taskless_orphan_count (int): raw len(find_taskless_markers(...)).
              Overlaps with the kind-orphan predicate for members missing
              BOTH kind and task_id, so it does not subtract cleanly from
              orphan_count — see the inline NOTE above its assignment.
            - deleted (int, only when apply=True)
            - failed (list[str], only when apply=True)
            - after (dict with counts, only when apply=True)
    """
    project_id: str = getattr(args, 'project_id', 'dark_factory')

    # --- Before counts (deterministic Qdrant payload-filter, not semantic) ---
    source_filter = {'source': MARKER_SOURCE}
    kind_filter = {'source': MARKER_SOURCE, 'kind': MARKER_KIND}

    total_source = await memory_service.count_memories_by_metadata(
        project_id=project_id, filters=source_filter,
    )
    total_with_kind = await memory_service.count_memories_by_metadata(
        project_id=project_id, filters=kind_filter,
    )
    before = {'total_source': total_source, 'total_with_kind': total_with_kind}

    # --- Enumerate via scroll (NOT semantic search) ---
    scroll_limit: int = getattr(args, 'limit', 1000)
    members = await memory_service.get_memories_by_metadata(
        project_id=project_id, filters=source_filter, limit=scroll_limit,
    )
    # Cross-check: if the scroll returned fewer records than the count, enumeration
    # was capped and some markers were silently skipped — log a warning per the
    # no-silent-caps convention so the operator knows the sweep is incomplete.
    if len(members) < total_source:
        logger.warning(
            'sweep_orphan_flag_markers: enumerated %d of %d source markers '
            '(scroll limit=%d) — scroll cap reached; re-run with a higher '
            '--limit value to ensure all orphans are covered.',
            len(members), total_source, scroll_limit,
        )
    # Orphans are the id-deduplicated, order-preserving union of the two
    # independent orphan predicates: missing kind (find_orphan_markers) and
    # missing task_id (find_taskless_markers, task 2108). A member missing
    # BOTH is caught by each predicate but must be deleted exactly once.
    kind_orphans = find_orphan_markers(members)
    taskless = find_taskless_markers(members)

    orphans: list[dict] = []
    seen_ids: set = set()
    for m in (*kind_orphans, *taskless):
        if m['id'] not in seen_ids:
            seen_ids.add(m['id'])
            orphans.append(m)

    orphan_ids = [o['id'] for o in orphans]

    # NOTE: taskless_orphan_count is the raw size of the taskless predicate
    # (find_taskless_markers), not a "taskless-only" diagnostic — it includes
    # members that are ALSO kind-orphans (e.g. a member missing both kind and
    # task_id). It therefore does not subtract from / sum cleanly against
    # orphan_count to reconstruct the union; it exists purely to make the
    # taskless predicate's contribution to the sweep observable.
    report: dict = {
        'dry_run': not args.apply,
        'before': before,
        'orphan_count': len(orphans),
        'orphan_ids': orphan_ids,
        'taskless_orphan_count': len(taskless),
    }

    if args.apply:
        delete_result = await delete_orphan_markers(
            memory_service, project_id, orphans,
        )
        report['deleted'] = delete_result['deleted']
        report['failed'] = delete_result['failed']

        # After counts
        after_source = await memory_service.count_memories_by_metadata(
            project_id=project_id, filters=source_filter,
        )
        after_kind = await memory_service.count_memories_by_metadata(
            project_id=project_id, filters=kind_filter,
        )
        report['after'] = {'total_source': after_source, 'total_with_kind': after_kind}

    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """Parse CLI args, build a live MemoryService, and run the sweep."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
    )
    parser = argparse.ArgumentParser(
        description=(
            'Detect (and optionally delete) stage1_flag_marker records missing '
            'kind or task_id.'
        ),
    )
    parser.add_argument(
        '--apply', action='store_true', default=False,
        help=(
            'Commit deletions of stage1_flag_marker records missing kind or '
            'task_id (default: dry-run only).'
        ),
    )
    parser.add_argument(
        '--project-id', dest='project_id', default='dark_factory',
        help='Project to sweep (default: dark_factory).',
    )
    parser.add_argument(
        '--limit', type=int, default=1000,
        help='Maximum records to enumerate per scroll (default: 1000). '
             'Increase if count_memories_by_metadata shows >1000 source markers.',
    )
    args = parser.parse_args()

    async def _run_live() -> dict:
        from fused_memory.config.schema import FusedMemoryConfig  # noqa: PLC0415
        from fused_memory.services.memory_service import MemoryService  # noqa: PLC0415

        config = FusedMemoryConfig()
        memory = MemoryService(config)
        try:
            await memory.initialize()
            return await run(args, memory)
        finally:
            if hasattr(memory, 'close'):
                await memory.close()

    report = asyncio.run(_run_live())
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
