#!/usr/bin/env python3
"""One-shot sweep: detect and optionally delete stage1_flag_marker records in Mem0/Qdrant
that are missing the ``kind='stage1_flag_marker'`` metadata key (task-1659 orphans).

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
            - orphan_count (int)
            - orphan_ids (list[str])
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
    orphans = find_orphan_markers(members)
    orphan_ids = [o['id'] for o in orphans]

    report: dict = {
        'dry_run': not args.apply,
        'before': before,
        'orphan_count': len(orphans),
        'orphan_ids': orphan_ids,
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
        description='Detect (and optionally delete) stage1_flag_marker records missing kind.',
    )
    parser.add_argument(
        '--apply', action='store_true', default=False,
        help='Commit deletions (default: dry-run only).',
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
