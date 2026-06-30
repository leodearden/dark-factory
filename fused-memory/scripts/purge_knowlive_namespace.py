#!/usr/bin/env python3
"""One-shot purge of the legacy 'knowlive' (no-separator) Graphiti graph and Mem0
namespace, plus retirement of the two Stage-1 flag memories that keep re-surfacing
this issue in reconciliation cycles (task 1937).

Background
----------
Task 515 ("re-key 'knowlive' -> 'know_live'") was deliberately CANCELLED. It left
~461 orphaned Graphiti nodes in a standalone 'knowlive' FalkorDB graph and ~22
orphaned Mem0 memories scoped to project_id='knowlive'. Two dark_factory flag
edges (c2b5ac3f-e8c5-41c2-a455-73f60d821679, af5dec02-bb63-4e3d-8552-f08da8d7517b)
record "N records awaiting re-keying" and keep resurfacing in reconciliation
because the re-key never happened and never will.

Purge, not re-key
------------------
'know-live'/'know_live' is a LIVE project. Merging a year-old, separately
cancelled and contamination-tainted blob of orphan data into it is the riskier
choice, and there is no clean re-key primitive to do it with anyway (Graphiti's
project_id IS the FalkorDB graph name, so re-key would mean a cross-graph node
move; Mem0 has no payload-update primitive). This script purges the legacy
namespace outright and retires (invalidates, does NOT delete) the two flag edges
that record the cancelled migration as still-pending.

What this script does
----------------------
  1. Enumerates, then (with --apply) DETACH DELETEs every node in the 'knowlive'
     FalkorDB graph via a single Cypher statement scoped to that graph only.
  2. Enumerates, then (with --apply) deletes every Mem0 memory scoped to
     project_id='knowlive' (best-effort -- one failure does not abort the batch).
  3. Invalidates (via update_edge(invalid_at=...), NOT delete) the two stale-flag
     edges in the dark_factory graph, preserving their audit trail. The task-515
     "cancelled" context edge (ea78d7a9) is left untouched -- it remains true.

This is IRREVERSIBLE for steps 1 and 2 (FalkorDB DETACH DELETE / Qdrant point
delete are not soft-deletes). Step 3 is reversible via
``update_edge(..., clear_invalid_at=True)``.

Usage
-----
  # Dry run (default): print a full JSON manifest, touch nothing. Redirect to a
  # file BEFORE running --apply -- the dry-run report doubles as the only
  # recovery record of what is about to be deleted.
  python scripts/purge_knowlive_namespace.py > knowlive_purge_manifest.json

  # Commit the purge + flag retirement.
  python scripts/purge_knowlive_namespace.py --apply

  # Override defaults (rarely needed).
  python scripts/purge_knowlive_namespace.py --apply \
      --namespace knowlive --flag-host-project dark_factory \
      --flag-uuids c2b5ac3f-e8c5-41c2-a455-73f60d821679 af5dec02-bb63-4e3d-8552-f08da8d7517b
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

LEGACY_NAMESPACE: str = 'knowlive'
FLAG_HOST_PROJECT: str = 'dark_factory'
STALE_FLAG_EDGE_UUIDS: tuple[str, ...] = (
    'c2b5ac3f-e8c5-41c2-a455-73f60d821679',
    'af5dec02-bb63-4e3d-8552-f08da8d7517b',
)

logger = logging.getLogger('purge_knowlive_namespace')


# ---------------------------------------------------------------------------
# Pure core
# ---------------------------------------------------------------------------

def build_purge_report(
    namespace: str,
    graphiti_rows: list[dict],
    mem0_members: list[dict],
    flag_uuids: tuple[str, ...] | list[str],
    *,
    dry_run: bool,
) -> dict:
    """Assemble the report skeleton dict from already-enumerated inputs. No I/O."""
    return {
        'namespace': namespace,
        'dry_run': dry_run,
        'graphiti': {
            'count': len(graphiti_rows),
            'node_uuids': [row['uuid'] for row in graphiti_rows],
        },
        'mem0': {
            'count': len(mem0_members),
            'memory_ids': [m['id'] for m in mem0_members],
        },
        'stale_flags': {
            'uuids': list(flag_uuids),
        },
    }


# ---------------------------------------------------------------------------
# Graphiti: enumerate + purge
# ---------------------------------------------------------------------------

async def enumerate_graphiti_namespace(
    graphiti: Any,
    namespace: str,
    *,
    limit: int = 1000,
) -> list[dict]:
    """Read-only enumeration of every node in the *namespace* FalkorDB graph."""
    graph = graphiti._graph_for(namespace)
    result = await graph.ro_query(
        'MATCH (n) RETURN n.uuid, labels(n), n.name LIMIT $limit',
        {'limit': limit},
    )
    rows = result.result_set or []
    if len(rows) >= limit:
        logger.warning(
            "purge_knowlive_namespace: enumerated %d node(s) in graph '%s', which "
            "hit limit=%d -- enumeration may be incomplete. Re-run with a higher "
            "--limit value to ensure the full namespace is covered.",
            len(rows), namespace, limit,
        )
    return [
        {'uuid': row[0], 'labels': row[1], 'name': row[2]}
        for row in rows
    ]


async def purge_graphiti_namespace(graphiti: Any, namespace: str) -> dict:
    """Best-effort ``MATCH (n) DETACH DELETE n`` against the *namespace* graph."""
    graph = graphiti._graph_for(namespace)
    try:
        await graph.query('MATCH (n) DETACH DELETE n')
    except Exception as e:
        logger.warning(
            "purge_knowlive_namespace: failed to purge graph '%s': %s",
            namespace, e,
        )
        return {'ok': False, 'error': str(e)}
    return {'ok': True, 'error': None}


# ---------------------------------------------------------------------------
# Mem0: enumerate + purge
# ---------------------------------------------------------------------------

async def enumerate_mem0_namespace(
    memory_service: Any,
    namespace: str,
    *,
    limit: int = 1000,
) -> tuple[list[dict], int]:
    """Deterministic (non-semantic) enumeration of Mem0 memories scoped to *namespace*."""
    count = await memory_service.count_memories_by_metadata(
        project_id=namespace, filters={},
    )
    members = await memory_service.get_memories_by_metadata(
        project_id=namespace, filters={}, limit=limit,
    )
    if len(members) < count:
        logger.warning(
            "purge_knowlive_namespace: enumerated %d of %d mem0 memories in "
            "'%s' (scroll limit=%d) -- scroll cap reached; re-run with a "
            "higher --limit value to ensure all records are covered.",
            len(members), count, namespace, limit,
        )
    return members, count


async def purge_mem0_namespace(
    memory_service: Any,
    namespace: str,
    members: list[dict],
) -> dict:
    """Best-effort delete of every Mem0 *members* entry scoped to *namespace*."""
    if not members:
        return {'deleted': 0, 'failed': []}

    async def _delete_one(member: dict):
        return await memory_service.delete_memory(
            memory_id=member['id'],
            store='mem0',
            project_id=namespace,
            _source='purge_knowlive_namespace',
        )

    results = await asyncio.gather(
        *(_delete_one(m) for m in members),
        return_exceptions=True,
    )

    deleted = 0
    failed: list[str] = []
    for member, result in zip(members, results, strict=False):
        if isinstance(result, BaseException):
            logger.warning(
                'purge_knowlive_namespace: failed to delete mem0 memory %s: %s',
                member['id'], result,
            )
            failed.append(member['id'])
        else:
            deleted += 1

    return {'deleted': deleted, 'failed': failed}


# ---------------------------------------------------------------------------
# Stale-flag retirement (dark_factory)
# ---------------------------------------------------------------------------

async def retire_stale_flags(
    memory_service: Any,
    edge_uuids: tuple[str, ...] | list[str] = STALE_FLAG_EDGE_UUIDS,
    host_project: str = FLAG_HOST_PROJECT,
    *,
    invalidation_time: datetime,
) -> dict:
    """Best-effort ``update_edge(invalid_at=...)`` for each stale-flag edge uuid."""
    async def _retire_one(edge_uuid: str):
        return await memory_service.update_edge(
            edge_uuid=edge_uuid,
            project_id=host_project,
            invalid_at=invalidation_time,
            _source='purge_knowlive_namespace',
        )

    results = await asyncio.gather(
        *(_retire_one(u) for u in edge_uuids),
        return_exceptions=True,
    )

    invalidated = 0
    failed: list[str] = []
    for edge_uuid, result in zip(edge_uuids, results, strict=False):
        if isinstance(result, BaseException):
            logger.warning(
                'purge_knowlive_namespace: failed to invalidate stale-flag edge %s: %s',
                edge_uuid, result,
            )
            failed.append(edge_uuid)
        else:
            invalidated += 1

    return {'invalidated': invalidated, 'failed': failed}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def run(
    args: Any,
    memory_service: Any,
    *,
    invalidation_time: datetime,
) -> dict:
    """Enumerate the legacy namespace and, with ``args.apply``, purge + retire flags."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """Parse CLI args, build a live MemoryService, and run the purge."""
    raise NotImplementedError


if __name__ == '__main__':
    sys.exit(main())
