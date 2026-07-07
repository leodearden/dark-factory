#!/usr/bin/env python3
"""Consolidation script for the cross-graph entity leak cleanup (CGL-θ, task 2274).

Three independent, order-insensitive operations against the FalkorDB
(Graphiti) and Qdrant (Mem0) backends. Each is dry-run by default and each
is guarded so a partial/incomplete state is reported UNRESOLVED rather than
destroying data:

  1. GRAPH-FAMILY MERGES with IDENTITY REWRITE -- sibling Graphiti graphs
     that are the same logical project under a different key (hyphenated,
     no-separator, ...) are merged into their underscore-canonical graph.
     Reuses ε's move primitive (``move_entity_across_graphs``,
     ``fused_memory.maintenance.cross_graph_move``, task 2271) with
     ``rewrite_group_id=canonical`` -- the Phase-2 identity rewrite (PRD
     decision 6) that ε's Phase-1 moves did not need.
  2. QDRANT COLLECTION MERGES -- legacy/divergent Mem0 collections (from
     historical ``collection_prefix`` values, RCA §4) are merged into their
     ``fused_<project>`` target: scrolled with vectors, payload ``user_id``
     rewritten to the canonical project id, upserted into the target, and
     the source collection deleted ONLY once fully drained.
  3. GUARDED JUNK-KEY DELETION -- known-junk FalkorDB graph keys (and any
     family sibling emptied by step 1) are removed via ``GRAPH.DELETE``
     (``graphiti._graph_for(key).delete()``) -- NOT the ``MATCH (n) DETACH
     DELETE n`` pattern in ``purge_knowlive_namespace.py``, which empties a
     graph but leaves its key in ``GRAPH.LIST``. Guarded on a live node
     count of exactly 0; a non-empty key is reported UNRESOLVED and its
     deletion is blocked.

Reviewable, static module-level constants (``GRAPH_FAMILY_ALIASES``,
``COLLECTION_MERGES``, ``JUNK_KEYS``) are the human-reviewable artifact PRD
decision 4 requires -- this script ships its OWN config (a sibling to, not
shared with, the ζ migration script's alias map).

Contract (S7, per ``plans/cross-graph-entity-leak-prd.md``): dry-run by
default, prints a full JSON manifest; ``--apply`` performs the mutations
described above plus a post-verify recount, and exits non-zero if any
section of the manifest carries an ``UNRESOLVED`` disposition (see
``has_unresolved``).

Scope: this script + its test suite are MOCK-unit only (MagicMock graphs,
AsyncMock Qdrant client) -- no live FalkorDB/Qdrant, and no assertion of
embedding byte-fidelity or ``GRAPH.LIST`` cleanliness. Those are the LIVE
B7 signal, mandated at the ι live throwaway-graph rehearsal (PRD decision
5), not asserted here.

Usage
-----
  # Dry run (default): print the full JSON manifest, touch nothing.
  python scripts/consolidate_namespace_families.py > consolidation_manifest.json

  # Commit the merges + junk-key deletions (exits non-zero on UNRESOLVED).
  python scripts/consolidate_namespace_families.py --apply
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from typing import Any

from fused_memory.maintenance.cross_graph_move import MoveResult, move_entity_across_graphs

logger = logging.getLogger('consolidate_namespace_families')


# ---------------------------------------------------------------------------
# Module-level constants (reviewable config -- PRD decision 4)
# ---------------------------------------------------------------------------

# Sibling FalkorDB graph key -> canonical underscore-form key. Canonical is
# always the underscore spelling (config.yaml, registry keys, factory-init
# rule -- PRD decision 6 identity rewrite target). The solar family
# (my_solar_challenge / solar_challenge_platform) is DELIBERATELY EXCLUDED:
# PRD Open Q1's default is keep-separate absent an explicit human decision;
# flipping it to a merge is a one-line addition at ι, not a default here.
GRAPH_FAMILY_ALIASES: dict[str, str] = {
    'know-live': 'know_live',
    'knowlive': 'know_live',
    'pump-web-ui': 'pump_web_ui',
}

# Legacy/divergent Qdrant collection -> canonical fused_<project> target.
# These collections hold REAL data from old per-project collection_prefix
# configs (RCA §4) -- the contract is merge, never plain-delete. The
# ambiguous collections (reify_ -- empty project id; fused_fused_memory) are
# DELIBERATELY OMITTED: PRD Open Q2 defers their disposition to ι human
# inspection, since auto-merging them risks mis-routing real data into the
# wrong project.
COLLECTION_MERGES: dict[str, str] = {
    'fused_dark-factory': 'fused_dark_factory',
    'reify_reify': 'fused_reify',
    'autopilot_video_autopilot_video': 'fused_autopilot_video',
}

# Known-junk FalkorDB graph keys with no legitimate data (RCA §"empty junk
# keys" inventory) -- removal is guarded on a live node count of 0 (see
# delete_junk_key), so listing a key here never itself causes data loss.
JUNK_KEYS: tuple[str, ...] = (
    'dark-factory',
    '-home-leo-src-dark-factory',
    'my-project',
    'test-project',
    'default',
    '1098',
)


# ---------------------------------------------------------------------------
# Pure core
# ---------------------------------------------------------------------------

def rewrite_point_payload_user_id(payload: dict, canonical_user_id: str) -> dict:
    """Return a COPY of *payload* with 'user_id' set to *canonical_user_id*.

    Every other payload key is preserved unchanged; the input dict is never
    mutated (the caller may still need the original for logging/comparison).
    """
    rewritten = dict(payload)
    rewritten['user_id'] = canonical_user_id
    return rewritten


def canonical_user_id_for(target_collection: str) -> str:
    """Derive the canonical project id (Mem0 user_id) from a fused_<project>
    target collection name, by stripping the collection prefix.

    Mirrors ``Scope.mem0_user_id`` == ``project_id`` and
    ``Scope.mem0_collection_name`` == ``f'{prefix}_{project_id}'`` (models/
    scope.py); every COLLECTION_MERGES target is already in that
    fused_<project> shape.
    """
    return target_collection.removeprefix('fused_')


def build_consolidation_report(
    graph_family_items: list[dict],
    collection_items: list[dict],
    junk_key_items: list[dict],
    *,
    dry_run: bool,
) -> dict:
    """Assemble the manifest dict from already-computed per-item lists. No I/O."""
    return {
        'dry_run': dry_run,
        'graph_family_merges': list(graph_family_items),
        'collection_merges': list(collection_items),
        'junk_key_deletions': list(junk_key_items),
    }


# ---------------------------------------------------------------------------
# Graph inspection (read-only)
# ---------------------------------------------------------------------------

async def enumerate_graph_entity_nodes(
    graphiti: Any,
    key: str,
    *,
    limit: int = 1000,
) -> list[dict]:
    """Read-only enumeration of every :Entity node in the *key* FalkorDB graph.

    Scoped to :Entity (not every label) -- the family-merge move primitive
    (``move_entity_across_graphs``) only moves Entity nodes; any Episodic-
    only residual left behind in *key* is exactly what should make the
    junk-key guard (``delete_junk_key``) classify it UNRESOLVED rather than
    silently GRAPH.DELETE-ing it.
    """
    graph = graphiti._graph_for(key)
    result = await graph.ro_query(
        'MATCH (n:Entity) RETURN n.uuid, n.name LIMIT $limit',
        {'limit': limit},
    )
    rows = result.result_set or []
    if len(rows) >= limit:
        logger.warning(
            "consolidate_namespace_families: enumerated %d Entity node(s) in "
            "graph '%s', which hit limit=%d -- enumeration may be incomplete. "
            "Re-run with a higher --limit value to ensure the full graph is "
            "covered.",
            len(rows), key, limit,
        )
    return [{'uuid': row[0], 'name': row[1]} for row in rows]


async def count_graph_nodes(graphiti: Any, key: str) -> int:
    """Read-only total node count (every label) for the *key* FalkorDB graph.

    Used as the guard for ``delete_junk_key``: GRAPH.DELETE is only safe
    when this is exactly 0.
    """
    graph = graphiti._graph_for(key)
    result = await graph.ro_query('MATCH (n) RETURN count(n)')
    return int(result.result_set[0][0])


# ---------------------------------------------------------------------------
# Graph family merge (mutating)
# ---------------------------------------------------------------------------

async def merge_graph_family(
    graphiti: Any,
    sibling: str,
    canonical: str,
    node_rows: list[dict],
) -> dict:
    """Move every *node_rows* entry from *sibling* into *canonical*.

    Calls ``move_entity_across_graphs(graphiti, uuid, sibling, canonical,
    rewrite_group_id=canonical)`` once per row -- the Phase-2 identity
    rewrite (PRD decision 6) -- and tallies the returned MoveResults.
    """
    summary = {
        'nodes_moved': 0,
        'edges_moved': 0,
        'edges_skipped': 0,
        'mentions_moved': 0,
        'mentions_skipped': 0,
    }
    for row in node_rows:
        result = await move_entity_across_graphs(
            graphiti, row['uuid'], sibling, canonical, rewrite_group_id=canonical,
        )
        summary['nodes_moved'] += 1
        summary['edges_moved'] += result.edges_moved
        summary['edges_skipped'] += result.edges_skipped
        summary['mentions_moved'] += result.mentions_moved
        summary['mentions_skipped'] += result.mentions_skipped
    return summary


# ---------------------------------------------------------------------------
# Qdrant collection merge
# ---------------------------------------------------------------------------

async def scroll_collection_points(
    qdrant_client: Any,
    collection: str,
    *,
    limit: int = 1000,
) -> list:
    """Read-only scroll of every point in *collection*, WITH vectors.

    ``with_vectors=True`` is essential: omitting it drops embeddings from
    the returned points, which would silently destroy them once re-upserted
    into the target collection (see ``merge_collection``).
    """
    points, _next_offset = await qdrant_client.scroll(
        collection_name=collection,
        with_payload=True,
        with_vectors=True,
        limit=limit,
    )
    if len(points) >= limit:
        logger.warning(
            "consolidate_namespace_families: scrolled %d point(s) from "
            "collection '%s', which hit limit=%d -- scroll may be "
            "incomplete/capped. Re-run with a higher --limit value before "
            "merging, or the source collection will not be deleted "
            '(see merge_collection).',
            len(points), collection, limit,
        )
    return points


async def merge_collection(
    qdrant_client: Any,
    source: str,
    target: str,
    canonical_user_id: str,
    points: list,
    *,
    capped: bool,
) -> dict:
    """Upsert *points* (payload user_id rewritten to canonical) into *target*.

    Preserves each point's original id and vector -- only the payload's
    user_id is rewritten. The *source* collection is deleted ONLY when
    *capped* is False (the scroll that produced *points* was NOT capped,
    i.e. fully drained): a capped scroll means the enumeration may be
    incomplete, so deleting source would risk losing un-migrated data --
    the caller marks that case UNRESOLVED instead.
    """
    from qdrant_client.http import models as qmodels  # noqa: PLC0415

    upsert_points = [
        qmodels.PointStruct(
            id=point.id,
            vector=point.vector,
            payload=rewrite_point_payload_user_id(dict(point.payload or {}), canonical_user_id),
        )
        for point in points
    ]
    if upsert_points:
        await qdrant_client.upsert(collection_name=target, points=upsert_points)

    source_deleted = False
    if not capped:
        await qdrant_client.delete_collection(source)
        source_deleted = True

    return {'points_upserted': len(upsert_points), 'source_deleted': source_deleted}


# ---------------------------------------------------------------------------
# Guarded junk-key deletion
# ---------------------------------------------------------------------------

async def delete_junk_key(graphiti: Any, key: str, node_count: int) -> str:
    """GRAPH.DELETE the *key* graph (removes it from GRAPH.LIST) -- but ONLY
    when its live *node_count* is exactly 0.

    Guards against destroying a key that unexpectedly still holds data: a
    non-zero count returns 'UNRESOLVED' without ever calling ``.delete()``
    (deletion blocked, no data loss). Best-effort: a raising ``.delete()``
    is caught and reported as UNRESOLVED rather than propagating.
    """
    if node_count > 0:
        return 'UNRESOLVED'
    graph = graphiti._graph_for(key)
    try:
        await graph.delete()
    except Exception as e:
        logger.warning(
            "consolidate_namespace_families: failed to GRAPH.DELETE junk key '%s': %s",
            key, e,
        )
        return 'UNRESOLVED'
    return 'DELETE'
