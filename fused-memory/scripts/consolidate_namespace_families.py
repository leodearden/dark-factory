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
