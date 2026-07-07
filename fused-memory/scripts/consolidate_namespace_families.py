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
