#!/usr/bin/env python3
"""Migrate misrouted cross-graph Graphiti nodes to their resolved home graph
(CGL-ζ, task 2272, PRD ``plans/cross-graph-entity-leak-prd.md`` contract seam
S7, Phase 1 LEAF).

Background
----------
``scripts/investigate_cross_graph_duplication.py`` (task 2116) confirmed that
``GraphitiBackend._graph_for``/``_driver_for`` use a node's ``group_id``
verbatim as the FalkorDB graph name, with no canonicalization -- so a node
whose ``group_id`` names a spelling variant, alias, or otherwise-wrong graph
key ends up "foreign" to the graph it actually lives in (``n.group_id <>
$graph_key``). This script census-enumerates every such foreign node across
every populated graph, classifies each one's correct disposition, and (with
human review of the emitted manifest) re-homes it via the CGL-ε primitives.

ζ is a pure CONSUMER of ε (task 2271, ``fused_memory.maintenance.cross_graph_move``,
merged to main): ``move_entity_across_graphs`` (S5) and
``merge_foreign_duplicate`` (S6). PRD decision G4 centralizes all Cypher and
the byte-exact ``vecf32`` raw-transport read in ε to avoid duplicating that
logic (and the file-lock contention a second implementation would invite) --
this script never re-implements a cross-graph move or a raw embedding read.

Environment requirements (standing ops-script lesson -- cf.
``purge_knowlive_namespace.py`` / ``prune_recon_cycle_summaries.py``)
-----------------------------------------------------------------------------
Run this script under the fused-memory SERVICE environment, not a bare
checkout shell: ``source .env`` first, and ensure ``PROJECT_ROOT`` +
``DASHBOARD_KNOWN_PROJECT_ROOTS`` are set (or pass ``--config`` to point at a
specific config file, which sets ``CONFIG_PATH`` before ``FusedMemoryConfig``
loads). Running with an incomplete environment silently narrows which
graphs/projects are visible to the census -- it does not raise -- so a
manifest produced from a bad environment can under-report the true
population.

Contract
--------
  * Dry run is the DEFAULT (no ``--apply``): the census + classification runs
    and a JSON manifest is printed. Nothing is mutated.
  * The emitted manifest IS the recovery record -- there is no other log of
    what was classified and why. Redirect dry-run output to a file and have
    it human-reviewed BEFORE passing it to ``--apply``.
  * ``--apply`` consumes an EXISTING, human-reviewed manifest file (via
    ``--manifest``) -- it does NOT recompute a fresh plan. It refuses to run
    without one.
  * Destructive steps are flagged: every mutation ``--apply`` performs is a
    call into an ε primitive (``move_entity_across_graphs`` /
    ``merge_foreign_duplicate``), both of which are CREATE-before-DELETE
    (crash-safe) and idempotent on re-run.
  * UNRESOLVED nodes (no populated-home and no ``ALIAS_MAP`` entry) are never
    silently dropped or routed to a new/unpopulated graph -- they block
    ``--apply`` for that node only and force a non-zero exit.

Scope note
----------
This module's test suite is mock-only (per project convention) and asserts
correctness at the manifest/classification/routing level (dry-run-touches-
nothing, apply dispatches to the right ε primitive, post-verify catches a
count mismatch). LIVE byte-fidelity (zero foreign nodes remaining, byte-exact
vector transport, against a REAL FalkorDB) is explicitly η's live
throwaway-graph rehearsal, not this task -- see ε's module docstring and PRD
decision 5. Foreign Episodic nodes are classified and listed here too, but
ε's ``move_entity_across_graphs`` is :Entity-scoped (it only reattaches, does
not relocate, Episodic nodes) -- their live relocation is also η's concern.

Usage
-----
  # Dry run (default): print a full JSON manifest, touch nothing. Redirect to
  # a file and have it reviewed BEFORE running --apply.
  python scripts/migrate_cross_graph_leak.py > cgl_migration_manifest.json

  # Commit the migration from a reviewed manifest.
  python scripts/migrate_cross_graph_leak.py --apply --manifest cgl_migration_manifest.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fused_memory.maintenance.cross_graph_move import (
    merge_foreign_duplicate,
    move_entity_across_graphs,
)

logger = logging.getLogger('migrate_cross_graph_leak')

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Explicit, human-reviewable alias map: orphan group_id spellings with no
# populated graph of their own -> their canonical populated home. No silent
# canonicalization beyond this table -- see resolve_target_graph (PRD
# decision 4).
ALIAS_MAP: dict[str, str] = {
    'knowlive': 'know_live',
    'know-live': 'know_live',
    'pump-web-ui': 'pump_web_ui',
    'dark-factory': 'dark_factory',
}

# Node disposition constants (see disposition_for).
MOVE: str = 'MOVE'
MERGE: str = 'MERGE'
UNRESOLVED: str = 'UNRESOLVED'

DEFAULT_PAGE_SIZE: int = 1000

# Hard safety cap on the number of SKIP/LIMIT pages census_foreign_nodes will
# fetch for a single graph, purely to bound worst-case pagination against a
# pathological/huge foreign population -- normal use never approaches it.
# Tests exercise the cap-hit WARNING path by monkeypatching this down.
MAX_CENSUS_PAGES: int = 1000


# ---------------------------------------------------------------------------
# Pure core
# ---------------------------------------------------------------------------

def resolve_target_graph(
    group_id: str,
    populated_graphs: set[str] | list[str] | frozenset[str],
    alias_map: dict[str, str],
) -> str | None:
    """Resolve *group_id* to its correct home graph, or None if UNRESOLVED.

    Resolution order (PRD decision 4 -- no silent canonicalization beyond
    this explicit, human-reviewable table):
      1. *group_id* itself names a real, already-populated graph -> that is
         its home (a displaced-only node: it just needs to move back).
      2. *group_id* is an orphan (not populated) but has an explicit
         *alias_map* entry -> the mapped canonical target.
      3. Neither -> None (UNRESOLVED). Never falls back to a generic
         normalization (e.g. blind hyphen->underscore rewrite) and never
         invents/targets a new, unpopulated graph.
    """
    if group_id in populated_graphs:
        return group_id
    if group_id in alias_map:
        return alias_map[group_id]
    return None


def disposition_for(target_graph: str | None, present_in_target: bool) -> str:
    """Classify a foreign node's migration disposition.

    - *target_graph* is None (resolve_target_graph could not resolve a home)
      -> UNRESOLVED, regardless of *present_in_target*.
    - *target_graph* resolved AND already present there -> MERGE (S6): a
      genuine duplicate uuid also lives in the home graph.
    - *target_graph* resolved AND absent there -> MOVE (S5): a displaced-only
      node with no copy at home yet.
    """
    if target_graph is None:
        return UNRESOLVED
    if present_in_target:
        return MERGE
    return MOVE


def build_manifest(
    classified_nodes: list[dict],
    census_counts: dict[str, int],
    *,
    dry_run: bool,
    alias_map: dict[str, str] = ALIAS_MAP,
) -> dict:
    """Assemble the migration manifest dict from already-classified inputs. No I/O.

    The manifest IS the recovery record (see module docstring): it echoes
    *alias_map* (so a human reviewer sees exactly which mapping produced
    each MOVE/MERGE target), the per-graph *census_counts*, every classified
    node record verbatim, summary tallies, and the uuids of any UNRESOLVED
    node (which block --apply for that node only).
    """
    summary = {MOVE: 0, MERGE: 0, UNRESOLVED: 0}
    unresolved_uuids: list[str] = []
    for node in classified_nodes:
        disposition = node['disposition']
        summary[disposition] = summary.get(disposition, 0) + 1
        if disposition == UNRESOLVED:
            unresolved_uuids.append(node['uuid'])
    summary['total'] = len(classified_nodes)

    return {
        'dry_run': dry_run,
        'alias_map': dict(alias_map),
        'nodes': list(classified_nodes),
        'census': dict(census_counts),
        'summary': summary,
        'unresolved_uuids': unresolved_uuids,
    }


# ---------------------------------------------------------------------------
# Graphiti: read-only census
# ---------------------------------------------------------------------------

async def census_foreign_nodes(
    graphiti: Any,
    graph_key: str,
    *,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> list[dict]:
    """Read-only, paged enumeration of every node foreign to *graph_key*.

    "Foreign" means ``n.group_id <> graph_key`` -- the node lives in this
    FalkorDB graph but its own ``group_id`` property names a different graph.
    Paginates via SKIP/LIMIT (never ``collect()``, which truncates) until a
    short page confirms the true end; counts are therefore always freshly
    recomputed, never a baked-in/cached figure.

    A hard ``MAX_CENSUS_PAGES`` safety cap bounds worst-case pagination
    against a pathological/huge foreign population. If that cap is reached
    while the last page fetched was still full (so there may be more), a
    WARNING is logged -- the caller must not treat the returned list as
    guaranteed-complete in that case (no silent caps).
    """
    graph = graphiti._graph_for(graph_key)
    rows: list[dict] = []
    skip = 0
    for _page_num in range(MAX_CENSUS_PAGES):
        result = await graph.ro_query(
            'MATCH (n) WHERE n.group_id <> $graph_key '
            'RETURN n.uuid, n.name, n.group_id, labels(n) '
            'SKIP $skip LIMIT $limit',
            {'graph_key': graph_key, 'skip': skip, 'limit': page_size},
        )
        page = result.result_set or []
        rows.extend(
            {
                'uuid': row[0],
                'name': row[1],
                'group_id': row[2],
                'labels': row[3],
                'source_graph': graph_key,
            }
            for row in page
        )
        if len(page) < page_size:
            return rows
        skip += page_size

    logger.warning(
        "census_foreign_nodes: graph '%s' hit the %d-page cap (page_size=%d) "
        'while the last page was still full -- enumeration may be '
        'incomplete. Re-run with a larger --page-size to ensure the full '
        'foreign population is covered.',
        graph_key, MAX_CENSUS_PAGES, page_size,
    )
    return rows


async def node_present_in_graph(graph: Any, uuid: str) -> bool:
    """Read-only presence probe: True iff a node with *uuid* exists in *graph*.

    Distinguishes MOVE (absent from the resolved target -- displaced-only)
    from MERGE (already present in the resolved target -- a genuine
    duplicate) in disposition_for.
    """
    result = await graph.ro_query(
        'MATCH (n {uuid: $uuid}) RETURN n.uuid LIMIT 1',
        {'uuid': uuid},
    )
    return bool(result.result_set)


async def count_node_edges_episodes(graph: Any, uuid: str) -> dict:
    """Read-only {'edges', 'episodes'} counts incident to *uuid* in *graph*.

    Surfaced on each manifest node so a human reviewer can see how much
    topology a MOVE/MERGE will carry before approving --apply. Both reads
    are read-only counts -- never .query.
    """
    edges_result = await graph.ro_query(
        'MATCH (n {uuid: $uuid})-[e:RELATES_TO]-(m) RETURN count(DISTINCT e)',
        {'uuid': uuid},
    )
    episodes_result = await graph.ro_query(
        'MATCH (ep:Episodic)-[e:MENTIONS]->(n {uuid: $uuid}) RETURN count(e)',
        {'uuid': uuid},
    )
    edges = edges_result.result_set[0][0] if edges_result.result_set else 0
    episodes = episodes_result.result_set[0][0] if episodes_result.result_set else 0
    return {'edges': edges, 'episodes': episodes}
