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
$graph_key`` -- or ``group_id`` missing/``NULL`` entirely, which Cypher's
three-valued logic would otherwise silently exclude from a naive ``<>``
predicate: ``NULL <> $graph_key`` evaluates to ``NULL``, not ``TRUE``). This
script census-enumerates every such foreign node across every populated
graph, classifies each one's correct disposition, and (with human review of
the emitted manifest) re-homes it via the CGL-ε primitives. A ``NULL``/
missing ``group_id`` has no meaningful value to resolve against
``populated_graphs``/``ALIAS_MAP``, so it always classifies ``UNRESOLVED``
(see ``resolve_target_graph``) -- it is listed for visibility, never
silently dropped, but blocks ``--apply`` for that node only.

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
  * A MOVE/MERGE that completes but silently drops topology (an ε primitive's
    ``MoveResult.edges_skipped``/``mentions_skipped``, or a
    ``MergeResult.home_edge_count_after`` that does not reconcile with
    ``home_edge_count_before + edges_recreated`` -- both cases the other
    endpoint being absent from the target/home graph) is surfaced as a
    blocked ``apply_results`` entry with a descriptive ``error``, not a clean
    exit -- see ``_move_result_entry``/``_merge_result_entry``.
  * UNRESOLVED nodes (no populated-home and no ``ALIAS_MAP`` entry) are never
    silently dropped or routed to a new/unpopulated graph -- they block
    ``--apply`` for that node only and force a non-zero exit.
  * A manifest node record missing a required field (``uuid``/
    ``disposition``/``source_graph``/``target_graph`` -- e.g. from hand
    editing) blocks that node only, with a descriptive error, rather than
    raising a bare ``KeyError`` that would abort the rest of the batch after
    earlier nodes were already mutated.

Scope note
----------
This module's test suite is mock-only (per project convention) and asserts
correctness at the manifest/classification/routing level (dry-run-touches-
nothing, apply dispatches to the right ε primitive, post-verify catches a
count mismatch). LIVE byte-fidelity (zero foreign nodes remaining, byte-exact
vector transport, against a REAL FalkorDB) is explicitly η's live
throwaway-graph rehearsal, not this task -- see ε's module docstring and PRD
decision 5. Foreign non-``:Entity`` nodes (Episodic, Community, ...) are
classified ``EPISODIC_SKIP`` -- listed in the manifest for visibility but
NEVER dispatched to an ``:Entity``-scoped primitive (ε's
``move_entity_across_graphs`` / ``merge_foreign_duplicate`` and this script's
own ``rekey_node_in_place`` all require an ``:Entity`` match) -- their live
relocation remains η's concern.

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
import types
from pathlib import Path
from typing import Any

from fused_memory.maintenance.cross_graph_move import (
    SubgraphEdgeResult,
    create_moved_node,
    delete_source_node,
    recreate_subgraph_relationships,
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
# A resolved target_graph that equals the node's OWN source_graph: the node
# already physically lives in its home graph and only its group_id property
# is a variant spelling (e.g. 'know-live' aliasing to the 'know_live' graph
# it's already in). This is an in-place re-key, never a cross-graph MOVE/
# MERGE -- see classify_node and rekey_node_in_place.
REKEY: str = 'REKEY'
# A foreign node whose labels do NOT positively confirm :Entity (Episodic,
# Community, or no labels at all). The epsilon primitives
# (move_entity_across_graphs / merge_foreign_duplicate) and this script's own
# rekey_node_in_place all MATCH (n:Entity {uuid}) -- dispatching a non-Entity
# node to one would either raise NodeNotFoundError (aborting the whole apply)
# or silently no-op (surfacing only as a post-verify count mismatch). Such a
# node is listed in the manifest for visibility but never actioned here; see
# classify_node.
EPISODIC_SKIP: str = 'EPISODIC_SKIP'

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
    group_id: str | None,
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

    *group_id* may be None (a node with a missing/``NULL`` ``group_id``
    property -- see ``census_foreign_nodes``): None is never a member of
    *populated_graphs* or a key in *alias_map*, so it always falls through
    to case 3 (UNRESOLVED), same as any other unmapped orphan.
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
    FalkorDB graph but its own ``group_id`` property names a different graph
    -- OR ``n.group_id IS NULL`` (missing entirely). The ``IS NULL`` half is
    required, not defensive belt-and-suspenders: Cypher's ``<>`` uses
    three-valued logic, so ``NULL <> $graph_key`` evaluates to ``NULL`` (not
    ``TRUE``) and a plain ``WHERE n.group_id <> $graph_key`` would silently
    exclude every ``NULL``-``group_id`` node from the census entirely, with
    no error and no log line. Paginates via SKIP/LIMIT (never ``collect()``,
    which truncates) until a short page confirms the true end; counts are
    therefore always freshly recomputed, never a baked-in/cached figure.

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
            'MATCH (n) WHERE n.group_id IS NULL OR n.group_id <> $graph_key '
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


async def classify_node(
    graphiti: Any,
    node: dict,
    populated_graphs: set[str] | list[str] | frozenset[str],
    *,
    alias_map: dict[str, str] = ALIAS_MAP,
) -> dict:
    """Classify one census row (see census_foreign_nodes) into a manifest record.

    Orchestrates the pure helpers above over a single foreign-node row:
    resolve_target_graph decides where the node belongs.

    - Labels do NOT positively confirm :Entity (Episodic, Community, or no
      labels at all): EPISODIC_SKIP, no presence probe. The :Entity-scoped
      epsilon primitives and rekey_node_in_place can never act on such a
      node -- dispatching/probing it is a trap (see the EPISODIC_SKIP
      constant). target_graph is still recorded informationally (where the
      node would belong, for eta's live relocation).
    - Unresolved (target is None): UNRESOLVED, no presence probe (no target
      to probe).
    - Resolved target EQUALS the node's own source_graph: REKEY, no presence
      probe. The node already physically lives in its home graph and only
      its group_id property is a variant spelling -- probing the target here
      would find the node ITSELF (present=True), misclassifying it as MERGE
      and sending it through merge_foreign_duplicate(wrong==home), which
      DETACH DELETEs the node's only copy (see the REKEY constant and
      rekey_node_in_place). disposition_for is not consulted for this case:
      the target==source decision is made here, not there.
    - Resolved target DIFFERS from source_graph: a presence probe against
      the target feeds disposition_for's MOVE/MERGE split, as before.

    edge_count/episode_count are always read from the SOURCE graph (not the
    target), so a human reviewer can see how much topology a MOVE/MERGE/
    REKEY will carry before approving --apply.
    """
    target_graph = resolve_target_graph(node['group_id'], populated_graphs, alias_map)
    labels = node.get('labels') or []
    if 'Entity' not in labels:
        disposition = EPISODIC_SKIP
    elif target_graph is None:
        disposition = UNRESOLVED
    elif target_graph == node['source_graph']:
        disposition = REKEY
    else:
        present_in_target = await node_present_in_graph(
            graphiti._graph_for(target_graph), node['uuid'],
        )
        disposition = disposition_for(target_graph, present_in_target)

    counts = await count_node_edges_episodes(
        graphiti._graph_for(node['source_graph']), node['uuid'],
    )

    return {
        'uuid': node['uuid'],
        'name': node['name'],
        'source_graph': node['source_graph'],
        'target_graph': target_graph,
        'disposition': disposition,
        'edge_count': counts['edges'],
        'episode_count': counts['episodes'],
    }


# ---------------------------------------------------------------------------
# Graphiti: in-place mutation (REKEY only)
# ---------------------------------------------------------------------------

async def rekey_node_in_place(graph: Any, uuid: str, new_group_id: str) -> None:
    """In-place group_id re-key for a REKEY-dispositioned node.

    Unlike MOVE (``move_entity_across_graphs``) and MERGE
    (``merge_foreign_duplicate``), a REKEY node already physically lives in
    its home graph -- only its ``group_id`` property is a variant spelling.
    This rewrites ``group_id`` via ``SET``, uniformly across the node and
    its incident RELATES_TO edges and Episodic MENTIONS links (mirroring
    ``move_entity_across_graphs``' uniform ``new_group_id`` treatment across
    a moved subgraph) -- and ONLY via ``SET``. It never issues ``CREATE`` or
    ``DETACH DELETE``, so there is no crash window in which the node's only
    copy could be lost (the data-loss trap this whole REKEY path exists to
    avoid -- see the REKEY constant and classify_node).
    """
    await graph.query(
        'MATCH (n:Entity {uuid: $uuid}) SET n.group_id = $gid',
        {'uuid': uuid, 'gid': new_group_id},
    )
    await graph.query(
        'MATCH (n:Entity {uuid: $uuid})-[e:RELATES_TO]-(m:Entity) '
        'SET e.group_id = $gid',
        {'uuid': uuid, 'gid': new_group_id},
    )
    await graph.query(
        'MATCH (ep:Episodic)-[e:MENTIONS]->(n:Entity {uuid: $uuid}) '
        'SET e.group_id = $gid',
        {'uuid': uuid, 'gid': new_group_id},
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def load_reviewed_manifest(path: str | Path) -> dict:
    """Load a previously-emitted, human-reviewed manifest JSON file.

    ``--apply`` consumes this EXISTING file verbatim -- it never recomputes
    a fresh census/classification (see the module docstring's Contract
    section): the dry-run manifest is the sole recovery record, and
    recomputing at apply time would bypass human review and could act on a
    census that has drifted since it was reviewed.
    """
    return json.loads(Path(path).read_text())


# Required keys on every manifest node record dispatched by run()'s apply
# loop. A classify_node-produced record always carries all four (plus
# name/edge_count/episode_count, not needed for dispatch) -- this constant
# exists for a hand-edited/malformed manifest file, see
# missing_required_node_keys.
_REQUIRED_NODE_KEYS: tuple[str, ...] = ('uuid', 'disposition', 'source_graph', 'target_graph')


def missing_required_node_keys(node: dict) -> list[str]:
    """Return which of _REQUIRED_NODE_KEYS are absent from *node*.

    A manifest file is just JSON -- nothing enforces that a hand-edited copy
    still carries every key a classify_node record does. run()'s apply loop
    calls this FIRST, before indexing ``node['uuid']``/``node['disposition']``/
    etc., so a malformed record blocks that node only (a descriptive error),
    rather than a bare ``KeyError`` escaping mid-batch and aborting every
    node after it -- including ones already mutated by an earlier iteration.
    """
    return [key for key in _REQUIRED_NODE_KEYS if key not in node]


def _move_result_entry(uuid: str, disposition: str, result: Any, target_graph: str) -> dict:
    """Build a MOVE apply_results entry from move_entity_across_graphs' MoveResult.

    MoveResult.edges_skipped/mentions_skipped count a RELATES_TO edge or
    Episodic MENTIONS link whose OTHER endpoint was absent from
    *target_graph* -- FalkorDB's CREATE silently matched nothing, so the
    edge/mention was never recreated, even though the source node is still
    unconditionally DETACH DELETEd (move_entity_across_graphs never rolls
    that back). Discarding the returned MoveResult (as run() used to) hides
    this from the manifest entirely -- a lossy move still reported
    ``applied`` and a clean post-verify exit, since post-verify only checks
    the foreign-node COUNT, not its topology. Folding the counts in here
    (and blocking the node when either is non-zero) surfaces the loss to the
    operator instead.
    """
    edges_skipped = result.edges_skipped
    mentions_skipped = result.mentions_skipped
    lossy = bool(edges_skipped or mentions_skipped)
    entry = {
        'uuid': uuid, 'disposition': disposition, 'applied': True, 'blocked': lossy,
        'edges_moved': result.edges_moved, 'edges_skipped': edges_skipped,
        'mentions_moved': result.mentions_moved, 'mentions_skipped': mentions_skipped,
    }
    if lossy:
        entry['error'] = (
            f'{edges_skipped} RELATES_TO edge(s) and {mentions_skipped} MENTIONS '
            f'link(s) were silently skipped moving {uuid} into {target_graph!r} '
            "(the edge/mention's other endpoint is not yet present there) -- "
            'topology loss, needs manual review'
        )
    return entry


def _merge_result_entry(uuid: str, disposition: str, result: Any, home_graph: str) -> dict:
    """Build a MERGE apply_results entry from merge_foreign_duplicate's MergeResult.

    MergeResult.home_edge_count_after is a genuine RE-READ of the home
    copy's RELATES_TO edge count after the recreate loop -- comparing it
    against ``home_edge_count_before + edges_recreated`` is a real
    no-edge-lost check (an edge recreate whose CREATE silently matched no
    endpoint in the home graph would leave the actual count short of that
    sum). Discarding the returned MergeResult (as run() used to) hides this
    from the manifest entirely. Folding the counts in here (and blocking the
    node on a mismatch) surfaces the loss to the operator instead.
    """
    expected_after = result.home_edge_count_before + result.edges_recreated
    lossy = result.home_edge_count_after != expected_after
    entry = {
        'uuid': uuid, 'disposition': disposition, 'applied': True, 'blocked': lossy,
        'edges_recreated': result.edges_recreated,
        'home_edge_count_before': result.home_edge_count_before,
        'home_edge_count_after': result.home_edge_count_after,
    }
    if lossy:
        entry['error'] = (
            f'home graph {home_graph!r} RELATES_TO edge count after merging '
            f'{uuid} ({result.home_edge_count_after}) does not match '
            f'home_edge_count_before + edges_recreated ({expected_after}) -- an '
            'edge recreate may have silently matched nothing, needs manual review'
        )
    return entry


async def run(args: Any, memory_service: Any) -> dict:
    """Census -> classify -> manifest (dry-run), or apply a reviewed manifest.

    Dry-run (default, ``args.apply`` False): enumerates every populated
    graph via ``list_graphs()``, census-enumerates each one's foreign nodes,
    classifies every foreign node found, and returns the assembled manifest
    (``build_manifest`` output, ``dry_run=True``) with ``exit_code=0``
    attached. Nothing is mutated on this path -- no ``graph.query`` is
    issued and neither ε primitive is invoked.

    Apply (``args.apply`` True) requires an EXISTING, human-reviewed
    manifest file (``args.manifest``) -- it refuses (non-zero ``exit_code``,
    zero mutations, no census recompute) if none is given, rather than
    silently falling back to a freshly recomputed census.
    """
    graphiti = memory_service.graphiti

    if not args.apply:
        populated = set(await graphiti.list_graphs())
        census_counts: dict[str, int] = {}
        classified: list[dict] = []
        for graph_key in populated:
            foreign = await census_foreign_nodes(graphiti, graph_key, page_size=args.page_size)
            census_counts[graph_key] = len(foreign)
            for node in foreign:
                classified.append(await classify_node(graphiti, node, populated))

        manifest = build_manifest(classified, census_counts, dry_run=True)
        manifest['exit_code'] = 0
        return manifest

    if not args.manifest:
        return {
            'dry_run': False,
            'nodes': [],
            'error': (
                '--apply requires an existing, human-reviewed manifest file '
                '(--manifest PATH) -- it never recomputes a fresh census. '
                'Run a dry-run first, review the emitted manifest, then '
                're-run with --apply --manifest PATH.'
            ),
            'exit_code': 1,
        }

    manifest = load_reviewed_manifest(args.manifest)

    # Partition manifest['nodes'] by disposition, preserving every existing
    # guard from the old single-pass loop (malformed record, MOVE/MERGE
    # source==target refusal, EPISODIC_SKIP, UNRESOLVED) -- each guard
    # decides the SAME way it always did. Only MOVE/MERGE nodes that clear
    # every guard are deferred into move_specs/merge_specs for the
    # three-phase barrier below, instead of being dispatched immediately.
    apply_results: list[dict] = []
    move_specs: list[dict] = []
    merge_specs: list[dict] = []

    for node in manifest['nodes']:
        missing_keys = missing_required_node_keys(node)
        if missing_keys:
            # A hand-edited/malformed manifest record missing a required
            # field blocks THIS node only -- indexing node['uuid'] etc.
            # below unguarded would raise a bare KeyError that escapes run()
            # entirely, aborting every node after it (including ones
            # already mutated by an earlier phase). See
            # missing_required_node_keys.
            apply_results.append({
                'uuid': node.get('uuid', '<missing-uuid>'),
                'disposition': node.get('disposition', '<missing-disposition>'),
                'applied': False,
                'blocked': True,
                'error': (
                    'malformed manifest node record: missing required '
                    f'key(s) {missing_keys!r} -- refusing to dispatch'
                ),
            })
            continue

        uuid = node['uuid']
        disposition = node['disposition']

        if disposition in (MOVE, MERGE) and node['source_graph'] == node['target_graph']:
            # Defensive guard: a correctly-classified in-place re-key is
            # REKEY, never MOVE/MERGE (see classify_node) -- a MOVE/MERGE
            # reaching here with source_graph == target_graph can only come
            # from a malformed/hand-edited manifest. Dispatching it would
            # make create_moved_node / delete_source_node treat the node's
            # only copy as both source AND target (wrong==home). Refuse it:
            # block this node only, never add it to move_specs/merge_specs.
            apply_results.append({
                'uuid': uuid, 'disposition': disposition, 'applied': False, 'blocked': True,
                'error': (
                    f'refusing {disposition} with source_graph == target_graph '
                    f"({node['source_graph']!r}) -- malformed manifest; this "
                    'node should have classified REKEY'
                ),
            })
            continue

        if disposition == MOVE:
            move_specs.append(node)
        elif disposition == MERGE:
            merge_specs.append(node)
        elif disposition == REKEY:
            # A REKEY node already physically lives in its home graph --
            # an in-place group_id SET, never a create/delete, so it carries
            # none of the barrier-ordering constraints the MOVE/MERGE phases
            # below exist for. Applied immediately; isolated in its own
            # try/except so one bad REKEY node cannot abort the batch.
            try:
                await rekey_node_in_place(
                    graphiti._graph_for(node['source_graph']), uuid, node['target_graph'],
                )
            except Exception as exc:
                apply_results.append({
                    'uuid': uuid, 'disposition': disposition, 'applied': False, 'blocked': True,
                    'error': str(exc),
                })
            else:
                apply_results.append(
                    {'uuid': uuid, 'disposition': disposition, 'applied': True, 'blocked': False},
                )
        elif disposition == EPISODIC_SKIP:
            # Non-:Entity foreign node (Episodic/Community/unlabeled): the
            # phase primitives and rekey_node_in_place are all :Entity-
            # scoped, so this node is never dispatched to any of them.
            # Blocks this node only; its live relocation is eta's concern.
            apply_results.append({
                'uuid': uuid, 'disposition': disposition, 'applied': False, 'blocked': True,
                'reason': (
                    'non-:Entity node -- epsilon primitives are :Entity-scoped; '
                    'deferred to eta live relocation'
                ),
            })
        else:
            # UNRESOLVED: never dispatched to a primitive (PRD decision 4 --
            # no silent drop, no silent routing to a new/unpopulated graph).
            # Blocks this node only; the report's non-zero exit_code (added
            # by the post-verify step) surfaces it to the operator.
            apply_results.append(
                {'uuid': uuid, 'disposition': disposition, 'applied': False, 'blocked': True},
            )

    # --- Three-phase barrier-ordered apply (CGL-eta follow-up, task 2415) ---
    #
    # The old single-call-per-node loop dispatched each MOVE/MERGE node's
    # create + edge-recreate + DETACH DELETE as one atomic unit, one node at
    # a time. When the migrating set contained a co-moving SUBGRAPH (two
    # migrating nodes joined by a shared RELATES_TO edge), the first-
    # processed endpoint would find its neighbour not yet in target (the
    # neighbour's own turn hadn't come yet), silently skip recreating the
    # shared edge, and then unconditionally DETACH DELETE its own source
    # copy -- destroying the edge's only remaining reference before the
    # second endpoint was ever processed. See cross_graph_move.py's
    # "Residual hazard" note and this task's analysis.
    #
    # Restructuring into three barrier-ordered PASSES over the WHOLE batch
    # closes this: every migrating node is created in its target (Phase A)
    # before ANY edge is recreated (Phase B), and no source node is deleted
    # (Phase C) until every edge/mention has been recreated. A co-moving
    # edge's two endpoints are therefore both already present in target by
    # the time Phase B reads/recreates it.

    # Phase A: CREATE every MOVE node's target-graph copy. Isolated per-node
    # -- a single node's create failure (a hand-edited manifest mislabeling
    # an Episodic node as MOVE, a ForeignDuplicateSuspectedError, a
    # transient FalkorDB error, ...) must not abort the whole batch; it is
    # recorded blocked and excluded from Phase B/C below (its edges cannot
    # be safely recreated, and its non-existent target copy must not have
    # its source deleted out from under it).
    create_failed: dict[str, Exception] = {}
    for spec in move_specs:
        try:
            await create_moved_node(
                graphiti, spec['uuid'], spec['source_graph'], spec['target_graph'],
                rewrite_group_id=spec['target_graph'],
            )
        except Exception as exc:
            create_failed[spec['uuid']] = exc

    # Phase B: recreate every RELATES_TO edge / Episodic MENTIONS link
    # incident to the batch, in ONE call spanning every MOVE spec that
    # survived Phase A plus every MERGE spec (a MERGE node's home copy
    # already exists -- no Phase-A create needed). A single batched call --
    # not one per node -- is what lets a co-moving edge shared between two
    # specs in THIS batch be deduped and recreated exactly once (see
    # recreate_subgraph_relationships). If the call itself raises, every
    # spec offered to it is blocked below (none of their edges are
    # known-safe to skip past into Phase C) -- but nodes already resolved
    # above (REKEY/EPISODIC_SKIP/UNRESOLVED/malformed/refused) are
    # unaffected, and post-verify below still runs.
    phase_b_specs = [spec for spec in move_specs if spec['uuid'] not in create_failed]
    phase_b_specs += merge_specs
    edge_result = SubgraphEdgeResult()
    phase_b_error: Exception | None = None
    if phase_b_specs:
        try:
            edge_result = await recreate_subgraph_relationships(graphiti, phase_b_specs)
        except Exception as exc:
            phase_b_error = exc

    # Phase C: DETACH DELETE every MOVE/MERGE source node whose earlier
    # phases succeeded -- deleting a source before its edges are known-
    # recreated is exactly the bug this task fixes, so a node whose Phase-A
    # create failed, or whose batch's Phase-B call failed, is left in place
    # (blocked, with the underlying error) rather than deleted.
    for spec in move_specs:
        uuid = spec['uuid']
        if uuid in create_failed:
            apply_results.append({
                'uuid': uuid, 'disposition': MOVE, 'applied': False, 'blocked': True,
                'error': str(create_failed[uuid]),
            })
            continue
        if phase_b_error is not None:
            apply_results.append({
                'uuid': uuid, 'disposition': MOVE, 'applied': False, 'blocked': True,
                'error': f'Phase B (recreate_subgraph_relationships) failed: {phase_b_error}',
            })
            continue
        try:
            await delete_source_node(graphiti, uuid, spec['source_graph'])
        except Exception as exc:
            apply_results.append({
                'uuid': uuid, 'disposition': MOVE, 'applied': False, 'blocked': True,
                'error': str(exc),
            })
        else:
            apply_results.append(
                {'uuid': uuid, 'disposition': MOVE, 'applied': True, 'blocked': False},
            )

    for spec in merge_specs:
        uuid = spec['uuid']
        if phase_b_error is not None:
            apply_results.append({
                'uuid': uuid, 'disposition': MERGE, 'applied': False, 'blocked': True,
                'error': f'Phase B (recreate_subgraph_relationships) failed: {phase_b_error}',
            })
            continue
        try:
            await delete_source_node(graphiti, uuid, spec['source_graph'])
        except Exception as exc:
            apply_results.append({
                'uuid': uuid, 'disposition': MERGE, 'applied': False, 'blocked': True,
                'error': str(exc),
            })
        else:
            apply_results.append(
                {'uuid': uuid, 'disposition': MERGE, 'applied': True, 'blocked': False},
            )

    # POST-VERIFY: re-census foreign counts per graph and compare to the
    # expected residual -- the per-graph count of UNRESOLVED + EPISODIC_SKIP
    # manifest nodes, which are the only foreign nodes a clean apply should
    # ever leave behind (EPISODIC_SKIP nodes are never actioned -- see the
    # EPISODIC_SKIP constant -- so their foreign copy necessarily survives).
    # A mismatch means a MOVE/MERGE silently failed to clear its source
    # copy; any UNRESOLVED/EPISODIC_SKIP node blocks success outright
    # regardless of whether the counts otherwise reconcile (PRD decision 4).
    populated = set(await graphiti.list_graphs())
    after_counts: dict[str, int] = {}
    for graph_key in populated:
        foreign = await census_foreign_nodes(graphiti, graph_key, page_size=args.page_size)
        after_counts[graph_key] = len(foreign)

    # .get() (not node['disposition']/node['source_graph']) so a malformed
    # manifest node record already blocked above by the missing-keys guard
    # does not ALSO raise a KeyError here -- this loop re-walks the same
    # manifest['nodes'] independently of the apply loop above.
    expected_residual: dict[str, int] = dict.fromkeys(populated, 0)
    for node in manifest['nodes']:
        if node.get('disposition') in (UNRESOLVED, EPISODIC_SKIP):
            source_graph = node.get('source_graph')
            if source_graph is not None:
                expected_residual[source_graph] = expected_residual.get(source_graph, 0) + 1

    matched = after_counts == expected_residual
    has_unresolved = bool(manifest.get('unresolved_uuids'))
    has_blocked = any(r['blocked'] for r in apply_results)
    # A SubgraphEdgeResult.edges_skipped/mentions_skipped > 0 means Phase B
    # found an edge/mention whose other endpoint was genuinely absent from
    # the resolved target (never recreated) -- the same silent-topology-loss
    # signal the old per-node MoveResult.edges_skipped/mentions_skipped
    # surfaced via _move_result_entry, now read off the WHOLE batch's single
    # SubgraphEdgeResult instead of one node's own result. Folded into
    # exit_code the same way has_unresolved/has_blocked are: it forces a
    # non-zero, blocking exit even when the post-verify count re-census
    # otherwise reconciles (a lost edge leaves no foreign-node residual of
    # its own to be caught by the count comparison above).
    has_edge_loss = bool(edge_result.edges_skipped or edge_result.mentions_skipped)
    # An edge whose two endpoints resolve to DIFFERENT target graphs cannot
    # be recreated in EITHER graph (FalkorDB RELATES_TO edges are
    # single-graph) -- Phase B reports each as a dropped-with-reason record
    # instead of silently losing it (see recreate_subgraph_relationships).
    # Surfacing it here (never omitted) and forcing a blocking exit mirrors
    # has_edge_loss/has_unresolved/has_blocked: a human must review it.
    has_dropped_cross_target = bool(edge_result.dropped_cross_target)

    report = dict(manifest)
    report['dry_run'] = False
    report['apply_results'] = apply_results
    report['edges_recreated'] = edge_result.edges_recreated
    report['edges_skipped'] = edge_result.edges_skipped
    report['mentions_recreated'] = edge_result.mentions_recreated
    report['mentions_skipped'] = edge_result.mentions_skipped
    report['dropped_cross_target_edges'] = edge_result.dropped_cross_target
    report['post_verify'] = {
        'matched': matched,
        'expected': expected_residual,
        'actual': after_counts,
    }
    report['exit_code'] = (
        0
        if (
            matched and not has_unresolved and not has_blocked
            and not has_edge_loss and not has_dropped_cross_target
        )
        else 1
    )
    return report


# ---------------------------------------------------------------------------
# Backend construction
# ---------------------------------------------------------------------------

async def build_memory_service(args: Any, config: Any) -> Any:
    """Build+initialize the backend/service run() drives, branching on args.apply.

    Dry-run (``args.apply`` False): the census/classify path only ever
    READS (``list_graphs``, ``ro_query``) -- it never needs a full
    ``MemoryService``, whose ``initialize()`` unconditionally runs the
    W6-ε startup identity scan (a dup-uuid-edge REPAIR, i.e. a write) and
    builds indices. Build a lean ``GraphitiBackend`` directly and
    initialize it with ``skip_maintenance=True`` so the default
    ``python migrate_cross_graph_leak.py > manifest.json`` invocation no
    longer mutates on startup or contends with a running service's own
    maintenance sweep. Returned as a tiny namespace exposing ``.graphiti``
    (the backend itself) and an awaitable ``.close``, mirroring the shape
    ``run()``/``_run_live`` expect from a full ``MemoryService``.

    Apply (``args.apply`` True): ``--apply`` legitimately mutates, so build
    and initialize the full ``MemoryService``, unskipped, exactly as
    before.
    """
    from fused_memory.backends.graphiti_client import GraphitiBackend  # noqa: PLC0415
    from fused_memory.services.memory_service import MemoryService  # noqa: PLC0415

    if not args.apply:
        backend = GraphitiBackend(config)
        await backend.initialize(skip_maintenance=True)
        return types.SimpleNamespace(graphiti=backend, close=backend.close)

    memory = MemoryService(config)
    await memory.initialize()
    return memory


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser (split out from main() so it's testable)."""
    parser = argparse.ArgumentParser(
        description=(
            'Census, classify, and (with --apply) migrate cross-graph-leaked '
            'Graphiti nodes to their resolved home graph.'
        ),
    )
    parser.add_argument(
        '--apply', action='store_true', default=False,
        help=(
            'Apply an existing, reviewed manifest (requires --manifest). '
            'Default: dry-run only -- census, classify, print the manifest, exit.'
        ),
    )
    parser.add_argument(
        '--manifest', default=None,
        help='Path to a previously emitted, human-reviewed manifest JSON file (required with --apply).',
    )
    parser.add_argument(
        '--page-size', type=int, default=DEFAULT_PAGE_SIZE,
        help=f'Census page size (default: {DEFAULT_PAGE_SIZE}).',
    )
    parser.add_argument(
        '--config', default=None,
        help='Path to a fused-memory config file (sets CONFIG_PATH before loading).',
    )
    return parser


def main() -> int:
    """Parse CLI args, build a live MemoryService, and run the census/apply."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
    )
    args = build_arg_parser().parse_args()

    if args.config:
        import os  # noqa: PLC0415
        os.environ['CONFIG_PATH'] = str(args.config)

    async def _run_live() -> dict:
        from fused_memory.config.schema import FusedMemoryConfig  # noqa: PLC0415

        config = FusedMemoryConfig()
        built = await build_memory_service(args, config)
        try:
            return await run(args, built)
        finally:
            if hasattr(built, 'close'):
                await built.close()

    report = asyncio.run(_run_live())
    print(json.dumps(report, indent=2, default=str))
    return report['exit_code']


if __name__ == '__main__':
    sys.exit(main())
