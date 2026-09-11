#!/usr/bin/env python3
"""Consolidation script for the cross-graph entity leak cleanup (CGL-θ, task 2274).

Four independent, order-insensitive operations against the FalkorDB
(Graphiti) and Qdrant (Mem0) backends. Each is dry-run by default and each
is guarded so a partial/incomplete state is reported UNRESOLVED rather than
destroying data:

  1. GRAPH-FAMILY MERGES with IDENTITY REWRITE -- sibling Graphiti graphs
     that are the same logical project under a different key (hyphenated,
     no-separator, ...) are merged into their underscore-canonical graph via
     the three-phase barrier-ordered apply (CGL-η follow-up, task 2415,
     extended to Episodic-node relocation by task 2502): Phase A creates
     every sibling Entity (``create_moved_node``) AND every sibling Episodic
     node (``create_moved_episode``) in the canonical graph, each with
     ``rewrite_group_id=canonical`` (the Phase-2 identity rewrite, PRD
     decision 6); Phase B recreates every intra-family RELATES_TO edge and
     MENTIONS link in ONE batched ``recreate_subgraph_relationships`` call;
     Phase C deletes every non-blocked source. This closes two bugs the OLD
     per-node ``move_entity_across_graphs`` loop had: an intra-family
     RELATES_TO edge between two co-moving nodes was destroyed by the first
     endpoint's delete before the second endpoint was ever processed, and a
     MENTIONS link from a sibling-resident Episodic node was silently
     dropped because episodes were never relocated at all. See
     ``fused_memory.maintenance.cross_graph_move`` and
     ``merge_graph_family``'s own docstring for the full barrier contract.
  2. QDRANT COLLECTION MERGES -- legacy/divergent Mem0 collections (from
     historical ``collection_prefix`` values, RCA §4) are merged into their
     ``fused_<project>`` target: counted in a read-only preflight, then
     streamed with vectors in bounded chunks with each payload ``user_id``
     rewritten to the canonical project id and upserted into the target,
     and the source collection deleted ONLY once fully drained.
  3. GUARDED JUNK-KEY DELETION -- known-junk FalkorDB graph keys (and any
     family sibling emptied by step 1) are removed via ``GRAPH.DELETE``
     (``graphiti._graph_for(key).delete()``) -- NOT the ``MATCH (n) DETACH
     DELETE n`` pattern in ``purge_knowlive_namespace.py``, which empties a
     graph but leaves its key in ``GRAPH.LIST``. Guarded on a live node
     count of exactly 0; a non-empty key is reported UNRESOLVED and its
     deletion is blocked.
  4. GUARDED EMPTY-COLLECTION DELETION -- known-empty divergent/junk Qdrant
     collections (``EMPTY_COLLECTION_CLEANUP``) that hold no real data and
     have no merge target are removed via ``delete_collection``, guarded on
     a live point count of exactly 0 (``count_collection_points``) --
     mirroring step 3's count-0 GRAPH.DELETE guard, applied to Qdrant. A
     non-empty collection is reported UNRESOLVED and its deletion is
     blocked.

Reviewable, static module-level constants (``GRAPH_FAMILY_ALIASES``,
``COLLECTION_MERGES``, ``JUNK_KEYS``, ``EMPTY_COLLECTION_CLEANUP``) are the
human-reviewable artifact PRD decision 4 requires -- this script ships its
OWN config (a sibling to, not shared with, the ζ migration script's alias
map).

Contract (S7, per ``plans/cross-graph-entity-leak-prd.md``): dry-run by
default, prints a full JSON manifest; ``--apply`` performs the mutations
described above, and exits non-zero if any section of the manifest carries
an ``UNRESOLVED`` disposition (see ``has_unresolved``). INTENTIONAL S7
DEVIATION (task 2502): ``--apply`` here is constants-driven -- there is no
manifest-path argument to replay a previously-printed dry-run report, and
no separate post-verify recount pass after the mutations. Both are safe to
omit because every mutating section is safe-by-construction: graph-family
merges are create-before-delete (a source is only removed after its
target-graph copy -- and every intra-family edge/mention -- has been
successfully recreated, per ``merge_graph_family``'s three-phase barrier),
and every deletion (junk key or empty collection) is guarded on a live
count of exactly 0 recomputed at run time, never trusted from a stale
manifest. The PRD's own S7 wording is left to the ι human-review step
(shared-doc scope), not edited by this task.

Scope: this script + its test suite are MOCK-unit only (MagicMock graphs,
AsyncMock Qdrant client) -- no live FalkorDB/Qdrant, and no assertion of
embedding byte-fidelity or ``GRAPH.LIST`` cleanliness. Those are the LIVE
B7 signal, mandated at the ι live throwaway-graph rehearsal (PRD decision
5), not asserted here.

Usage
-----
  # Dry run (default): print the full JSON manifest, touch nothing.
  python scripts/consolidate_namespace_families.py > consolidation_manifest.json

  # Commit the merges + junk-key/empty-collection deletions (exits non-zero
  # on UNRESOLVED).
  python scripts/consolidate_namespace_families.py --apply
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from typing import Any

from fused_memory.backends.mem0_client import (
    DEFAULT_SCROLL_MAX_PAGES,
    ScrollPageBudgetExhausted,
)
from fused_memory.maintenance.cross_graph_move import (  # noqa: F401
    SubgraphEdgeResult,
    create_moved_episode,
    create_moved_node,
    delete_source_episode,
    delete_source_node,
    recreate_subgraph_relationships,
)
from fused_memory.utils.store_mutation_preflight import (
    StoreMutationUnavailable,
    assert_store_mutation_allowed,
)

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
    'fused_pump-web-ui': 'fused_pump_web_ui',
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

# Empty divergent/junk Qdrant collections with no merge target -- unlike
# COLLECTION_MERGES (which scrolls -> upserts -> deletes a POPULATED
# source), these strays hold no real data and are simply removed once
# verified empty (count_collection_points == 0), mirroring the
# JUNK_KEYS/delete_junk_key count-0 guard for FalkorDB graph keys. reify_
# (empty project id) and fused_fused_memory are DELIBERATELY OMITTED here
# too -- PRD Open Q2 defers their disposition to ι human review, same as
# COLLECTION_MERGES above.
EMPTY_COLLECTION_CLEANUP: tuple[str, ...] = (
    'fused_knowlive',
    'fused_know-live',
    'fused_autopilot-video',
    'fused_my-project',
    'fused_1098',
    'fused_default',
    'fused_-home-leo-src-dark-factory',
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
    empty_collection_items: list[dict],
    *,
    dry_run: bool,
) -> dict:
    """Assemble the manifest dict from already-computed per-item lists. No I/O."""
    return {
        'dry_run': dry_run,
        'graph_family_merges': list(graph_family_items),
        'collection_merges': list(collection_items),
        'junk_key_deletions': list(junk_key_items),
        'empty_collection_deletions': list(empty_collection_items),
    }


def has_unresolved(report: dict) -> bool:
    """True iff any item in any of the four manifest sections carries an
    'UNRESOLVED' disposition.

    This is the S7 exit-code predicate: ``main()`` returns non-zero on
    ``--apply`` when this is True. Pure function, no I/O.
    """
    sections = (
        report['graph_family_merges'],
        report['collection_merges'],
        report['junk_key_deletions'],
        report['empty_collection_deletions'],
    )
    return any(item.get('disposition') == 'UNRESOLVED' for section in sections for item in section)


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

    Scoped to :Entity (not every label) -- Episodic nodes are enumerated
    separately, via ``enumerate_graph_episodic_nodes``, and relocated by
    ``merge_graph_family``'s own Phase A (``create_moved_episode``). Any
    OTHER-labeled (e.g. Community) residual left behind in *key* after a
    clean merge is exactly what should make the junk-key guard
    (``delete_junk_key``, which counts every label) classify it UNRESOLVED
    rather than silently GRAPH.DELETE-ing it.

    Single-page fetch: this issues exactly ONE ``LIMIT $limit`` query and
    never follows up with a second page, so a graph with more than *limit*
    Entity nodes is permanently reported UNRESOLVED at the given --limit.
    Callers must pass a *limit* larger than the true Entity-node count of
    every graph they intend to migrate in this run; the full result set is
    held in memory, so raising *limit* trades completeness for peak memory.
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


async def enumerate_graph_episodic_nodes(
    graphiti: Any,
    key: str,
    *,
    limit: int = 1000,
) -> list[dict]:
    """Read-only enumeration of every :Episodic node in the *key* FalkorDB graph.

    Mirrors ``enumerate_graph_entity_nodes``, scoped to :Episodic instead of
    :Entity. Every sibling-resident Episodic node must be discovered here so
    ``merge_graph_family`` can relocate it (``create_moved_episode``) into
    the canonical graph BEFORE Phase B recreates its MENTIONS links --
    otherwise the episode stays behind in the sibling and every MENTIONS
    link onto a moved Entity is silently dropped (Phase B's episode-present
    MATCH finds nothing).

    Single-page fetch: this issues exactly ONE ``LIMIT $limit`` query and
    never follows up with a second page, so a graph with more than *limit*
    Episodic nodes is permanently reported UNRESOLVED at the given --limit --
    the same no-silent-caps convention as ``enumerate_graph_entity_nodes``.
    Callers must pass a *limit* larger than the true Episodic-node count of
    every graph they intend to migrate in this run.
    """
    graph = graphiti._graph_for(key)
    result = await graph.ro_query(
        'MATCH (n:Episodic) RETURN n.uuid LIMIT $limit',
        {'limit': limit},
    )
    rows = result.result_set or []
    if len(rows) >= limit:
        logger.warning(
            "consolidate_namespace_families: enumerated %d Episodic node(s) in "
            "graph '%s', which hit limit=%d -- enumeration may be incomplete. "
            "Re-run with a higher --limit value to ensure the full graph is "
            "covered.",
            len(rows), key, limit,
        )
    return [{'uuid': row[0]} for row in rows]


async def count_graph_nodes(graphiti: Any, key: str) -> int:
    """Read-only total node count (every label) for the *key* FalkorDB graph.

    Used as the guard for ``delete_junk_key``: GRAPH.DELETE is only safe
    when this is exactly 0. Defensively treats a missing/empty
    ``result_set`` (e.g. a transient backend hiccup, or a RO query against a
    graph key absent from ``GRAPH.LIST`` -- read-only queries do not
    auto-create) as a count of 0 rather than raising, mirroring
    ``enumerate_graph_entity_nodes``'s ``result.result_set or []`` guard.
    """
    graph = graphiti._graph_for(key)
    result = await graph.ro_query('MATCH (n) RETURN count(n)')
    rows = result.result_set or []
    return int(rows[0][0]) if rows else 0


# ---------------------------------------------------------------------------
# Graph family merge (mutating)
# ---------------------------------------------------------------------------

async def merge_graph_family(
    graphiti: Any,
    sibling: str,
    canonical: str,
    entity_rows: list[dict],
    episode_rows: list[dict],
) -> dict:
    """Move every *entity_rows* Entity and *episode_rows* Episodic node from
    *sibling* into *canonical* via the three-phase barrier-ordered apply
    (CGL-η follow-up, task 2502 -- template: ``migrate_cross_graph_leak.py``'s
    ``run()``, which closed the identical edge-loss bug for the ζ migration
    script; this brings the same fix -- plus Episodic/MENTIONS relocation --
    to the θ family merge).

    The OLD implementation drove ``move_entity_across_graphs`` once per node,
    which DETACH-DELETEd each source node (with its edges) immediately after
    creating it in target, BEFORE the next node moved -- destroying any
    RELATES_TO edge between two nodes in the SAME family (the first
    endpoint's delete took the edge; the second endpoint never saw it). It
    also never relocated Episodic nodes, so every MENTIONS link from a
    sibling-resident episode onto a moved entity was lost too (edges are
    single-graph; the episode stayed behind).

    Phase A: CREATE every entity's (``create_moved_node``) and every
    episode's (``create_moved_episode``) target-graph copy, each isolated --
    a single item's create failure is recorded in ``create_failed``/
    ``episode_create_failed`` (never aborts the batch) and excluded from
    Phase B/C below. Relocating sibling Episodic nodes here (not just Entity
    nodes) is what lets Phase B's UNCHANGED MENTIONS recreate succeed: it
    only recreates a MENTIONS link when the episode is already present in
    the entity's resolved target graph -- absent, it silently counts the
    link in ``mentions_skipped`` instead.

    Phase B: recreate every intra-family RELATES_TO edge and MENTIONS link in
    ONE batched ``recreate_subgraph_relationships`` call, built from a MOVE
    spec (``uuid``/``disposition='MOVE'``/``source_graph=sibling``/
    ``target_graph=canonical``) for every entity that survived Phase A.
    Episodes are never passed as their own specs -- only entities carry
    RELATES_TO/MENTIONS specs in this module's schema; an episode's
    relevance to Phase B is only as the MENTIONS target, already satisfied
    by its Phase-A relocation. A single batched call (not one per entity) is
    what dedupes a co-moving edge shared by two entities in this family,
    recreating it exactly once. If the call raises, its ``exc.partial_result``
    (always attached by ``recreate_subgraph_relationships`` -- see that
    function's docstring) is recovered so a partial tally is never discarded
    in favor of an all-zero default. A raise ALSO records the exception in
    ``phase_b_error``, which Phase C below checks FIRST, before any per-uuid
    reasoning: a SYSTEMIC failure (a per-spec gather ``ro_query``, or a lost
    falkor-client acquisition mid-batch -- the two failure modes
    ``recreate_subgraph_relationships`` documents as raising rather than
    isolating onto ``blocked``) means the recovered ``partial_result.blocked``
    names only the items that failed in ISOLATION before the abort -- it says
    nothing about the specs the batch never reached, whose edges/mentions now
    exist ONLY in source. Treating that recovered partial as a definitive
    per-uuid block-list would therefore delete sources the batch never
    actually confirmed recreating; ``phase_b_error`` instead withholds the
    WHOLE batch unconditionally (see Phase C).

    Phase C: DETACH DELETE every entity's (``delete_source_node``) and every
    episode's (``delete_source_episode``) source copy. When Phase B raised a
    SYSTEMIC error (``phase_b_error is not None``), EVERY entity and episode
    in the batch is withheld UNCONDITIONALLY -- see the Phase B paragraph
    above for why a recovered ``partial_result.blocked`` cannot be trusted as
    a definitive per-uuid list in that case. Otherwise, a source is withheld
    for any uuid that failed Phase A, or named in Phase B's
    ``SubgraphEdgeResult.blocked`` (an entity via ``node_uuids``, which are
    always entity uuids even for a blocked MENTIONS item -- see that
    dataclass's docstring). BOTH an episode's AND an entity's deletion is
    ADDITIONALLY withheld whenever ANY MENTIONS link incident to it was not
    confirmed recreated, closing three more create-before-delete gaps the
    base entity/episode Phase-A-failure guarantee (and the ``phase_b_error``
    whole-batch gate above) do not cover by themselves: (1) a Phase-B
    blocked mention names its episode via the blocked item's
    ``episode_uuid`` key; (2) an entity whose Phase-A create FAILED is
    dropped from Phase B's batch entirely, so a mention ONTO it is never
    even read/recreated there -- caught by a guarded sibling
    MENTIONS-topology probe (``MATCH (ep:Episodic)-[:MENTIONS]->(n:Entity
    {uuid: $uuid})``, run only when there are both episodes to withhold and
    Phase-A entity failures to check, AND Phase B did not itself abort
    systemically -- every source is already withheld in that case, and
    issuing more reads against a backend that just failed systemically is
    pointless/risky); and (3) the mirror image (reviewer follow-up) -- an
    episode whose Phase-A create FAILED correctly stays behind in sibling
    (withheld via ``episode_create_failed``), but any entity it MENTIONS is
    unaffected by THAT failure: the entity's OWN Phase-A create succeeded,
    so Phase B's MENTIONS CREATE still runs, MATCHes the
    never-created-in-target episode, finds nothing, and silently counts the
    link in ``mentions_skipped`` (a WARNING+skip, not a raise, so it never
    reaches ``blocked`` either) -- caught by the reverse-direction guarded
    probe (``MATCH (ep:Episodic {uuid: $uuid})-[:MENTIONS]->(n:Entity)
    RETURN n.uuid``, run only when there are both entities to withhold and
    Phase-A episode failures to check, likewise short-circuited on a
    systemic Phase-B abort). In every case, deleting a source whose
    edge/mention was never successfully recreated elsewhere would destroy
    the only remaining copy.

    Returns:
        A summary dict: ``nodes_moved``, ``episodes_moved``,
        ``nodes_blocked``, ``episodes_blocked`` (counts -- Phase-A failures
        plus Phase-B ``blocked``/MENTIONS-topology withholding, entities and
        episodes alike), ``edges_recreated``, ``edges_skipped``,
        ``mentions_recreated``, ``mentions_skipped`` (straight from the
        batch's ``SubgraphEdgeResult``), ``dropped_cross_target`` and
        ``blocked`` (both lists, passed through from the
        ``SubgraphEdgeResult`` verbatim -- never silently dropped), plus
        ``merge_mentions_dropped`` and ``merge_mentions_dropped_uuids``
        (task 4183's census of the MENTIONS links AT RISK -- destroyed if a
        MERGE spec's wrong copy is deleted -- likewise passed through
        verbatim).
        This script builds MOVE specs only (see the ``move_specs`` assembly
        below), so the census is structurally always 0/``[]`` here -- it
        fires only for MERGE specs -- but both keys are surfaced anyway
        under the same "never silently dropped" convention as
        ``dropped_cross_target``/``blocked``, and so the field is already
        wired should MERGE specs ever be added.
    """
    # --- Phase A: create every entity + episode in canonical ---------------
    create_failed: dict[str, Exception] = {}
    for row in entity_rows:
        try:
            await create_moved_node(
                graphiti, row['uuid'], sibling, canonical, rewrite_group_id=canonical,
            )
        except Exception as exc:
            create_failed[row['uuid']] = exc

    episode_create_failed: dict[str, Exception] = {}
    for row in episode_rows:
        try:
            await create_moved_episode(
                graphiti, row['uuid'], sibling, canonical, rewrite_group_id=canonical,
            )
        except Exception as exc:
            episode_create_failed[row['uuid']] = exc

    # --- Phase B: batch-recreate intra-family edges + MENTIONS --------------
    entity_specs = [
        {
            'uuid': row['uuid'], 'disposition': 'MOVE',
            'source_graph': sibling, 'target_graph': canonical,
        }
        for row in entity_rows if row['uuid'] not in create_failed
    ]
    edge_result = SubgraphEdgeResult()
    phase_b_error: Exception | None = None
    if entity_specs:
        try:
            edge_result = await recreate_subgraph_relationships(graphiti, entity_specs)
        except Exception as exc:
            phase_b_error = exc
            partial_result = getattr(exc, 'partial_result', None)
            if partial_result is not None:
                edge_result = partial_result

    # blocked_node_uuids (mirrors migrate_cross_graph_leak.py's run()): every
    # uuid named in a Phase-B blocked item's node_uuids -- both endpoints for
    # a blocked RELATES_TO edge, the entity uuid for a blocked MENTIONS link
    # -- must have its Phase-C deletion withheld too, or the un-recreated
    # edge/mention (which now exists only in source) would be destroyed.
    blocked_node_uuids: set[str] = set()
    for blocked_item in edge_result.blocked:
        for node_uuid in blocked_item.get('node_uuids', []):
            blocked_node_uuids.add(node_uuid)

    # blocked_episode_uuids (reviewer follow-up, data-loss-barrier-gap): an
    # episode's Phase-C deletion must be withheld whenever ANY incident
    # MENTIONS link was not confirmed recreated -- otherwise the un-recreated
    # link (which now exists only in source) is destroyed right alongside it,
    # reintroducing for MENTIONS the exact create-before-delete loss the
    # three-phase barrier eliminates for entities. Two real paths destroy a
    # source-only MENTIONS link without this:
    #   (1) a MENTIONS recreate that raised lands in edge_result.blocked with
    #       kind='mention' and an episode_uuid -- the episode's own Phase-A
    #       create succeeded, so it is not in episode_create_failed, and
    #       node_uuids only ever names the ENTITY -- so without reading
    #       episode_uuid here, nothing would withhold this episode.
    #   (2) an entity whose Phase-A create FAILED is filtered out of
    #       entity_specs entirely, so Phase B never reads/recreates its
    #       incident MENTIONS (not counted in blocked or mentions_skipped) --
    #       the mentioning episode (Phase-A create ok) would otherwise be
    #       deleted anyway.
    blocked_episode_uuids: set[str] = set()
    for blocked_item in edge_result.blocked:
        if blocked_item.get('kind') == 'mention':
            episode_uuid = blocked_item.get('episode_uuid')
            if episode_uuid is not None:
                blocked_episode_uuids.add(episode_uuid)

    # Guarded sibling MENTIONS-topology probe for case (2) above -- only run
    # when there is both an episode that COULD be withheld and a Phase-A
    # entity failure to check MENTIONS against, so the common (no episodes,
    # or no Phase-A failures) path issues no extra read.
    if phase_b_error is None and episode_rows and create_failed:
        for failed_uuid in create_failed:
            topology_result = await graphiti._graph_for(sibling).ro_query(
                'MATCH (ep:Episodic)-[:MENTIONS]->(n:Entity {uuid: $uuid}) RETURN ep.uuid',
                {'uuid': failed_uuid},
            )
            for topology_row in topology_result.result_set or []:
                blocked_episode_uuids.add(topology_row[0])

    # Guarded sibling MENTIONS-topology probe, REVERSE direction (reviewer
    # follow-up, data-loss-barrier-gap case (3) above): when an episode's
    # Phase-A create_moved_episode FAILS, it is correctly withheld via
    # episode_create_failed -- but any entity it MENTIONS is unaffected by
    # THAT failure: the entity's own Phase-A create succeeded, so it is
    # offered to Phase B, whose MENTIONS CREATE MATCHes the
    # never-created-in-target episode and finds nothing -- silently counted
    # in edge_result.mentions_skipped (a WARNING+skip, NOT a raise, so it
    # never lands in edge_result.blocked either). Without this probe the
    # entity's uuid never reaches blocked_node_uuids, so its Phase-C DETACH
    # DELETE would proceed and destroy the source-only MENTIONS edge from
    # the still-sibling-resident failed episode. Only run when there is both
    # an entity that COULD be withheld and a Phase-A episode failure to
    # check MENTIONS against, so the common (no entities, or no Phase-A
    # episode failures) path issues no extra read.
    if phase_b_error is None and entity_rows and episode_create_failed:
        for failed_episode_uuid in episode_create_failed:
            topology_result = await graphiti._graph_for(sibling).ro_query(
                'MATCH (ep:Episodic {uuid: $uuid})-[:MENTIONS]->(n:Entity) RETURN n.uuid',
                {'uuid': failed_episode_uuid},
            )
            for topology_row in topology_result.result_set or []:
                blocked_node_uuids.add(topology_row[0])

    # --- Phase C: delete every non-blocked, non-failed source ---------------
    nodes_moved = 0
    nodes_blocked = 0
    for row in entity_rows:
        uuid = row['uuid']
        if phase_b_error is not None or uuid in create_failed or uuid in blocked_node_uuids:
            nodes_blocked += 1
            continue
        await delete_source_node(graphiti, uuid, sibling)
        nodes_moved += 1

    episodes_moved = 0
    episodes_blocked = 0
    for row in episode_rows:
        uuid = row['uuid']
        if (
            phase_b_error is not None
            or uuid in episode_create_failed
            or uuid in blocked_episode_uuids
        ):
            episodes_blocked += 1
            continue
        await delete_source_episode(graphiti, uuid, sibling)
        episodes_moved += 1

    return {
        'nodes_moved': nodes_moved,
        'episodes_moved': episodes_moved,
        'nodes_blocked': nodes_blocked,
        'episodes_blocked': episodes_blocked,
        'edges_recreated': edge_result.edges_recreated,
        'edges_skipped': edge_result.edges_skipped,
        'mentions_recreated': edge_result.mentions_recreated,
        'mentions_skipped': edge_result.mentions_skipped,
        'dropped_cross_target': edge_result.dropped_cross_target,
        'blocked': edge_result.blocked,
        'merge_mentions_dropped': edge_result.merge_mentions_dropped,
        'merge_mentions_dropped_uuids': edge_result.merge_mentions_dropped_uuids,
    }


# ---------------------------------------------------------------------------
# Qdrant collection merge
# ---------------------------------------------------------------------------

async def preflight_collection_points(
    qdrant_client: Any,
    collection: str,
    *,
    page_size: int = 1000,
    max_pages: int = DEFAULT_SCROLL_MAX_PAGES,
) -> tuple[int | None, bool]:
    """Read-only PREFLIGHT: how many points *collection* holds, and whether
    ``merge_collection``'s drain could enumerate all of them.

    O(1) -- ONE ``count`` round-trip through this module's own
    ``count_collection_points`` (defined below), not a paged scroll. This
    used to drain ``Mem0Backend.scroll_collection_pages`` page by page just
    to fold the points into a counter, which cost one round-trip per page
    (~200 for a 200k-point collection) and, under ``--apply``, enumerated
    every collection TWICE -- once here and again in ``merge_collection``.
    Qdrant's count API answers the same question in one call, and the only
    other bit the drain yielded, ``capped``, is exactly derivable (below).

    *collection* is passed VERBATIM -- ``COLLECTION_MERGES`` holds legacy
    mis-named collections (``fused_dark-factory``, ``reify_reify``) that a
    ``Scope`` structurally cannot produce, which is why this addresses the
    raw collection-name-keyed count API rather than a ``Scope``-addressed
    ``Mem0Backend.count``. The BACKEND is still what pages: task 3225 moved
    the offset/next_offset walk there, and ``merge_collection`` drives it as
    the pass's sole drain.

    Returns:
        ``(point_count, capped)``.  ``capped`` is True when *collection*
        holds more points than ``merge_collection``'s drain could enumerate
        under this budget -- ``point_count > page_size * max_pages``. That
        boundary matches the pager exactly: ``scroll_collection_pages``
        raises only when ``max_pages`` pages are consumed with
        ``next_offset`` STILL live, so a collection of exactly
        ``page_size * max_pages`` points drains cleanly and is not capped.
        The caller's ``if args.apply and not capped`` guard is what stops
        ``merge_collection`` running at all against a collection this
        preflight says cannot be fully enumerated, so an UNRESOLVED item
        still means nothing was written.

        ``point_count`` is None when the count itself could not be read --
        the same 'unreadable count' convention ``run()``'s junk-key loop
        already reports (``'node_count': None``), so a failure is never
        rendered as a plausible-looking 0.

    A raising count is CAUGHT, never propagated: an unreadable count, a
    ``TimeoutError``, or a transport error.  A raising sub-operation must
    not abort the whole consolidation run, because earlier keys/sections of
    the same ``--apply`` pass may already hold committed mutations -- the
    same except-``Exception`` -> WARNING -> 'UNRESOLVED' idiom the sibling
    ``delete_empty_collection`` uses.  Every failure maps onto the existing
    capped contract -- item UNRESOLVED, no upsert, no source delete, non-zero
    exit -- so the externally visible behaviour is unchanged.

    The two branches log DISTINCT warnings: only the over-budget branch
    carries the raise-the-budget remediation, because that advice is wrong
    after a transport failure.
    """
    try:
        point_count = await count_collection_points(qdrant_client, collection)
    except Exception as e:
        # Deliberately NO raise-the-budget advice here: this is not a cap,
        # and sending an operator to re-run a bigger scroll against a
        # transport that is failing is wrong advice.
        logger.warning(
            "consolidate_namespace_families: the point count for collection "
            "'%s' could not be read: %s -- the collection is reported "
            'UNRESOLVED and its source will not be deleted.',
            collection, e, exc_info=True,
        )
        return None, True

    budget = page_size * max_pages
    if point_count > budget:
        logger.warning(
            "consolidate_namespace_families: collection '%s' holds %d point(s), "
            'more than the merge drain could enumerate under its page budget '
            '(%d page(s) of %d = %d) -- migrating it would be INCOMPLETE, so '
            'this collection is reported UNRESOLVED and its source will not be '
            'deleted (see merge_collection). Re-run with a higher --max-pages '
            '(page budget) or --limit (page size).',
            collection, point_count, max_pages, page_size, budget,
        )
        return point_count, True
    return point_count, False


async def merge_collection(
    backend: Any,
    qdrant_client: Any,
    source: str,
    target: str,
    canonical_user_id: str,
    *,
    page_size: int = 1000,
    max_pages: int = DEFAULT_SCROLL_MAX_PAGES,
) -> dict:
    """STREAM *source* into *target*, rewriting each payload's user_id.

    This is the pass's SOLE with-vectors drain, and it upserts in CHUNKS of
    at most *page_size* points: it never holds more than one chunk resident
    and never issues a request larger than one chunk. Draining the whole
    collection first (with vectors) and issuing a single upsert made both
    peak memory and the request size scale with the collection -- up to
    *page_size* x *max_pages* points carrying embeddings -- on exactly the
    multi-page collections task 3225 first made reachable.

    Preserves each point's original id and vector -- only the payload's
    user_id is rewritten. Qdrant upsert is id-keyed: this ASSUMES every
    *source* point id is globally unique with respect to *target* (true for
    Mem0's UUID point ids), so a same-id point already present in *target*
    would be silently overwritten rather than merged. No pre-upsert
    existence check is performed -- COLLECTION_MERGES entries must only ever
    pair collections whose point ids cannot collide.

    The *source* collection is deleted ONLY after this function's own drain
    reaches exhaustion. A drain that dies mid-stream leaves the target
    holding an idempotent, id-keyed PARTIAL copy while the source stays
    fully intact: a re-run with a larger budget completes the migration,
    harmlessly re-upserting the points that already landed. No data is lost,
    only duplicated work. Deleting the source there would destroy the
    un-migrated remainder, so the delete is withheld and
    ``enumeration_incomplete`` is returned True for the caller to fold into
    an UNRESOLVED disposition.

    Returns:
        ``{'points_upserted': n, 'source_deleted': bool,
        'enumeration_incomplete': bool, 'delete_failed': bool}``.
        ``points_upserted`` tallies what ACTUALLY reached *target*, not what
        was enumerated -- a chunk still buffered when the drain died was
        never sent and is not counted. ``enumeration_incomplete`` and
        ``delete_failed`` are DISTINCT because they describe different
        states: the first means part of the source never reached the target
        (a re-run must finish the migration), the second means every point
        landed but the now-redundant source could not be dropped (a re-run
        or a manual delete tidies up, and no data is at risk). ``run()``
        folds EITHER into an UNRESOLVED disposition.

    ANY failure is CAUGHT, never propagated, for the same reason
    ``preflight_collection_points`` catches its own: budget exhaustion, a
    ``TimeoutError`` from ``scroll_collection_pages``'s per-page
    ``asyncio.wait_for``, a transport error, or a raising
    ``delete_collection`` -- a raising sub-operation must not abort the whole
    consolidation run, because earlier keys/sections of the same ``--apply``
    pass may already hold committed mutations, and the later sections (junk
    keys, empty collections) would be skipped entirely, handing the operator
    a traceback instead of a manifest (the ``delete_empty_collection``
    idiom). Every branch leaves the source in place; only the budget branch
    advises raising the budget.
    """
    from qdrant_client.http import models as qmodels  # noqa: PLC0415

    points_upserted = 0
    chunk: list = []

    async def _flush() -> None:
        nonlocal points_upserted
        if not chunk:
            return
        await qdrant_client.upsert(collection_name=target, points=list(chunk))
        points_upserted += len(chunk)
        chunk.clear()

    try:
        async for point in backend.scroll_collection_pages(
            source,
            page_size=page_size,
            max_pages=max_pages,
            with_vectors=True,
        ):
            chunk.append(
                qmodels.PointStruct(
                    id=point.id,
                    vector=point.vector,
                    payload=rewrite_point_payload_user_id(
                        dict(point.payload or {}), canonical_user_id,
                    ),
                ),
            )
            if len(chunk) >= page_size:
                await _flush()
        await _flush()
    except ScrollPageBudgetExhausted:
        logger.warning(
            "consolidate_namespace_families: the merge drain of collection "
            "'%s' exhausted its page budget (%d page(s) of %d) after "
            '%d point(s) were upserted into %r -- the migration is '
            'INCOMPLETE, so this collection is reported UNRESOLVED and its '
            'source is NOT deleted. The points already upserted are id-keyed '
            'and idempotent: re-run with a higher --max-pages (page budget) '
            'or --limit (page size) to complete it.',
            source, max_pages, page_size, points_upserted, target,
        )
        return {
            'points_upserted': points_upserted,
            'source_deleted': False,
            'enumeration_incomplete': True,
            'delete_failed': False,
        }
    except Exception as e:
        # Same withheld delete, no raise-the-budget advice -- see the sibling
        # branch in scroll_collection_points.
        logger.warning(
            "consolidate_namespace_families: the merge drain of collection "
            "'%s' FAILED after %d point(s) were upserted into %r: %s -- the "
            'migration is INCOMPLETE, so this collection is reported '
            'UNRESOLVED and its source is NOT deleted. The points already '
            'upserted are id-keyed and idempotent, so a re-run completes it.',
            source, points_upserted, target, e, exc_info=True,
        )
        return {
            'points_upserted': points_upserted,
            'source_deleted': False,
            'enumeration_incomplete': True,
            'delete_failed': False,
        }

    # The delete is guarded for the SAME reason the drain is: a raising
    # sub-operation must not abort a pass whose earlier keys/sections may
    # already hold committed mutations. Withholding it is also harmless --
    # every point is already in the target, so the source is now a redundant
    # copy an operator can drop by hand or on a re-run (which re-upserts
    # id-keyed and idempotent).
    try:
        await qdrant_client.delete_collection(source)
    except Exception as e:
        logger.warning(
            "consolidate_namespace_families: collection '%s' was fully merged "
            'into %r (%d point(s)) but deleting the source FAILED: %s -- the '
            'collection is reported UNRESOLVED so the run exits non-zero. No '
            'data is at risk: the target holds every point, and the source is '
            'now a redundant copy that can be dropped by hand or by re-running.',
            source, target, points_upserted, e, exc_info=True,
        )
        return {
            'points_upserted': points_upserted,
            'source_deleted': False,
            'enumeration_incomplete': False,
            'delete_failed': True,
        }
    return {
        'points_upserted': points_upserted,
        'source_deleted': True,
        'enumeration_incomplete': False,
        'delete_failed': False,
    }


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


# ---------------------------------------------------------------------------
# Guarded empty-collection deletion (Qdrant)
# ---------------------------------------------------------------------------

async def count_collection_points(qdrant_client: Any, collection: str) -> int:
    """Read-only exact point count for *collection* via Qdrant's count API.

    THE single home for "how many points does this collection hold", with
    two consumers: the guard for ``delete_empty_collection`` (deletion is
    only safe when this is exactly 0) and ``preflight_collection_points``
    (which turns the same number into the collection-merge report's
    ``point_count`` and its ``capped`` budget verdict). An INDETERMINATE
    result -- a response
    object with no ``.count`` attribute at all, or an explicit ``.count is
    None`` -- RAISES (``ValueError``) rather than defaulting to 0
    (reviewer follow-up: for a deletion guard, an unreadable count must
    fail CLOSED and block the delete, not fail OPEN and authorize one).
    ``run()``'s empty-collection loop already catches a raising call here
    (as does ``preflight_collection_points``) and reports that item
    UNRESOLVED, exactly like its sibling
    ``count_graph_nodes``/``delete_junk_key`` guard, so raising is both
    consistent with that existing handler and strictly safer than the old
    silent-0 default. A genuine empty collection (``.count == 0``) is
    unaffected -- 0 is a valid, DETERMINATE count, not treated as missing.
    In practice Qdrant's count API always returns an int ``.count``, so
    this is defensive-only.
    """
    result = await qdrant_client.count(collection_name=collection)
    count = getattr(result, 'count', None)
    if count is None:
        raise ValueError(
            f"count_collection_points: Qdrant count() for collection "
            f"'{collection}' returned no usable .count (got {result!r}) -- "
            'refusing to treat an indeterminate count as empty.',
        )
    return int(count)


async def delete_empty_collection(qdrant_client: Any, collection: str, point_count: int) -> str:
    """Delete the *collection* -- but ONLY when its live *point_count* is
    exactly 0.

    Mirrors ``delete_junk_key``'s count-0 guard: a non-zero count returns
    'UNRESOLVED' without ever calling ``delete_collection()`` (deletion
    blocked, no data loss). Best-effort: a raising ``delete_collection()``
    is caught and reported as UNRESOLVED rather than propagating.
    """
    if point_count > 0:
        return 'UNRESOLVED'
    try:
        await qdrant_client.delete_collection(collection)
    except Exception as e:
        logger.warning(
            "consolidate_namespace_families: failed to delete empty Qdrant "
            "collection '%s': %s",
            collection, e,
        )
        return 'UNRESOLVED'
    return 'DELETE'


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def run(
    args: Any,
    memory_service: Any,
    *,
    limit: int = 1000,
    max_pages: int = DEFAULT_SCROLL_MAX_PAGES,
) -> dict:
    """Enumerate/scroll/count every configured family, collection, junk key,
    and empty-collection candidate and, with ``args.apply``, perform the
    merges + guarded deletions.

    Dry-run (``args.apply`` falsy) performs ZERO mutations: every section's
    ``disposition`` is a PREVIEW of what ``--apply`` would do, computed from
    read-only enumeration/count/scroll alone -- ``merge_graph_family`` and
    ``merge_collection`` are only invoked when ``args.apply`` is true AND the
    corresponding read was not capped (an UNRESOLVED item is never mutated,
    not even partially); the mutating halves of ``delete_junk_key`` and
    ``delete_empty_collection`` are only invoked when ``args.apply`` is true
    (each guarded internally on its own count being exactly 0).

    A capped read is reported ``UNRESOLVED`` rather than ``MERGE`` for that
    item, mirroring the junk-key count>0 guard: a partial read must never be
    mistaken for a clean, complete one (no-silent-caps). A graph enumeration
    is capped when its row count hits *limit*; a collection is capped when
    its point count exceeds what the merge drain could enumerate under
    ``limit * max_pages`` (see ``preflight_collection_points``).

    A graph-family item's ``disposition`` is DOWNGRADED from ``MERGE`` to
    ``UNRESOLVED`` after a clean (uncapped) ``--apply`` merge whenever the
    returned summary carries any loss/blocked signal (``edges_skipped``,
    ``mentions_skipped``, a non-empty ``dropped_cross_target``, a non-empty
    ``blocked``, or any ``nodes_blocked``/``episodes_blocked``) -- otherwise
    a family that tallies skipped edges/mentions but empties cleanly (no
    episodes left behind) would exit 0 despite losing data, the exact
    silent-signal failure task 2502 fixes. This mirrors
    ``migrate_cross_graph_leak.py``'s ``has_edge_loss``/
    ``has_dropped_cross_target``/``has_blocked_items`` exit-code folding,
    adapted to this script's per-item disposition rather than a parallel
    top-level predicate -- ``has_unresolved`` already scans every section
    for ``'UNRESOLVED'``, so no new predicate is needed. The summary's
    ``merge_mentions_dropped``/``merge_mentions_dropped_uuids`` census is
    DELIBERATELY excluded from that predicate (task 4183, operator ruling
    2026-08-12: visibility only) -- it is informational, like
    ``migrate_cross_graph_leak.py``'s ``embedding_omitted``, and must not
    flip an otherwise-clean family to ``UNRESOLVED``; the omission is a
    decision, not an oversight. A source whose
    deletion was withheld (Phase-A create failure or Phase-B ``blocked``)
    leaves recoverable residue in the sibling graph, which also keeps that
    sibling's junk-key node count > 0 -- so its GRAPH.DELETE (step 3 below)
    is correctly guarded off too, without any extra bookkeeping here.
    """
    # Fail-CLOSED capability preflight, one probe per run, BEFORE the scan.
    #
    # Hoisted to the top of ``run`` because this script has FOUR independent
    # ``args.apply`` gates -- graph-family merge, collection merge, junk-key
    # GRAPH.DELETE and empty-collection delete_collection -- each inside its
    # own per-item loop, and NONE dominates the other three. A guard at any one
    # of them would leave the other three phases unprotected; this is the only
    # single site that covers all four, and the only one where the probe is
    # once per RUN rather than once per item. It also sits above
    # ``_get_async_qdrant()`` below, so a refused run never even opens the
    # Qdrant transport.
    if args.apply:
        try:
            assert_store_mutation_allowed(
                operation='consolidate_namespace_families --apply'
            )
        except StoreMutationUnavailable:
            logger.error(
                'consolidate_namespace_families: --apply NOT started '
                "(fail-closed) -- this process cannot write mem0's history "
                'directory, so a consolidation would upsert points and delete '
                'source nodes without recording either, and a run interrupted '
                'part-way strands records in the sibling namespace with the '
                'canonical copy already written -- across four phases that '
                'each drop a collection, DETACH DELETE a graph, or delete a '
                'source node. Nothing was enumerated and nothing was mutated, '
                'and no Qdrant client was opened. Route the consolidation '
                'through the fused-memory MCP server (the unsandboxed owner '
                'of the store), or re-run from an unsandboxed operator shell. '
                'To obtain the preview report safely from anywhere, re-run '
                'without --apply.'
            )
            raise

    graphiti = memory_service.graphiti
    qdrant_client = await memory_service.mem0._get_async_qdrant()

    # --- 1. Graph-family merges (identity rewrite) -------------------------
    graph_family_items: list[dict] = []
    for sibling, canonical in GRAPH_FAMILY_ALIASES.items():
        entity_rows = await enumerate_graph_entity_nodes(graphiti, sibling, limit=limit)
        episode_rows = await enumerate_graph_episodic_nodes(graphiti, sibling, limit=limit)
        capped = len(entity_rows) >= limit or len(episode_rows) >= limit
        item: dict[str, Any] = {
            'sibling': sibling,
            'canonical': canonical,
            'node_count': len(entity_rows),
            'node_uuids': [row['uuid'] for row in entity_rows],
            'episode_count': len(episode_rows),
            'episode_uuids': [row['uuid'] for row in episode_rows],
            'disposition': 'UNRESOLVED' if capped else 'MERGE',
        }
        if args.apply and not capped:
            summary = await merge_graph_family(graphiti, sibling, canonical, entity_rows, episode_rows)
            item.update(summary)
            # Fold edge/mention loss + dropped_cross_target + blocked + Phase-A
            # create failures into UNRESOLVED -- the exact silent-signal
            # failure this task fixes (merge_graph_family tallies skipped
            # counts, but a family with no episodes empties cleanly and used
            # to exit 0 even after losing edges). Mirrors
            # migrate_cross_graph_leak.py's has_edge_loss/
            # has_dropped_cross_target/has_blocked_items exit-code folding,
            # adapted from that script's per-manifest predicate to this
            # script's per-item disposition -- has_unresolved (already
            # scanning for 'UNRESOLVED') picks this up with no new predicate.
            if (
                summary['edges_skipped']
                or summary['mentions_skipped']
                or summary['dropped_cross_target']
                or summary['blocked']
                or summary['nodes_blocked']
                or summary['episodes_blocked']
            ):
                item['disposition'] = 'UNRESOLVED'
        graph_family_items.append(item)

    # --- 2. Qdrant collection merges ----------------------------------------
    collection_items: list[dict] = []
    for source, target in COLLECTION_MERGES.items():
        # O(1) preflight: ONE count round-trip, not a paged scroll. `capped`
        # is the flag it returns, NOT point_count >= limit -- with real
        # paging a mergeable collection can hold any count; what disqualifies
        # it is holding more than --limit x --max-pages. merge_collection
        # below runs the pass's sole drain.
        point_count, capped = await preflight_collection_points(
            qdrant_client, source, page_size=limit, max_pages=max_pages,
        )
        canonical_user_id = canonical_user_id_for(target)
        item = {
            'source': source,
            'target': target,
            'canonical_user_id': canonical_user_id,
            'point_count': point_count,
            'disposition': 'UNRESOLVED' if capped else 'MERGE',
        }
        # Guarded exactly like the graph-family branch above: a capped
        # (possibly-incomplete) scroll must not mutate the target at all,
        # not even a partial upsert -- an UNRESOLVED item must mean nothing
        # was written, matching the "reported UNRESOLVED rather than acted
        # on" contract (module docstring).
        if args.apply and not capped:
            summary = await merge_collection(
                memory_service.mem0, qdrant_client, source, target, canonical_user_id,
                page_size=limit, max_pages=max_pages,
            )
            item.update(summary)
            # A drain that died mid-merge upserted only part of the source,
            # and a merge whose terminal delete failed left the source behind
            # -- neither is a clean MERGE, so fold both into UNRESOLVED
            # exactly like the graph-family post-merge downgrade above, so
            # has_unresolved picks them up and the run exits non-zero rather
            # than reporting an incomplete migration as done.
            if summary['enumeration_incomplete'] or summary['delete_failed']:
                item['disposition'] = 'UNRESOLVED'
        collection_items.append(item)

    # --- 3. Guarded junk-key deletion (JUNK_KEYS + emptied siblings) --------
    junk_key_items: list[dict] = []
    for key in (*JUNK_KEYS, *GRAPH_FAMILY_ALIASES.keys()):
        try:
            node_count = await count_graph_nodes(graphiti, key)
        except Exception as e:
            # A raising count must never abort the whole consolidation run --
            # earlier keys/sections in this same --apply pass may already
            # hold committed mutations. Report this key UNRESOLVED and move
            # on, exactly like the delete_junk_key guard does for a raising
            # .delete().
            logger.warning(
                "consolidate_namespace_families: failed to count nodes for "
                "graph key '%s': %s -- reporting UNRESOLVED rather than "
                "aborting the run.",
                key, e,
            )
            junk_key_items.append({
                'key': key,
                'node_count': None,
                'disposition': 'UNRESOLVED',
            })
            continue
        if args.apply:
            disposition = await delete_junk_key(graphiti, key, node_count)
        else:
            disposition = 'DELETE' if node_count == 0 else 'UNRESOLVED'
        junk_key_items.append({
            'key': key,
            'node_count': node_count,
            'disposition': disposition,
        })

    # --- 4. Guarded empty-collection deletion (EMPTY_COLLECTION_CLEANUP) ---
    empty_collection_items: list[dict] = []
    for collection in EMPTY_COLLECTION_CLEANUP:
        try:
            point_count = await count_collection_points(qdrant_client, collection)
        except Exception as e:
            # Mirrors the junk-key count guard above: a raising count must
            # never abort the whole consolidation run -- earlier keys/
            # sections in this same --apply pass may already hold committed
            # mutations. Report this collection UNRESOLVED and move on.
            logger.warning(
                "consolidate_namespace_families: failed to count points for "
                "Qdrant collection '%s': %s -- reporting UNRESOLVED rather "
                "than aborting the run.",
                collection, e,
            )
            empty_collection_items.append({
                'collection': collection,
                'point_count': None,
                'disposition': 'UNRESOLVED',
            })
            continue
        if args.apply:
            disposition = await delete_empty_collection(qdrant_client, collection, point_count)
        else:
            disposition = 'DELETE' if point_count == 0 else 'UNRESOLVED'
        empty_collection_items.append({
            'collection': collection,
            'point_count': point_count,
            'disposition': disposition,
        })

    return build_consolidation_report(
        graph_family_items, collection_items, junk_key_items, empty_collection_items,
        dry_run=not args.apply,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser.

    Extracted from ``main()`` so the flags are testable without running the
    consolidation, mirroring ``census_memory_metadata.py::_build_parser``.
    """
    parser = argparse.ArgumentParser(
        description=(
            'Consolidate cross-graph namespace families: merge sibling Graphiti '
            'graphs (with identity rewrite), merge legacy Qdrant collections '
            '(with user_id rewrite), and delete guarded junk keys and empty '
            'Qdrant collections.'
        ),
    )
    parser.add_argument(
        '--apply', action='store_true', default=False,
        help='Commit the merges + junk-key/empty-collection deletions '
             '(default: dry-run only, report and exit).',
    )
    parser.add_argument(
        '--limit', type=int, default=1000,
        help='Page/row size (default: 1000). For the GRAPH enumerations it is '
             'a single-page row cap: it must exceed the true row count of the '
             'largest graph being migrated, or that item is permanently '
             'reported UNRESOLVED. For the COLLECTION scroll it is the PAGE '
             'SIZE -- pages are followed to exhaustion, so a collection larger '
             'than this now migrates fully instead of being permanently '
             'UNRESOLVED; --max-pages bounds the total. It is also the merge '
             'upsert CHUNK size, so peak memory scales with this value, not '
             'with the collection. Increase if the dry-run report logs a '
             'row-cap WARNING.',
    )
    parser.add_argument(
        '--max-pages', dest='max_pages', type=int, default=DEFAULT_SCROLL_MAX_PAGES,
        help='Per-collection PAGE BUDGET for the Qdrant scroll (default: '
             f'{DEFAULT_SCROLL_MAX_PAGES}). Total points enumerated per '
             'collection is bounded by --limit x --max-pages; exceeding it '
             'reports that collection UNRESOLVED with its source undeleted. '
             'Raise it when the report logs a page-budget WARNING.',
    )
    parser.add_argument(
        '--config', default=None,
        help='Path to a fused-memory config file (sets CONFIG_PATH before loading).',
    )
    return parser


def main() -> int:
    """Parse CLI args, build a live MemoryService, and run the consolidation."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
    )
    args = _build_parser().parse_args()

    if args.config:
        import os  # noqa: PLC0415
        os.environ['CONFIG_PATH'] = str(args.config)

    async def _run_live() -> dict:
        from fused_memory.config.schema import FusedMemoryConfig  # noqa: PLC0415
        from fused_memory.services.memory_service import MemoryService  # noqa: PLC0415

        config = FusedMemoryConfig()
        memory = MemoryService(config)
        try:
            await memory.initialize()
            return await run(args, memory, limit=args.limit, max_pages=args.max_pages)
        finally:
            if hasattr(memory, 'close'):
                await memory.close()

    report = asyncio.run(_run_live())
    print(json.dumps(report, indent=2, default=str))
    if not args.apply:
        logger.info('Dry run -- nothing was modified. Use --apply to commit the consolidation.')
    return 1 if (args.apply and has_unresolved(report)) else 0


if __name__ == '__main__':
    sys.exit(main())
