"""Cross-graph entity move + foreign-duplicate-merge primitives (CGL-ε, task 2271).

Delivers the cross-graph re-key/move primitive that does not exist anywhere
else in this codebase today: ``GraphitiBackend.merge_entities``
(backends/graphiti_client.py:1013) only merges nodes within a single graph,
and ``scripts/purge_knowlive_namespace.py``'s docstring states outright that
there is no clean re-key/cross-graph-move primitive to re-home orphaned data.

Validated approach (``plans/cross-graph-entity-leak-rca.md`` §6 Phase 1,
RCA-validated 2026-07-06 experiment): FalkorDB's decoded/textual float form
for a ``vecf32`` property truncates to 6 decimal places and is LOSSY. Reading
via the raw ``GRAPH.RO_QUERY ... --compact`` transport instead yields the
EXACT float32 decimal string as it exists on the wire. This module therefore
never calls ``float()`` on a vector component -- embeddings are carried as
opaque strings from read straight through to the recreated node/edge's
``vecf32([...])`` Cypher literal (see ``parse_compact_vector_reply`` /
``format_vecf32_literal``).

Scope (Phase-1 foundation only, per ``plans/cross-graph-entity-leak-prd.md``
contract seams S5+S6): primitives only -- no CLI/``run_*`` entrypoint, no
live-data run. Consumed by the migration (ζ) and consolidation (θ) scripts,
which are separate tasks. Byte-fidelity against a REAL FalkorDB is
deliberately NOT asserted by this module's test suite (mock-only, per
project convention) -- that is mandated in the η live throwaway-graph
rehearsal (PRD decision 5).

All Cypher and the raw-embedding transport live in this module (reached via
``graphiti._graph_for(name)`` and ``graphiti._require_falkor_client()``)
rather than as new ``GraphitiBackend`` methods, to avoid file-lock
contention with the γ/W6 normalization work landing in graphiti_client.py
concurrently (PRD G4).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from fused_memory.backends.graphiti_client import NodeNotFoundError

logger = logging.getLogger(__name__)

# falkordb.query_result.ResultSetScalarTypes.VALUE_VECTORF32 -- the compact-
# reply scalar-type tag for a vecf32 column (see _read_compact_vector).
_VALUE_VECTORF32 = 12


# ---------------------------------------------------------------------------
# Byte-exact vecf32 passthrough (pure functions -- never call float() here)
# ---------------------------------------------------------------------------

def parse_compact_vector_reply(reply: str) -> list[str]:
    """Parse a ``GRAPH.RO_QUERY ... --compact`` vector reply into exact tokens.

    *reply* is the bracketed, comma-separated vector text as read from the
    raw compact transport (e.g. ``'[0.5, -0.25]'``). Returns each element as
    its verbatim string token -- whitespace-stripped, but otherwise untouched.

    Never calls ``float()`` (or any other numeric coercion) on a token: the
    whole point of this function is to preserve the exact float32 decimal
    string as it existed on the wire, since the alternative (decoded/textual)
    read path is lossy (see module docstring).
    """
    text = reply.strip()
    if text.startswith('[') and text.endswith(']'):
        text = text[1:-1]
    text = text.strip()
    if not text:
        return []
    return [token.strip() for token in text.split(',')]


def format_vecf32_literal(tokens: list[str]) -> str:
    """Render exact float32 string tokens as a ``vecf32([...])`` Cypher literal.

    *tokens* are embedded verbatim (comma-space joined) -- never re-parsed or
    reformatted -- so that the exact decimal strings read via the raw
    ``--compact`` transport survive untouched into the Cypher literal used to
    recreate the node/edge's vector property on the target/home graph.
    """
    return f"vecf32([{', '.join(tokens)}])"


def _quote_cypher_string(value: str) -> str:
    """Inline *value* as a single-quoted Cypher string literal, escaping ``\\``/``'``.

    Used only for uuids (program-generated, low-risk) in the hand-rolled
    Cypher issued through the raw ``--compact`` transport, which bypasses the
    normal ``graph.query(cypher, params)`` parameter-binding path (see
    ``_read_compact_vector``).
    """
    escaped = value.replace('\\', '\\\\').replace("'", "\\'")
    return f"'{escaped}'"


async def _read_compact_vector(falkor_client: Any, *, group_id: str, cypher: str) -> str:
    """Injectable raw ``--compact`` transport seam for a single vecf32 column.

    Issues *cypher* (expected to ``RETURN`` exactly one vecf32-typed value)
    against *group_id* directly through the raw FalkorDB client (bypassing
    ``falkordb.Graph.query()``/``.ro_query()``, whose ``QueryResult.parse()``
    -- specifically ``__parse_vectorf32``/``__parse_double`` -- calls
    ``float()`` on each component and so silently adopts FalkorDB's lossy
    textual double representation; this is the RCA-validated truncation this
    module exists to avoid). Returns the raw bracketed, comma-separated
    vector text (the same shape ``parse_compact_vector_reply`` consumes),
    extracted from the reply WITHOUT ever calling ``float()``.

    This default implementation is a best-effort reading of the ``falkordb``
    package's own (already-parsed-elsewhere) reply-shape convention
    (``falkordb.query_result.parse_scalar``: ``[scalar_type, value]`` cells,
    scalar_type 12 == VECTORF32) -- it is NOT exercised by this (mock-only)
    test suite. Tests monkeypatch this module attribute directly with a fake
    returning a recorded fixture string. Byte-fidelity against a REAL
    FalkorDB is validated separately, in the η live throwaway-graph
    rehearsal (see ``plans/cross-graph-entity-leak-prd.md`` decision 5).
    """
    reply = await falkor_client.execute_command('GRAPH.RO_QUERY', group_id, cypher, '--compact')
    row = reply[1][0]
    cell = row[0]
    scalar_type, value = int(cell[0]), cell[1]
    if scalar_type != _VALUE_VECTORF32:
        raise ValueError(
            f'_read_compact_vector: expected a VECTORF32 (12) cell, got type {scalar_type}'
        )
    tokens = [v.decode() if isinstance(v, bytes) else str(v) for v in value]
    return '[' + ', '.join(tokens) + ']'


@dataclass
class MoveResult:
    """Result of a ``move_entity_across_graphs`` call.

    Attributes:
        uuid: UUID of the moved (or already-moved) Entity node.
        source_graph: Graph the node was moved from.
        target_graph: Graph the node was moved to.
        already_moved: True when the idempotency probe short-circuited the
            call as a no-op (uuid already present in target, absent from
            source) -- see step-13/14. Defaults to False.
        edges_moved: Count of RELATES_TO edges reattached to the moved node.
        mentions_moved: Count of Episodic MENTIONS links reattached.
    """

    uuid: str
    source_graph: str
    target_graph: str
    already_moved: bool = False
    edges_moved: int = 0
    mentions_moved: int = 0


async def move_entity_across_graphs(
    graphiti: Any,
    uuid: str,
    source_graph: str,
    target_graph: str,
    *,
    rewrite_group_id: str | None = None,
) -> MoveResult:
    """Move the Entity node *uuid* from *source_graph* to *target_graph*.

    Node-core (step-5/6): reads the node's scalar props from source via the
    normal (non-lossy) ``ro_query`` path, reads its exact ``name_embedding``
    via the raw ``--compact`` transport, and ``CREATE``s the node in target
    with a byte-exact ``vecf32([...])`` literal.

    Edges (step-7/8): every RELATES_TO edge incident to the node (either
    direction) is recreated on the corresponding target-graph node with a
    byte-exact ``fact_embedding``, provided the OTHER endpoint already
    exists in target_graph (silently skipped otherwise -- see inline note).

    Mentions (step-9/10): every Episodic MENTIONS link onto the node is
    recreated in target_graph, provided the episode node already exists
    there (same silent-skip rule as edges; MENTIONS carries no embedding).

    Delete (step-11/12): once every target CREATE above has been awaited,
    the source node is ``DETACH DELETE``d -- always the LAST mutation, so a
    crash mid-move still leaves a recoverable duplicate rather than losing
    the only copy.

    The idempotency probe is added in a later step (13-14);
    ``rewrite_group_id`` substitution for the node/edges/mentions lands in
    step-16 (applied here as a forward-compatible no-op when
    ``rewrite_group_id`` is None).

    Args:
        graphiti: An initialized GraphitiBackend (or compatible object
            exposing ``_graph_for`` / ``_require_falkor_client``).
        uuid: UUID of the Entity node to move.
        source_graph: Graph the node currently lives in.
        target_graph: Graph to move the node into.
        rewrite_group_id: When given, substituted for the node's ``group_id``
            property on recreate; when None (Phase-1 default), the source
            node's own group_id is carried through unchanged.

    Returns:
        MoveResult describing the move.

    Raises:
        NodeNotFoundError: if no Entity node with *uuid* exists in
            *source_graph*.
    """
    source = graphiti._graph_for(source_graph)
    target = graphiti._graph_for(target_graph)

    node_result = await source.ro_query(
        'MATCH (n:Entity {uuid: $uuid}) '
        'RETURN n.uuid, n.name, n.group_id, n.summary, n.created_at',
        {'uuid': uuid},
    )
    if not node_result.result_set:
        raise NodeNotFoundError(f'Entity node not found in source_graph {source_graph!r}: {uuid}')
    node_uuid, name, group_id, summary, created_at = node_result.result_set[0]

    falkor_client = graphiti._require_falkor_client()
    embedding_cypher = (
        f'MATCH (n:Entity {{uuid: {_quote_cypher_string(uuid)}}}) RETURN n.name_embedding'
    )
    embedding_reply = await _read_compact_vector(
        falkor_client, group_id=source_graph, cypher=embedding_cypher,
    )
    embedding_literal = format_vecf32_literal(parse_compact_vector_reply(embedding_reply))

    new_group_id = group_id if rewrite_group_id is None else rewrite_group_id

    await target.query(
        'CREATE (n:Entity {uuid: $uuid}) '
        'SET n.name = $name, '
        '    n.group_id = $group_id, '
        '    n.summary = $summary, '
        '    n.created_at = $created_at, '
        f'    n.name_embedding = {embedding_literal}',
        {
            'uuid': node_uuid,
            'name': name,
            'group_id': new_group_id,
            'summary': summary,
            'created_at': created_at,
        },
    )

    # --- RELATES_TO edges (step-7/8) ---
    # Undirected match + WITH DISTINCT e mirrors GraphitiBackend.get_valid_edges_for_node's
    # established self-loop dedup idiom; startNode()/endNode() recover the edge's TRUE
    # direction regardless of which side matched $uuid, so it can be preserved on recreate.
    edges_result = await source.ro_query(
        'MATCH (n:Entity {uuid: $uuid})-[e:RELATES_TO]-(m:Entity) '
        'WITH DISTINCT e, startNode(e) AS s, endNode(e) AS t '
        'RETURN e.uuid, e.name, e.fact, e.valid_at, e.invalid_at, e.created_at, '
        '       e.group_id, e.episodes, s.uuid, t.uuid',
        {'uuid': uuid},
    )
    edges_moved = 0
    for row in edges_result.result_set or []:
        (edge_uuid, edge_name, fact, valid_at, invalid_at, edge_created_at,
         _edge_group_id, episodes, src_uuid, dst_uuid) = row

        edge_embedding_cypher = (
            f'MATCH ()-[e:RELATES_TO {{uuid: {_quote_cypher_string(edge_uuid)}}}]-() '
            'RETURN e.fact_embedding'
        )
        edge_embedding_reply = await _read_compact_vector(
            falkor_client, group_id=source_graph, cypher=edge_embedding_cypher,
        )
        edge_embedding_literal = format_vecf32_literal(
            parse_compact_vector_reply(edge_embedding_reply)
        )

        # Both endpoints are MATCHed (never CREATEd) by uuid: the moved node was
        # just created above, and the other endpoint must already exist in
        # target_graph for the edge to be recreated -- otherwise this MATCH
        # yields no rows and the edge is silently skipped (left for the
        # caller to move the other endpoint too, or accept the drop).
        await target.query(
            'MATCH (a:Entity {uuid: $src_uuid}), (b:Entity {uuid: $dst_uuid}) '
            'CREATE (a)-[r:RELATES_TO]->(b) '
            'SET r.uuid = $edge_uuid, '
            '    r.name = $name, '
            '    r.fact = $fact, '
            '    r.valid_at = $valid_at, '
            '    r.invalid_at = $invalid_at, '
            '    r.created_at = $created_at, '
            '    r.group_id = $group_id, '
            '    r.episodes = $episodes, '
            f'    r.fact_embedding = {edge_embedding_literal}',
            {
                'src_uuid': src_uuid,
                'dst_uuid': dst_uuid,
                'edge_uuid': edge_uuid,
                'name': edge_name,
                'fact': fact,
                'valid_at': valid_at,
                'invalid_at': invalid_at,
                'created_at': edge_created_at,
                'group_id': new_group_id,
                'episodes': episodes,
            },
        )
        edges_moved += 1

    # --- Episodic MENTIONS links (step-9/10) ---
    # MENTIONS carries no embedding (graphiti_core's own save query, see
    # models/edges/edge_db_queries.py EPISODIC_EDGE_SAVE, only ever sets
    # uuid/group_id/created_at) -- no raw-transport read needed here.
    mentions_result = await source.ro_query(
        'MATCH (ep:Episodic)-[e:MENTIONS]->(n:Entity {uuid: $uuid}) '
        'RETURN e.uuid, e.group_id, e.created_at, ep.uuid',
        {'uuid': uuid},
    )
    mentions_moved = 0
    for row in mentions_result.result_set or []:
        mention_uuid, _mention_group_id, mention_created_at, episode_uuid = row
        # As with RELATES_TO's other endpoint: the Episodic node is not moved
        # by this primitive and must already exist in target_graph, or this
        # MATCH yields no rows and the mention is silently skipped.
        await target.query(
            'MATCH (ep:Episodic {uuid: $episode_uuid}), (n:Entity {uuid: $entity_uuid}) '
            'CREATE (ep)-[e:MENTIONS]->(n) '
            'SET e.uuid = $edge_uuid, '
            '    e.group_id = $group_id, '
            '    e.created_at = $created_at',
            {
                'episode_uuid': episode_uuid,
                'entity_uuid': node_uuid,
                'edge_uuid': mention_uuid,
                'group_id': new_group_id,
                'created_at': mention_created_at,
            },
        )
        mentions_moved += 1

    # --- source DETACH DELETE (step-11/12) ---
    # Always the LAST mutation: every target CREATE (node, edges, mentions)
    # above has already been awaited, so a crash here still leaves a
    # recoverable duplicate in target_graph rather than losing the only
    # copy. Never touches target_graph.
    await source.query(
        'MATCH (n:Entity {uuid: $uuid}) DETACH DELETE n',
        {'uuid': uuid},
    )

    logger.info(
        'move_entity_across_graphs: node moved uuid=%s source=%s target=%s '
        'edges_moved=%d mentions_moved=%d',
        uuid, source_graph, target_graph, edges_moved, mentions_moved,
    )
    return MoveResult(
        uuid=uuid,
        source_graph=source_graph,
        target_graph=target_graph,
        edges_moved=edges_moved,
        mentions_moved=mentions_moved,
    )
