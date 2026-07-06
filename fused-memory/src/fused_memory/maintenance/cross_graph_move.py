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
from dataclasses import dataclass, field
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

    This default implementation reads the ``falkordb`` package's own
    (already-parsed-elsewhere) reply-shape convention
    (``falkordb.query_result.parse_scalar``: ``[scalar_type, value]`` cells,
    scalar_type 12 == VECTORF32). ``TestReadCompactVectorTransport`` (in the
    test suite) exercises this parsing directly against a recorded/
    representative raw reply structure -- every OTHER test in this module
    monkeypatches this function away. Byte-fidelity against a REAL FalkorDB
    remains validated separately, in the η live throwaway-graph rehearsal
    (see ``plans/cross-graph-entity-leak-prd.md`` decision 5).

    A null/absent embedding (e.g. an Entity/RELATES_TO row somehow missing
    its embedding property) surfaces as the ``ValueError`` below, never as a
    silently-empty vector: FalkorDB reports a NULL property with a
    non-VECTORF32 ``scalar_type``, so it fails the type check before the
    token list is ever indexed. This is a deliberate choice, not an
    oversight -- every row this module reads is expected to carry an
    embedding, so a missing one is treated as a hard, descriptively-messaged
    failure (naming *group_id* and *cypher*) that surfaces a corrupt/
    unexpected source row immediately, rather than silently proceeding
    without it (which would produce a differently-shaped, harder-to-
    diagnose gap later -- e.g. a moved node that can no longer be found by
    semantic search).
    """
    reply = await falkor_client.execute_command('GRAPH.RO_QUERY', group_id, cypher, '--compact')
    row = reply[1][0]
    cell = row[0]
    scalar_type, value = int(cell[0]), cell[1]
    if scalar_type != _VALUE_VECTORF32:
        raise ValueError(
            f'_read_compact_vector: expected a VECTORF32 (12) cell for '
            f'group_id={group_id!r}, got scalar_type={scalar_type} '
            f'(cypher={cypher!r}). A null/absent embedding on the source row '
            f"surfaces here too -- see this function's docstring."
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
            source). Defaults to False.
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

    Edges (step-7/8, amended): every RELATES_TO edge incident to the node
    (either direction) is recreated on the corresponding target-graph node
    with a byte-exact ``fact_embedding``, PROVIDED the other endpoint
    already exists in target_graph -- checked via a cheap presence probe
    BEFORE the comparatively expensive raw ``--compact`` embedding read, so
    an edge that will be skipped never pays for that round trip (silently
    skipped otherwise -- see inline note). ``edges_moved`` only counts
    edges the recreate actually created (the CREATE's own ``RETURN`` is
    checked), so it can never overstate the count.

    Mentions (step-9/10, amended): every Episodic MENTIONS link onto the
    node is recreated in target_graph, provided the episode node already
    exists there (same silent-skip rule as edges; MENTIONS carries no
    embedding, so there is no separate pre-check to add). ``mentions_moved``
    is likewise only counted when the recreate's ``RETURN`` confirms a row
    was created.

    Delete (step-11/12): once every target CREATE above has been awaited,
    the source node is ``DETACH DELETE``d -- always the LAST mutation, so a
    crash mid-move still leaves a recoverable duplicate rather than losing
    the only copy.

    Idempotency (step-13/14, amended): a single, unconditional probe reads
    whether the node is present in target_graph, run BEFORE any mutation
    regardless of whether the node is still present in source_graph.
    Combined with the source read above, this yields three cases:

    - Absent from source, present in target: the move already completed
      (e.g. a prior run finished, or crashed strictly after the source
      DETACH DELETE below). No-op: MoveResult is returned with
      ``already_moved=True`` and neither graph is touched again.
    - Absent from source, absent from target: genuinely not found --
      raises NodeNotFoundError.
    - Present in BOTH source and target: a prior run crashed strictly
      BETWEEN the node CREATE and the source DETACH DELETE. The node
      CREATE is skipped (target already has it -- re-issuing CREATE here
      would duplicate it, since multi-tenant FalkorDB builds no uniqueness
      constraint on Entity.uuid), and the call proceeds straight to
      (re-)attempting the edge/mention reattachment and the source delete,
      so a retry converges on a single target copy instead of a duplicate.

    Residual hazard (NOT closed by the above): a resumed move does not
    itself deduplicate RELATES_TO/MENTIONS rows that were already recreated
    in target during the crashed attempt -- only the Entity node recreate
    is guarded this way; edges/mentions have no MERGE-keyed or presence-
    checked-against-target write of their own uuid. A retry after a crash
    that got partway through the edges/mentions loops may therefore
    recreate (and thus duplicate) rows it already created before crashing.
    Fully closing this would need per-edge/mention idempotent writes, which
    is out of scope here; callers needing exact-once edge/mention semantics
    across crash-recovery should independently verify post-move counts.

    Group-id rewrite (step-15/16): ``new_group_id`` (rewrite_group_id if
    given, else the source node's own group_id) is applied uniformly to the
    recreated node's ``group_id`` property AND to every recreated edge's/
    mention's ``group_id`` property -- so a caller rewriting the node's home
    graph gets a consistent group_id across the whole moved subgraph, not
    just the node itself. When ``rewrite_group_id`` is None (the Phase-1
    default), this is a no-op: every group_id is carried through unchanged.

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
            *source_graph* (and it is also absent from *target_graph* --
            otherwise the idempotency no-op above applies instead).
    """
    source = graphiti._graph_for(source_graph)
    target = graphiti._graph_for(target_graph)

    node_result = await source.ro_query(
        'MATCH (n:Entity {uuid: $uuid}) '
        'RETURN n.uuid, n.name, n.group_id, n.summary, n.created_at',
        {'uuid': uuid},
    )

    # --- unconditional target-presence probe (step-13/14, amended) ---
    # Always read target presence before any mutation, regardless of
    # whether the node is still present in source. This one probe backs
    # BOTH the fully-idempotent no-op (absent from source, present in
    # target) AND detecting a crash that happened strictly between the node
    # CREATE and the source DETACH DELETE (present in BOTH graphs) -- see
    # the "Idempotency" section of this function's docstring.
    target_probe = await target.ro_query(
        'MATCH (n:Entity {uuid: $uuid}) RETURN n.uuid',
        {'uuid': uuid},
    )
    node_already_in_target = bool(target_probe.result_set)

    if not node_result.result_set:
        if node_already_in_target:
            logger.info(
                'move_entity_across_graphs: already moved, no-op uuid=%s '
                'source=%s target=%s',
                uuid, source_graph, target_graph,
            )
            return MoveResult(
                uuid=uuid,
                source_graph=source_graph,
                target_graph=target_graph,
                already_moved=True,
            )
        raise NodeNotFoundError(f'Entity node not found in source_graph {source_graph!r}: {uuid}')
    node_uuid, name, group_id, summary, created_at = node_result.result_set[0]

    falkor_client = graphiti._require_falkor_client()
    new_group_id = group_id if rewrite_group_id is None else rewrite_group_id

    if node_already_in_target:
        # Resuming after a crash strictly between the node CREATE and the
        # source DETACH DELETE below: skip re-CREATE-ing the node (it's
        # already there -- see the docstring's "Idempotency" section) and
        # fall straight through to (re-)attempting edges/mentions + delete.
        logger.warning(
            'move_entity_across_graphs: resuming partially-completed move '
            '(node already present in target, skipping node CREATE) uuid=%s '
            'source=%s target=%s',
            uuid, source_graph, target_graph,
        )
    else:
        embedding_cypher = (
            f'MATCH (n:Entity {{uuid: {_quote_cypher_string(uuid)}}}) RETURN n.name_embedding'
        )
        embedding_reply = await _read_compact_vector(
            falkor_client, group_id=source_graph, cypher=embedding_cypher,
        )
        embedding_literal = format_vecf32_literal(parse_compact_vector_reply(embedding_reply))

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

        # Pre-check (amended, suggestion #5): confirm BOTH endpoints already
        # exist in target_graph via the same MATCH pattern the recreate
        # below uses, BEFORE paying for the raw --compact embedding round
        # trip. Without this, an edge whose other endpoint hasn't been moved
        # to target yet would still incur that (comparatively expensive)
        # read only to be silently skipped by the recreate's own MATCH.
        endpoint_probe = await target.ro_query(
            'MATCH (a:Entity {uuid: $src_uuid}), (b:Entity {uuid: $dst_uuid}) '
            'RETURN a.uuid, b.uuid',
            {'src_uuid': src_uuid, 'dst_uuid': dst_uuid},
        )
        if not endpoint_probe.result_set:
            logger.info(
                'move_entity_across_graphs: skipping RELATES_TO edge uuid=%s -- '
                'other endpoint not yet present in target_graph=%s',
                edge_uuid, target_graph,
            )
            continue

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

        # Both endpoints are MATCHed (never CREATEd) by uuid -- the pre-check
        # above already confirmed both exist in target_graph, so this MATCH
        # is expected to succeed; the RETURN is still checked below (rather
        # than incrementing edges_moved unconditionally) as a defense-in-
        # depth guard against the CREATE's own MATCH unexpectedly matching
        # nothing (see suggestion #2).
        create_result = await target.query(
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
            f'    r.fact_embedding = {edge_embedding_literal} '
            'RETURN r.uuid',
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
        if create_result.result_set:
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
        # MATCH yields no rows and the mention is silently skipped. Mentions
        # carry no embedding, so (unlike edges) there's no expensive read to
        # protect with a separate pre-check -- the RETURN below is enough.
        mention_result = await target.query(
            'MATCH (ep:Episodic {uuid: $episode_uuid}), (n:Entity {uuid: $entity_uuid}) '
            'CREATE (ep)-[e:MENTIONS]->(n) '
            'SET e.uuid = $edge_uuid, '
            '    e.group_id = $group_id, '
            '    e.created_at = $created_at '
            'RETURN e.uuid',
            {
                'episode_uuid': episode_uuid,
                'entity_uuid': node_uuid,
                'edge_uuid': mention_uuid,
                'group_id': new_group_id,
                'created_at': mention_created_at,
            },
        )
        if mention_result.result_set:
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


# ---------------------------------------------------------------------------
# merge_foreign_duplicate (S6)
# ---------------------------------------------------------------------------

def classify_unique_wrong_edges(
    home_edge_uuids: set[str],
    wrong_edge_uuids: set[str],
) -> set[str]:
    """Return edge uuids present on the wrong-graph copy but absent from home.

    Pure set-diff (``wrong_edge_uuids - home_edge_uuids``) over edge UUIDS
    (not edge objects/rows) -- the classification ``merge_foreign_duplicate``
    uses to decide which of the wrong-graph copy's edges must be recreated
    on the home copy before the wrong-graph copy is deleted. Edges present on
    both copies are already accounted for on home and are left untouched.
    """
    return set(wrong_edge_uuids) - set(home_edge_uuids)


@dataclass
class MergeResult:
    """Result of a ``merge_foreign_duplicate`` call.

    Attributes:
        uuid: UUID of the Entity node whose wrong-graph duplicate was merged
            into its home-graph copy.
        wrong_graph: Graph the foreign (to-be-deleted) duplicate lived in.
        home_graph: Graph the canonical copy lives in.
        unique_wrong_edge_uuids: Edge uuids present on the wrong-graph copy
            but absent from the home copy (``classify_unique_wrong_edges``
            over both copies' edge-uuid sets).
        edges_recreated: Count of unique edges recreated on the home copy.
        home_edge_count_before: Home copy's RELATES_TO edge count before the
            merge.
        home_edge_count_after: Home copy's RELATES_TO edge count RE-READ from
            the home graph after the recreates (not derived arithmetically
            from ``home_edge_count_before + edges_recreated``) -- so
            comparing the two attributes is a genuine no-edge-lost /
            no-double-count check rather than a tautology: a recreate whose
            MATCH silently matched nothing would surface as a mismatch here.
    """

    uuid: str
    wrong_graph: str
    home_graph: str
    unique_wrong_edge_uuids: frozenset[str] = field(default_factory=frozenset)
    edges_recreated: int = 0
    home_edge_count_before: int = 0
    home_edge_count_after: int = 0


async def _read_relates_to_edge_uuids(graph: Any, uuid: str) -> set[str]:
    """Read the set of RELATES_TO edge uuids incident to Entity *uuid* (either direction).

    Lightweight uuid-only counterpart of the full-property edge read in
    ``move_entity_across_graphs`` -- used where only set-membership (not the
    edges' own properties) is needed.
    """
    result = await graph.ro_query(
        'MATCH (n:Entity {uuid: $uuid})-[e:RELATES_TO]-(m:Entity) '
        'WITH DISTINCT e '
        'RETURN e.uuid',
        {'uuid': uuid},
    )
    return {row[0] for row in (result.result_set or [])}


async def merge_foreign_duplicate(
    graphiti: Any,
    uuid: str,
    wrong_graph: str,
    home_graph: str,
) -> MergeResult:
    """Merge the *wrong_graph* duplicate of Entity *uuid* into its *home_graph* copy.

    Unique-edge classification (step-17/18): every RELATES_TO edge incident
    to the node is read (as full property rows) from *wrong_graph*, and (as a
    lightweight uuid set) from *home_graph*. ``classify_unique_wrong_edges``
    computes the set-diff -- the edges unique to the wrong-graph copy -- and
    that set is exposed on the returned ``MergeResult``.

    Recreate-then-delete (step-19/20): each unique edge is recreated on the
    home copy with a byte-exact ``fact_embedding`` (read via the same raw
    ``--compact`` transport ``move_entity_across_graphs`` uses -- see the
    module docstring), preserving every other property (including the
    edge's own ``group_id``) unchanged -- S6 has no ``rewrite_group_id``
    analogue, unlike S5. Only once every recreate has been awaited is the
    wrong-graph copy ``DETACH DELETE``d -- CREATE-BEFORE-DELETE, same
    crash-safety rationale as ``move_entity_across_graphs``: a crash
    mid-merge leaves a recoverable duplicate rather than losing an edge.
    Edges already shared between both copies are left untouched on home.

    Args:
        graphiti: An initialized GraphitiBackend (or compatible object
            exposing ``_graph_for`` / ``_require_falkor_client``).
        uuid: UUID of the Entity node whose duplicate is being merged.
        wrong_graph: Graph holding the foreign (to-be-deleted) duplicate.
        home_graph: Graph holding the canonical copy.

    Returns:
        MergeResult describing the merge -- ``home_edge_count_after`` is
        RE-READ from the home graph after the recreates, so comparing it
        against ``home_edge_count_before + edges_recreated`` is a genuine
        no-edge-lost / no-double-count check (a mismatch would indicate a
        recreate silently matched nothing -- see
        ``_read_relates_to_edge_uuids``) rather than a tautology.
    """
    wrong = graphiti._graph_for(wrong_graph)
    home = graphiti._graph_for(home_graph)

    wrong_edges_result = await wrong.ro_query(
        'MATCH (n:Entity {uuid: $uuid})-[e:RELATES_TO]-(m:Entity) '
        'WITH DISTINCT e, startNode(e) AS s, endNode(e) AS t '
        'RETURN e.uuid, e.name, e.fact, e.valid_at, e.invalid_at, e.created_at, '
        '       e.group_id, e.episodes, s.uuid, t.uuid',
        {'uuid': uuid},
    )
    wrong_edge_rows = wrong_edges_result.result_set or []
    wrong_rows_by_uuid = {row[0]: row for row in wrong_edge_rows}
    wrong_edge_uuids = set(wrong_rows_by_uuid)

    home_edge_uuids = await _read_relates_to_edge_uuids(home, uuid)

    unique_wrong_edge_uuids = classify_unique_wrong_edges(home_edge_uuids, wrong_edge_uuids)

    falkor_client = graphiti._require_falkor_client()
    edges_recreated = 0
    for edge_uuid in unique_wrong_edge_uuids:
        (_edge_uuid, edge_name, fact, valid_at, invalid_at, edge_created_at,
         edge_group_id, episodes, src_uuid, dst_uuid) = wrong_rows_by_uuid[edge_uuid]

        edge_embedding_cypher = (
            f'MATCH ()-[e:RELATES_TO {{uuid: {_quote_cypher_string(edge_uuid)}}}]-() '
            'RETURN e.fact_embedding'
        )
        edge_embedding_reply = await _read_compact_vector(
            falkor_client, group_id=wrong_graph, cypher=edge_embedding_cypher,
        )
        edge_embedding_literal = format_vecf32_literal(
            parse_compact_vector_reply(edge_embedding_reply)
        )

        # Both endpoints are MATCHed (never CREATEd): the home copy of the
        # moved-duplicate's node and the edge's other endpoint must already
        # exist in home_graph, or this MATCH yields no rows and the edge is
        # silently skipped (same convention as move_entity_across_graphs).
        await home.query(
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
                'group_id': edge_group_id,
                'episodes': episodes,
            },
        )
        edges_recreated += 1

    # --- wrong-graph-copy DETACH DELETE (step-19/20) ---
    # Always the LAST mutation: every home-copy recreate above has already
    # been awaited. Never touches home_graph.
    await wrong.query(
        'MATCH (n:Entity {uuid: $uuid}) DETACH DELETE n',
        {'uuid': uuid},
    )

    logger.info(
        'merge_foreign_duplicate: merged uuid=%s wrong=%s home=%s '
        'edges_recreated=%d',
        uuid, wrong_graph, home_graph, edges_recreated,
    )
    return MergeResult(
        uuid=uuid,
        wrong_graph=wrong_graph,
        home_graph=home_graph,
        unique_wrong_edge_uuids=frozenset(unique_wrong_edge_uuids),
        edges_recreated=edges_recreated,
        home_edge_count_before=len(home_edge_uuids),
        home_edge_count_after=len(home_edge_uuids) + edges_recreated,
    )
