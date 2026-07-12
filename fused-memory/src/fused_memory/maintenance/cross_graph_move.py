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

# falkordb.query_result.ResultSetScalarTypes.VALUE_NULL -- the compact-reply
# scalar-type tag for a NULL/absent property (see _read_compact_vector).
_VALUE_NULL = 1


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


def _embedding_set_clause(assignment_target: str, reply: str | None) -> str:
    """Render an optional, comma-prefixed embedding SET fragment.

    This is the single seam that makes the embedding property optional at
    every recreate site (node ``name_embedding`` / edge ``fact_embedding``):
    when *reply* is ``None`` (a null/absent source embedding -- see
    ``_read_compact_vector``), returns ``''`` so the caller's CREATE omits
    the embedding property entirely, preserving every other property on the
    recreated node/edge. When *reply* is a real ``--compact`` vector reply,
    returns ``f', {assignment_target} = {vecf32([...])}'`` -- reusing the
    UNCHANGED byte-exact ``format_vecf32_literal(parse_compact_vector_reply(
    ...))`` passthrough, so the non-null path stays byte-identical to before
    this helper existed.
    """
    if reply is None:
        return ''
    return f', {assignment_target} = {format_vecf32_literal(parse_compact_vector_reply(reply))}'


def _quote_cypher_string(value: str) -> str:
    """Inline *value* as a single-quoted Cypher string literal, escaping ``\\``/``'``.

    Used only for uuids (program-generated, low-risk) in the hand-rolled
    Cypher issued through the raw ``--compact`` transport, which bypasses the
    normal ``graph.query(cypher, params)`` parameter-binding path (see
    ``_read_compact_vector``).
    """
    escaped = value.replace('\\', '\\\\').replace("'", "\\'")
    return f"'{escaped}'"


async def _read_compact_vector(
    falkor_client: Any, *, group_id: str, cypher: str,
) -> str | None:
    """Injectable raw ``--compact`` transport seam for a single vecf32 column.

    Issues *cypher* (expected to ``RETURN`` exactly one vecf32-typed value)
    against *group_id* directly through the raw FalkorDB client (bypassing
    ``falkordb.Graph.query()``/``.ro_query()``, whose ``QueryResult.parse()``
    -- specifically ``__parse_vectorf32``/``__parse_double`` -- calls
    ``float()`` on each component and so silently adopts FalkorDB's lossy
    textual double representation; this is the RCA-validated truncation this
    module exists to avoid). Returns the raw bracketed, comma-separated
    vector text (the same shape ``parse_compact_vector_reply`` consumes),
    extracted from the reply WITHOUT ever calling ``float()`` -- or ``None``
    when the source property is null/absent (see below).

    This default implementation reads the ``falkordb`` package's own
    (already-parsed-elsewhere) reply-shape convention
    (``falkordb.query_result.parse_scalar``: ``[scalar_type, value]`` cells,
    scalar_type 12 == VECTORF32). ``TestReadCompactVectorTransport`` (in the
    test suite) exercises this parsing directly against a recorded/
    representative raw reply structure -- every OTHER test in this module
    monkeypatches this function away. Byte-fidelity against a REAL FalkorDB
    remains validated separately, in the η live throwaway-graph rehearsal
    (see ``plans/cross-graph-entity-leak-prd.md`` decision 5).

    A null/absent embedding (e.g. an Entity/RELATES_TO row that was
    persisted without its embedding property -- confirmed live: ~10% of
    RELATES_TO edges) surfaces as a ``scalar_type`` of ``_VALUE_NULL`` (1).
    This is now recognized as valid real data, not corruption: this function
    returns ``None`` for exactly that tag, letting callers recreate the
    node/edge embedding-less instead of failing. Any OTHER non-VECTORF32
    ``scalar_type`` (a genuinely wrong-typed column -- e.g. VALUE_ARRAY) is
    still a hard, descriptively-messaged failure (naming *group_id* and
    *cypher*) that surfaces a corrupt/unexpected source row immediately,
    rather than silently proceeding without it (which would produce a
    differently-shaped, harder-to-diagnose gap later -- e.g. a moved node
    that can no longer be found by semantic search). A reply with zero rows
    (e.g. a transient race where the source row was deleted between an
    earlier existence check and this read, or an unexpected reply shape)
    raises the same kind of descriptive ``ValueError`` rather than a bare
    ``IndexError``.

    CAVEAT (reviewer follow-up, task 2451 amendment): the ``scalar_type ==
    _VALUE_NULL`` (1) tag match immediately below is exercised only by a
    hand-built mock reply (``TestReadCompactVectorTransport::
    test_null_absent_embedding_returns_none_sentinel``'s
    ``_compact_reply([_VALUE_NULL_TAG, None])``). The live incident that
    motivated this fix (edge f0fc1aba in reify, 2026-07-11) confirms that
    null/absent embeddings EXIST at ~10% prevalence in production -- it
    does not, by itself, confirm this exact compact-reply cell shape for a
    null property against a real FalkorDB. If the live shape ever differs
    (a different tag or a wrapped cell), this function falls through to the
    ValueError branch below instead -- safe (matches the pre-fix behaviour,
    no data corruption) but silently non-fixing for the ~10% scenario this
    task targets, with every unit test still green. The η live
    throwaway-graph rehearsal (``plans/cross-graph-entity-leak-prd.md``
    decision 5) MUST be extended with a probe case -- seed a
    ``_probe``-prefixed edge with a persisted-null ``fact_embedding`` and
    assert this function returns ``None`` against that real row -- before
    this null-tolerance path is trusted at full-census scale. That PRD/
    runbook update is outside ``fused_memory/maintenance``'s locked scope
    for this amendment pass and is tracked as follow-up, not done here.
    """
    reply = await falkor_client.execute_command('GRAPH.RO_QUERY', group_id, cypher, '--compact')
    rows = reply[1]
    if not rows:
        raise ValueError(
            f'_read_compact_vector: expected exactly one row for '
            f'group_id={group_id!r} (cypher={cypher!r}), got zero rows -- '
            'the source row may have been deleted between an existence '
            'check and this read, or the reply shape is unexpected.'
        )
    cell = rows[0][0]
    scalar_type, value = int(cell[0]), cell[1]
    if scalar_type == _VALUE_NULL:
        # Mock-verified only -- see this function's "CAVEAT" docstring
        # paragraph for why the η live rehearsal must confirm this tag
        # against a real FalkorDB null property before full-census trust.
        return None
    if scalar_type != _VALUE_VECTORF32:
        raise ValueError(
            f'_read_compact_vector: expected a VECTORF32 (12) cell for '
            f'group_id={group_id!r}, got scalar_type={scalar_type} '
            f'(cypher={cypher!r}). A null/absent embedding on the source row '
            f'surfaces as scalar_type={_VALUE_NULL} (VALUE_NULL) and is '
            "returned as None instead -- see this function's docstring."
        )
    tokens = [v.decode() if isinstance(v, bytes) else str(v) for v in value]
    return '[' + ', '.join(tokens) + ']'


class ForeignDuplicateSuspectedError(Exception):
    """Raised when a uuid present in both graphs looks like a genuine duplicate.

    ``move_entity_across_graphs``'s target-presence probe (see its docstring's
    "Idempotency" section) treats any uuid already present in *both*
    source_graph and target_graph as a partially-completed move ("resume"):
    a prior run crashed strictly between the node CREATE and the source
    DETACH DELETE. But this problem domain is cross-graph duplication, so the
    uuid may instead legitimately name a genuine, DIVERGENT duplicate in
    target_graph -- exactly the scenario ``merge_foreign_duplicate`` (S6)
    exists to handle -- rather than a resumable remnant of the SAME node.

    The cheap divergence guard (name + created_at mismatch between the
    source and target copies) raises this error instead of silently treating
    a genuine duplicate as a resume, which would otherwise overwrite the
    target's relationship topology and delete the source copy with no
    signal. Callers hitting this should route the uuid through
    ``merge_foreign_duplicate`` instead of ``move_entity_across_graphs``.
    """


@dataclass
class _NodeCreateOutcome:
    """Internal result of ``_create_entity_in_target`` -- shared by
    ``create_moved_node`` and ``move_entity_across_graphs``.

    Attributes:
        already_moved: True ONLY for the fully-idempotent short-circuit case
            (absent from source, present in target) -- a caller processing
            edges/mentions/delete (``move_entity_across_graphs``) MUST do NO
            further work at all when this is True; the whole move already
            happened.
        created: True when this call issued a fresh node CREATE in target.
            False covers both ``already_moved`` and the matching-resume
            case (present in both graphs, name/created_at match -- CREATE
            is skipped because the node is already there, but a caller
            processing edges/mentions/delete still proceeds to do so).
        node_uuid: The node's own uuid, as read from source (or the probed
            uuid when ``already_moved`` short-circuited before any source
            row was read).
        new_group_id: ``rewrite_group_id`` if given, else the source node's
            own group_id -- None when ``already_moved`` (no further group_id
            use is possible; the caller returns immediately).
        source: The resolved source-graph handle (``graphiti._graph_for``),
            so callers continuing past the node-create step don't need to
            re-resolve it.
        target: The resolved target-graph handle, likewise.
        falkor_client: The raw transport client (``graphiti._require_falkor_client()``),
            likewise -- None when ``already_moved``.
    """

    already_moved: bool
    created: bool
    node_uuid: str | None
    new_group_id: str | None
    source: Any
    target: Any
    falkor_client: Any


async def _create_entity_in_target(
    graphiti: Any,
    uuid: str,
    source_graph: str,
    target_graph: str,
    *,
    rewrite_group_id: str | None = None,
) -> _NodeCreateOutcome:
    """Node-core read+CREATE shared by ``create_moved_node`` (Phase A) and
    ``move_entity_across_graphs`` (S5, still a single-call primitive).

    Reads the node's scalar props from source (plain ``ro_query``) and runs
    the unconditional target-presence probe, then either: short-circuits as
    already-fully-moved (absent from source, present in target); raises
    ``NodeNotFoundError`` (absent from both); raises
    ``ForeignDuplicateSuspectedError`` (present in both, diverging
    name/created_at); skips the CREATE as a genuine resume (present in both,
    matching); or reads the exact ``name_embedding`` via the raw
    ``--compact`` transport and CREATEs the node in target. See
    ``move_entity_across_graphs``'s docstring "Idempotency" section for the
    full case analysis -- this helper implements exactly that analysis, up
    to (but not including) edges/mentions/delete, which remain the caller's
    responsibility.
    """
    source = graphiti._graph_for(source_graph)
    target = graphiti._graph_for(target_graph)

    node_result = await source.ro_query(
        'MATCH (n:Entity {uuid: $uuid}) '
        'RETURN n.uuid, n.name, n.group_id, n.summary, n.created_at',
        {'uuid': uuid},
    )

    # --- unconditional target-presence probe ---
    # Always read target presence before any mutation, regardless of
    # whether the node is still present in source. This one probe backs
    # BOTH the fully-idempotent no-op (absent from source, present in
    # target) AND detecting a crash that happened strictly between the node
    # CREATE and the source DETACH DELETE (present in BOTH graphs).
    target_probe = await target.ro_query(
        'MATCH (n:Entity {uuid: $uuid}) RETURN n.uuid, n.name, n.created_at',
        {'uuid': uuid},
    )
    node_already_in_target = bool(target_probe.result_set)

    if not node_result.result_set:
        if node_already_in_target:
            logger.info(
                '_create_entity_in_target: already moved, no-op uuid=%s '
                'source=%s target=%s',
                uuid, source_graph, target_graph,
            )
            return _NodeCreateOutcome(
                already_moved=True,
                created=False,
                node_uuid=uuid,
                new_group_id=None,
                source=source,
                target=target,
                falkor_client=None,
            )
        raise NodeNotFoundError(f'Entity node not found in source_graph {source_graph!r}: {uuid}')
    node_uuid, name, group_id, summary, created_at = node_result.result_set[0]

    falkor_client = graphiti._require_falkor_client()
    new_group_id = group_id if rewrite_group_id is None else rewrite_group_id

    if node_already_in_target:
        # Present in BOTH graphs: distinguish a genuine resume (this SAME
        # node, recreated by a prior run that crashed strictly between the
        # node CREATE and the source DETACH DELETE) from a genuine foreign
        # duplicate (a DIFFERENT, divergent Entity that happens to share
        # this uuid in target_graph -- merge_foreign_duplicate's territory,
        # not this function's). This is a cheap heuristic (name +
        # created_at), not a full deep-equality check.
        _target_uuid, target_name, target_created_at = target_probe.result_set[0]
        if target_name != name or target_created_at != created_at:
            raise ForeignDuplicateSuspectedError(
                f'_create_entity_in_target: uuid={uuid} exists in target_graph '
                f'{target_graph!r} with a diverging name/created_at from the '
                f'source_graph {source_graph!r} copy (target name={target_name!r} '
                f'created_at={target_created_at!r}; source name={name!r} '
                f'created_at={created_at!r}) -- this looks like a genuine '
                'cross-graph duplicate, not a partially-completed move. Route '
                'it through merge_foreign_duplicate instead.'
            )
        # Resuming after a crash strictly between the node CREATE and the
        # source DETACH DELETE: skip re-CREATE-ing the node (it's already
        # there) and let the caller fall straight through to (re-)attempting
        # edges/mentions + delete.
        logger.warning(
            '_create_entity_in_target: resuming partially-completed move '
            '(node already present in target, skipping node CREATE) uuid=%s '
            'source=%s target=%s',
            uuid, source_graph, target_graph,
        )
        return _NodeCreateOutcome(
            already_moved=False,
            created=False,
            node_uuid=node_uuid,
            new_group_id=new_group_id,
            source=source,
            target=target,
            falkor_client=falkor_client,
        )

    embedding_cypher = (
        f'MATCH (n:Entity {{uuid: {_quote_cypher_string(uuid)}}}) RETURN n.name_embedding'
    )
    embedding_reply = await _read_compact_vector(
        falkor_client, group_id=source_graph, cypher=embedding_cypher,
    )

    await target.query(
        'CREATE (n:Entity {uuid: $uuid}) '
        'SET n.name = $name, '
        '    n.group_id = $group_id, '
        '    n.summary = $summary, '
        '    n.created_at = $created_at'
        f'{_embedding_set_clause("n.name_embedding", embedding_reply)}',
        {
            'uuid': node_uuid,
            'name': name,
            'group_id': new_group_id,
            'summary': summary,
            'created_at': created_at,
        },
    )
    return _NodeCreateOutcome(
        already_moved=False,
        created=True,
        node_uuid=node_uuid,
        new_group_id=new_group_id,
        source=source,
        target=target,
        falkor_client=falkor_client,
    )


@dataclass
class CreateResult:
    """Result of a ``create_moved_node`` call (Phase A: node-only create).

    Attributes:
        uuid: UUID of the created (or already-present) Entity node.
        source_graph: Graph the node is being moved from -- read-only in
            this phase; ``create_moved_node`` never mutates source_graph.
        target_graph: Graph the node was (or already is) created in.
        already_created: True when the node was already present in
            target_graph, so no CREATE was (re-)issued -- covers BOTH the
            fully-idempotent no-op (absent from source, present in target)
            and the matching-resume case (present in both, name/created_at
            match). False when this call issued a fresh node CREATE.
    """

    uuid: str
    source_graph: str
    target_graph: str
    already_created: bool = False


async def create_moved_node(
    graphiti: Any,
    uuid: str,
    source_graph: str,
    target_graph: str,
    *,
    rewrite_group_id: str | None = None,
) -> CreateResult:
    """Phase A of the three-phase barrier-ordered apply (CGL-η follow-up,
    task 2415): CREATE *uuid*'s Entity node in *target_graph* -- and ONLY
    the node.

    Issues NO RELATES_TO/MENTIONS recreate and NO source DETACH DELETE --
    those are ``recreate_subgraph_relationships`` (Phase B) and
    ``delete_source_node`` (Phase C) respectively. Running Phase A for
    every MOVE node in a batch BEFORE any Phase B/C call is what guarantees
    both endpoints of a co-moving RELATES_TO edge are present in their
    targets before that edge is recreated -- the barrier ordering that
    closes the edge-loss bug in the old single-call recreate-then-delete
    loop (see ``move_entity_across_graphs``'s "Residual hazard" note and
    this module's ``run()`` caller in ``scripts/migrate_cross_graph_leak.py``).

    Delegates the full read+probe+guard+create case analysis to
    ``_create_entity_in_target`` (shared with ``move_entity_across_graphs``,
    whose external behavior/Cypher/call-sequence is unchanged by this
    delegation) and reports the outcome as a ``CreateResult``.

    Args:
        graphiti: An initialized GraphitiBackend (or compatible object
            exposing ``_graph_for`` / ``_require_falkor_client``).
        uuid: UUID of the Entity node to create in target_graph.
        source_graph: Graph the node currently lives in (read-only).
        target_graph: Graph to CREATE the node into.
        rewrite_group_id: When given, substituted for the node's
            ``group_id`` property on recreate; when None, the source node's
            own group_id is carried through unchanged.

    Returns:
        CreateResult describing the create (or no-op).

    Raises:
        NodeNotFoundError: if no Entity node with *uuid* exists in
            *source_graph* (and it is also absent from *target_graph*).
        ForeignDuplicateSuspectedError: if *uuid* is present in BOTH graphs
            but target's name/created_at diverge from source's -- callers
            hitting this should route *uuid* through ``merge_foreign_duplicate``
            instead.
    """
    outcome = await _create_entity_in_target(
        graphiti, uuid, source_graph, target_graph, rewrite_group_id=rewrite_group_id,
    )
    return CreateResult(
        uuid=uuid,
        source_graph=source_graph,
        target_graph=target_graph,
        already_created=not outcome.created,
    )


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
        edges_moved: Count of RELATES_TO edges actually reattached to the
            moved node -- verified via the target CREATE's own
            ``relationships_created`` stat, NOT merely attempted. An edge
            whose CREATE silently matched no endpoint in target_graph is
            excluded from this count (see edges_skipped).
        edges_skipped: Count of RELATES_TO edges whose target CREATE
            matched no endpoint (the edge's other endpoint not yet present
            in target_graph) and was therefore silently skipped by
            FalkorDB. Exposed so callers can detect the drop from the
            returned MoveResult alone, without independently re-querying
            either graph.
        mentions_moved: Count of Episodic MENTIONS links actually
            reattached (same relationships_created verification as
            edges_moved).
        mentions_skipped: Count of MENTIONS links silently skipped (same
            cause as edges_skipped -- the episode node not yet present in
            target_graph).
    """

    uuid: str
    source_graph: str
    target_graph: str
    already_moved: bool = False
    edges_moved: int = 0
    edges_skipped: int = 0
    mentions_moved: int = 0
    mentions_skipped: int = 0


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
    with a byte-exact ``fact_embedding``, provided the OTHER endpoint
    already exists in target_graph. FalkorDB silently matches nothing (and
    creates nothing) when that endpoint is absent -- this is detected via
    the CREATE's own ``relationships_created`` stat, so it is counted in
    ``MoveResult.edges_skipped`` rather than being miscounted as moved.

    Mentions (step-9/10, amended): every Episodic MENTIONS link onto the
    node is recreated in target_graph, provided the episode node already
    exists there (same silent-skip detection as edges, via
    ``relationships_created``; MENTIONS carries no embedding). Skips are
    counted in ``MoveResult.mentions_skipped``.

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
    - Present in BOTH source and target: EITHER a prior run crashed
      strictly BETWEEN the node CREATE and the source DETACH DELETE
      (genuine resume), OR target_graph happens to hold a genuine,
      divergent duplicate of this uuid (this problem domain is cross-graph
      duplication, so that is not a remote possibility -- see
      ``merge_foreign_duplicate``). A cheap divergence guard (amended)
      distinguishes the two: target's ``name``/``created_at`` are compared
      against source's. On a match (genuine resume), the node CREATE is
      skipped (target already has it -- re-issuing CREATE here would
      duplicate it, since multi-tenant FalkorDB builds no uniqueness
      constraint on Entity.uuid), and the call proceeds straight to
      (re-)attempting the edge/mention reattachment and the source delete,
      so a retry converges on a single target copy instead of a duplicate.
      On a mismatch, ``ForeignDuplicateSuspectedError`` is raised instead --
      before any edge/mention read or mutation on either graph -- rather
      than silently overwriting the target's divergent topology and
      deleting the source copy.

    Residual hazard (NOT closed by the above): a resumed move does not
    itself deduplicate RELATES_TO/MENTIONS rows that were already recreated
    in target during the crashed attempt -- only the Entity node recreate
    is guarded this way; edges/mentions have no MERGE-keyed or presence-
    checked-against-target write of their own uuid. A retry after a crash
    that got partway through the edges/mentions loops may therefore
    recreate (and thus duplicate) rows it already created before crashing.
    Fully closing this would need per-edge/mention idempotent writes, which
    is out of scope here; callers needing exact-once edge/mention semantics
    across crash-recovery should independently verify post-move counts
    (distinct from the silent-endpoint-missing skips that edges_skipped/
    mentions_skipped already surface directly on the returned MoveResult).

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
        ForeignDuplicateSuspectedError: if *uuid* is present in BOTH graphs
            but target's name/created_at diverge from source's -- see the
            "Idempotency" section above. Callers hitting this should route
            *uuid* through ``merge_foreign_duplicate`` instead.
    """
    outcome = await _create_entity_in_target(
        graphiti, uuid, source_graph, target_graph, rewrite_group_id=rewrite_group_id,
    )
    if outcome.already_moved:
        return MoveResult(
            uuid=uuid,
            source_graph=source_graph,
            target_graph=target_graph,
            already_moved=True,
        )

    source = outcome.source
    target = outcome.target
    node_uuid = outcome.node_uuid
    new_group_id = outcome.new_group_id
    falkor_client = outcome.falkor_client

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
    edges_skipped = 0
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

        # Both endpoints are MATCHed (never CREATEd) by uuid: the other
        # endpoint (dst_uuid) must already exist in target_graph for the
        # edge to be recreated -- otherwise this MATCH yields no rows and
        # the edge is silently skipped (left for the caller to move the
        # other endpoint too, or accept the drop; see the docstring's
        # "Residual hazard" note). relationships_created (FalkorDB's own
        # CREATE stat) distinguishes the two outcomes -- edges_moved only
        # counts a genuine create, never a silent no-op MATCH.
        edge_create_result = await target.query(
            'MATCH (a:Entity {uuid: $src_uuid}), (b:Entity {uuid: $dst_uuid}) '
            'CREATE (a)-[r:RELATES_TO]->(b) '
            'SET r.uuid = $edge_uuid, '
            '    r.name = $name, '
            '    r.fact = $fact, '
            '    r.valid_at = $valid_at, '
            '    r.invalid_at = $invalid_at, '
            '    r.created_at = $created_at, '
            '    r.group_id = $group_id, '
            '    r.episodes = $episodes'
            f'{_embedding_set_clause("r.fact_embedding", edge_embedding_reply)}',
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
        if edge_create_result.relationships_created:
            edges_moved += 1
        else:
            edges_skipped += 1
            logger.warning(
                'move_entity_across_graphs: RELATES_TO edge uuid=%s silently '
                'skipped -- other endpoint uuid=%s not present in '
                'target_graph=%s (uuid=%s source=%s)',
                edge_uuid, dst_uuid, target_graph, uuid, source_graph,
            )

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
    mentions_skipped = 0
    for row in mentions_result.result_set or []:
        mention_uuid, _mention_group_id, mention_created_at, episode_uuid = row
        # As with RELATES_TO's other endpoint: the Episodic node is not moved
        # by this primitive and must already exist in target_graph, or this
        # MATCH yields no rows and the mention is silently skipped --
        # relationships_created distinguishes the two outcomes, same as the
        # RELATES_TO loop above.
        mention_create_result = await target.query(
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
        if mention_create_result.relationships_created:
            mentions_moved += 1
        else:
            mentions_skipped += 1
            logger.warning(
                'move_entity_across_graphs: MENTIONS link uuid=%s silently '
                'skipped -- episode uuid=%s not present in target_graph=%s '
                '(uuid=%s source=%s)',
                mention_uuid, episode_uuid, target_graph, uuid, source_graph,
            )

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
        'edges_moved=%d edges_skipped=%d mentions_moved=%d mentions_skipped=%d',
        uuid, source_graph, target_graph, edges_moved, edges_skipped,
        mentions_moved, mentions_skipped,
    )
    return MoveResult(
        uuid=uuid,
        source_graph=source_graph,
        target_graph=target_graph,
        edges_moved=edges_moved,
        edges_skipped=edges_skipped,
        mentions_moved=mentions_moved,
        mentions_skipped=mentions_skipped,
    )


@dataclass
class SubgraphEdgeResult:
    """Result of a ``recreate_subgraph_relationships`` call (Phase B: batch
    edge/mention recreate of the three-phase barrier-ordered apply).

    Attributes:
        edges_recreated: Count of RELATES_TO edges actually recreated across
            the WHOLE batch -- verified via the target CREATE's own
            ``relationships_created`` stat, same convention as
            ``MoveResult.edges_moved``. Each distinct edge (keyed by edge
            uuid) is counted at most once, however many specs it is incident
            to (see the module's co-moving-pair fix).
        edges_skipped: Count of RELATES_TO edges whose target CREATE
            matched no endpoint (the edge's other endpoint not present in
            the resolved target graph) and was therefore silently skipped by
            FalkorDB -- same meaning as ``MoveResult.edges_skipped``. Does
            NOT include an edge that was skipped because it was already
            present in target (a genuine idempotent no-op, not a loss).
        mentions_recreated: Count of Episodic MENTIONS links actually
            recreated (same ``relationships_created`` verification).
        mentions_skipped: Count of MENTIONS links silently skipped because
            the episode node was not present in the resolved target graph.
        dropped_cross_target: Records for edges whose two endpoints resolve
            to two DIFFERENT target graphs (FalkorDB RELATES_TO edges are
            single-graph, so such an edge cannot be recreated in EITHER
            graph) -- each a dict carrying the edge uuid, both endpoint
            uuids, both resolved targets, and a human-readable reason.
            Reported for human review, never silently lost. Populated by
            the cross-target detection extension (task 2415 step-8) --
            always empty for a batch with no such edges.
        blocked: Records for individual edges/mentions whose embedding read
            or CREATE raised (e.g. a transient FalkorDB error) -- each a
            dict with ``kind`` (``'edge'`` or ``'mention'``), the item's own
            ``uuid``, a human-readable ``reason`` (``str(exc)``), and
            ``node_uuids`` (the incident node uuid(s): both endpoints for a
            RELATES_TO edge, the entity uuid for a MENTIONS link). Per-item
            isolation (CGL-η follow-up, task 2451) means a single bad
            edge/mention costs only itself -- the batch continues and this
            item is surfaced here for human review, same "never silently
            lost" convention as ``dropped_cross_target``, rather than
            aborting the whole batch the way it used to (see this
            function's docstring). A caller MUST withhold Phase C
            source-deletion for every uuid named here, or the un-recreated
            edge/mention -- which still exists only in source -- would be
            destroyed (see ``scripts/migrate_cross_graph_leak.py``'s
            ``run()``). Always empty for a batch with no such failures.
        embedding_omitted: Count of RELATES_TO edges counted in
            ``edges_recreated`` (a subset of it, never mentions -- MENTIONS
            links carry no embedding property) whose source
            ``fact_embedding`` read null/absent, so the recreated edge
            landed WITHOUT that property (see ``_embedding_set_clause`` /
            ``_read_compact_vector``'s null-tolerance, CGL-η follow-up
            reviewer amendment, task 2451). An embedding-less RELATES_TO
            edge is invisible to vector/semantic search, so this is a
            quality signal surfaced for operator visibility -- purely
            informational, it does NOT gate ``blocked``/exit_code the way
            ``dropped_cross_target``/``blocked`` do, since a null embedding
            is valid data, not a failure. Always 0 for a batch where every
            recreated edge carried a real embedding.
    """

    edges_recreated: int = 0
    edges_skipped: int = 0
    mentions_recreated: int = 0
    mentions_skipped: int = 0
    dropped_cross_target: list = field(default_factory=list)
    blocked: list = field(default_factory=list)
    embedding_omitted: int = 0


async def _entity_present_in_graph(graph: Any, uuid: str) -> bool:
    """Read-only presence probe: True iff an ``:Entity`` node with *uuid*
    exists in *graph*.

    Used by ``recreate_subgraph_relationships`` to decide whether a
    RELATES_TO edge's non-migrating endpoint (one with no spec of its own in
    this batch, so it has no ``target_of`` entry) is deliverable into the
    OTHER, migrating endpoint's resolved target -- i.e. whether it is
    already a home-resident there. Scoped to ``:Entity`` (unlike
    ``scripts/migrate_cross_graph_leak.py``'s broader, any-label
    ``node_present_in_graph``) because RELATES_TO only ever connects
    Entity<->Entity, matching this module's Entity-only scope.
    """
    result = await graph.ro_query(
        'MATCH (n:Entity {uuid: $uuid}) RETURN n.uuid LIMIT 1',
        {'uuid': uuid},
    )
    return bool(result.result_set)


async def _relates_to_edge_already_in_target(graph: Any, edge_uuid: str) -> bool:
    """Read-only presence probe: True iff a RELATES_TO edge with *edge_uuid*
    already exists in *graph* (either direction).

    Unlike an Entity node's uuid (guarded by ``_create_entity_in_target``'s
    unconditional target-presence probe), FalkorDB enforces no uniqueness
    constraint on a relationship's ``uuid`` property -- a blind re-``CREATE``
    on a crash-resumed or fully-idempotent re-run of
    ``recreate_subgraph_relationships`` would duplicate an edge it already
    recreated in a prior pass, rather than silently no-op the way the node
    CREATE does. This probe is what lets a re-run skip an edge instead.
    """
    result = await graph.ro_query(
        'MATCH ()-[e:RELATES_TO {uuid: $uuid}]-() RETURN e.uuid LIMIT 1',
        {'uuid': edge_uuid},
    )
    return bool(result.result_set)


async def _mentions_link_already_in_target(graph: Any, mention_uuid: str) -> bool:
    """Read-only presence probe: True iff a MENTIONS link with *mention_uuid*
    already exists in *graph*.

    Mirrors ``_relates_to_edge_already_in_target``: FalkorDB enforces no
    uniqueness constraint on a relationship's ``uuid`` property either, so a
    blind re-``CREATE`` on a crash-resumed or fully-idempotent re-run of
    ``recreate_subgraph_relationships`` would duplicate a MENTIONS link it
    already recreated in a prior pass -- the exact same hazard already
    guarded for RELATES_TO. This probe is what lets a re-run skip a MENTIONS
    link instead.
    """
    result = await graph.ro_query(
        'MATCH ()-[e:MENTIONS]->() WHERE e.uuid = $uuid RETURN e.uuid LIMIT 1',
        {'uuid': mention_uuid},
    )
    return bool(result.result_set)


async def recreate_subgraph_relationships(graphiti: Any, specs: list[dict]) -> SubgraphEdgeResult:
    """Phase B of the three-phase barrier-ordered apply (CGL-η follow-up,
    task 2415): recreate every RELATES_TO edge and Episodic MENTIONS link
    incident to *specs*, deduped by uuid across the WHOLE batch.

    *specs* is a list of manifest-node-shaped dicts (``uuid``,
    ``disposition``, ``source_graph``, ``target_graph``) -- the same shape
    ``run()``'s apply loop partitions manifest nodes into (move_specs +
    merge_specs, task 2415 step-12). Called ONLY after ``create_moved_node``
    (Phase A) has run for every MOVE spec in the batch, so every migrating
    node already exists in its target graph before any edge is read here --
    this is what lets a co-moving neighbour's shared RELATES_TO edge survive
    (the old single-call ``move_entity_across_graphs`` loop would have
    DETACH DELETEd the source before the second endpoint was ever
    processed; see this module's "Residual hazard" note and
    ``scripts/migrate_cross_graph_leak.py``'s ``run()``).

    MOVE edges: for every non-MERGE spec, every incident RELATES_TO edge
    (full property row, read via the same ``startNode``/``endNode`` SELECT
    ``move_entity_across_graphs`` uses) and every incident Episodic MENTIONS
    link is read from the spec's ``source_graph``. Edges are accumulated
    into a single dict keyed by edge uuid -- so a co-moving edge incident to
    TWO specs in this batch (read once per incident spec) is recreated only
    ONCE, not twice. For each distinct edge, both endpoints' target graphs
    are looked up via a ``target_of`` map built from every spec's own
    ``uuid``/``target_graph``; when an endpoint has no spec of its own (a
    non-migrating home-resident), its deliverability is decided by a
    presence probe (``_entity_present_in_graph``) against the OTHER,
    migrating endpoint's resolved target. When both endpoints resolve to the
    SAME target graph T, the edge is recreated in T (byte-exact
    ``fact_embedding`` via the raw ``--compact`` transport, ``group_id``
    rewritten to T -- mirroring ``create_moved_node``'s
    ``rewrite_group_id=target_graph`` convention for Phase A), skipped as a
    genuine idempotent no-op if an edge with that uuid already exists in T
    (``_relates_to_edge_already_in_target``), or counted in
    ``edges_skipped`` if the target CREATE's own ``relationships_created``
    stat reports the other endpoint was silently unmatched. When the two
    endpoints do NOT resolve to one shared target graph (differing targets,
    or an endpoint that is neither migrating nor already present in the
    other's target) -- cross-graph RELATES_TO edges are unsupported -- the
    edge is recreated in NEITHER graph and a dropped-with-reason record
    (edge uuid, both endpoint uuids, both resolved targets, a reason) is
    appended to ``dropped_cross_target`` instead, so it is surfaced for
    human review rather than silently lost the way a bare ``edges_skipped``
    count would be. MENTIONS are recreated the same way
    ``move_entity_across_graphs`` does (MATCH the episode + entity in the
    entity's resolved target, ``relationships_created`` distinguishes
    recreate from silent skip), skipped as a genuine idempotent no-op if
    already present in target (``_mentions_link_already_in_target`` -- the
    same re-run-safety guard as RELATES_TO's
    ``_relates_to_edge_already_in_target``, since FalkorDB enforces no
    uniqueness constraint on a MENTIONS relationship's uuid either).

    MERGE fold: run AFTER the MOVE-edge/mentions passes above have fully
    completed for the batch. For every MERGE spec, every RELATES_TO edge
    incident to its wrong-copy (``source_graph``) is read (full property
    rows, same SELECT as above) and compared -- via
    ``classify_unique_wrong_edges`` over uuid sets -- against the home
    copy's (``target_graph``) OWN current incident edge-uuid set (``_read_
    relates_to_edge_uuids``, S6's uuid-only reader). Only the edges unique
    to the wrong copy (absent from home) are recreated on the home copy,
    preserving the wrong copy's OWN ``group_id`` verbatim (no rewrite -- S6
    has no ``rewrite_group_id`` analogue, mirroring
    ``merge_foreign_duplicate``); an edge already shared with home is left
    untouched. Because this pass re-reads home's edge-uuid set fresh for
    EACH merge spec, an edge the MOVE-edge pass (or an earlier merge spec in
    this same batch) already landed on that home graph is correctly
    excluded here too -- no double-create for a MOVE<->MERGE or
    MERGE<->MERGE shared edge. MERGE has no MENTIONS-fold analogue (mirrors
    ``merge_foreign_duplicate``, which never touches MENTIONS either).

    Issues NO ``DETACH DELETE`` anywhere -- that is ``delete_source_node``'s
    job (Phase C), which callers must run only after this function has
    completed for the WHOLE batch (every edge recreated before any source is
    deleted is the barrier ordering that closes the edge-loss bug).

    Args:
        graphiti: An initialized GraphitiBackend (or compatible object
            exposing ``_graph_for`` / ``_require_falkor_client``).
        specs: Manifest-node-shaped dicts for every MOVE/MERGE node in this
            apply batch (``uuid``, ``disposition``, ``source_graph``,
            ``target_graph``).

    Returns:
        SubgraphEdgeResult tallying edges/mentions recreated vs. skipped
        (plus, from step-8 onward, any cross-target-dropped edges, and from
        the CGL-η follow-up onward, any per-item ``blocked`` failures).

    Raises:
        A single edge/mention's embedding read or CREATE raising (e.g. a
        transient FalkorDB error) is per-item isolated (CGL-η follow-up,
        task 2451): it does NOT raise out of this function -- it is instead
        recorded on the returned result's ``blocked`` list and the batch
        continues, so one bad item costs one item, not the whole census.
        What DOES still raise out of this function is a systemic/batch-level
        failure -- a per-spec gather ``ro_query`` (reading a spec's incident
        edges/mentions, or a MERGE spec's wrong-copy/home edge sets) or a
        lost falkor-client acquisition -- since that is not a single item's
        problem. This primitive mutates target graphs incrementally as it
        walks the batch, so a mid-batch systemic failure can still leave
        real, non-zero edges/mentions counts (and blocked entries) recorded
        before the raise -- that partial tally is attached to the exception
        as ``exc.partial_result`` (a ``SubgraphEdgeResult``) instead of
        being discarded, so a caller (e.g.
        ``scripts/migrate_cross_graph_leak.py``'s ``run()``) can surface
        accurate partial-progress counts instead of reporting an all-zero
        default for a batch that was actually partway mutated (task 2415
        amendment round 2).
    """
    result = SubgraphEdgeResult()
    try:
        await _recreate_subgraph_relationships_batch(graphiti, specs, result)
    except Exception as exc:
        exc.partial_result = result  # type: ignore[attr-defined]
        raise
    return result


async def _recreate_subgraph_relationships_batch(
    graphiti: Any, specs: list[dict], result: SubgraphEdgeResult,
) -> None:
    """Phase-B batch work for ``recreate_subgraph_relationships``, mutating
    *result* in place as it recreates edges/mentions.

    Split out of the public wrapper so it can hold a reference to the SAME
    ``SubgraphEdgeResult`` it returns on success and attach it to
    ``exc.partial_result`` on failure -- see
    ``recreate_subgraph_relationships``'s docstring.
    """
    falkor_client = graphiti._require_falkor_client()
    target_of: dict[str, str] = {spec['uuid']: spec['target_graph'] for spec in specs}

    # MERGE specs' edges are folded separately (step-10) -- only MOVE specs'
    # incident edges/mentions are read here. Compared against the literal
    # 'MERGE' string (not scripts/migrate_cross_graph_leak.py's MERGE
    # constant) -- this module must not import from its ζ-script consumer.
    move_specs = [spec for spec in specs if spec.get('disposition') != 'MERGE']

    edges_by_uuid: dict[str, tuple[list, str]] = {}
    mentions_by_uuid: dict[str, tuple[list, str, str]] = {}
    for spec in move_specs:
        source = graphiti._graph_for(spec['source_graph'])

        edges_result = await source.ro_query(
            'MATCH (n:Entity {uuid: $uuid})-[e:RELATES_TO]-(m:Entity) '
            'WITH DISTINCT e, startNode(e) AS s, endNode(e) AS t '
            'RETURN e.uuid, e.name, e.fact, e.valid_at, e.invalid_at, e.created_at, '
            '       e.group_id, e.episodes, s.uuid, t.uuid',
            {'uuid': spec['uuid']},
        )
        for row in edges_result.result_set or []:
            edges_by_uuid.setdefault(row[0], (row, spec['source_graph']))

        mentions_result = await source.ro_query(
            'MATCH (ep:Episodic)-[e:MENTIONS]->(n:Entity {uuid: $uuid}) '
            'RETURN e.uuid, e.group_id, e.created_at, ep.uuid',
            {'uuid': spec['uuid']},
        )
        for row in mentions_result.result_set or []:
            mentions_by_uuid.setdefault(row[0], (row, spec['source_graph'], spec['uuid']))

    for edge_uuid, (row, source_graph) in edges_by_uuid.items():
        (_edge_uuid, edge_name, fact, valid_at, invalid_at, edge_created_at,
         _edge_group_id, episodes, src_uuid, dst_uuid) = row

        src_target = target_of.get(src_uuid)
        dst_target = target_of.get(dst_uuid)

        # A non-migrating endpoint (no spec of its own -- absent from
        # target_of) is deliverable only if it is already a home-resident in
        # the OTHER (migrating) endpoint's resolved target -- a presence
        # probe decides it. At least one of src_target/dst_target is always
        # resolved here, since every edge in edges_by_uuid was read via a
        # MOVE spec's own incident-edge query (so that spec's uuid is always
        # one of the two endpoints).
        if src_target is None and dst_target is not None:
            if await _entity_present_in_graph(graphiti._graph_for(dst_target), src_uuid):
                src_target = dst_target
        elif (
            dst_target is None and src_target is not None
            and await _entity_present_in_graph(graphiti._graph_for(src_target), dst_uuid)
        ):
            dst_target = src_target

        if src_target is None or dst_target is None or src_target != dst_target:
            # The two endpoints do not share one destination graph --
            # cross-graph RELATES_TO edges are unsupported (FalkorDB edges
            # are single-graph), so this edge cannot be recreated in EITHER
            # graph. Reported for human review, never silently lost (unlike
            # the old move_entity_across_graphs' edges_skipped, which only
            # ever surfaced a bare count with no record of why).
            result.dropped_cross_target.append({
                'edge_uuid': edge_uuid,
                'src_uuid': src_uuid,
                'dst_uuid': dst_uuid,
                'src_target': src_target,
                'dst_target': dst_target,
                'reason': (
                    'edge endpoints resolve to different target graphs (or '
                    'an undeliverable non-migrating endpoint) -- cross-graph '
                    'RELATES_TO edges are unsupported; needs manual review'
                ),
            })
            continue

        target_graph_name = src_target
        target = graphiti._graph_for(target_graph_name)

        if await _relates_to_edge_already_in_target(target, edge_uuid):
            # Genuine idempotent no-op (re-run after a prior completed/
            # partial Phase B) -- not a loss, so neither counter is
            # incremented.
            continue

        try:
            edge_embedding_cypher = (
                f'MATCH ()-[e:RELATES_TO {{uuid: {_quote_cypher_string(edge_uuid)}}}]-() '
                'RETURN e.fact_embedding'
            )
            edge_embedding_reply = await _read_compact_vector(
                falkor_client, group_id=source_graph, cypher=edge_embedding_cypher,
            )
            embedding_clause = _embedding_set_clause('r.fact_embedding', edge_embedding_reply)

            edge_create_result = await target.query(
                'MATCH (a:Entity {uuid: $src_uuid}), (b:Entity {uuid: $dst_uuid}) '
                'CREATE (a)-[r:RELATES_TO]->(b) '
                'SET r.uuid = $edge_uuid, '
                '    r.name = $name, '
                '    r.fact = $fact, '
                '    r.valid_at = $valid_at, '
                '    r.invalid_at = $invalid_at, '
                '    r.created_at = $created_at, '
                '    r.group_id = $group_id, '
                '    r.episodes = $episodes'
                f'{embedding_clause}',
                {
                    'src_uuid': src_uuid,
                    'dst_uuid': dst_uuid,
                    'edge_uuid': edge_uuid,
                    'name': edge_name,
                    'fact': fact,
                    'valid_at': valid_at,
                    'invalid_at': invalid_at,
                    'created_at': edge_created_at,
                    'group_id': target_graph_name,
                    'episodes': episodes,
                },
            )
            if edge_create_result.relationships_created:
                result.edges_recreated += 1
                if not embedding_clause:
                    result.embedding_omitted += 1
            else:
                result.edges_skipped += 1
                logger.warning(
                    'recreate_subgraph_relationships: RELATES_TO edge uuid=%s '
                    'silently skipped -- other endpoint (src=%s dst=%s) not '
                    'present in target_graph=%s',
                    edge_uuid, src_uuid, dst_uuid, target_graph_name,
                )
        except Exception as exc:
            # Per-item isolation (CGL-η follow-up, task 2451): a single bad
            # edge's embedding read or CREATE (e.g. a transient FalkorDB
            # error) costs only this ONE edge, not the whole batch -- unlike
            # a per-spec gather-read failure (outside this try/except),
            # which remains a systemic/batch-level abort (see this module's
            # recreate_subgraph_relationships docstring). Both incident node
            # uuids are recorded so a caller can withhold Phase C
            # source-deletion for them, preserving create-before-delete for
            # this un-recreated edge.
            result.blocked.append({
                'kind': 'edge',
                'uuid': edge_uuid,
                'reason': str(exc),
                'node_uuids': [src_uuid, dst_uuid],
            })
            logger.warning(
                'recreate_subgraph_relationships: RELATES_TO edge uuid=%s '
                'blocked -- %s: %s (src=%s dst=%s target_graph=%s); batch '
                'continues',
                edge_uuid, type(exc).__name__, exc, src_uuid, dst_uuid,
                target_graph_name,
            )

    for mention_uuid, (row, _source_graph, entity_uuid) in mentions_by_uuid.items():
        _mention_uuid, _mention_group_id, mention_created_at, episode_uuid = row
        target_graph_name = target_of[entity_uuid]
        target = graphiti._graph_for(target_graph_name)

        if await _mentions_link_already_in_target(target, mention_uuid):
            # Genuine idempotent no-op (re-run after a prior completed/
            # partial Phase B) -- not a loss, so neither counter is
            # incremented. Mirrors the RELATES_TO skip above.
            continue

        try:
            mention_create_result = await target.query(
                'MATCH (ep:Episodic {uuid: $episode_uuid}), (n:Entity {uuid: $entity_uuid}) '
                'CREATE (ep)-[e:MENTIONS]->(n) '
                'SET e.uuid = $edge_uuid, '
                '    e.group_id = $group_id, '
                '    e.created_at = $created_at',
                {
                    'episode_uuid': episode_uuid,
                    'entity_uuid': entity_uuid,
                    'edge_uuid': mention_uuid,
                    'group_id': target_graph_name,
                    'created_at': mention_created_at,
                },
            )
            if mention_create_result.relationships_created:
                result.mentions_recreated += 1
            else:
                result.mentions_skipped += 1
                logger.warning(
                    'recreate_subgraph_relationships: MENTIONS link uuid=%s '
                    'silently skipped -- episode uuid=%s not present in '
                    'target_graph=%s (entity_uuid=%s)',
                    mention_uuid, episode_uuid, target_graph_name, entity_uuid,
                )
        except Exception as exc:
            # Per-item isolation, same rationale as the RELATES_TO edge pass
            # above: a single bad MENTIONS CREATE costs only this ONE link.
            result.blocked.append({
                'kind': 'mention',
                'uuid': mention_uuid,
                'reason': str(exc),
                'node_uuids': [entity_uuid],
            })
            logger.warning(
                'recreate_subgraph_relationships: MENTIONS link uuid=%s '
                'blocked -- %s: %s (entity_uuid=%s target_graph=%s); batch '
                'continues',
                mention_uuid, type(exc).__name__, exc, entity_uuid,
                target_graph_name,
            )

    # --- MERGE fold ---
    # Run AFTER the MOVE-edge pass above has fully completed (sequential
    # awaits, so this is guaranteed): if a MOVE-pass edge already landed on
    # this MERGE spec's home copy (e.g. a shared MOVE<->MERGE edge whose
    # MOVE endpoint was processed first), _read_relates_to_edge_uuids below
    # observes it in home_edge_uuids and classify_unique_wrong_edges
    # correctly excludes it here -- no double-create.
    merge_specs = [spec for spec in specs if spec.get('disposition') == 'MERGE']
    for spec in merge_specs:
        uuid = spec['uuid']
        wrong_graph = spec['source_graph']
        home_graph = spec['target_graph']
        wrong = graphiti._graph_for(wrong_graph)
        home = graphiti._graph_for(home_graph)

        wrong_edges_result = await wrong.ro_query(
            'MATCH (n:Entity {uuid: $uuid})-[e:RELATES_TO]-(m:Entity) '
            'WITH DISTINCT e, startNode(e) AS s, endNode(e) AS t '
            'RETURN e.uuid, e.name, e.fact, e.valid_at, e.invalid_at, e.created_at, '
            '       e.group_id, e.episodes, s.uuid, t.uuid',
            {'uuid': uuid},
        )
        wrong_rows_by_uuid = {row[0]: row for row in (wrong_edges_result.result_set or [])}
        wrong_edge_uuids = set(wrong_rows_by_uuid)

        home_edge_uuids = await _read_relates_to_edge_uuids(home, uuid)
        unique_wrong_edge_uuids = classify_unique_wrong_edges(home_edge_uuids, wrong_edge_uuids)

        for edge_uuid in unique_wrong_edge_uuids:
            (_edge_uuid, edge_name, fact, valid_at, invalid_at, edge_created_at,
             edge_group_id, episodes, src_uuid, dst_uuid) = wrong_rows_by_uuid[edge_uuid]

            try:
                edge_embedding_cypher = (
                    f'MATCH ()-[e:RELATES_TO {{uuid: {_quote_cypher_string(edge_uuid)}}}]-() '
                    'RETURN e.fact_embedding'
                )
                edge_embedding_reply = await _read_compact_vector(
                    falkor_client, group_id=wrong_graph, cypher=edge_embedding_cypher,
                )
                embedding_clause = _embedding_set_clause('r.fact_embedding', edge_embedding_reply)

                # Both endpoints are MATCHed (never CREATEd): the home copy
                # of this MERGE node and the edge's other endpoint must
                # already exist in home_graph, or this MATCH yields no rows
                # and the edge is silently skipped -- same convention as the
                # MOVE-edge pass above / merge_foreign_duplicate. group_id is
                # preserved from the wrong copy verbatim -- MERGE has no
                # rewrite_group_id analogue (mirrors merge_foreign_duplicate).
                edge_create_result = await home.query(
                    'MATCH (a:Entity {uuid: $src_uuid}), (b:Entity {uuid: $dst_uuid}) '
                    'CREATE (a)-[r:RELATES_TO]->(b) '
                    'SET r.uuid = $edge_uuid, '
                    '    r.name = $name, '
                    '    r.fact = $fact, '
                    '    r.valid_at = $valid_at, '
                    '    r.invalid_at = $invalid_at, '
                    '    r.created_at = $created_at, '
                    '    r.group_id = $group_id, '
                    '    r.episodes = $episodes'
                    f'{embedding_clause}',
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
                if edge_create_result.relationships_created:
                    result.edges_recreated += 1
                    if not embedding_clause:
                        result.embedding_omitted += 1
                else:
                    result.edges_skipped += 1
                    logger.warning(
                        'recreate_subgraph_relationships: MERGE-fold RELATES_TO '
                        'edge uuid=%s silently skipped -- other endpoint '
                        '(src=%s dst=%s) not present in home_graph=%s',
                        edge_uuid, src_uuid, dst_uuid, home_graph,
                    )
            except Exception as exc:
                # Per-item isolation, same rationale as the MOVE-edge pass
                # above: a single bad MERGE-fold edge costs only itself.
                result.blocked.append({
                    'kind': 'edge',
                    'uuid': edge_uuid,
                    'reason': str(exc),
                    'node_uuids': [src_uuid, dst_uuid],
                })
                logger.warning(
                    'recreate_subgraph_relationships: MERGE-fold RELATES_TO '
                    'edge uuid=%s blocked -- %s: %s (src=%s dst=%s '
                    'home_graph=%s); batch continues',
                    edge_uuid, type(exc).__name__, exc, src_uuid, dst_uuid,
                    home_graph,
                )


async def delete_source_node(graphiti: Any, uuid: str, source_graph: str) -> None:
    """Phase C of the three-phase barrier-ordered apply (CGL-η follow-up,
    task 2415): ``DETACH DELETE`` *uuid*'s Entity node from *source_graph* --
    and ONLY that.

    Reuses the exact DETACH DELETE Cypher ``move_entity_across_graphs``
    issues as its final mutation. Callers driving the three-phase apply must
    call this ONLY after every ``create_moved_node`` (Phase A) and
    ``recreate_subgraph_relationships`` (Phase B) call for the WHOLE batch
    has completed -- deleting a source before its edges are recreated
    elsewhere is exactly the bug this task fixes (a co-moving neighbour's
    shared edge silently skipped by the OTHER endpoint's target CREATE, then
    destroyed here before it is ever recreated).

    Touches ONLY source_graph -- never resolves or queries target_graph.
    Idempotent: FalkorDB's ``MATCH ... DETACH DELETE`` matches (and deletes)
    nothing when the node is already gone, so a re-run after a completed
    apply -- or a retry after Phase C partially completed -- is a safe
    no-op; this function performs no existence pre-check of its own.

    Args:
        graphiti: An initialized GraphitiBackend (or compatible object
            exposing ``_graph_for``).
        uuid: UUID of the Entity node to delete from source_graph.
        source_graph: Graph to delete the node from.

    Returns:
        None.
    """
    source = graphiti._graph_for(source_graph)
    await source.query(
        'MATCH (n:Entity {uuid: $uuid}) DETACH DELETE n',
        {'uuid': uuid},
    )
    logger.info(
        'delete_source_node: deleted uuid=%s source=%s', uuid, source_graph,
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
            '    r.episodes = $episodes'
            f'{_embedding_set_clause("r.fact_embedding", edge_embedding_reply)}',
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

    # --- independent re-read (step-21/22) ---
    # home_edge_count_after is a genuine re-read of home's RELATES_TO edge
    # set, taken after the recreate loop above -- NOT derived arithmetically
    # from home_edge_count_before + edges_recreated. A recreate whose CREATE
    # silently matched no endpoint (the documented silent-skip when the
    # edge's other endpoint is absent from home) would leave home's actual
    # edge count unchanged; only an independent re-read surfaces that
    # mismatch instead of masking it behind arithmetic.
    home_edge_uuids_after = await _read_relates_to_edge_uuids(home, uuid)

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
        home_edge_count_after=len(home_edge_uuids_after),
    )
