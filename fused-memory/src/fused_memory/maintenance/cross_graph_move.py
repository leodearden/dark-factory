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
    with a byte-exact ``vecf32([...])`` literal. Edge/mentions reattachment,
    the source ``DETACH DELETE``, and the idempotency probe are added in
    later steps (7-14); ``rewrite_group_id`` substitution lands in step-16.

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

    logger.info(
        'move_entity_across_graphs: node moved uuid=%s source=%s target=%s',
        uuid, source_graph, target_graph,
    )
    return MoveResult(uuid=uuid, source_graph=source_graph, target_graph=target_graph)
