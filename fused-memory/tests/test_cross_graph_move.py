"""Tests for cross-graph move + foreign-duplicate-merge primitives (task 2271).

Covers ``fused_memory.maintenance.cross_graph_move``:
- ``parse_compact_vector_reply`` / ``format_vecf32_literal`` (pure byte-exact
  vecf32 passthrough functions)
- ``move_entity_across_graphs`` (S5): node + RELATES_TO edges + Episodic
  MENTIONS reattachment, create-before-delete ordering, idempotency,
  ``rewrite_group_id``
- ``merge_foreign_duplicate`` (S6): unique-edge classification, recreate +
  delete, edge-count invariant

Mock-only per project convention (MagicMock graphs; no live-FalkorDB
fixture). Byte-fidelity against a REAL FalkorDB is NOT asserted here -- see
the module docstring and ``plans/cross-graph-entity-leak-prd.md`` decision 5.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Recorded/representative `GRAPH.RO_QUERY ... --compact` vector-reply fixture.
#
# Real FalkorDB (per the RCA) transmits vecf32 components as exact float32
# decimal strings on the wire when queried with --compact; this fixture
# stands in for that reply (mock-only suite; no live FalkorDB -- see module
# docstring). The second and third tokens are the FULL terminating decimal
# expansion of an IEEE-754 float32 value (e.g. 0.10000000149011611938 is the
# exact value nearest to 0.1 as a 32-bit float) -- far more digits than
# either a naive `f'{float(tok):.6f}'` truncation or a `str(float(tok))`
# shortest-round-trip re-encoding would preserve. A lossy implementation
# (one that ever calls `float()` on a token) alters these strings; a byte-
# exact passthrough implementation does not. This makes the RED tests below
# genuinely fail against a lossy impl and pass only for a string-passthrough
# impl -- an exactness claim satisfied by construction, not numeric guessing.
# ---------------------------------------------------------------------------
COMPACT_VECTOR_REPLY_FIXTURE = (
    '[0.5, 0.10000000149011611938, -0.987654321098765432]'
)
EXPECTED_VECF32_LITERAL_FIXTURE = (
    'vecf32([0.5, 0.10000000149011611938, -0.987654321098765432])'
)
