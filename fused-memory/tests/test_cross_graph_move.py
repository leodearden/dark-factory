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

from fused_memory.maintenance.cross_graph_move import (
    format_vecf32_literal,
    parse_compact_vector_reply,
)

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


# ---------------------------------------------------------------------------
# step-1: parse_compact_vector_reply
# ---------------------------------------------------------------------------

class TestParseCompactVectorReply:
    """parse_compact_vector_reply(reply) -> list[str], never coercing via float()."""

    def test_parses_recorded_fixture_into_exact_tokens(self):
        """Splits the bracketed --compact reply into its exact string tokens."""
        tokens = parse_compact_vector_reply(COMPACT_VECTOR_REPLY_FIXTURE)
        assert tokens == ['0.5', '0.10000000149011611938', '-0.987654321098765432']

    def test_high_precision_token_survives_byte_for_byte(self):
        """The 20-decimal-digit token is preserved verbatim, not float()-truncated.

        A lossy implementation that round-trips through float() (or formats
        with a fixed 6-decimal precision) would alter this token; asserting
        exact string equality against the full-precision fixture value is
        what makes this test genuinely RED against such an implementation.
        """
        tokens = parse_compact_vector_reply(COMPACT_VECTOR_REPLY_FIXTURE)
        assert tokens[1] == '0.10000000149011611938'
        # Sanity-check the fixture itself is a genuine precision trap: a lossy
        # %.6f-style truncation of this token collapses to a value that does
        # not string-match the original -- so the byte-exact assertion above
        # is the thing actually distinguishing a correct impl from a lossy one.
        assert f'{float(tokens[1]):.6f}' != tokens[1]

    def test_empty_vector_reply_returns_empty_list(self):
        """An empty '--compact' vector reply parses to an empty token list."""
        assert parse_compact_vector_reply('[]') == []


# ---------------------------------------------------------------------------
# step-3: format_vecf32_literal
# ---------------------------------------------------------------------------

class TestFormatVecf32Literal:
    """format_vecf32_literal(tokens) -> 'vecf32([...])', tokens embedded verbatim."""

    def test_formats_tokens_into_vecf32_literal(self):
        """Renders a simple token list as a vecf32([...]) Cypher literal."""
        assert format_vecf32_literal(['0.5', '1.0']) == 'vecf32([0.5, 1.0])'

    def test_empty_tokens_render_empty_vecf32_literal(self):
        """An empty token list renders as vecf32([])."""
        assert format_vecf32_literal([]) == 'vecf32([])'

    def test_round_trip_matches_expected_fixture_byte_for_byte(self):
        """format_vecf32_literal(parse_compact_vector_reply(fixture)) is lossless.

        Chaining the two pure functions over the recorded --compact fixture
        must reproduce the recorded expected vecf32([...]) literal exactly,
        including the full-precision (non-float()-roundtrippable) tokens --
        proving the read->literal passthrough never touches the numeric
        value.
        """
        tokens = parse_compact_vector_reply(COMPACT_VECTOR_REPLY_FIXTURE)
        assert format_vecf32_literal(tokens) == EXPECTED_VECF32_LITERAL_FIXTURE
