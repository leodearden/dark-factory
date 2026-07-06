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

from unittest.mock import AsyncMock, MagicMock

import pytest
from _fm_helpers import extract_cypher, extract_params

from fused_memory.maintenance import cross_graph_move
from fused_memory.maintenance.cross_graph_move import (
    MoveResult,
    format_vecf32_literal,
    move_entity_across_graphs,
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


# ---------------------------------------------------------------------------
# move_entity_across_graphs (S5) fixtures + shared test helper
# ---------------------------------------------------------------------------

NODE_UUID_FIXTURE = 'node-aaaa-1111'
SOURCE_GRAPH_FIXTURE = 'dark_factory'
TARGET_GRAPH_FIXTURE = 'know_live'

# (uuid, name, group_id, summary, created_at) -- the scalar Entity props
# returned by the plain (non-lossy) ro_query read.
NODE_ROW_FIXTURE = [
    NODE_UUID_FIXTURE, 'Alice', SOURCE_GRAPH_FIXTURE, 'Alice is a person.', '2026-01-01T00:00:00+00:00',
]


def _route_graphs(mapping: dict) -> MagicMock:
    """MagicMock side_effect routing backend._driver._get_graph(name) -> mapping[name]."""
    return MagicMock(side_effect=lambda name: mapping[name])


# ---------------------------------------------------------------------------
# step-5: move_entity_across_graphs -- node core
# ---------------------------------------------------------------------------

class TestMoveEntityAcrossGraphsNodeCore:
    """S5 node-move core: read node props + exact name_embedding, CREATE in target."""

    @pytest.mark.asyncio
    async def test_creates_node_in_target_with_byte_exact_name_embedding(
        self, mock_config, make_backend, make_graph_mock, monkeypatch,
    ):
        """Reads the node from source (props via ro_query, embedding via the raw
        transport) and CREATEs it in target with a byte-exact vecf32 literal.
        """
        backend = make_backend(mock_config)
        source_mock = make_graph_mock()
        # ro_query is called three times: (1) node scalar props, (2) this
        # node's RELATES_TO edges (step-7/8), (3) its Episodic MENTIONS links
        # (step-9/10) -- both empty here, so no edge/mention is recreated and
        # the CREATE/embedding-read assertions below are unaffected.
        source_mock.ro_query = AsyncMock(side_effect=[
            MagicMock(result_set=[NODE_ROW_FIXTURE]),
            MagicMock(result_set=[]),
            MagicMock(result_set=[]),
        ])
        target_mock = make_graph_mock()
        backend._driver._get_graph = _route_graphs({
            SOURCE_GRAPH_FIXTURE: source_mock,
            TARGET_GRAPH_FIXTURE: target_mock,
        })

        fake_read_compact = AsyncMock(return_value=COMPACT_VECTOR_REPLY_FIXTURE)
        monkeypatch.setattr(cross_graph_move, '_read_compact_vector', fake_read_compact)

        result = await move_entity_across_graphs(
            backend, NODE_UUID_FIXTURE, SOURCE_GRAPH_FIXTURE, TARGET_GRAPH_FIXTURE,
        )

        # (a) reads node props from source via ro_query, and the exact
        # name_embedding from source via the raw transport.
        assert source_mock.ro_query.await_count == 3
        read_params = extract_params(source_mock.ro_query.call_args_list[0])
        assert read_params.get('uuid') == NODE_UUID_FIXTURE
        fake_read_compact.assert_awaited_once()
        assert fake_read_compact.call_args.kwargs.get('group_id') == SOURCE_GRAPH_FIXTURE

        # (b) issues a CREATE against target_graph carrying the byte-exact
        # vecf32 literal plus the node's uuid/name/group_id params.
        target_mock.query.assert_awaited_once()
        cypher = extract_cypher(target_mock.query.call_args)
        params = extract_params(target_mock.query.call_args)
        assert 'CREATE' in cypher
        assert EXPECTED_VECF32_LITERAL_FIXTURE in cypher
        assert params.get('uuid') == NODE_UUID_FIXTURE
        assert params.get('name') == 'Alice'
        assert params.get('group_id') == SOURCE_GRAPH_FIXTURE

        # source's mutating .query is used solely for the final DETACH
        # DELETE (step-11/12); ordering vs. the target CREATE is covered in
        # TestMoveEntityAcrossGraphsOrdering.
        source_mock.query.assert_awaited_once()
        delete_cypher = extract_cypher(source_mock.query.call_args)
        delete_params = extract_params(source_mock.query.call_args)
        assert 'DETACH DELETE' in delete_cypher
        assert delete_params.get('uuid') == NODE_UUID_FIXTURE

        assert isinstance(result, MoveResult)
        assert result.uuid == NODE_UUID_FIXTURE
        assert result.source_graph == SOURCE_GRAPH_FIXTURE
        assert result.target_graph == TARGET_GRAPH_FIXTURE


# ---------------------------------------------------------------------------
# step-7: move_entity_across_graphs -- RELATES_TO edge reattachment
# ---------------------------------------------------------------------------

OTHER_NODE_UUID_FIXTURE = 'node-bbbb-2222'
EDGE_UUID_FIXTURE = 'edge-cccc-3333'

# (e.uuid, e.name, e.fact, e.valid_at, e.invalid_at, e.created_at, e.group_id,
#  e.episodes, startNode(e).uuid, endNode(e).uuid) -- an outgoing edge from
# the moved node to another (unmoved) entity.
EDGE_ROW_FIXTURE = [
    EDGE_UUID_FIXTURE, 'is_related_to', 'Alice is related to Bob.',
    '2026-01-01T00:00:00+00:00', None, '2026-01-01T00:00:00+00:00',
    SOURCE_GRAPH_FIXTURE, ['episode-uuid-1'],
    NODE_UUID_FIXTURE, OTHER_NODE_UUID_FIXTURE,
]


class TestMoveEntityAcrossGraphsEdges:
    """S5 edge reattachment: recreate the moved node's RELATES_TO edges in target."""

    @pytest.mark.asyncio
    async def test_recreates_relates_to_edge_with_byte_exact_fact_embedding(
        self, mock_config, make_backend, make_graph_mock, monkeypatch,
    ):
        backend = make_backend(mock_config)
        source_mock = make_graph_mock()
        source_mock.ro_query = AsyncMock(side_effect=[
            MagicMock(result_set=[NODE_ROW_FIXTURE]),
            MagicMock(result_set=[EDGE_ROW_FIXTURE]),
            MagicMock(result_set=[]),  # no MENTIONS links in this scenario (step-9/10)
        ])
        target_mock = make_graph_mock()
        backend._driver._get_graph = _route_graphs({
            SOURCE_GRAPH_FIXTURE: source_mock,
            TARGET_GRAPH_FIXTURE: target_mock,
        })

        fake_read_compact = AsyncMock(return_value=COMPACT_VECTOR_REPLY_FIXTURE)
        monkeypatch.setattr(cross_graph_move, '_read_compact_vector', fake_read_compact)

        result = await move_entity_across_graphs(
            backend, NODE_UUID_FIXTURE, SOURCE_GRAPH_FIXTURE, TARGET_GRAPH_FIXTURE,
        )

        # edges are read from source via a second ro_query call
        assert source_mock.ro_query.await_count == 3
        edges_read_cypher = extract_cypher(source_mock.ro_query.call_args_list[1])
        assert 'RELATES_TO' in edges_read_cypher
        assert 'DISTINCT' in edges_read_cypher

        # fact_embedding is read via the raw transport for the node AND the edge
        assert fake_read_compact.await_count == 2
        edge_embed_call = fake_read_compact.call_args_list[1]
        assert edge_embed_call.kwargs.get('group_id') == SOURCE_GRAPH_FIXTURE
        assert EDGE_UUID_FIXTURE in edge_embed_call.kwargs.get('cypher', '')

        # target.query is called twice: (1) node CREATE, (2) edge CREATE
        assert target_mock.query.await_count == 2
        edge_create_call = target_mock.query.call_args_list[1]
        cypher = extract_cypher(edge_create_call)
        params = extract_params(edge_create_call)
        assert 'CREATE' in cypher
        assert 'RELATES_TO' in cypher
        assert EXPECTED_VECF32_LITERAL_FIXTURE in cypher
        assert params.get('src_uuid') == NODE_UUID_FIXTURE
        assert params.get('dst_uuid') == OTHER_NODE_UUID_FIXTURE
        assert params.get('edge_uuid') == EDGE_UUID_FIXTURE
        assert params.get('name') == 'is_related_to'
        assert params.get('fact') == 'Alice is related to Bob.'
        assert params.get('group_id') == SOURCE_GRAPH_FIXTURE
        assert params.get('episodes') == ['episode-uuid-1']

        assert result.edges_moved == 1


# ---------------------------------------------------------------------------
# step-9: move_entity_across_graphs -- Episodic MENTIONS reattachment
# ---------------------------------------------------------------------------

EPISODE_UUID_FIXTURE = 'episode-dddd-4444'
MENTION_UUID_FIXTURE = 'mention-eeee-5555'

# (e.uuid, e.group_id, e.created_at, ep.uuid) -- a MENTIONS link from an
# Episodic node onto the moved Entity node.
MENTION_ROW_FIXTURE = [
    MENTION_UUID_FIXTURE, SOURCE_GRAPH_FIXTURE, '2026-01-01T00:00:00+00:00', EPISODE_UUID_FIXTURE,
]


class TestMoveEntityAcrossGraphsMentions:
    """S5 mentions reattachment: recreate the moved node's Episodic MENTIONS links."""

    @pytest.mark.asyncio
    async def test_recreates_mentions_link_for_episode(
        self, mock_config, make_backend, make_graph_mock, monkeypatch,
    ):
        backend = make_backend(mock_config)
        source_mock = make_graph_mock()
        source_mock.ro_query = AsyncMock(side_effect=[
            MagicMock(result_set=[NODE_ROW_FIXTURE]),
            MagicMock(result_set=[]),  # no RELATES_TO edges in this scenario
            MagicMock(result_set=[MENTION_ROW_FIXTURE]),
        ])
        target_mock = make_graph_mock()
        backend._driver._get_graph = _route_graphs({
            SOURCE_GRAPH_FIXTURE: source_mock,
            TARGET_GRAPH_FIXTURE: target_mock,
        })

        fake_read_compact = AsyncMock(return_value=COMPACT_VECTOR_REPLY_FIXTURE)
        monkeypatch.setattr(cross_graph_move, '_read_compact_vector', fake_read_compact)

        result = await move_entity_across_graphs(
            backend, NODE_UUID_FIXTURE, SOURCE_GRAPH_FIXTURE, TARGET_GRAPH_FIXTURE,
        )

        # MENTIONS links are read from source via a third ro_query call
        assert source_mock.ro_query.await_count == 3
        mentions_read_cypher = extract_cypher(source_mock.ro_query.call_args_list[2])
        assert 'MENTIONS' in mentions_read_cypher
        assert 'Episodic' in mentions_read_cypher

        # target.query: (1) node CREATE, (2) mentions CREATE (no edges here)
        assert target_mock.query.await_count == 2
        mention_create_call = target_mock.query.call_args_list[1]
        cypher = extract_cypher(mention_create_call)
        params = extract_params(mention_create_call)
        assert 'MENTIONS' in cypher
        assert 'Episodic' in cypher
        assert params.get('episode_uuid') == EPISODE_UUID_FIXTURE
        assert params.get('entity_uuid') == NODE_UUID_FIXTURE
        assert params.get('edge_uuid') == MENTION_UUID_FIXTURE
        assert params.get('group_id') == SOURCE_GRAPH_FIXTURE

        assert result.mentions_moved == 1


# ---------------------------------------------------------------------------
# step-11: move_entity_across_graphs -- CREATE-BEFORE-DELETE ordering
# ---------------------------------------------------------------------------

class TestMoveEntityAcrossGraphsOrdering:
    """S5 ordering: the source DETACH DELETE runs strictly after every target CREATE."""

    @pytest.mark.asyncio
    async def test_source_detach_delete_runs_after_all_target_creates(
        self, mock_config, make_backend, make_graph_mock, monkeypatch,
    ):
        """Records mutating .query() call order across BOTH graphs into one
        shared list (each graph's .query is wrapped to append (role, cypher,
        params)), so the true cross-graph ordering of the target CREATEs vs.
        the source DETACH DELETE can be asserted directly -- not just each
        mock's own internal call order, which wouldn't distinguish "source
        deleted before target finished creating" from the reverse.
        """
        backend = make_backend(mock_config)
        call_order: list[tuple[str, str, dict]] = []

        source_mock = make_graph_mock()
        source_mock.ro_query = AsyncMock(side_effect=[
            MagicMock(result_set=[NODE_ROW_FIXTURE]),
            MagicMock(result_set=[EDGE_ROW_FIXTURE]),
            MagicMock(result_set=[MENTION_ROW_FIXTURE]),
        ])

        async def _record_source_query(cypher, params=None):
            call_order.append(('source', cypher, params or {}))
            return MagicMock()

        source_mock.query = AsyncMock(side_effect=_record_source_query)

        target_mock = make_graph_mock()

        async def _record_target_query(cypher, params=None):
            call_order.append(('target', cypher, params or {}))
            return MagicMock()

        target_mock.query = AsyncMock(side_effect=_record_target_query)

        backend._driver._get_graph = _route_graphs({
            SOURCE_GRAPH_FIXTURE: source_mock,
            TARGET_GRAPH_FIXTURE: target_mock,
        })

        fake_read_compact = AsyncMock(return_value=COMPACT_VECTOR_REPLY_FIXTURE)
        monkeypatch.setattr(cross_graph_move, '_read_compact_vector', fake_read_compact)

        await move_entity_across_graphs(
            backend, NODE_UUID_FIXTURE, SOURCE_GRAPH_FIXTURE, TARGET_GRAPH_FIXTURE,
        )

        # node CREATE, edge CREATE, mention CREATE (all target) THEN the
        # source DETACH DELETE -- exactly 4 mutating calls, in this order.
        assert len(call_order) == 4
        roles = [role for role, _, _ in call_order]
        assert roles == ['target', 'target', 'target', 'source']
        for _role, cypher, _params in call_order[:3]:
            assert 'CREATE' in cypher

        # the delete is the final mutating call, targets source_graph only
        # (never target_graph), and is scoped to the moved node's uuid.
        final_role, final_cypher, final_params = call_order[-1]
        assert final_role == 'source'
        assert 'DETACH DELETE' in final_cypher
        assert final_params.get('uuid') == NODE_UUID_FIXTURE


# ---------------------------------------------------------------------------
# step-13: move_entity_across_graphs -- idempotency short-circuit
# ---------------------------------------------------------------------------

class TestMoveEntityAcrossGraphsIdempotency:
    """S5 idempotency: already-moved (present in target, absent in source) is a no-op."""

    @pytest.mark.asyncio
    async def test_already_moved_short_circuits_before_any_write(
        self, mock_config, make_backend, make_graph_mock, monkeypatch,
    ):
        """When the uuid is already present in target_graph and absent from
        source_graph, the call performs NO CREATE and NO DETACH DELETE, never
        touches the raw --compact embedding transport, and returns a
        MoveResult flagged already-moved. The already-moved probe (reading
        target) must run before any write -- there must be zero mutating
        calls on either graph.
        """
        backend = make_backend(mock_config)

        source_mock = make_graph_mock()
        source_mock.ro_query = AsyncMock(return_value=MagicMock(result_set=[]))

        target_mock = make_graph_mock()
        target_mock.ro_query = AsyncMock(
            return_value=MagicMock(result_set=[[NODE_UUID_FIXTURE]])
        )

        backend._driver._get_graph = _route_graphs({
            SOURCE_GRAPH_FIXTURE: source_mock,
            TARGET_GRAPH_FIXTURE: target_mock,
        })

        fake_read_compact = AsyncMock(return_value=COMPACT_VECTOR_REPLY_FIXTURE)
        monkeypatch.setattr(cross_graph_move, '_read_compact_vector', fake_read_compact)

        result = await move_entity_across_graphs(
            backend, NODE_UUID_FIXTURE, SOURCE_GRAPH_FIXTURE, TARGET_GRAPH_FIXTURE,
        )

        # the probe reads target for presence ...
        target_mock.ro_query.assert_awaited_once()
        probe_params = extract_params(target_mock.ro_query.call_args)
        assert probe_params.get('uuid') == NODE_UUID_FIXTURE

        # ... and short-circuits before any mutation or embedding read.
        target_mock.query.assert_not_awaited()
        source_mock.query.assert_not_awaited()
        fake_read_compact.assert_not_awaited()

        assert isinstance(result, MoveResult)
        assert result.uuid == NODE_UUID_FIXTURE
        assert result.source_graph == SOURCE_GRAPH_FIXTURE
        assert result.target_graph == TARGET_GRAPH_FIXTURE
        assert result.already_moved is True
