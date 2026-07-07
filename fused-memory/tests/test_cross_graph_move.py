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
    ForeignDuplicateSuspectedError,
    MergeResult,
    MoveResult,
    classify_unique_wrong_edges,
    format_vecf32_literal,
    merge_foreign_duplicate,
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
# _read_compact_vector: the ONE non-pure, byte-exactness-critical read path.
#
# Every OTHER test in this module monkeypatches _read_compact_vector away, so
# this class is the module's only coverage of its raw `--compact` reply
# parsing: the reply[1][0]/cell indexing, the int(cell[0]) scalar-type
# extraction, the non-VECTORF32 ValueError path (which is ALSO how a
# null/absent embedding surfaces), and the bytes-token materialization. A
# regression in any of these (reply-shape change, wrong scalar-type constant)
# would otherwise ship green. It feeds a recorded/representative raw reply
# structure through the REAL function.
#
# Compact reply shape (per falkordb's own parse_scalar convention, mirrored in
# _read_compact_vector's docstring): reply == [header, rows, stats]; rows[0] is
# the first row; row[0] is the first cell == [scalar_type, value]; scalar_type
# 12 == VALUE_VECTORF32 and value is the list of exact float32 token bytes.
# ---------------------------------------------------------------------------

_VALUE_VECTORF32_TAG = 12  # falkordb ResultSetScalarTypes.VALUE_VECTORF32
_VALUE_NULL_TAG = 1        # falkordb ResultSetScalarTypes.VALUE_NULL


def _compact_reply(cell) -> list:
    """Wrap a single result *cell* in the [header, rows, stats] compact shape."""
    return ['header', [[cell]], 'stats']


class TestReadCompactVectorTransport:
    """_read_compact_vector(falkor_client, *, group_id, cypher) raw-reply parsing."""

    def _client_returning(self, reply) -> MagicMock:
        client = MagicMock()
        client.execute_command = AsyncMock(return_value=reply)
        return client

    @pytest.mark.asyncio
    async def test_returns_exact_bracketed_token_string(self):
        """A VECTORF32 cell's byte-exact tokens materialize to the bracketed reply string.

        Uses the full-precision recorded fixture tokens, so a regression that
        (say) round-tripped a token through float() would alter the returned
        string and fail here. Also asserts the raw `--compact` command is
        issued verbatim against the given group_id/cypher.
        """
        cypher = 'MATCH (n:Entity {uuid: \'x\'}) RETURN n.name_embedding'
        reply = _compact_reply([
            _VALUE_VECTORF32_TAG,
            [b'0.5', b'0.10000000149011611938', b'-0.987654321098765432'],
        ])
        client = self._client_returning(reply)

        out = await cross_graph_move._read_compact_vector(
            client, group_id=SOURCE_GRAPH_FIXTURE, cypher=cypher,
        )

        # (a) exact bracketed token string, byte-for-byte the --compact fixture.
        assert out == COMPACT_VECTOR_REPLY_FIXTURE
        # ...and it flows losslessly into the vecf32 literal via the pure funcs.
        assert (
            format_vecf32_literal(parse_compact_vector_reply(out))
            == EXPECTED_VECF32_LITERAL_FIXTURE
        )
        # It issued the raw GRAPH.RO_QUERY ... --compact command as given.
        client.execute_command.assert_awaited_once_with(
            'GRAPH.RO_QUERY', SOURCE_GRAPH_FIXTURE, cypher, '--compact',
        )

    @pytest.mark.asyncio
    async def test_str_tokens_are_materialized_without_float_coercion(self):
        """Non-bytes (str) tokens pass through via str(v), never float()-coerced."""
        reply = _compact_reply([_VALUE_VECTORF32_TAG, ['0.5', '0.10000000149011611938']])
        client = self._client_returning(reply)

        out = await cross_graph_move._read_compact_vector(
            client, group_id=SOURCE_GRAPH_FIXTURE, cypher='RETURN n.name_embedding',
        )

        assert out == '[0.5, 0.10000000149011611938]'

    @pytest.mark.asyncio
    async def test_non_vectorf32_scalar_type_raises_naming_group_id_and_cypher(self):
        """A non-VECTORF32 scalar-type tag raises ValueError citing group_id + cypher.

        Guards the scalar-type constant (_VALUE_VECTORF32): a wrong tag, or a
        genuinely wrong-typed column, must fail loudly with a diagnostic
        naming both the group_id and the cypher rather than mis-parsing.
        """
        cypher = 'RETURN e.fact_embedding'
        # scalar_type 10 (e.g. VALUE_ARRAY) != VECTORF32 -> type check fires.
        reply = _compact_reply([10, [b'not', b'a', b'vector']])
        client = self._client_returning(reply)

        with pytest.raises(ValueError) as excinfo:
            await cross_graph_move._read_compact_vector(
                client, group_id='know_live', cypher=cypher,
            )

        msg = str(excinfo.value)
        assert 'know_live' in msg
        assert cypher in msg
        assert '10' in msg  # the offending scalar_type is reported

    @pytest.mark.asyncio
    async def test_null_absent_embedding_raises_not_empty_vector(self):
        """A NULL/absent embedding raises rather than yielding an empty vector.

        FalkorDB reports a missing property as a non-VECTORF32 (VALUE_NULL)
        scalar_type with a null value; the type check fires BEFORE the value is
        ever indexed, so a corrupt/embedding-less source row surfaces as a hard
        ValueError instead of a silently-empty vecf32([]) literal.
        """
        reply = _compact_reply([_VALUE_NULL_TAG, None])
        client = self._client_returning(reply)

        with pytest.raises(ValueError):
            await cross_graph_move._read_compact_vector(
                client, group_id=SOURCE_GRAPH_FIXTURE, cypher='RETURN n.name_embedding',
            )

    @pytest.mark.asyncio
    async def test_empty_rows_raises_naming_group_id_and_cypher(self):
        """amend (reviewer_comprehensive robustness): zero rows raises a
        descriptive ValueError, not a bare IndexError.

        A transient race (the source row deleted between an earlier
        existence check and this read) or an unexpected reply shape must
        surface as a hard, diagnosable failure naming group_id/cypher --
        consistent with the non-VECTORF32 path just above -- rather than an
        undiagnosable IndexError from indexing an empty rows list.
        """
        cypher = 'RETURN n.name_embedding'
        reply = ['header', [], 'stats']  # zero rows
        client = self._client_returning(reply)

        with pytest.raises(ValueError) as excinfo:
            await cross_graph_move._read_compact_vector(
                client, group_id=SOURCE_GRAPH_FIXTURE, cypher=cypher,
            )

        msg = str(excinfo.value)
        assert SOURCE_GRAPH_FIXTURE in msg
        assert cypher in msg


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

    @pytest.mark.asyncio
    async def test_edge_silently_matched_nothing_is_not_counted_as_moved(
        self, mock_config, make_backend, make_graph_mock, monkeypatch,
    ):
        """amend (reviewer_comprehensive correctness): the edge CREATE is a
        `MATCH (a),(b) CREATE (a)-[r]->(b)` that silently produces nothing
        when the other endpoint is absent from target_graph. When FalkorDB's
        own relationships_created stat reports 0 for that CREATE, the edge
        must NOT be counted in edges_moved -- and the drop must be visible
        via edges_skipped, from the return value alone.
        """
        backend = make_backend(mock_config)
        source_mock = make_graph_mock()
        source_mock.ro_query = AsyncMock(side_effect=[
            MagicMock(result_set=[NODE_ROW_FIXTURE]),
            MagicMock(result_set=[EDGE_ROW_FIXTURE]),
            MagicMock(result_set=[]),  # no MENTIONS links in this scenario
        ])
        target_mock = make_graph_mock()
        # node CREATE succeeds (unchecked by the code -- not MATCH-gated);
        # the edge CREATE's MATCH finds no endpoint -- the documented
        # silent-skip -- so relationships_created reports 0.
        target_mock.query = AsyncMock(side_effect=[
            MagicMock(),
            MagicMock(relationships_created=0),
        ])
        backend._driver._get_graph = _route_graphs({
            SOURCE_GRAPH_FIXTURE: source_mock,
            TARGET_GRAPH_FIXTURE: target_mock,
        })

        fake_read_compact = AsyncMock(return_value=COMPACT_VECTOR_REPLY_FIXTURE)
        monkeypatch.setattr(cross_graph_move, '_read_compact_vector', fake_read_compact)

        result = await move_entity_across_graphs(
            backend, NODE_UUID_FIXTURE, SOURCE_GRAPH_FIXTURE, TARGET_GRAPH_FIXTURE,
        )

        assert result.edges_moved == 0
        assert result.edges_skipped == 1


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

    @pytest.mark.asyncio
    async def test_mention_silently_matched_nothing_is_not_counted_as_moved(
        self, mock_config, make_backend, make_graph_mock, monkeypatch,
    ):
        """amend (reviewer_comprehensive correctness): same silent-skip class
        as RELATES_TO edges -- when the MENTIONS CREATE's MATCH finds no
        Episodic node in target_graph, relationships_created reports 0.
        mentions_moved must not count it, and mentions_skipped must expose
        the drop.
        """
        backend = make_backend(mock_config)
        source_mock = make_graph_mock()
        source_mock.ro_query = AsyncMock(side_effect=[
            MagicMock(result_set=[NODE_ROW_FIXTURE]),
            MagicMock(result_set=[]),  # no RELATES_TO edges in this scenario
            MagicMock(result_set=[MENTION_ROW_FIXTURE]),
        ])
        target_mock = make_graph_mock()
        # node CREATE succeeds; the mention CREATE's MATCH finds no episode
        # node -- the documented silent-skip -- so relationships_created
        # reports 0.
        target_mock.query = AsyncMock(side_effect=[
            MagicMock(),
            MagicMock(relationships_created=0),
        ])
        backend._driver._get_graph = _route_graphs({
            SOURCE_GRAPH_FIXTURE: source_mock,
            TARGET_GRAPH_FIXTURE: target_mock,
        })

        fake_read_compact = AsyncMock(return_value=COMPACT_VECTOR_REPLY_FIXTURE)
        monkeypatch.setattr(cross_graph_move, '_read_compact_vector', fake_read_compact)

        result = await move_entity_across_graphs(
            backend, NODE_UUID_FIXTURE, SOURCE_GRAPH_FIXTURE, TARGET_GRAPH_FIXTURE,
        )

        assert result.mentions_moved == 0
        assert result.mentions_skipped == 1


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


# ---------------------------------------------------------------------------
# amend (reviewer_comprehensive correctness): move_entity_across_graphs --
# foreign-duplicate guard on the target-presence probe
# ---------------------------------------------------------------------------

class TestMoveEntityAcrossGraphsForeignDuplicateGuard:
    """S5 resume-vs-duplicate: a uuid present in BOTH graphs is only treated
    as a partially-completed move (resume) when target's name/created_at
    match source's; a divergence raises instead of silently overwriting the
    target's topology and deleting the source copy.
    """

    @pytest.mark.asyncio
    async def test_diverging_target_copy_raises_instead_of_silent_resume(
        self, mock_config, make_backend, make_graph_mock, monkeypatch,
    ):
        """uuid present in both graphs, but the target copy's name/created_at
        diverge from source's -- a genuine foreign duplicate (the scenario
        merge_foreign_duplicate exists to handle), not a resumed move. Must
        raise ForeignDuplicateSuspectedError before any read of edges/
        mentions or any mutation on either graph.
        """
        backend = make_backend(mock_config)
        source_mock = make_graph_mock()
        source_mock.ro_query = AsyncMock(return_value=MagicMock(result_set=[NODE_ROW_FIXTURE]))

        target_mock = make_graph_mock()
        # Present in target, but with a DIFFERENT name/created_at than the
        # source copy -- a divergent, genuine duplicate rather than a
        # partial-move remnant of the SAME node.
        target_mock.ro_query = AsyncMock(
            return_value=MagicMock(
                result_set=[[NODE_UUID_FIXTURE, 'Bob', '2020-06-01T00:00:00+00:00']]
            )
        )

        backend._driver._get_graph = _route_graphs({
            SOURCE_GRAPH_FIXTURE: source_mock,
            TARGET_GRAPH_FIXTURE: target_mock,
        })

        fake_read_compact = AsyncMock(return_value=COMPACT_VECTOR_REPLY_FIXTURE)
        monkeypatch.setattr(cross_graph_move, '_read_compact_vector', fake_read_compact)

        with pytest.raises(ForeignDuplicateSuspectedError):
            await move_entity_across_graphs(
                backend, NODE_UUID_FIXTURE, SOURCE_GRAPH_FIXTURE, TARGET_GRAPH_FIXTURE,
            )

        # No mutation on either graph, and the RELATES_TO/MENTIONS reads and
        # the embedding transport are never reached -- the guard fires
        # before any of that.
        target_mock.query.assert_not_awaited()
        source_mock.query.assert_not_awaited()
        fake_read_compact.assert_not_awaited()
        assert source_mock.ro_query.await_count == 1

    @pytest.mark.asyncio
    async def test_matching_target_copy_proceeds_as_genuine_resume(
        self, mock_config, make_backend, make_graph_mock, monkeypatch,
    ):
        """uuid present in both graphs, target copy's name/created_at MATCH
        source's -- a genuine resume (the earlier attempt crashed strictly
        between the node CREATE and the source DETACH DELETE). Proceeds
        without raising: the node CREATE is skipped (already there), and
        the source DETACH DELETE still runs.
        """
        backend = make_backend(mock_config)
        source_mock = make_graph_mock()
        source_mock.ro_query = AsyncMock(side_effect=[
            MagicMock(result_set=[NODE_ROW_FIXTURE]),
            MagicMock(result_set=[]),  # no RELATES_TO edges
            MagicMock(result_set=[]),  # no MENTIONS links
        ])

        target_mock = make_graph_mock()
        # Same uuid/name/created_at as NODE_ROW_FIXTURE -- a genuine resume.
        target_mock.ro_query = AsyncMock(
            return_value=MagicMock(
                result_set=[[NODE_UUID_FIXTURE, 'Alice', '2026-01-01T00:00:00+00:00']]
            )
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

        # node CREATE is skipped (already present) -- target.query is never
        # awaited (no edges/mentions in this scenario either) -- but the
        # source DETACH DELETE still runs.
        target_mock.query.assert_not_awaited()
        source_mock.query.assert_awaited_once()
        delete_cypher = extract_cypher(source_mock.query.call_args)
        assert 'DETACH DELETE' in delete_cypher

        assert isinstance(result, MoveResult)
        assert result.already_moved is False


# ---------------------------------------------------------------------------
# step-15: move_entity_across_graphs -- rewrite_group_id
# ---------------------------------------------------------------------------

class TestMoveEntityAcrossGraphsRewriteGroupId:
    """S5 rewrite_group_id: rewrites (or preserves) group_id on node/edge/mention."""

    @pytest.mark.asyncio
    async def test_rewrite_group_id_applies_to_node_edge_and_mention(
        self, mock_config, make_backend, make_graph_mock, monkeypatch,
    ):
        """When rewrite_group_id is given, the recreated node's CREATE sets
        group_id to the rewritten value, and the recreated edge/mention
        (both of which store a group_id) carry the SAME rewritten value --
        never the source's original group_id.
        """
        backend = make_backend(mock_config)
        source_mock = make_graph_mock()
        source_mock.ro_query = AsyncMock(side_effect=[
            MagicMock(result_set=[NODE_ROW_FIXTURE]),
            MagicMock(result_set=[EDGE_ROW_FIXTURE]),
            MagicMock(result_set=[MENTION_ROW_FIXTURE]),
        ])
        target_mock = make_graph_mock()
        backend._driver._get_graph = _route_graphs({
            SOURCE_GRAPH_FIXTURE: source_mock,
            TARGET_GRAPH_FIXTURE: target_mock,
        })

        fake_read_compact = AsyncMock(return_value=COMPACT_VECTOR_REPLY_FIXTURE)
        monkeypatch.setattr(cross_graph_move, '_read_compact_vector', fake_read_compact)

        rewritten_group_id = 'know_live'
        await move_entity_across_graphs(
            backend, NODE_UUID_FIXTURE, SOURCE_GRAPH_FIXTURE, TARGET_GRAPH_FIXTURE,
            rewrite_group_id=rewritten_group_id,
        )

        assert target_mock.query.await_count == 3
        node_params = extract_params(target_mock.query.call_args_list[0])
        edge_params = extract_params(target_mock.query.call_args_list[1])
        mention_params = extract_params(target_mock.query.call_args_list[2])

        assert node_params.get('group_id') == rewritten_group_id
        assert edge_params.get('group_id') == rewritten_group_id
        assert mention_params.get('group_id') == rewritten_group_id
        # never the source's original group_id
        assert SOURCE_GRAPH_FIXTURE not in (
            node_params.get('group_id'), edge_params.get('group_id'), mention_params.get('group_id'),
        )

    @pytest.mark.asyncio
    async def test_no_rewrite_preserves_original_group_id(
        self, mock_config, make_backend, make_graph_mock, monkeypatch,
    ):
        """When rewrite_group_id is None (the Phase-1 default), the node's
        original group_id property is preserved unchanged on the recreated
        node/edge/mention.
        """
        backend = make_backend(mock_config)
        source_mock = make_graph_mock()
        source_mock.ro_query = AsyncMock(side_effect=[
            MagicMock(result_set=[NODE_ROW_FIXTURE]),
            MagicMock(result_set=[EDGE_ROW_FIXTURE]),
            MagicMock(result_set=[MENTION_ROW_FIXTURE]),
        ])
        target_mock = make_graph_mock()
        backend._driver._get_graph = _route_graphs({
            SOURCE_GRAPH_FIXTURE: source_mock,
            TARGET_GRAPH_FIXTURE: target_mock,
        })

        fake_read_compact = AsyncMock(return_value=COMPACT_VECTOR_REPLY_FIXTURE)
        monkeypatch.setattr(cross_graph_move, '_read_compact_vector', fake_read_compact)

        await move_entity_across_graphs(
            backend, NODE_UUID_FIXTURE, SOURCE_GRAPH_FIXTURE, TARGET_GRAPH_FIXTURE,
        )

        assert target_mock.query.await_count == 3
        node_params = extract_params(target_mock.query.call_args_list[0])
        edge_params = extract_params(target_mock.query.call_args_list[1])
        mention_params = extract_params(target_mock.query.call_args_list[2])

        assert node_params.get('group_id') == SOURCE_GRAPH_FIXTURE
        assert edge_params.get('group_id') == SOURCE_GRAPH_FIXTURE
        assert mention_params.get('group_id') == SOURCE_GRAPH_FIXTURE


# ---------------------------------------------------------------------------
# step-17: merge_foreign_duplicate -- unique-edge classification (pure)
# ---------------------------------------------------------------------------

class TestClassifyUniqueWrongEdges:
    """classify_unique_wrong_edges(home, wrong) -> wrong-only edge uuids (set-diff)."""

    def test_returns_edges_unique_to_wrong_copy(self):
        """Given home={a,b} and wrong={b,c,d}, unique(wrong) == {c,d}."""
        home_edge_uuids = {'a', 'b'}
        wrong_edge_uuids = {'b', 'c', 'd'}
        assert classify_unique_wrong_edges(home_edge_uuids, wrong_edge_uuids) == {'c', 'd'}

    def test_empty_when_wrong_copy_has_no_edges(self):
        """An empty wrong-copy edge set classifies as no unique edges."""
        assert classify_unique_wrong_edges({'a', 'b'}, set()) == set()

    def test_empty_when_all_wrong_edges_already_on_home(self):
        """Every wrong-copy edge already present on home classifies as empty."""
        assert classify_unique_wrong_edges({'a', 'b', 'c'}, {'a', 'b'}) == set()


# ---------------------------------------------------------------------------
# step-19: merge_foreign_duplicate -- recreate-then-delete, no edge lost
# ---------------------------------------------------------------------------

HOME_GRAPH_FIXTURE = SOURCE_GRAPH_FIXTURE
WRONG_GRAPH_FIXTURE = TARGET_GRAPH_FIXTURE

SHARED_EDGE_UUID_FIXTURE = 'edge-shared-9999'
HOME_ONLY_EDGE_UUID_FIXTURE = 'edge-home-only-8888'
UNIQUE_WRONG_EDGE_UUID_FIXTURE = 'edge-unique-wrong-7777'

# Full-property row for the edge shared by both copies (already accounted
# for on home) -- must NOT be recreated.
SHARED_EDGE_ROW_FIXTURE = [
    SHARED_EDGE_UUID_FIXTURE, 'is_related_to', 'Alice is related to Bob.',
    '2026-01-01T00:00:00+00:00', None, '2026-01-01T00:00:00+00:00',
    WRONG_GRAPH_FIXTURE, ['episode-uuid-1'],
    NODE_UUID_FIXTURE, OTHER_NODE_UUID_FIXTURE,
]
# Full-property row for the wrong-copy's edge that's UNIQUE to it (absent
# from home) -- must be recreated on home with a byte-exact fact_embedding.
UNIQUE_WRONG_EDGE_ROW_FIXTURE = [
    UNIQUE_WRONG_EDGE_UUID_FIXTURE, 'is_related_to', 'Alice is related to Carol.',
    '2026-01-02T00:00:00+00:00', None, '2026-01-02T00:00:00+00:00',
    WRONG_GRAPH_FIXTURE, ['episode-uuid-2'],
    NODE_UUID_FIXTURE, OTHER_NODE_UUID_FIXTURE,
]


class TestMergeForeignDuplicate:
    """S6 merge: recreate the wrong-copy's unique edges on home, then delete wrong."""

    @pytest.mark.asyncio
    async def test_recreates_unique_edges_then_deletes_wrong_copy_no_edge_lost(
        self, mock_config, make_backend, make_graph_mock, monkeypatch,
    ):
        """home has 2 edges (one shared with wrong, one home-only); wrong has
        2 edges (the shared one + one unique to it). Only the unique edge
        should be recreated on home -- with a byte-exact fact_embedding --
        and the wrong-graph-copy DETACH DELETE must run strictly after that
        recreate. Post: home_edge_count_after == home_edge_count_before + 1,
        matching len(unique(wrong)); the shared edge is counted once, not
        duplicated.
        """
        backend = make_backend(mock_config)
        call_order: list[tuple[str, str, dict]] = []

        wrong_mock = make_graph_mock()
        wrong_mock.ro_query = AsyncMock(
            return_value=MagicMock(
                result_set=[SHARED_EDGE_ROW_FIXTURE, UNIQUE_WRONG_EDGE_ROW_FIXTURE]
            )
        )

        async def _record_wrong_query(cypher, params=None):
            call_order.append(('wrong', cypher, params or {}))
            return MagicMock()

        wrong_mock.query = AsyncMock(side_effect=_record_wrong_query)

        home_mock = make_graph_mock()
        # Independent before/after re-reads (not a single static
        # return_value): before-read == 2 edges (shared + home-only);
        # after-read == 3 edges, reflecting the landed unique-edge recreate.
        # home_edge_count_after must come from the after-read, not from
        # arithmetic on home_edge_count_before + edges_recreated.
        home_mock.ro_query = AsyncMock(
            side_effect=[
                MagicMock(
                    result_set=[[SHARED_EDGE_UUID_FIXTURE], [HOME_ONLY_EDGE_UUID_FIXTURE]]
                ),
                MagicMock(
                    result_set=[
                        [SHARED_EDGE_UUID_FIXTURE],
                        [HOME_ONLY_EDGE_UUID_FIXTURE],
                        [UNIQUE_WRONG_EDGE_UUID_FIXTURE],
                    ]
                ),
            ]
        )

        async def _record_home_query(cypher, params=None):
            call_order.append(('home', cypher, params or {}))
            return MagicMock()

        home_mock.query = AsyncMock(side_effect=_record_home_query)

        backend._driver._get_graph = _route_graphs({
            WRONG_GRAPH_FIXTURE: wrong_mock,
            HOME_GRAPH_FIXTURE: home_mock,
        })

        fake_read_compact = AsyncMock(return_value=COMPACT_VECTOR_REPLY_FIXTURE)
        monkeypatch.setattr(cross_graph_move, '_read_compact_vector', fake_read_compact)

        result = await merge_foreign_duplicate(
            backend, NODE_UUID_FIXTURE, WRONG_GRAPH_FIXTURE, HOME_GRAPH_FIXTURE,
        )

        # (a) exactly one mutating home call (the unique edge's CREATE) and
        # one mutating wrong call (the DETACH DELETE) -- the shared edge is
        # NOT recreated.
        assert len(call_order) == 2
        create_role, create_cypher, create_params = call_order[0]
        assert create_role == 'home'
        assert 'CREATE' in create_cypher
        assert 'RELATES_TO' in create_cypher
        assert EXPECTED_VECF32_LITERAL_FIXTURE in create_cypher
        assert create_params.get('edge_uuid') == UNIQUE_WRONG_EDGE_UUID_FIXTURE
        assert create_params.get('src_uuid') == NODE_UUID_FIXTURE
        assert create_params.get('dst_uuid') == OTHER_NODE_UUID_FIXTURE
        assert create_params.get('fact') == 'Alice is related to Carol.'

        fake_read_compact.assert_awaited_once()
        assert fake_read_compact.call_args.kwargs.get('group_id') == WRONG_GRAPH_FIXTURE
        assert UNIQUE_WRONG_EDGE_UUID_FIXTURE in fake_read_compact.call_args.kwargs.get('cypher', '')

        # (b) the wrong-graph-copy DETACH DELETE runs strictly AFTER the
        # recreate, and is scoped to wrong_graph (never home_graph).
        delete_role, delete_cypher, delete_params = call_order[1]
        assert delete_role == 'wrong'
        assert 'DETACH DELETE' in delete_cypher
        assert delete_params.get('uuid') == NODE_UUID_FIXTURE

        # (c) no edge lost, none double-counted -- home_edge_count_after is
        # sourced from the independent after-read (3), not the arithmetic
        # home_edge_count_before + edges_recreated (which would also be 3
        # here, but for the wrong reason -- see the dedicated mismatch test
        # below that tells the two apart).
        assert isinstance(result, MergeResult)
        assert result.home_edge_count_before == 2
        assert result.edges_recreated == 1
        assert result.home_edge_count_after == 3
        assert result.unique_wrong_edge_uuids == {UNIQUE_WRONG_EDGE_UUID_FIXTURE}

    @pytest.mark.asyncio
    async def test_home_edge_count_after_is_independent_reread_not_arithmetic(
        self, mock_config, make_backend, make_graph_mock, monkeypatch,
    ):
        """home_edge_count_after must be an INDEPENDENT re-read of home's
        edge set taken after the recreate loop, not the arithmetic
        ``home_edge_count_before + edges_recreated``.

        Here the unique edge's CREATE silently matches no endpoint (the
        documented silent-skip when the other endpoint is absent from
        home), so a re-read of home after the recreate loop still shows
        the original 2 edges. A re-read impl reports
        ``home_edge_count_after == 2``; the arithmetic tautology
        (2 + 1 == 3) would mask the loss instead of surfacing it. This
        assertion fails against the current arithmetic impl (which yields
        3) and passes only for a genuine re-read impl.
        """
        backend = make_backend(mock_config)

        wrong_mock = make_graph_mock()
        wrong_mock.ro_query = AsyncMock(
            return_value=MagicMock(
                result_set=[SHARED_EDGE_ROW_FIXTURE, UNIQUE_WRONG_EDGE_ROW_FIXTURE]
            )
        )
        wrong_mock.query = AsyncMock(return_value=MagicMock())

        home_mock = make_graph_mock()
        # before-read == 2 edges; after-read STILL == 2 edges -- the unique
        # edge's CREATE matched no endpoint, so home is unchanged.
        unchanged_result_set = [
            [SHARED_EDGE_UUID_FIXTURE], [HOME_ONLY_EDGE_UUID_FIXTURE],
        ]
        home_mock.ro_query = AsyncMock(
            side_effect=[
                MagicMock(result_set=list(unchanged_result_set)),
                MagicMock(result_set=list(unchanged_result_set)),
            ]
        )
        home_mock.query = AsyncMock(return_value=MagicMock())

        backend._driver._get_graph = _route_graphs({
            WRONG_GRAPH_FIXTURE: wrong_mock,
            HOME_GRAPH_FIXTURE: home_mock,
        })

        fake_read_compact = AsyncMock(return_value=COMPACT_VECTOR_REPLY_FIXTURE)
        monkeypatch.setattr(cross_graph_move, '_read_compact_vector', fake_read_compact)

        result = await merge_foreign_duplicate(
            backend, NODE_UUID_FIXTURE, WRONG_GRAPH_FIXTURE, HOME_GRAPH_FIXTURE,
        )

        assert result.home_edge_count_before == 2
        assert result.edges_recreated == 1
        assert result.home_edge_count_after == 2
        assert result.home_edge_count_after != (
            result.home_edge_count_before + result.edges_recreated
        )
