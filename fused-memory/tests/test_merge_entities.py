"""Tests for merge_entities across backends, service, and MCP tool.

Covers:
- GraphitiBackend.redirect_node_edges()
- GraphitiBackend.delete_entity_node()
- GraphitiBackend.merge_entities()
- MemoryService.merge_entities()
- MCP tool merge_entities in tools.py
- DISALLOW_MEMORY_WRITES list in cli_stage_runner.py
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from _fm_helpers import assert_ro_query_only, extract_cypher, extract_params, make_rebuild_detail

from fused_memory.backends.graphiti_client import GraphitiBackend, NodeNotFoundError

# ---------------------------------------------------------------------------
# step-1: GraphitiBackend.redirect_node_edges
# ---------------------------------------------------------------------------

class TestRedirectNodeEdges:
    """GraphitiBackend.redirect_node_edges(deprecated_uuid, surviving_uuid) redirects edges."""

    @pytest.mark.asyncio
    async def test_redirects_outgoing_edges(self, mock_config, make_backend, make_graph_mock):
        """Creates new edges from surviving→target and deletes old edges from deprecated→target."""
        backend = make_backend(mock_config)
        # Simulate query calls: first (inter-node delete) returns empty,
        # second (outgoing redirect) returns empty, third (incoming) returns empty
        graph = make_graph_mock([])
        backend._driver._get_graph = MagicMock(return_value=graph)
        result = await backend.redirect_node_edges('dep-uuid', 'sur-uuid', group_id='test')
        # Should have called graph.query at least 3 times (inter-node, outgoing, incoming)
        assert graph.query.await_count >= 3
        assert 'outgoing_redirected' in result
        assert 'incoming_redirected' in result
        assert 'inter_node_deleted' in result

    @pytest.mark.asyncio
    async def test_deletes_inter_node_edges(self, mock_config, make_backend, make_graph_mock):
        """Edges between deprecated and surviving nodes are removed before redirect."""
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        backend._driver._get_graph = MagicMock(return_value=graph)
        dep_uuid = 'dep-abc'
        sur_uuid = 'sur-xyz'
        await backend.redirect_node_edges(dep_uuid, sur_uuid, group_id='test')
        # First query call should target inter-node edges
        first_call = graph.query.call_args_list[0]
        params = extract_params(first_call)
        # Both UUIDs should appear in params
        assert dep_uuid in params.values() or any(
            dep_uuid in str(v) for v in params.values()
        )

    @pytest.mark.asyncio
    async def test_returns_count_dict(self, mock_config, make_backend, make_graph_mock):
        """Returns dict with outgoing_redirected, incoming_redirected, inter_node_deleted counts."""
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        backend._driver._get_graph = MagicMock(return_value=graph)
        result = await backend.redirect_node_edges('dep-uuid', 'sur-uuid', group_id='test')
        assert isinstance(result['outgoing_redirected'], int)
        assert isinstance(result['incoming_redirected'], int)
        assert isinstance(result['inter_node_deleted'], int)

    @pytest.mark.asyncio
    async def test_raises_when_not_initialized(self, mock_config):
        """Raises RuntimeError when client is not initialized."""
        backend = GraphitiBackend(mock_config)  # client is None
        with pytest.raises(RuntimeError, match='not initialized'):
            await backend.redirect_node_edges('dep-uuid', 'sur-uuid', group_id='test')

    @pytest.mark.asyncio
    async def test_handles_empty_edge_sets(self, mock_config, make_backend, make_graph_mock):
        """When no edges exist, returns zeros for all counts."""
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        backend._driver._get_graph = MagicMock(return_value=graph)
        result = await backend.redirect_node_edges('dep-uuid', 'sur-uuid', group_id='test')
        assert result['outgoing_redirected'] == 0
        assert result['incoming_redirected'] == 0
        assert result['inter_node_deleted'] == 0


# ---------------------------------------------------------------------------
# step-3: GraphitiBackend.delete_entity_node
# ---------------------------------------------------------------------------

class TestDeleteEntityNode:
    """GraphitiBackend.delete_entity_node(uuid) removes an entity node."""

    @pytest.mark.asyncio
    async def test_executes_detach_delete(self, mock_config, make_backend, make_graph_mock):
        """Issues DETACH DELETE Cypher for the given UUID."""
        backend = make_backend(mock_config)
        # Pre-check uses ro_query; DETACH DELETE uses query
        graph = make_graph_mock(ro_rows=[['NodeName', 'some summary']], q_rows=[])
        backend._driver._get_graph = MagicMock(return_value=graph)
        await backend.delete_entity_node('node-uuid-1', group_id='test')
        assert graph.ro_query.await_count == 1
        assert graph.query.await_count == 1
        # The single query call should contain DETACH DELETE
        cypher = extract_cypher(graph.query.call_args)
        assert 'DETACH DELETE' in cypher

    @pytest.mark.asyncio
    async def test_passes_uuid_as_param(self, mock_config, make_backend, make_graph_mock):
        """Passes node UUID as parameter to both the pre-check and delete Cypher calls."""
        backend = make_backend(mock_config)
        node_uuid = 'my-node-uuid'
        graph = make_graph_mock(ro_rows=[['Name', '']], q_rows=[])
        backend._driver._get_graph = MagicMock(return_value=graph)
        await backend.delete_entity_node(node_uuid, group_id='test')
        # Pre-check (ro_query) should pass uuid param
        ro_params = extract_params(graph.ro_query.call_args)
        assert ro_params.get('uuid') == node_uuid
        # Delete (query) should also pass uuid param
        q_params = extract_params(graph.query.call_args)
        assert q_params.get('uuid') == node_uuid
        # Exactly-once routing contract: each slot called exactly once
        graph.ro_query.assert_awaited_once()
        graph.query.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_raises_node_not_found_when_missing(self, mock_config, make_backend, make_graph_mock):
        """Raises NodeNotFoundError when pre-check returns no rows."""
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        backend._driver._get_graph = MagicMock(return_value=graph)
        with pytest.raises(NodeNotFoundError):
            await backend.delete_entity_node('missing-uuid', group_id='test')
        graph.ro_query.assert_awaited_once()  # pre-check ran exactly once
        graph.query.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_raises_when_not_initialized(self, mock_config):
        """Raises RuntimeError when client is not initialized."""
        backend = GraphitiBackend(mock_config)  # client is None
        with pytest.raises(RuntimeError, match='not initialized'):
            await backend.delete_entity_node('node-uuid-1', group_id='test')


# ---------------------------------------------------------------------------
# step-5: GraphitiBackend.merge_entities
# ---------------------------------------------------------------------------

class TestMergeEntities:
    """GraphitiBackend.merge_entities(deprecated_uuid, surviving_uuid) merges two nodes."""

    @pytest.fixture
    def backend_with_mocks(self, mock_config, make_backend):
        """GraphitiBackend with sub-methods mocked for orchestration testing."""
        backend = make_backend(mock_config)
        backend.get_node_text = AsyncMock(side_effect=[
            ('DeprecatedName', 'old dep summary'),   # first call: deprecated node
            ('SurvivingName', 'old sur summary'),     # second call: surviving node
        ])
        backend.redirect_node_edges = AsyncMock(return_value={
            'outgoing_redirected': 2,
            'incoming_redirected': 1,
            'inter_node_deleted': 0,
        })
        backend.delete_entity_node = AsyncMock()
        backend.dedup_valid_edges_for_node = AsyncMock(return_value=0)
        backend.refresh_entity_summary = AsyncMock(return_value=make_rebuild_detail(
            'sur-uuid', 'SurvivingName',
            old_summary='old sur summary',
            new_summary='SurvivingName knows Foo\nDeprecatedName knows Bar',
            edge_count=3,
        ))
        return backend

    @pytest.mark.asyncio
    async def test_validates_both_nodes_exist(self, backend_with_mocks):
        """Calls get_node_text for both UUIDs to validate existence."""
        backend = backend_with_mocks
        await backend.merge_entities('dep-uuid', 'sur-uuid', group_id='test')
        assert backend.get_node_text.await_count == 2
        calls = [c[0][0] for c in backend.get_node_text.call_args_list]
        assert 'dep-uuid' in calls
        assert 'sur-uuid' in calls

    @pytest.mark.asyncio
    async def test_raises_node_not_found_for_deprecated(self, mock_config, make_backend):
        """Raises NodeNotFoundError when deprecated node doesn't exist."""
        backend = make_backend(mock_config)
        backend.get_node_text = AsyncMock(side_effect=NodeNotFoundError('not found'))
        with pytest.raises(NodeNotFoundError):
            await backend.merge_entities('missing-dep', 'sur-uuid', group_id='test')

    @pytest.mark.asyncio
    async def test_raises_node_not_found_for_surviving(self, mock_config, make_backend):
        """Raises NodeNotFoundError when surviving node doesn't exist."""
        backend = make_backend(mock_config)
        backend.get_node_text = AsyncMock(side_effect=[
            ('DepName', ''),           # deprecated exists
            NodeNotFoundError('not found'),  # surviving missing
        ])
        with pytest.raises(NodeNotFoundError):
            await backend.merge_entities('dep-uuid', 'missing-sur', group_id='test')

    @pytest.mark.asyncio
    async def test_calls_in_correct_order(self, backend_with_mocks):
        """Calls redirect_node_edges, then delete_entity_node, then
        dedup_valid_edges_for_node, then refresh_entity_summary."""
        backend = backend_with_mocks
        call_order = []
        backend.redirect_node_edges = AsyncMock(
            side_effect=lambda *a, **kw: call_order.append('redirect') or {
                'outgoing_redirected': 0, 'incoming_redirected': 0, 'inter_node_deleted': 0
            }
        )
        backend.delete_entity_node = AsyncMock(
            side_effect=lambda *a, **kw: call_order.append('delete')
        )
        backend.dedup_valid_edges_for_node = AsyncMock(
            side_effect=lambda *a, **kw: call_order.append('dedup') or 0
        )
        backend.refresh_entity_summary = AsyncMock(
            side_effect=lambda *a, **kw: call_order.append('refresh') or make_rebuild_detail('sur-uuid', 'S')
        )
        await backend.merge_entities('dep-uuid', 'sur-uuid', group_id='test')
        assert call_order == ['redirect', 'delete', 'dedup', 'refresh']

    @pytest.mark.asyncio
    async def test_dedups_surviving_node_edges(self, backend_with_mocks):
        """Awaits dedup_valid_edges_for_node once with the surviving uuid and group_id."""
        backend = backend_with_mocks
        await backend.merge_entities('dep-uuid', 'sur-uuid', group_id='test')
        backend.dedup_valid_edges_for_node.assert_awaited_once_with('sur-uuid', group_id='test')

    @pytest.mark.asyncio
    async def test_audit_dict_reports_duplicate_edges_removed(self, backend_with_mocks):
        """Audit dict surfaces the duplicate_edges_removed count from dedup_valid_edges_for_node."""
        backend = backend_with_mocks
        backend.dedup_valid_edges_for_node = AsyncMock(return_value=2)
        result = await backend.merge_entities('dep-uuid', 'sur-uuid', group_id='test')
        assert result['duplicate_edges_removed'] == 2

    @pytest.mark.asyncio
    async def test_returns_audit_dict(self, backend_with_mocks):
        """Returns audit dict with expected keys."""
        backend = backend_with_mocks
        result = await backend.merge_entities('dep-uuid', 'sur-uuid', group_id='test')
        assert result['surviving_uuid'] == 'sur-uuid'
        assert result['deprecated_uuid'] == 'dep-uuid'
        assert result['surviving_name'] == 'SurvivingName'
        assert result['deprecated_name'] == 'DeprecatedName'
        assert 'edges_redirected' in result
        assert isinstance(result['edges_redirected'], dict)
        assert 'surviving_summary' in result

    @pytest.mark.asyncio
    async def test_merge_with_zero_edges_succeeds(self, mock_config, make_backend):
        """When deprecated node has zero edges, merge still succeeds."""
        backend = make_backend(mock_config)
        backend.get_node_text = AsyncMock(side_effect=[
            ('DepName', ''),
            ('SurName', 'existing summary'),
        ])
        backend.redirect_node_edges = AsyncMock(return_value={
            'outgoing_redirected': 0,
            'incoming_redirected': 0,
            'inter_node_deleted': 0,
        })
        backend.delete_entity_node = AsyncMock()
        backend.dedup_valid_edges_for_node = AsyncMock(return_value=0)
        backend.refresh_entity_summary = AsyncMock(return_value=make_rebuild_detail(
            'sur-uuid', 'SurName',
            old_summary='existing summary', new_summary='existing summary', edge_count=1,
        ))
        result = await backend.merge_entities('dep-uuid', 'sur-uuid', group_id='test')
        backend.delete_entity_node.assert_awaited_once_with('dep-uuid', group_id='test')
        backend.refresh_entity_summary.assert_awaited_once_with('sur-uuid', group_id='test')
        assert result['surviving_uuid'] == 'sur-uuid'


# ---------------------------------------------------------------------------
# task 2118 step-1/3: GraphitiBackend._duplicate_edge_uuids (pure helper)
# ---------------------------------------------------------------------------

class TestDuplicateEdgeUuids:
    """GraphitiBackend._duplicate_edge_uuids(rows) groups valid-edge rows shaped
    [neighbor_uuid, edge_uuid, fact, valid_at] by (neighbor_uuid, normalized fact,
    valid_at) and returns the non-survivor edge uuids to delete, keeping the
    lexicographically-lowest uuid per group as the survivor."""

    _FACT = (
        'The fix requires adding the _finalizing_head term to '
        '_inflight_speculative_count().'
    )
    _VALID_AT = '2026-07-06T00:00:00+00:00'

    def test_returns_non_survivor_uuids_for_duplicate_group(self):
        """Three edges from A to the same neighbor B, identical fact and
        valid_at (mirrors the real aec7014f/735efe05 case) -> the two higher
        uuids are returned; the lowest uuid ('e-1') survives."""
        rows = [
            ['B-uuid', 'e-3', self._FACT, self._VALID_AT],
            ['B-uuid', 'e-1', self._FACT, self._VALID_AT],
            ['B-uuid', 'e-2', self._FACT, self._VALID_AT],
        ]
        result = GraphitiBackend._duplicate_edge_uuids(rows)
        assert result == ['e-2', 'e-3']

    def test_single_edge_returns_empty(self):
        """A lone edge has nothing to dedup against."""
        rows = [['B-uuid', 'e-1', self._FACT, self._VALID_AT]]
        assert GraphitiBackend._duplicate_edge_uuids(rows) == []

    def test_all_distinct_returns_empty(self):
        """Edges with distinct neighbor/fact/valid_at are all preserved."""
        rows = [
            ['B-uuid', 'e-1', 'fact one', '2026-07-06T00:00:00+00:00'],
            ['C-uuid', 'e-2', 'fact two', '2026-07-06T01:00:00+00:00'],
        ]
        assert GraphitiBackend._duplicate_edge_uuids(rows) == []

    def test_different_valid_at_preserved(self):
        """Same neighbor and fact but different valid_at are temporally
        distinct assertions — not duplicates."""
        rows = [
            ['B-uuid', 'e-1', 'Auth depends on Redis.', '2026-07-06T00:00:00+00:00'],
            ['B-uuid', 'e-2', 'Auth depends on Redis.', '2026-07-06T01:00:00+00:00'],
        ]
        assert GraphitiBackend._duplicate_edge_uuids(rows) == []

    def test_different_neighbor_preserved(self):
        """Same fact and valid_at but different neighbor are a distinct pair —
        not duplicates."""
        rows = [
            ['B-uuid', 'e-1', 'Auth depends on Redis.', self._VALID_AT],
            ['C-uuid', 'e-2', 'Auth depends on Redis.', self._VALID_AT],
        ]
        assert GraphitiBackend._duplicate_edge_uuids(rows) == []

    def test_fact_normalized_for_case_and_whitespace(self):
        """Facts differing only by case/whitespace are treated as duplicates
        (mirrors MemoryService._normalize_fact / graphiti-core's
        _normalize_string_exact). Fails until fact normalization is applied
        to the grouping key."""
        rows = [
            ['B-uuid', 'e-2', 'Auth depends on Redis', self._VALID_AT],
            ['B-uuid', 'e-1', '  auth  depends  on  redis  ', self._VALID_AT],
        ]
        assert GraphitiBackend._duplicate_edge_uuids(rows) == ['e-2']

    def test_self_loop_double_match_returns_empty(self):
        """The undirected Cypher MATCH in dedup_valid_edges_for_node
        double-matches a self-loop edge (A)-[e]-(A), surfacing the same
        (neighbor_uuid, edge_uuid, fact, valid_at) row twice. The edge-uuid
        dedup guard (rows deduplicated by edge_uuid before grouping) must
        fold these back into a single row so the lone self-loop edge is not
        mistaken for a duplicate pair and returned for deletion. A
        regression that removed the edge-uuid dedup would risk exactly that
        for a self-loop with no other edges to disambiguate it."""
        rows = [
            ['A-uuid', 'e-1', self._FACT, self._VALID_AT],
            ['A-uuid', 'e-1', self._FACT, self._VALID_AT],
        ]
        assert GraphitiBackend._duplicate_edge_uuids(rows) == []

    def test_collapses_reverse_direction_duplicate_by_design(self):
        """The (neighbor_uuid, fact, valid_at) grouping key carries no edge
        direction, so a forward edge A->B and a reverse edge B->A sharing an
        identical fact and valid_at collapse into a single survivor exactly
        like a same-direction parallel duplicate would. This is deliberate,
        not an oversight (see task 2118's design decision to use an
        undirected MATCH): the merge scenario this helper targets only ever
        mints same-direction duplicates, and the undirected Cypher query in
        dedup_valid_edges_for_node reports the neighbor via `m.uuid`
        regardless of which side the edge originates from, so direction
        isn't even available to key on. Pinned here so a future change to
        add direction-sensitivity is a conscious, test-visible decision
        rather than a silent behavior change."""
        rows = [
            # Forward edge A->B, as seen from A's undirected query (m=B).
            ['B-uuid', 'e-2', 'Auth depends on Redis.', self._VALID_AT],
            # Reverse edge B->A, also surfaced with neighbor_uuid='B-uuid'
            # from A's perspective since the MATCH is undirected.
            ['B-uuid', 'e-1', 'Auth depends on Redis.', self._VALID_AT],
        ]
        assert GraphitiBackend._duplicate_edge_uuids(rows) == ['e-2']


# ---------------------------------------------------------------------------
# task 2118 step-5/6: GraphitiBackend.dedup_valid_edges_for_node
# ---------------------------------------------------------------------------

class TestDedupValidEdgesForNode:
    """GraphitiBackend.dedup_valid_edges_for_node(node_uuid, *, group_id) reads
    the node's valid incident edges and deletes non-survivor duplicates via
    bulk_remove_edges."""

    @pytest.mark.asyncio
    async def test_removes_duplicate_edges_and_returns_count(
        self, mock_config, make_backend, make_graph_mock,
    ):
        """One duplicate group (uuids 'e-2' and 'e-1') collapses to the
        non-survivor 'e-2'; an unrelated distinct edge is left alone."""
        backend = make_backend(mock_config)
        valid_at = '2026-07-06T00:00:00+00:00'
        rows = [
            ['B-uuid', 'e-2', 'Some fact.', valid_at],
            ['B-uuid', 'e-1', 'Some fact.', valid_at],
            ['C-uuid', 'e-3', 'Unrelated fact.', valid_at],
        ]
        graph = make_graph_mock(rows)
        backend._driver._get_graph = MagicMock(return_value=graph)
        backend.bulk_remove_edges = AsyncMock(return_value=1)

        result = await backend.dedup_valid_edges_for_node('A-uuid', group_id='test')

        cypher = extract_cypher(graph.ro_query.call_args)
        assert '-[e:RELATES_TO]-' in cypher
        assert 'invalid_at IS NULL' in cypher
        params = extract_params(graph.ro_query.call_args)
        assert params.get('uuid') == 'A-uuid'
        backend.bulk_remove_edges.assert_awaited_once_with(['e-2'], group_id='test')
        assert result == 1

    @pytest.mark.asyncio
    async def test_no_duplicates_returns_zero_without_deleting(
        self, mock_config, make_backend, make_graph_mock,
    ):
        """All-distinct rows -> returns 0 and bulk_remove_edges is never called."""
        backend = make_backend(mock_config)
        rows = [
            ['B-uuid', 'e-1', 'Fact one.', '2026-07-06T00:00:00+00:00'],
            ['C-uuid', 'e-2', 'Fact two.', '2026-07-06T01:00:00+00:00'],
        ]
        graph = make_graph_mock(rows)
        backend._driver._get_graph = MagicMock(return_value=graph)
        backend.bulk_remove_edges = AsyncMock(return_value=0)

        result = await backend.dedup_valid_edges_for_node('A-uuid', group_id='test')

        assert result == 0
        backend.bulk_remove_edges.assert_not_awaited()


# ---------------------------------------------------------------------------
# task 2073 step-1/2: GraphitiBackend.find_duplicate_entity_nodes
# ---------------------------------------------------------------------------

class TestFindDuplicateEntityNodes:
    """GraphitiBackend.find_duplicate_entity_nodes(name, *, group_id) returns every
    Entity node sharing an exact name, canonical-ordered (most valid edges, then
    oldest created_at, then uuid) so the post-write node-dedup sweep can pick
    matches[0] as the merge survivor."""

    @pytest.mark.asyncio
    async def test_uses_ro_query_only(self, mock_config, make_backend, make_graph_mock):
        """Read-only lookup: awaits graph.ro_query exactly once, never graph.query."""
        backend = make_backend(mock_config)
        rows = [
            ['canon-uuid', 100, 5],
            ['dup-uuid-1', 200, 2],
        ]
        await assert_ro_query_only(
            backend, make_graph_mock, rows, 'find_duplicate_entity_nodes',
            'orchestrator-reify.service', group_id='test',
        )

    @pytest.mark.asyncio
    async def test_cypher_matches_exact_name_and_counts_valid_edges(
        self, mock_config, make_backend, make_graph_mock,
    ):
        """Cypher matches Entity by exact $name param and counts only valid RELATES_TO edges."""
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        backend._driver._get_graph = MagicMock(return_value=graph)
        await backend.find_duplicate_entity_nodes('Reify', group_id='test')
        cypher = extract_cypher(graph.ro_query.call_args)
        assert '(n:Entity {name: $name})' in cypher
        assert 'RELATES_TO' in cypher
        assert 'invalid_at IS NULL' in cypher
        assert 'count(DISTINCT e)' in cypher
        params = extract_params(graph.ro_query.call_args)
        assert params.get('name') == 'Reify'

    @pytest.mark.asyncio
    async def test_cypher_orders_canonical_first(self, mock_config, make_backend, make_graph_mock):
        """ORDER BY ranks highest edge_count first, then oldest created_at, then uuid."""
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        backend._driver._get_graph = MagicMock(return_value=graph)
        await backend.find_duplicate_entity_nodes('Reify', group_id='test')
        cypher = extract_cypher(graph.ro_query.call_args)
        order_idx = cypher.find('ORDER BY')
        assert order_idx != -1, f'Expected ORDER BY clause in cypher: {cypher}'
        order_clause = cypher[order_idx:]
        assert 'edge_count DESC' in order_clause
        assert 'created_at ASC' in order_clause
        assert 'uuid ASC' in order_clause
        # edge_count must be the primary sort key, then created_at, then uuid
        assert order_clause.find('edge_count') < order_clause.find('created_at') < order_clause.find('uuid')

    @pytest.mark.asyncio
    async def test_returns_rows_preserving_order(self, mock_config, make_backend, make_graph_mock):
        """Returns list[dict] with uuid/created_at/edge_count, preserving DB row order."""
        backend = make_backend(mock_config)
        rows = [
            ['canon-uuid', 100, 5],
            ['dup-uuid-1', 200, 2],
            ['dup-uuid-2', 300, 1],
        ]
        graph = make_graph_mock(rows)
        backend._driver._get_graph = MagicMock(return_value=graph)
        result = await backend.find_duplicate_entity_nodes('Reify', group_id='test')
        assert result == [
            {'uuid': 'canon-uuid', 'created_at': 100, 'edge_count': 5},
            {'uuid': 'dup-uuid-1', 'created_at': 200, 'edge_count': 2},
            {'uuid': 'dup-uuid-2', 'created_at': 300, 'edge_count': 1},
        ]

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_matches(self, mock_config, make_backend, make_graph_mock):
        """Empty result_set -> []."""
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        backend._driver._get_graph = MagicMock(return_value=graph)
        result = await backend.find_duplicate_entity_nodes('Nonexistent', group_id='test')
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_single_element_list_when_one_match(
        self, mock_config, make_backend, make_graph_mock,
    ):
        """A single matching node -> single-element list (no duplicate to merge)."""
        backend = make_backend(mock_config)
        rows = [['only-uuid', 100, 0]]
        graph = make_graph_mock(rows)
        backend._driver._get_graph = MagicMock(return_value=graph)
        result = await backend.find_duplicate_entity_nodes('Unique', group_id='test')
        assert result == [{'uuid': 'only-uuid', 'created_at': 100, 'edge_count': 0}]

    @pytest.mark.asyncio
    async def test_raises_when_not_initialized(self, mock_config):
        """Raises RuntimeError when client is not initialized."""
        backend = GraphitiBackend(mock_config)  # client is None
        with pytest.raises(RuntimeError, match='not initialized'):
            await backend.find_duplicate_entity_nodes('Reify', group_id='test')


# ---------------------------------------------------------------------------
# step-7: MemoryService.merge_entities
# ---------------------------------------------------------------------------

class TestMemoryServiceMergeEntities:
    """MemoryService.merge_entities() delegates to graphiti backend with journaling."""

    @pytest.fixture
    def service(self, mock_config):
        """MemoryService with mocked graphiti backend."""
        from fused_memory.services.memory_service import MemoryService
        svc = MemoryService(mock_config)
        svc.graphiti = MagicMock()
        svc.graphiti.merge_entities = AsyncMock(return_value={
            'surviving_uuid': 'sur-uuid',
            'surviving_name': 'SurvivingName',
            'deprecated_uuid': 'dep-uuid',
            'deprecated_name': 'DeprecatedName',
            'edges_redirected': {'outgoing_redirected': 2, 'incoming_redirected': 1, 'inter_node_deleted': 0},
            'surviving_summary': {'before': 'old', 'after': 'new', 'edge_count': 3},
        })
        svc.mem0 = MagicMock()
        svc.durable_queue = MagicMock()
        svc.durable_queue.enqueue = AsyncMock(return_value=1)
        return svc

    @pytest.mark.asyncio
    async def test_delegates_to_graphiti_backend(self, service):
        """Calls graphiti.merge_entities with correct UUIDs."""
        result = await service.merge_entities(
            deprecated_uuid='dep-uuid',
            surviving_uuid='sur-uuid',
            project_id='dark_factory',
        )
        service.graphiti.merge_entities.assert_awaited_once_with('dep-uuid', 'sur-uuid', group_id='dark_factory')
        assert result['surviving_uuid'] == 'sur-uuid'

    @pytest.mark.asyncio
    async def test_logs_via_write_journal_when_present(self, service):
        """Logs write op via write journal when journal is set."""
        mock_journal = MagicMock()
        mock_journal.log_write_op = AsyncMock()
        service.set_write_journal(mock_journal)
        await service.merge_entities(
            deprecated_uuid='dep-uuid',
            surviving_uuid='sur-uuid',
            project_id='dark_factory',
            agent_id='test-agent',
        )
        mock_journal.log_write_op.assert_awaited_once()
        call_kwargs = mock_journal.log_write_op.call_args[1]
        assert call_kwargs.get('operation') == 'merge_entities'
        assert call_kwargs.get('project_id') == 'dark_factory'

    @pytest.mark.asyncio
    async def test_journal_params_contain_both_uuids(self, service):
        """Journal params dict contains both deprecated_uuid and surviving_uuid."""
        mock_journal = MagicMock()
        mock_journal.log_write_op = AsyncMock()
        service.set_write_journal(mock_journal)
        await service.merge_entities(
            deprecated_uuid='dep-uuid',
            surviving_uuid='sur-uuid',
            project_id='dark_factory',
        )
        call_kwargs = mock_journal.log_write_op.call_args[1]
        params = call_kwargs.get('params', {})
        assert params.get('deprecated_uuid') == 'dep-uuid'
        assert params.get('surviving_uuid') == 'sur-uuid'

    @pytest.mark.asyncio
    async def test_journal_failure_does_not_mask_success(self, service):
        """Journal failure does not prevent returning successful result."""
        mock_journal = MagicMock()
        mock_journal.log_write_op = AsyncMock(side_effect=RuntimeError('journal down'))
        service.set_write_journal(mock_journal)
        result = await service.merge_entities(
            deprecated_uuid='dep-uuid',
            surviving_uuid='sur-uuid',
            project_id='dark_factory',
        )
        assert result['surviving_uuid'] == 'sur-uuid'

    @pytest.mark.asyncio
    async def test_backend_failure_is_journaled(self, service):
        """When backend raises, journal is still called with success=False."""
        service.graphiti.merge_entities = AsyncMock(
            side_effect=ValueError('node not found')
        )
        mock_journal = MagicMock()
        mock_journal.log_write_op = AsyncMock()
        service.set_write_journal(mock_journal)
        with pytest.raises(ValueError, match='node not found'):
            await service.merge_entities(
                deprecated_uuid='dep-uuid',
                surviving_uuid='sur-uuid',
                project_id='dark_factory',
            )
        mock_journal.log_write_op.assert_awaited_once()
        call_kwargs = mock_journal.log_write_op.call_args[1]
        assert call_kwargs.get('success') is False

    @pytest.mark.asyncio
    async def test_works_without_journal(self, service):
        """Works correctly when no write journal is set."""
        service._write_journal = None
        result = await service.merge_entities(
            deprecated_uuid='dep-uuid',
            surviving_uuid='sur-uuid',
            project_id='dark_factory',
        )
        assert result['surviving_uuid'] == 'sur-uuid'


# ---------------------------------------------------------------------------
# step-9: MCP tool merge_entities
# ---------------------------------------------------------------------------

class TestMergeEntitiesMcpTool:
    """MCP tool merge_entities is registered and delegates correctly."""

    @pytest.fixture
    def mock_service(self):
        """Mock MemoryService for tool registration."""
        svc = AsyncMock()
        svc.merge_entities = AsyncMock(return_value={
            'surviving_uuid': 'sur-uuid',
            'surviving_name': 'SurvivingName',
            'deprecated_uuid': 'dep-uuid',
            'deprecated_name': 'DeprecatedName',
            'edges_redirected': {'outgoing_redirected': 2, 'incoming_redirected': 1, 'inter_node_deleted': 0},
            'surviving_summary': {'before': 'old', 'after': 'new', 'edge_count': 3},
        })
        return svc

    @pytest.fixture
    def mcp_server(self, mock_service):
        """MCP server with mock memory service."""
        from fused_memory.server.tools import create_mcp_server
        return create_mcp_server(mock_service)

    @pytest.mark.asyncio
    async def test_tool_is_registered(self, mcp_server):
        """merge_entities is registered as an MCP tool."""
        tool_names = [t.name for t in await mcp_server.list_tools()]
        assert 'merge_entities' in tool_names

    @pytest.mark.asyncio
    async def test_delegates_to_memory_service(self, mcp_server, mock_service):
        """Tool calls memory_service.merge_entities with correct UUIDs."""
        await mcp_server._tool_manager.call_tool(
            'merge_entities',
            {
                'deprecated_uuid': 'dep-uuid',
                'surviving_uuid': 'sur-uuid',
                'project_id': 'dark_factory',
            },
        )
        mock_service.merge_entities.assert_awaited_once()
        call_kwargs = mock_service.merge_entities.call_args[1]
        assert call_kwargs.get('deprecated_uuid') == 'dep-uuid'
        assert call_kwargs.get('surviving_uuid') == 'sur-uuid'
        assert call_kwargs.get('project_id') == 'dark_factory'

    @pytest.mark.asyncio
    async def test_empty_project_id_returns_error(self, mcp_server, mock_service):
        """Empty project_id returns validation error dict."""
        import json
        result = await mcp_server._tool_manager.call_tool(
            'merge_entities',
            {'deprecated_uuid': 'dep-uuid', 'surviving_uuid': 'sur-uuid', 'project_id': ''},
        )
        if isinstance(result, list):
            content = result[0].text if hasattr(result[0], 'text') else str(result[0])
            parsed = json.loads(content)
        else:
            parsed = result
        assert 'error' in parsed
        assert parsed.get('error_type') == 'ValidationError'
        mock_service.merge_entities.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_deprecated_uuid_returns_error(self, mcp_server, mock_service):
        """Empty deprecated_uuid returns validation error dict."""
        import json
        result = await mcp_server._tool_manager.call_tool(
            'merge_entities',
            {'deprecated_uuid': '', 'surviving_uuid': 'sur-uuid', 'project_id': 'dark_factory'},
        )
        if isinstance(result, list):
            content = result[0].text if hasattr(result[0], 'text') else str(result[0])
            parsed = json.loads(content)
        else:
            parsed = result
        assert 'error' in parsed
        assert parsed.get('error_type') == 'ValidationError'
        mock_service.merge_entities.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_surviving_uuid_returns_error(self, mcp_server, mock_service):
        """Empty surviving_uuid returns validation error dict."""
        import json
        result = await mcp_server._tool_manager.call_tool(
            'merge_entities',
            {'deprecated_uuid': 'dep-uuid', 'surviving_uuid': '', 'project_id': 'dark_factory'},
        )
        if isinstance(result, list):
            content = result[0].text if hasattr(result[0], 'text') else str(result[0])
            parsed = json.loads(content)
        else:
            parsed = result
        assert 'error' in parsed
        assert parsed.get('error_type') == 'ValidationError'
        mock_service.merge_entities.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_same_uuid_returns_error(self, mcp_server, mock_service):
        """Same UUID for both params returns validation error."""
        import json
        result = await mcp_server._tool_manager.call_tool(
            'merge_entities',
            {'deprecated_uuid': 'same-uuid', 'surviving_uuid': 'same-uuid', 'project_id': 'dark_factory'},
        )
        if isinstance(result, list):
            content = result[0].text if hasattr(result[0], 'text') else str(result[0])
            parsed = json.loads(content)
        else:
            parsed = result
        assert 'error' in parsed
        assert parsed.get('error_type') == 'ValidationError'
        mock_service.merge_entities.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_exception_returns_error_dict(self, mcp_server, mock_service):
        """Exception from memory_service returns error dict (not raised)."""
        import json
        mock_service.merge_entities = AsyncMock(
            side_effect=RuntimeError('FalkorDB connection failed')
        )
        result = await mcp_server._tool_manager.call_tool(
            'merge_entities',
            {'deprecated_uuid': 'dep-uuid', 'surviving_uuid': 'sur-uuid', 'project_id': 'dark_factory'},
        )
        if isinstance(result, list):
            content = result[0].text if hasattr(result[0], 'text') else str(result[0])
            parsed = json.loads(content)
        else:
            parsed = result
        assert 'error' in parsed
        assert 'FalkorDB connection failed' in parsed['error']


# ---------------------------------------------------------------------------
# step-11: DISALLOW_MEMORY_WRITES list in cli_stage_runner.py
# ---------------------------------------------------------------------------

class TestDisallowListForMergeEntities:
    """merge_entities must be in DISALLOW_MEMORY_WRITES but NOT in STAGE1_DISALLOWED."""

    def test_merge_entities_in_disallow_memory_writes(self):
        """'mcp__fused-memory__merge_entities' must be in DISALLOW_MEMORY_WRITES
        so Stage 3 (read-only) cannot call it."""
        from fused_memory.reconciliation.cli_stage_runner import DISALLOW_MEMORY_WRITES
        assert 'mcp__fused-memory__merge_entities' in DISALLOW_MEMORY_WRITES

    def test_merge_entities_not_in_stage1_disallowed(self):
        """Stage 1 must be able to call merge_entities (not in STAGE1_DISALLOWED)."""
        from fused_memory.reconciliation.cli_stage_runner import STAGE1_DISALLOWED
        assert 'mcp__fused-memory__merge_entities' not in STAGE1_DISALLOWED

    def test_merge_entities_in_stage3_disallowed(self):
        """Stage 3 must NOT be able to call merge_entities (in STAGE3_DISALLOWED
        via DISALLOW_MEMORY_WRITES)."""
        from fused_memory.reconciliation.cli_stage_runner import STAGE3_DISALLOWED
        assert 'mcp__fused-memory__merge_entities' in STAGE3_DISALLOWED
