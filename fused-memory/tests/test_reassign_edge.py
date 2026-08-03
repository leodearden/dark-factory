"""Tests for reassign_edge across backend, service, and MCP tool.

reassign_edge re-points ONE existing RELATES_TO edge's endpoint (source or
target) to a different Entity node, LOSSLESSLY — preserving the edge uuid,
fact text, fact_embedding, valid_at/invalid_at/expired_at, created_at,
group_id, and episode links — then refreshes the two affected node summaries.
It is the single-edge, uuid-targeted sibling of redirect_node_edges (the
bulk merge-time endpoint mover).

Covers (layered like test_merge_entities.py):
- GraphitiBackend.reassign_edge()  — validation, endpoint-move Cypher, refresh
- A skipif-gated live-FalkorDB acceptance test (real topology move + preservation)
- MemoryService.reassign_edge()    — delegation, journaling, event emission
- MCP tool reassign_edge in tools.py
- DISALLOW_MEMORY_WRITES list in cli_stage_runner.py
"""
from __future__ import annotations

import contextlib
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from _fm_helpers import (
    FALKOR_HOST,
    FALKOR_PORT,
    extract_cypher,
    extract_params,
    falkor_skipif,
    unique_graph_name,
)
from falkordb.asyncio import FalkorDB
from graphiti_core.errors import EdgeNotFoundError

from fused_memory.backends.graphiti_client import (
    GraphitiBackend,
    NodeNotFoundError,
    _MultiTenantFalkorDriver,
)

# ---------------------------------------------------------------------------
# step-1: GraphitiBackend.reassign_edge — validation slice
# ---------------------------------------------------------------------------

class TestReassignEdgeBackendValidation:
    """GraphitiBackend.reassign_edge validates which_end, the target edge, the
    new endpoint node, and the initialized state before performing any move."""

    @pytest.mark.asyncio
    async def test_invalid_which_end_raises_value_error(
        self, mock_config, make_backend, make_graph_mock,
    ):
        """which_end must be 'source' or 'target' — anything else raises ValueError."""
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        backend._driver._get_graph = MagicMock(return_value=graph)
        with pytest.raises(ValueError, match='which_end'):
            await backend.reassign_edge(
                'edge-uuid', 'new-endpoint-uuid', which_end='sideways', group_id='test',
            )

    @pytest.mark.asyncio
    async def test_missing_edge_raises_edge_not_found(
        self, mock_config, make_backend, make_graph_mock,
    ):
        """When the topology read returns no rows, the edge does not exist."""
        backend = make_backend(mock_config)
        # ro_query returns empty -> topology read finds no edge.
        graph = make_graph_mock([])
        backend._driver._get_graph = MagicMock(return_value=graph)
        with pytest.raises(EdgeNotFoundError):
            await backend.reassign_edge(
                'missing-edge', 'new-endpoint-uuid', which_end='source', group_id='test',
            )

    @pytest.mark.asyncio
    async def test_missing_new_endpoint_raises_node_not_found(
        self, mock_config, make_backend, make_graph_mock,
    ):
        """Edge exists (topology read returns rows) but the new endpoint node
        does not -> NodeNotFoundError."""
        backend = make_backend(mock_config)

        async def router(cypher, params=None):
            result = MagicMock()
            if 'RELATES_TO' in cypher:
                # Topology endpoint read — edge exists (source, target).
                result.result_set = [['src-uuid', 'tgt-uuid']]
            else:
                # New endpoint node existence read — absent.
                result.result_set = []
            return result

        graph = make_graph_mock([])
        graph.ro_query = AsyncMock(side_effect=router)
        backend._driver._get_graph = MagicMock(return_value=graph)
        with pytest.raises(NodeNotFoundError):
            await backend.reassign_edge(
                'edge-uuid', 'missing-endpoint', which_end='source', group_id='test',
            )

    @pytest.mark.asyncio
    async def test_raises_when_not_initialized(self, mock_config):
        """Raises RuntimeError when the backend is not initialized (driver is None)."""
        backend = GraphitiBackend(mock_config)  # client/_driver are None
        with pytest.raises(RuntimeError, match='not initialized'):
            await backend.reassign_edge(
                'edge-uuid', 'new-endpoint-uuid', which_end='source', group_id='test',
            )

    @pytest.mark.asyncio
    async def test_duplicate_uuid_edge_raises_and_issues_no_write(
        self, mock_config, make_backend, make_graph_mock,
    ):
        """When the topology read returns MORE THAN ONE row for the uuid (a
        duplicate-uuid RELATES_TO edge), reassign refuses loudly rather than
        operating on an arbitrary result_set[0] — and issues no CREATE/DELETE."""
        backend = make_backend(mock_config)

        async def router(cypher, params=None):
            result = MagicMock()
            if 'RELATES_TO' in cypher:
                # Two edges share the uuid — a data-integrity anomaly the
                # uuid-keyed move below could not disambiguate.
                result.result_set = [['s1', 't1'], ['s2', 't2']]
            else:
                result.result_set = [['node']]
            return result

        graph = make_graph_mock([])
        graph.ro_query = AsyncMock(side_effect=router)
        backend._driver._get_graph = MagicMock(return_value=graph)
        with pytest.raises(RuntimeError, match='duplicate-uuid'):
            await backend.reassign_edge(
                'dup-edge', 'new-endpoint-uuid', which_end='source', group_id='test',
            )
        graph.query.assert_not_awaited()


# ---------------------------------------------------------------------------
# step-3: GraphitiBackend.reassign_edge — endpoint-move Cypher
# ---------------------------------------------------------------------------

class TestReassignEdgeBackendMove:
    """The endpoint-move issues a single atomic CREATE-new + DELETE-old query
    that PRESERVES the edge uuid, copies every property by direct old.<prop>
    reference, sets the moved endpoint from the new-endpoint param, and pins the
    UNCHANGED endpoint from the matched TOPOLOGY node (never the unreliable
    old.<end>_node_uuid property). A no-op guard skips the write entirely when
    the new endpoint already equals the endpoint being moved."""

    _COPIED_PROPS = (
        'name', 'fact', 'fact_embedding', 'valid_at', 'invalid_at',
        'expired_at', 'created_at', 'group_id', 'episodes',
    )

    @staticmethod
    def _wire_graph(
        backend, make_graph_mock, *, source_uuid, target_uuid, node_exists=True,
    ):
        """Route ro_query: topology endpoint read -> [source, target]; new-node
        existence read -> one row (or empty). graph.query stays a plain AsyncMock."""
        async def ro_router(cypher, params=None):
            result = MagicMock()
            if 'RELATES_TO' in cypher:
                result.result_set = [[source_uuid, target_uuid]]
            else:
                result.result_set = [['node']] if node_exists else []
            return result

        graph = make_graph_mock([])
        graph.ro_query = AsyncMock(side_effect=ro_router)
        backend._driver._get_graph = MagicMock(return_value=graph)
        return graph

    @pytest.mark.asyncio
    async def test_source_move_cypher_shape(self, mock_config, make_backend, make_graph_mock):
        """which_end='source': CREATE (new_src)->(target), uuid preserved,
        props copied, source set from param, target pinned from topology."""
        backend = make_backend(mock_config)
        graph = self._wire_graph(
            backend, make_graph_mock, source_uuid='src-uuid', target_uuid='tgt-uuid',
        )
        await backend.reassign_edge(
            'edge-uuid', 'new-src-uuid', which_end='source', group_id='test',
        )

        create_calls = [c for c in graph.query.call_args_list if 'CREATE' in extract_cypher(c)]
        assert len(create_calls) == 1, (
            f'expected exactly one CREATE query, got {len(create_calls)}: '
            f'{[extract_cypher(c) for c in create_calls]}'
        )
        cypher = extract_cypher(create_calls[0])
        params = extract_params(create_calls[0])

        # New source node on the left, unchanged target on the right.
        assert 'CREATE (new_src)-[new:RELATES_TO]->(target)' in cypher
        # uuid PRESERVED (not re-minted like redirect_node_edges does).
        assert 'new.uuid = old.uuid' in cypher
        assert '$new_uuid' not in cypher
        # Every preserved property copied by direct old.<prop> reference.
        for prop in self._COPIED_PROPS:
            assert f'new.{prop} = old.{prop}' in cypher, f'missing copy of {prop}'
        # Moved endpoint set from the new-endpoint param.
        assert 'new.source_node_uuid = $new_endpoint_uuid' in cypher
        # UNCHANGED endpoint pinned from TOPOLOGY node, NOT the old property.
        assert 'new.target_node_uuid = target.uuid' in cypher
        assert 'old.target_node_uuid' not in cypher
        # Audit trail + old edge removed.
        assert 'new.reassigned_from_node_uuid' in cypher
        assert 'DELETE old' in cypher
        # Params carry the edge, the new endpoint, and the audit source.
        assert params.get('edge_uuid') == 'edge-uuid'
        assert params.get('new_endpoint_uuid') == 'new-src-uuid'
        assert params.get('old_endpoint_uuid') == 'src-uuid'

    @pytest.mark.asyncio
    async def test_target_move_cypher_shape(self, mock_config, make_backend, make_graph_mock):
        """which_end='target': CREATE (source)->(new_tgt), target set from param,
        source pinned from topology."""
        backend = make_backend(mock_config)
        graph = self._wire_graph(
            backend, make_graph_mock, source_uuid='src-uuid', target_uuid='tgt-uuid',
        )
        await backend.reassign_edge(
            'edge-uuid', 'new-tgt-uuid', which_end='target', group_id='test',
        )

        create_calls = [c for c in graph.query.call_args_list if 'CREATE' in extract_cypher(c)]
        assert len(create_calls) == 1, (
            f'expected exactly one CREATE query, got {len(create_calls)}: '
            f'{[extract_cypher(c) for c in create_calls]}'
        )
        cypher = extract_cypher(create_calls[0])
        params = extract_params(create_calls[0])

        # Unchanged source on the left, new target node on the right.
        assert 'CREATE (source)-[new:RELATES_TO]->(new_tgt)' in cypher
        assert 'new.uuid = old.uuid' in cypher
        for prop in self._COPIED_PROPS:
            assert f'new.{prop} = old.{prop}' in cypher, f'missing copy of {prop}'
        # Moved endpoint set from param; unchanged endpoint pinned from topology.
        assert 'new.target_node_uuid = $new_endpoint_uuid' in cypher
        assert 'new.source_node_uuid = source.uuid' in cypher
        assert 'old.source_node_uuid' not in cypher
        assert 'new.reassigned_from_node_uuid' in cypher
        assert 'DELETE old' in cypher
        assert params.get('old_endpoint_uuid') == 'tgt-uuid'
        assert params.get('new_endpoint_uuid') == 'new-tgt-uuid'

    @pytest.mark.asyncio
    async def test_noop_when_new_equals_current_endpoint(
        self, mock_config, make_backend, make_graph_mock,
    ):
        """When new_endpoint_uuid already equals the endpoint being moved,
        NO CREATE/DELETE query is issued and moved is False."""
        backend = make_backend(mock_config)
        graph = self._wire_graph(
            backend, make_graph_mock, source_uuid='src-uuid', target_uuid='tgt-uuid',
        )
        # Moving the source end onto the SAME source uuid — a no-op.
        result = await backend.reassign_edge(
            'edge-uuid', 'src-uuid', which_end='source', group_id='test',
        )
        graph.query.assert_not_awaited()
        assert result['moved'] is False

    @pytest.mark.asyncio
    async def test_self_loop_rejected_and_issues_no_write(
        self, mock_config, make_backend, make_graph_mock,
    ):
        """Moving an endpoint onto the UNCHANGED endpoint would form a
        self-referential edge (e.g. src->tgt, which_end='source', new=tgt =>
        tgt->tgt). This is rejected with ValueError and issues no write."""
        backend = make_backend(mock_config)
        graph = self._wire_graph(
            backend, make_graph_mock, source_uuid='src-uuid', target_uuid='tgt-uuid',
        )
        with pytest.raises(ValueError, match='self-referential'):
            await backend.reassign_edge(
                'edge-uuid', 'tgt-uuid', which_end='source', group_id='test',
            )
        graph.query.assert_not_awaited()


def _wire_topology_graph(
    backend, make_graph_mock, *, source_uuid, target_uuid, node_exists=True,
):
    """ro_query router: RELATES_TO topology read -> [source, target]; the
    new-endpoint node existence read -> one row (or empty when node_exists is
    False). graph.query stays the default AsyncMock."""
    async def ro_router(cypher, params=None):
        result = MagicMock()
        if 'RELATES_TO' in cypher:
            result.result_set = [[source_uuid, target_uuid]]
        else:
            result.result_set = [['node']] if node_exists else []
        return result

    graph = make_graph_mock([])
    graph.ro_query = AsyncMock(side_effect=ro_router)
    backend._driver._get_graph = MagicMock(return_value=graph)
    return graph


# ---------------------------------------------------------------------------
# step-5: GraphitiBackend.reassign_edge — post-move refresh + audit dict
# ---------------------------------------------------------------------------

class TestReassignEdgePostMove:
    """After a move, exactly the OLD (lost the edge) and NEW (gained it)
    endpoint summaries are refreshed — best-effort — and a full audit dict is
    returned. The UNCHANGED endpoint keeps the identical edge, so it is not
    refreshed."""

    @pytest.mark.asyncio
    async def test_refreshes_old_and_new_endpoints_only(
        self, mock_config, make_backend, make_graph_mock,
    ):
        """refresh_entity_summary is awaited for the OLD and NEW endpoints, and
        NOT for the unchanged endpoint."""
        backend = make_backend(mock_config)
        _wire_topology_graph(
            backend, make_graph_mock, source_uuid='src-uuid', target_uuid='tgt-uuid',
        )
        backend.refresh_entity_summary = AsyncMock(return_value={})
        await backend.reassign_edge(
            'edge-uuid', 'new-src-uuid', which_end='source', group_id='test',
        )
        refreshed_uuids = [c.args[0] for c in backend.refresh_entity_summary.call_args_list]
        assert set(refreshed_uuids) == {'src-uuid', 'new-src-uuid'}
        assert 'tgt-uuid' not in refreshed_uuids  # unchanged endpoint untouched

    @pytest.mark.asyncio
    async def test_audit_dict_has_expected_keys(
        self, mock_config, make_backend, make_graph_mock,
    ):
        """The returned audit dict has exactly the documented keys and values."""
        backend = make_backend(mock_config)
        _wire_topology_graph(
            backend, make_graph_mock, source_uuid='src-uuid', target_uuid='tgt-uuid',
        )
        backend.refresh_entity_summary = AsyncMock(return_value={})
        result = await backend.reassign_edge(
            'edge-uuid', 'new-src-uuid', which_end='source', group_id='test',
        )
        assert set(result.keys()) == {
            'uuid', 'which_end', 'old_endpoint_uuid', 'new_endpoint_uuid',
            'unchanged_endpoint_uuid', 'moved', 'refreshed_nodes',
        }
        assert result['uuid'] == 'edge-uuid'
        assert result['which_end'] == 'source'
        assert result['old_endpoint_uuid'] == 'src-uuid'
        assert result['new_endpoint_uuid'] == 'new-src-uuid'
        assert result['unchanged_endpoint_uuid'] == 'tgt-uuid'
        assert result['moved'] is True
        assert result['refreshed_nodes'] == ['src-uuid', 'new-src-uuid']

    @pytest.mark.asyncio
    async def test_refresh_failure_is_swallowed(
        self, mock_config, make_backend, make_graph_mock,
    ):
        """A refresh raising for one node is logged and swallowed (best-effort):
        reassign still returns and that node is absent from refreshed_nodes."""
        backend = make_backend(mock_config)
        _wire_topology_graph(
            backend, make_graph_mock, source_uuid='src-uuid', target_uuid='tgt-uuid',
        )

        async def refresh_side_effect(node_uuid, *args, **kwargs):
            if node_uuid == 'src-uuid':
                raise RuntimeError('refresh boom')
            return {}

        backend.refresh_entity_summary = AsyncMock(side_effect=refresh_side_effect)
        result = await backend.reassign_edge(
            'edge-uuid', 'new-src-uuid', which_end='source', group_id='test',
        )
        assert result['moved'] is True
        assert 'src-uuid' not in result['refreshed_nodes']  # the failed node
        assert 'new-src-uuid' in result['refreshed_nodes']


# ---------------------------------------------------------------------------
# Live-FalkorDB integration harness (mirrors test_merge_entities.py:980-1132)
# ---------------------------------------------------------------------------
#
# Pins the REAL topology move + lossless preservation + uuid-uniqueness that
# the mock tests above cannot: mocks can only assert the Cypher string/params
# shape, not FalkorDB's actual relationship relocation, per-uuid count(*), or
# byte-for-byte vecf32/temporal/episode preservation. Skipped automatically
# when FalkorDB is unreachable (the mock tests guarantee coverage then) and
# deselected by the default `-m 'not integration'`; runs only under an explicit
# `-m integration` with FalkorDB reachable. Duplicated (not shared via
# conftest) per the established per-module convention that sidesteps the
# sys.modules['conftest'] collision documented in _fm_helpers.py.
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def reassign_edge_live_graph():
    """Provision a throwaway, uniquely-named FalkorDB graph, yield it, clean up."""
    graph_name = unique_graph_name('3009_reassign_edge')
    client = FalkorDB(host=FALKOR_HOST, port=FALKOR_PORT)
    with contextlib.suppress(Exception):
        stale = client.select_graph(graph_name)
        await stale.delete()
    graph = client.select_graph(graph_name)
    try:
        yield graph_name, graph
    finally:
        with contextlib.suppress(Exception):
            await graph.delete()
        with contextlib.suppress(Exception):
            await client.aclose()


@falkor_skipif()
@pytest.mark.timeout(15)
@pytest.mark.integration
class TestReassignEdgeLiveFalkorDB:
    """Pin the real-DB topology move + losslessness against a REAL FalkorDB
    server: the edge uuid is preserved (count(*)==1, no dup), the endpoint is
    genuinely relocated, and fact/valid_at/invalid_at/episodes/fact_embedding
    survive byte-for-byte — for both which_end='source' and which_end='target'.
    """

    @staticmethod
    async def _seed_nodes(graph, uuids):
        parts = ', '.join(f"(:Entity {{uuid: '{u}', name: '{u}', summary: ''}})" for u in uuids)
        await graph.query(f'CREATE {parts}')

    @staticmethod
    async def _seed_edge(graph, src, dst, edge_uuid):
        await graph.query(
            'MATCH (a:Entity {uuid: $src}), (b:Entity {uuid: $dst}) '
            'CREATE (a)-[e:RELATES_TO {uuid: $edge_uuid, name: $name, fact: $fact, '
            'valid_at: $valid_at, invalid_at: $invalid_at, episodes: $episodes}]->(b) '
            'SET e.fact_embedding = vecf32($embedding)',
            {
                'src': src, 'dst': dst, 'edge_uuid': edge_uuid, 'name': 'rel',
                'fact': 'A relates to B', 'valid_at': '2026-01-01T00:00:00Z',
                'invalid_at': '2026-06-01T00:00:00Z', 'episodes': ['ep1', 'ep2'],
                'embedding': [1.0, 2.0, 3.0],
            },
        )

    @pytest.mark.asyncio
    async def test_source_move_preserves_edge_losslessly(
        self, mock_config, reassign_edge_live_graph,
    ):
        graph_name, graph = reassign_edge_live_graph
        await self._seed_nodes(graph, ['A', 'B', 'C'])
        await self._seed_edge(graph, 'A', 'B', 'e1')

        backend = GraphitiBackend(mock_config)
        backend._driver = _MultiTenantFalkorDriver(host=FALKOR_HOST, port=FALKOR_PORT)
        try:
            result = await backend.reassign_edge('e1', 'C', which_end='source', group_id=graph_name)
            assert result['moved'] is True

            # Topology moved: C-[e1]->B exists, carrying every preserved property.
            moved = await graph.query(
                "MATCH (c:Entity {uuid: 'C'})-[e:RELATES_TO {uuid: 'e1'}]->(b:Entity {uuid: 'B'}) "
                'RETURN e.fact, e.valid_at, e.invalid_at, e.episodes, e.fact_embedding, '
                'e.reassigned_from_node_uuid'
            )
            assert len(moved.result_set) == 1
            fact, valid_at, invalid_at, episodes, embedding, reassigned_from = moved.result_set[0]
            assert fact == 'A relates to B'
            assert valid_at == '2026-01-01T00:00:00Z'
            assert invalid_at == '2026-06-01T00:00:00Z'
            assert list(episodes) == ['ep1', 'ep2']
            assert list(embedding) == pytest.approx([1.0, 2.0, 3.0])
            assert reassigned_from == 'A'  # audit trail

            # A no longer connected via e1.
            a_gone = await graph.query(
                "MATCH (a:Entity {uuid: 'A'})-[e:RELATES_TO {uuid: 'e1'}]->() RETURN count(e)"
            )
            assert a_gone.result_set[0][0] == 0

            # uuid PRESERVED and unique: exactly one RELATES_TO edge with uuid 'e1'.
            dup = await graph.query(
                "MATCH ()-[e:RELATES_TO {uuid: 'e1'}]->() RETURN count(e)"
            )
            assert dup.result_set[0][0] == 1
        finally:
            await backend.close()

    @pytest.mark.asyncio
    async def test_target_move_preserves_edge_losslessly(
        self, mock_config, reassign_edge_live_graph,
    ):
        graph_name, graph = reassign_edge_live_graph
        await self._seed_nodes(graph, ['A', 'B', 'C'])
        await self._seed_edge(graph, 'A', 'B', 'e2')

        backend = GraphitiBackend(mock_config)
        backend._driver = _MultiTenantFalkorDriver(host=FALKOR_HOST, port=FALKOR_PORT)
        try:
            result = await backend.reassign_edge('e2', 'C', which_end='target', group_id=graph_name)
            assert result['moved'] is True

            # Topology moved: A-[e2]->C exists, carrying every preserved property.
            moved = await graph.query(
                "MATCH (a:Entity {uuid: 'A'})-[e:RELATES_TO {uuid: 'e2'}]->(c:Entity {uuid: 'C'}) "
                'RETURN e.fact, e.valid_at, e.invalid_at, e.episodes, e.fact_embedding, '
                'e.reassigned_from_node_uuid'
            )
            assert len(moved.result_set) == 1
            fact, valid_at, invalid_at, episodes, embedding, reassigned_from = moved.result_set[0]
            assert fact == 'A relates to B'
            assert valid_at == '2026-01-01T00:00:00Z'
            assert invalid_at == '2026-06-01T00:00:00Z'
            assert list(episodes) == ['ep1', 'ep2']
            assert list(embedding) == pytest.approx([1.0, 2.0, 3.0])
            assert reassigned_from == 'B'  # audit trail (old target)

            # B no longer the target via e2.
            b_gone = await graph.query(
                "MATCH ()-[e:RELATES_TO {uuid: 'e2'}]->(b:Entity {uuid: 'B'}) RETURN count(e)"
            )
            assert b_gone.result_set[0][0] == 0

            # uuid preserved and unique.
            dup = await graph.query(
                "MATCH ()-[e:RELATES_TO {uuid: 'e2'}]->() RETURN count(e)"
            )
            assert dup.result_set[0][0] == 1
        finally:
            await backend.close()


# ---------------------------------------------------------------------------
# step-7: MemoryService.reassign_edge
# ---------------------------------------------------------------------------

class TestMemoryServiceReassignEdge:
    """MemoryService.reassign_edge() delegates to the graphiti backend with
    journaling + a memory_updated event, mirroring update_edge/merge_entities."""

    @pytest.fixture
    def service(self, mock_config):
        """MemoryService with a mocked graphiti backend."""
        from fused_memory.services.memory_service import MemoryService
        svc = MemoryService(mock_config)
        svc.graphiti = MagicMock()
        svc.graphiti.reassign_edge = AsyncMock(return_value={
            'uuid': 'edge-uuid',
            'which_end': 'source',
            'old_endpoint_uuid': 'old-uuid',
            'new_endpoint_uuid': 'new-uuid',
            'unchanged_endpoint_uuid': 'unchanged-uuid',
            'moved': True,
            'refreshed_nodes': ['old-uuid', 'new-uuid'],
        })
        svc.mem0 = MagicMock()
        svc.durable_queue = MagicMock()
        svc.durable_queue.enqueue = AsyncMock(return_value=1)
        return svc

    @pytest.mark.asyncio
    async def test_delegates_to_graphiti_backend(self, service):
        """Calls graphiti.reassign_edge with edge/new-endpoint/which_end and
        group_id=project_id, and returns a reassigned/graphiti status dict."""
        result = await service.reassign_edge(
            edge_uuid='edge-uuid',
            new_endpoint_uuid='new-uuid',
            which_end='source',
            project_id='dark_factory',
        )
        service.graphiti.reassign_edge.assert_awaited_once_with(
            'edge-uuid', 'new-uuid', which_end='source', group_id='dark_factory',
        )
        assert result['status'] == 'reassigned'
        assert result['store'] == 'graphiti'
        # backend audit keys pass through
        assert result['moved'] is True
        assert result['old_endpoint_uuid'] == 'old-uuid'

    @pytest.mark.asyncio
    async def test_logs_via_write_journal_when_present(self, service):
        """Logs write op via write journal with operation='reassign_edge'."""
        mock_journal = MagicMock()
        mock_journal.log_write_op = AsyncMock()
        service.set_write_journal(mock_journal)
        await service.reassign_edge(
            edge_uuid='edge-uuid',
            new_endpoint_uuid='new-uuid',
            which_end='source',
            project_id='dark_factory',
            agent_id='test-agent',
        )
        mock_journal.log_write_op.assert_awaited_once()
        call_kwargs = mock_journal.log_write_op.call_args[1]
        assert call_kwargs.get('operation') == 'reassign_edge'
        assert call_kwargs.get('project_id') == 'dark_factory'

    @pytest.mark.asyncio
    async def test_journal_params_contain_move_args(self, service):
        """Journal params dict carries edge_uuid, new_endpoint_uuid, which_end."""
        mock_journal = MagicMock()
        mock_journal.log_write_op = AsyncMock()
        service.set_write_journal(mock_journal)
        await service.reassign_edge(
            edge_uuid='edge-uuid',
            new_endpoint_uuid='new-uuid',
            which_end='target',
            project_id='dark_factory',
        )
        params = mock_journal.log_write_op.call_args[1].get('params', {})
        assert params.get('edge_uuid') == 'edge-uuid'
        assert params.get('new_endpoint_uuid') == 'new-uuid'
        assert params.get('which_end') == 'target'

    @pytest.mark.asyncio
    async def test_emits_memory_updated_event(self, service):
        """A memory_updated ReconciliationEvent is emitted for the edge."""
        from fused_memory.models.reconciliation import EventType
        service._emit_event = AsyncMock()
        await service.reassign_edge(
            edge_uuid='edge-uuid',
            new_endpoint_uuid='new-uuid',
            which_end='source',
            project_id='dark_factory',
        )
        service._emit_event.assert_awaited_once()
        event = service._emit_event.call_args[0][0]
        assert event.type == EventType.memory_updated
        assert event.payload.get('edge_uuid') == 'edge-uuid'
        assert event.payload.get('store') == 'graphiti'

    @pytest.mark.asyncio
    async def test_noop_reassign_does_not_emit_event(self, service):
        """A no-op reassign (backend returns moved=False — the new endpoint
        already equals the current one) changed nothing in the graph, so it must
        NOT emit a memory_updated event (which would trigger spurious downstream
        reconciliation). The call still succeeds and returns the no-op audit."""
        service.graphiti.reassign_edge = AsyncMock(return_value={
            'uuid': 'edge-uuid',
            'which_end': 'source',
            'old_endpoint_uuid': 'same-uuid',
            'new_endpoint_uuid': 'same-uuid',
            'unchanged_endpoint_uuid': 'other-uuid',
            'moved': False,
            'refreshed_nodes': [],
        })
        service._emit_event = AsyncMock()
        result = await service.reassign_edge(
            edge_uuid='edge-uuid',
            new_endpoint_uuid='same-uuid',
            which_end='source',
            project_id='dark_factory',
        )
        service._emit_event.assert_not_awaited()
        assert result['status'] == 'reassigned'
        assert result['moved'] is False

    @pytest.mark.asyncio
    async def test_journal_failure_does_not_mask_success(self, service):
        """Journal failure does not prevent returning a successful result."""
        mock_journal = MagicMock()
        mock_journal.log_write_op = AsyncMock(side_effect=RuntimeError('journal down'))
        service.set_write_journal(mock_journal)
        result = await service.reassign_edge(
            edge_uuid='edge-uuid',
            new_endpoint_uuid='new-uuid',
            which_end='source',
            project_id='dark_factory',
        )
        assert result['status'] == 'reassigned'

    @pytest.mark.asyncio
    async def test_backend_failure_is_journaled(self, service):
        """When the backend raises, the journal is still called with success=False
        and the error is re-raised."""
        service.graphiti.reassign_edge = AsyncMock(
            side_effect=ValueError('edge not found')
        )
        mock_journal = MagicMock()
        mock_journal.log_write_op = AsyncMock()
        service.set_write_journal(mock_journal)
        with pytest.raises(ValueError, match='edge not found'):
            await service.reassign_edge(
                edge_uuid='edge-uuid',
                new_endpoint_uuid='new-uuid',
                which_end='source',
                project_id='dark_factory',
            )
        mock_journal.log_write_op.assert_awaited_once()
        assert mock_journal.log_write_op.call_args[1].get('success') is False

    @pytest.mark.asyncio
    async def test_works_without_journal(self, service):
        """Works correctly when no write journal is set."""
        service._write_journal = None
        result = await service.reassign_edge(
            edge_uuid='edge-uuid',
            new_endpoint_uuid='new-uuid',
            which_end='source',
            project_id='dark_factory',
        )
        assert result['status'] == 'reassigned'


# ---------------------------------------------------------------------------
# step-9: MCP tool reassign_edge
# ---------------------------------------------------------------------------

def _parse_tool_result(result):
    """Normalize a call_tool result (list of content blocks or a dict) to a dict."""
    import json
    if isinstance(result, list):
        content = result[0].text if hasattr(result[0], 'text') else str(result[0])
        return json.loads(content)
    return result


class TestReassignEdgeMcpTool:
    """MCP tool reassign_edge is registered, guards its inputs, and delegates."""

    @pytest.fixture
    def mock_service(self):
        svc = AsyncMock()
        svc.reassign_edge = AsyncMock(return_value={
            'status': 'reassigned', 'store': 'graphiti', 'uuid': 'edge-uuid',
            'which_end': 'source', 'moved': True,
        })
        return svc

    @pytest.fixture
    def mcp_server(self, mock_service):
        from fused_memory.server.tools import create_mcp_server
        return create_mcp_server(mock_service)

    @pytest.mark.asyncio
    async def test_tool_is_registered(self, mcp_server):
        tool_names = [t.name for t in await mcp_server.list_tools()]
        assert 'reassign_edge' in tool_names

    @pytest.mark.asyncio
    async def test_delegates_to_memory_service(self, mcp_server, mock_service):
        await mcp_server._tool_manager.call_tool(
            'reassign_edge',
            {
                'edge_uuid': 'edge-uuid',
                'new_endpoint_uuid': 'new-uuid',
                'which_end': 'source',
                'project_id': 'dark_factory',
            },
        )
        mock_service.reassign_edge.assert_awaited_once()
        call_kwargs = mock_service.reassign_edge.call_args[1]
        assert call_kwargs.get('edge_uuid') == 'edge-uuid'
        assert call_kwargs.get('new_endpoint_uuid') == 'new-uuid'
        assert call_kwargs.get('which_end') == 'source'
        assert call_kwargs.get('project_id') == 'dark_factory'

    @pytest.mark.asyncio
    async def test_empty_edge_uuid_returns_error(self, mcp_server, mock_service):
        result = await mcp_server._tool_manager.call_tool(
            'reassign_edge',
            {'edge_uuid': '  ', 'new_endpoint_uuid': 'new-uuid',
             'which_end': 'source', 'project_id': 'dark_factory'},
        )
        parsed = _parse_tool_result(result)
        assert parsed.get('error_type') == 'ValidationError'
        mock_service.reassign_edge.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_new_endpoint_uuid_returns_error(self, mcp_server, mock_service):
        result = await mcp_server._tool_manager.call_tool(
            'reassign_edge',
            {'edge_uuid': 'edge-uuid', 'new_endpoint_uuid': '',
             'which_end': 'source', 'project_id': 'dark_factory'},
        )
        parsed = _parse_tool_result(result)
        assert parsed.get('error_type') == 'ValidationError'
        mock_service.reassign_edge.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_invalid_which_end_returns_error(self, mcp_server, mock_service):
        result = await mcp_server._tool_manager.call_tool(
            'reassign_edge',
            {'edge_uuid': 'edge-uuid', 'new_endpoint_uuid': 'new-uuid',
             'which_end': 'sideways', 'project_id': 'dark_factory'},
        )
        parsed = _parse_tool_result(result)
        assert parsed.get('error_type') == 'ValidationError'
        mock_service.reassign_edge.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_project_id_returns_error(self, mcp_server, mock_service):
        result = await mcp_server._tool_manager.call_tool(
            'reassign_edge',
            {'edge_uuid': 'edge-uuid', 'new_endpoint_uuid': 'new-uuid',
             'which_end': 'source', 'project_id': ''},
        )
        parsed = _parse_tool_result(result)
        assert parsed.get('error_type') == 'ValidationError'
        mock_service.reassign_edge.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_exception_returns_error_dict(self, mcp_server, mock_service):
        mock_service.reassign_edge = AsyncMock(
            side_effect=RuntimeError('FalkorDB connection failed')
        )
        result = await mcp_server._tool_manager.call_tool(
            'reassign_edge',
            {'edge_uuid': 'edge-uuid', 'new_endpoint_uuid': 'new-uuid',
             'which_end': 'source', 'project_id': 'dark_factory'},
        )
        parsed = _parse_tool_result(result)
        assert 'error' in parsed
        assert 'FalkorDB connection failed' in parsed['error']


# ---------------------------------------------------------------------------
# step-11: reassign_edge disallow-list stage membership
# ---------------------------------------------------------------------------


class TestDisallowListForReassignEdge:
    """reassign_edge is a memory write: it must be in DISALLOW_MEMORY_WRITES
    (blocked in Stage 3 read-only) but NOT in STAGE1_DISALLOWED (Stage 1 may
    call it)."""

    def test_reassign_edge_in_disallow_memory_writes(self):
        """'mcp__fused-memory__reassign_edge' must be in DISALLOW_MEMORY_WRITES
        so Stage 3 (read-only) cannot call it."""
        from fused_memory.reconciliation.cli_stage_runner import DISALLOW_MEMORY_WRITES
        assert 'mcp__fused-memory__reassign_edge' in DISALLOW_MEMORY_WRITES

    def test_reassign_edge_not_in_stage1_disallowed(self):
        """Stage 1 must be able to call reassign_edge (not in STAGE1_DISALLOWED)."""
        from fused_memory.reconciliation.cli_stage_runner import STAGE1_DISALLOWED
        assert 'mcp__fused-memory__reassign_edge' not in STAGE1_DISALLOWED

    def test_reassign_edge_in_stage3_disallowed(self):
        """Stage 3 must NOT be able to call reassign_edge (in STAGE3_DISALLOWED
        via DISALLOW_MEMORY_WRITES)."""
        from fused_memory.reconciliation.cli_stage_runner import STAGE3_DISALLOWED
        assert 'mcp__fused-memory__reassign_edge' in STAGE3_DISALLOWED
