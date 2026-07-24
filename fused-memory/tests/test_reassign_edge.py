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
import os
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from _fm_helpers import extract_cypher, extract_params
from falkordb import FalkorDB as _SyncFalkorDB
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

FALKOR_HOST: str = os.environ.get('FALKOR_HOST', 'localhost')
FALKOR_PORT: int = int(os.environ.get('FALKOR_PORT', '6379'))


def _falkor_available() -> bool:
    """FalkorDB-native reachability probe (mirrors test_merge_entities.py)."""
    try:
        client = _SyncFalkorDB(host=FALKOR_HOST, port=FALKOR_PORT, socket_connect_timeout=2)
        try:
            client.select_graph('_probe').query('RETURN 1')
        finally:
            with contextlib.suppress(Exception):
                client.close()
        return True
    except Exception:
        return False


@pytest_asyncio.fixture
async def reassign_edge_live_graph():
    """Provision a throwaway, uniquely-named FalkorDB graph, yield it, clean up."""
    graph_name = f'_test_3009_reassign_edge_{uuid.uuid4().hex[:8]}'
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


@pytest.mark.skipif(not _falkor_available(), reason='FalkorDB not reachable')
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
