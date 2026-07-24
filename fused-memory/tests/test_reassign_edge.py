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
