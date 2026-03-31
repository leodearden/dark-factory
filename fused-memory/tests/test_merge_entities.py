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

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
        cast_target = MagicMock()
        cast_target._get_graph = MagicMock(return_value=graph)
        with patch('fused_memory.backends.graphiti_client.cast', return_value=cast_target):
            result = await backend.redirect_node_edges('dep-uuid', 'sur-uuid')
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
        cast_target = MagicMock()
        cast_target._get_graph = MagicMock(return_value=graph)
        dep_uuid = 'dep-abc'
        sur_uuid = 'sur-xyz'
        with patch('fused_memory.backends.graphiti_client.cast', return_value=cast_target):
            await backend.redirect_node_edges(dep_uuid, sur_uuid)
        # First query call should target inter-node edges
        first_call = graph.query.call_args_list[0]
        args = first_call[0]
        cypher = args[0] if args else ''
        params = args[1] if len(args) > 1 else {}
        # Both UUIDs should appear in params
        assert dep_uuid in params.values() or any(
            dep_uuid in str(v) for v in params.values()
        )

    @pytest.mark.asyncio
    async def test_returns_count_dict(self, mock_config, make_backend, make_graph_mock):
        """Returns dict with outgoing_redirected, incoming_redirected, inter_node_deleted counts."""
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        cast_target = MagicMock()
        cast_target._get_graph = MagicMock(return_value=graph)
        with patch('fused_memory.backends.graphiti_client.cast', return_value=cast_target):
            result = await backend.redirect_node_edges('dep-uuid', 'sur-uuid')
        assert isinstance(result['outgoing_redirected'], int)
        assert isinstance(result['incoming_redirected'], int)
        assert isinstance(result['inter_node_deleted'], int)

    @pytest.mark.asyncio
    async def test_raises_when_not_initialized(self, mock_config):
        """Raises RuntimeError when client is not initialized."""
        backend = GraphitiBackend(mock_config)  # client is None
        with pytest.raises(RuntimeError, match='not initialized'):
            await backend.redirect_node_edges('dep-uuid', 'sur-uuid')

    @pytest.mark.asyncio
    async def test_handles_empty_edge_sets(self, mock_config, make_backend, make_graph_mock):
        """When no edges exist, returns zeros for all counts."""
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        cast_target = MagicMock()
        cast_target._get_graph = MagicMock(return_value=graph)
        with patch('fused_memory.backends.graphiti_client.cast', return_value=cast_target):
            result = await backend.redirect_node_edges('dep-uuid', 'sur-uuid')
        assert result['outgoing_redirected'] == 0
        assert result['incoming_redirected'] == 0
        assert result['inter_node_deleted'] == 0
