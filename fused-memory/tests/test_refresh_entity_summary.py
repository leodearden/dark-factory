"""Tests for refresh_entity_summary across backends, service, and MCP tool.

Covers:
- GraphitiBackend.get_valid_edges_for_node()
- GraphitiBackend.update_node_summary()
- GraphitiBackend.refresh_entity_summary()
- MemoryService.refresh_entity_summary()
- MCP tool refresh_entity_summary in tools.py
- DISALLOW_MEMORY_WRITES list in cli_stage_runner.py
- STAGE1_SYSTEM_PROMPT in prompts/stage1.py
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fused_memory.backends.graphiti_client import GraphitiBackend


# ---------------------------------------------------------------------------
# step-1: GraphitiBackend.get_valid_edges_for_node
# ---------------------------------------------------------------------------

class TestGetValidEdgesForNode:
    """GraphitiBackend.get_valid_edges_for_node(node_uuid) returns valid edges."""

    @pytest.mark.asyncio
    async def test_returns_valid_edges(self, mock_config, make_backend, make_graph_mock):
        """Returns list of dicts with uuid/fact/name keys for matching edges."""
        backend = make_backend(mock_config)
        rows = [
            ['edge-1', 'Alice knows Bob', 'knows'],
            ['edge-2', 'Alice works at Acme', 'works_at'],
        ]
        graph = make_graph_mock(rows)
        cast_target = MagicMock()
        cast_target._get_graph = MagicMock(return_value=graph)
        with patch('fused_memory.backends.graphiti_client.cast', return_value=cast_target):
            result = await backend.get_valid_edges_for_node('node-uuid-1')
        assert len(result) == 2
        assert result[0]['uuid'] == 'edge-1'
        assert result[0]['fact'] == 'Alice knows Bob'
        assert result[0]['name'] == 'knows'
        assert result[1]['uuid'] == 'edge-2'
        assert result[1]['fact'] == 'Alice works at Acme'
        assert result[1]['name'] == 'works_at'

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_valid_edges(self, mock_config, make_backend, make_graph_mock):
        """Returns empty list when no valid edges remain."""
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        cast_target = MagicMock()
        cast_target._get_graph = MagicMock(return_value=graph)
        with patch('fused_memory.backends.graphiti_client.cast', return_value=cast_target):
            result = await backend.get_valid_edges_for_node('node-uuid-1')
        assert result == []

    @pytest.mark.asyncio
    async def test_raises_when_not_initialized(self, mock_config):
        """Raises RuntimeError when client is not initialized."""
        backend = GraphitiBackend(mock_config)  # client is None
        with pytest.raises(RuntimeError, match='not initialized'):
            await backend.get_valid_edges_for_node('node-uuid-1')

    @pytest.mark.asyncio
    async def test_passes_uuid_to_query(self, mock_config, make_backend, make_graph_mock):
        """Passes node UUID as parameter to the Cypher query."""
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        cast_target = MagicMock()
        cast_target._get_graph = MagicMock(return_value=graph)
        node_uuid = 'my-node-uuid'
        with patch('fused_memory.backends.graphiti_client.cast', return_value=cast_target):
            await backend.get_valid_edges_for_node(node_uuid)
        call_args = graph.query.call_args
        assert call_args is not None
        args, kwargs = call_args
        cypher_params = args[1] if len(args) > 1 else kwargs.get('params', {})
        assert cypher_params.get('uuid') == node_uuid

    @pytest.mark.asyncio
    async def test_filters_invalid_at_null(self, mock_config, make_backend, make_graph_mock):
        """Query uses WHERE e.invalid_at IS NULL to filter active edges only."""
        backend = make_backend(mock_config)
        graph = make_graph_mock([])
        cast_target = MagicMock()
        cast_target._get_graph = MagicMock(return_value=graph)
        with patch('fused_memory.backends.graphiti_client.cast', return_value=cast_target):
            await backend.get_valid_edges_for_node('node-uuid-1')
        call_args = graph.query.call_args
        args, kwargs = call_args
        cypher = args[0] if args else kwargs.get('query', '')
        assert 'invalid_at IS NULL' in cypher
