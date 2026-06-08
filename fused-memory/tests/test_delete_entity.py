"""Tests for delete_entity across backends, service, and MCP tool.

Covers:
- GraphitiBackend.get_connected_entity_uuids()
- GraphitiBackend.delete_entity()
- MemoryService.delete_entity()
- MCP tool delete_entity in tools.py
- DISALLOW_MEMORY_WRITES list in cli_stage_runner.py
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from _fm_helpers import extract_cypher, extract_params

from fused_memory.backends.graphiti_client import GraphitiBackend, NodeNotFoundError


# ---------------------------------------------------------------------------
# step-1: GraphitiBackend.get_connected_entity_uuids
# ---------------------------------------------------------------------------

class TestGetConnectedEntityUuids:
    """GraphitiBackend.get_connected_entity_uuids(uuid, *, group_id) returns neighbour UUIDs."""

    @pytest.mark.asyncio
    async def test_returns_neighbour_uuids(self, mock_config, make_backend, make_graph_mock):
        """Parses distinct neighbour UUID strings from ro_query result_set rows."""
        backend = make_backend(mock_config)
        graph = make_graph_mock(ro_rows=[['uuid-a'], ['uuid-b']])
        backend._driver._get_graph = MagicMock(return_value=graph)
        result = await backend.get_connected_entity_uuids('node-uuid', group_id='test')
        assert result == ['uuid-a', 'uuid-b']

    @pytest.mark.asyncio
    async def test_uses_ro_query_not_query(self, mock_config, make_backend, make_graph_mock):
        """Uses ro_query (read-only) and NOT the write query path."""
        backend = make_backend(mock_config)
        graph = make_graph_mock(ro_rows=[])
        backend._driver._get_graph = MagicMock(return_value=graph)
        await backend.get_connected_entity_uuids('node-uuid', group_id='test')
        graph.ro_query.assert_awaited_once()
        graph.query.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_passes_uuid_as_param(self, mock_config, make_backend, make_graph_mock):
        """Passes the target uuid as a Cypher parameter."""
        backend = make_backend(mock_config)
        graph = make_graph_mock(ro_rows=[])
        backend._driver._get_graph = MagicMock(return_value=graph)
        target = 'my-node-uuid'
        await backend.get_connected_entity_uuids(target, group_id='test')
        params = extract_params(graph.ro_query.call_args)
        assert params.get('uuid') == target

    @pytest.mark.asyncio
    async def test_cypher_filters_invalid_and_excludes_self(self, mock_config, make_backend, make_graph_mock):
        """Cypher matches RELATES_TO, filters invalid_at IS NULL, and excludes self (m.uuid <> $uuid)."""
        backend = make_backend(mock_config)
        graph = make_graph_mock(ro_rows=[])
        backend._driver._get_graph = MagicMock(return_value=graph)
        await backend.get_connected_entity_uuids('node-uuid', group_id='test')
        cypher = extract_cypher(graph.ro_query.call_args)
        assert 'RELATES_TO' in cypher
        assert 'invalid_at IS NULL' in cypher
        assert 'm.uuid <> $uuid' in cypher or "m.uuid <>" in cypher

    @pytest.mark.asyncio
    async def test_raises_when_not_initialized(self, mock_config):
        """Raises RuntimeError when constructed uninitialized (no client)."""
        backend = GraphitiBackend(mock_config)
        with pytest.raises(RuntimeError, match='not initialized'):
            await backend.get_connected_entity_uuids('node-uuid', group_id='test')

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_neighbours(self, mock_config, make_backend, make_graph_mock):
        """Returns empty list when no neighbours found."""
        backend = make_backend(mock_config)
        graph = make_graph_mock(ro_rows=[])
        backend._driver._get_graph = MagicMock(return_value=graph)
        result = await backend.get_connected_entity_uuids('isolated-uuid', group_id='test')
        assert result == []
