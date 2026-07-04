"""Tests for the get_entity_by_uuid diagnostic — service method + MCP tool.

Task 2086: read-only topological UUID→node lookup, exposing the existing
GraphitiBackend.get_node_text via a new service method and MCP tool. Unlike
get_entity (name-based, fuzzy, semantic edge-gather), this is a direct
UUID readback: {uuid, name, summary} with no edge gather.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.backends.graphiti_client import NodeNotFoundError
from fused_memory.services.memory_service import MemoryService


@pytest.fixture
def service(mock_config):
    """MemoryService with a mocked graphiti backend (no real DB needed)."""
    svc = MemoryService(mock_config)
    svc.graphiti = MagicMock()
    svc.graphiti.get_node_text = AsyncMock(return_value=('NodeName', 'a summary'))
    return svc


class TestGetEntityByUuidService:
    """MemoryService.get_entity_by_uuid — thin wrapper over graphiti.get_node_text."""

    @pytest.mark.asyncio
    async def test_hit_returns_uuid_name_summary(self, service):
        result = await service.get_entity_by_uuid('node-uuid-1', project_id='test')
        assert result == {'uuid': 'node-uuid-1', 'name': 'NodeName', 'summary': 'a summary'}
        service.graphiti.get_node_text.assert_awaited_once_with('node-uuid-1', group_id='test')

    @pytest.mark.asyncio
    async def test_empty_summary_is_passed_through(self, service):
        service.graphiti.get_node_text = AsyncMock(return_value=('N', ''))
        result = await service.get_entity_by_uuid('node-uuid-2', project_id='test')
        assert result['summary'] == ''

    @pytest.mark.asyncio
    async def test_miss_raises_node_not_found_error(self, service):
        service.graphiti.get_node_text = AsyncMock(side_effect=NodeNotFoundError('missing'))
        with pytest.raises(NodeNotFoundError):
            await service.get_entity_by_uuid('node-uuid-missing', project_id='test')
