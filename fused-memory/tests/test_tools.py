"""Tests for MCP tool-level behavior (tool handler parameter forwarding, etc.)."""

from unittest.mock import AsyncMock

import pytest

from fused_memory.server.tools import create_mcp_server


# ------------------------------------------------------------------
# get_entity valid_only parameter forwarding
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_entity_tool_passes_valid_only_true_to_service():
    """get_entity tool passes valid_only=True to memory_service.get_entity()."""
    mock_service = AsyncMock()
    mock_service.get_entity = AsyncMock(return_value={'nodes': [], 'edges': []})
    server = create_mcp_server(mock_service)

    await server._tool_manager.call_tool(
        'get_entity',
        {'name': 'TestEntity', 'project_id': 'proj', 'valid_only': True},
    )

    mock_service.get_entity.assert_called_once()
    _, kwargs = mock_service.get_entity.call_args
    assert kwargs.get('valid_only') is True, (
        f'Expected valid_only=True to be forwarded to memory_service.get_entity, '
        f'got call_args={mock_service.get_entity.call_args}'
    )


@pytest.mark.asyncio
async def test_get_entity_tool_valid_only_defaults_to_false():
    """get_entity tool defaults valid_only=False when not specified."""
    mock_service = AsyncMock()
    mock_service.get_entity = AsyncMock(return_value={'nodes': [], 'edges': []})
    server = create_mcp_server(mock_service)

    await server._tool_manager.call_tool(
        'get_entity',
        {'name': 'TestEntity', 'project_id': 'proj'},
    )

    mock_service.get_entity.assert_called_once()
    _, kwargs = mock_service.get_entity.call_args
    assert kwargs.get('valid_only') is False, (
        f'Expected valid_only to default to False, '
        f'got call_args={mock_service.get_entity.call_args}'
    )
