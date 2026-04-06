"""Tests for MCP tools: get_dead_letters and purge_dead_letters."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.server.tools import create_mcp_server


def _make_dead_item(item_id: int, group_id: str = 'dark_factory') -> dict:
    """Helper to create a fake dead-letter item dict."""
    return {
        'id': item_id,
        'group_id': group_id,
        'operation': 'add_memory_graphiti',
        'payload': {'content': f'test content {item_id}', 'group_id': group_id},
        'attempts': 5,
        'error': 'ResponseError: Query timed out',
        'created_at': time.time() - 3600,
    }


class TestGetDeadLetters:
    @pytest.mark.asyncio
    async def test_get_dead_letters_returns_items(self):
        """get_dead_letters returns dead_letters list with expected fields."""
        mock_service = AsyncMock()
        fake_items = [_make_dead_item(i) for i in range(1, 4)]
        mock_service.durable_queue.get_dead_items = AsyncMock(return_value=fake_items)

        server = create_mcp_server(mock_service)
        result = await server._tool_manager.call_tool(
            'get_dead_letters',
            {'project_id': 'dark_factory'},
        )

        assert isinstance(result, dict), f'Expected dict, got {type(result)}: {result!r}'
        assert 'dead_letters' in result, f'Expected dead_letters key: {result!r}'
        assert len(result['dead_letters']) == 3

        for item in result['dead_letters']:
            assert 'id' in item
            assert 'error' in item
            assert 'operation' in item
            assert 'attempts' in item
            assert 'payload_preview' in item

        mock_service.durable_queue.get_dead_items.assert_awaited_once_with(
            group_id='dark_factory'
        )

    @pytest.mark.asyncio
    async def test_get_dead_letters_rejects_missing_queue(self):
        """get_dead_letters returns ConfigurationError when durable_queue is None."""
        mock_service = AsyncMock()
        mock_service.durable_queue = None

        server = create_mcp_server(mock_service)
        result = await server._tool_manager.call_tool('get_dead_letters', {})

        assert isinstance(result, dict)
        assert 'error' in result
        assert result['error_type'] == 'ConfigurationError'


class TestPurgeDeadLetters:
    @pytest.mark.asyncio
    async def test_purge_dead_letters_delegates(self):
        """purge_dead_letters delegates to durable_queue.purge_dead with correct args."""
        mock_service = AsyncMock()
        mock_service.durable_queue.purge_dead = AsyncMock(return_value=2)

        server = create_mcp_server(mock_service)
        result = await server._tool_manager.call_tool(
            'purge_dead_letters',
            {'project_id': 'reify', 'error_pattern': 'NodeNotFoundError%'},
        )

        assert isinstance(result, dict), f'Expected dict, got {type(result)}: {result!r}'
        assert result == {'status': 'purged', 'items_purged': 2}, f'Unexpected result: {result!r}'

        mock_service.durable_queue.purge_dead.assert_awaited_once_with(
            group_id='reify',
            error_pattern='NodeNotFoundError%',
            ids=None,
            confirm_purge_all=False,
        )

    @pytest.mark.asyncio
    async def test_purge_dead_letters_surfaces_guard_error(self):
        """purge_dead_letters returns ValidationError when ValueError raised by purge_dead."""
        mock_service = AsyncMock()
        mock_service.durable_queue.purge_dead = AsyncMock(
            side_effect=ValueError(
                'purge_dead requires at least one filter or confirm_purge_all=True'
            )
        )

        server = create_mcp_server(mock_service)
        result = await server._tool_manager.call_tool('purge_dead_letters', {})

        assert isinstance(result, dict)
        assert result['error_type'] == 'ValidationError'
        assert 'confirm_purge_all' in result['error']
