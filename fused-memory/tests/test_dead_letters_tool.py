"""MCP-level tests for the get_dead_letters tool."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.server.tools import create_mcp_server


# ── helpers ────────────────────────────────────────────────────────────────


def _make_mock_service(get_dead_items_return=None):
    """Build a minimal mock MemoryService with a durable_queue."""
    svc = AsyncMock()
    svc.durable_queue = MagicMock()
    svc.durable_queue.get_dead_items = AsyncMock(
        return_value=get_dead_items_return or []
    )
    return svc


_FAKE_DEAD_ITEMS = [
    {
        'id': 11,
        'group_id': 'proj1',
        'operation': 'add_episode',
        'payload': {'content': 'newer', 'group_id': 'proj1'},
        'attempts': 3,
        'error': 'RuntimeError: timeout',
        'created_at': 1_700_000_011.0,
    },
    {
        'id': 10,
        'group_id': 'proj1',
        'operation': 'add_episode',
        'payload': {'content': 'older', 'group_id': 'proj1'},
        'attempts': 3,
        'error': 'RuntimeError: timeout',
        'created_at': 1_700_000_010.0,
    },
]


# ── step-9 tests ───────────────────────────────────────────────────────────


class TestGetDeadLettersDurableQueue:
    """get_dead_letters with only durable_queue (no event_queue)."""

    @pytest.mark.asyncio
    async def test_returns_items_from_durable_queue(self):
        """Tool merges durable-queue dead items into the 'items' list."""
        svc = _make_mock_service(get_dead_items_return=_FAKE_DEAD_ITEMS)
        server = create_mcp_server(svc)

        result = await server._tool_manager.call_tool(
            'get_dead_letters',
            {'project_id': 'proj1', 'limit': 50},
        )

        # Top-level structure
        assert 'items' in result, f'Expected items key; got: {result}'
        assert 'counts' in result, f'Expected counts key; got: {result}'

        items = result['items']
        assert len(items) == 2

        for item in items:
            assert item['source'] == 'durable_queue'
            assert 'operation' in item
            assert 'payload' in item
            assert 'error' in item
            assert 'attempts' in item
            assert 'timestamp' in item or 'created_at' in item

    @pytest.mark.asyncio
    async def test_passes_group_id_and_limit_to_get_dead_items(self):
        """get_dead_items is called with group_id=project_id, limit=limit."""
        svc = _make_mock_service(get_dead_items_return=_FAKE_DEAD_ITEMS)
        server = create_mcp_server(svc)

        await server._tool_manager.call_tool(
            'get_dead_letters',
            {'project_id': 'proj1', 'limit': 50},
        )

        svc.durable_queue.get_dead_items.assert_called_once_with(
            group_id='proj1', limit=50,
        )

    @pytest.mark.asyncio
    async def test_no_durable_queue_returns_empty(self):
        """When durable_queue is None, tool returns empty items without raising."""
        svc = AsyncMock()
        svc.durable_queue = None
        server = create_mcp_server(svc)

        result = await server._tool_manager.call_tool(
            'get_dead_letters',
            {},
        )

        assert 'items' in result
        assert result['items'] == []
        assert 'error' not in result
