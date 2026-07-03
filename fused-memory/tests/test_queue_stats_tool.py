"""MCP-level tests for the get_queue_stats tool."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.server.tools import create_mcp_server

# ── helpers ────────────────────────────────────────────────────────────────


def _make_mock_service(get_stats_return=None):
    """Build a minimal mock MemoryService with a durable_queue."""
    svc = AsyncMock()
    svc.durable_queue = MagicMock()
    svc.durable_queue.get_stats = AsyncMock(
        return_value=get_stats_return
        or {'counts': {'completed': 8648, 'dead': 3}, 'oldest_pending_age_seconds': None}
    )
    return svc


_FAKE_STATS = {'counts': {'completed': 8648, 'dead': 3}, 'oldest_pending_age_seconds': None}


# ── step-1 tests ───────────────────────────────────────────────────────────


class TestGetQueueStatsProjectScoping:
    """get_queue_stats forwards project_id through to durable_queue.get_stats."""

    @pytest.mark.asyncio
    async def test_forwards_project_id_as_group_id(self):
        """project_id is forwarded as group_id, and the tool returns get_stats' result."""
        svc = _make_mock_service(get_stats_return=_FAKE_STATS)
        server = create_mcp_server(svc)

        result = await server._tool_manager.call_tool(
            'get_queue_stats',
            {'project_id': 'proj1'},
        )

        svc.durable_queue.get_stats.assert_called_once_with(group_id='proj1')
        assert result == _FAKE_STATS

    @pytest.mark.asyncio
    async def test_no_project_id_is_global(self):
        """Omitting project_id preserves the backward-compatible global path."""
        svc = _make_mock_service(get_stats_return=_FAKE_STATS)
        server = create_mcp_server(svc)

        result = await server._tool_manager.call_tool(
            'get_queue_stats',
            {},
        )

        svc.durable_queue.get_stats.assert_called_once_with(group_id=None)
        assert result == _FAKE_STATS
