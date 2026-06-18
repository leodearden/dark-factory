"""Behaviour tests for the search MCP tool degraded-channel surfacing (task 1812).

step-15 (RED) / step-16 (GREEN):
  When memory_service.search returns a SearchResults with degraded=True, the MCP
  handler must surface degraded/failed_stores on the response.  Happy-path response
  shape must be unchanged (no degraded keys when degraded is False).
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from fused_memory.server.tools import create_mcp_server
from fused_memory.services.memory_service import SearchResults

_PROJECT_ID = 'dark_factory'


class TestSearchToolDegradedSurfacing:
    """search MCP tool must surface degraded/failed_stores only on fault path."""

    @pytest.mark.asyncio
    async def test_degraded_search_includes_degraded_keys(self):
        """When service.search returns degraded=True, response has degraded=True and failed_stores."""
        mock_service = AsyncMock()
        mock_service.search = AsyncMock(
            return_value=SearchResults([], degraded=True, failed_stores=['mem0'])
        )
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'search',
            {'query': 'test query', 'project_id': _PROJECT_ID},
        )

        assert isinstance(result, dict), f'Expected dict, got {type(result)}: {result!r}'
        assert 'error' not in result, f'Unexpected error in result: {result!r}'

        assert result.get('degraded') is True, (
            f"Expected result['degraded'] is True when search is degraded, "
            f"got result={result!r}. "
            "RED: handler does not yet surface degraded key."
        )
        assert result.get('failed_stores') == ['mem0'], (
            f"Expected result['failed_stores'] == ['mem0'], got {result.get('failed_stores')!r}. "
            "RED: handler does not yet surface failed_stores key."
        )

    @pytest.mark.asyncio
    async def test_clean_search_omits_degraded_keys(self):
        """When service.search returns degraded=False, response has NO degraded/failed_stores keys.

        Fault-only loudness: happy-path MCP response shape unchanged.
        """
        mock_service = AsyncMock()
        mock_service.search = AsyncMock(
            return_value=SearchResults([], degraded=False, failed_stores=[])
        )
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'search',
            {'query': 'test query', 'project_id': _PROJECT_ID},
        )

        assert isinstance(result, dict), f'Expected dict, got {type(result)}: {result!r}'
        assert 'error' not in result, f'Unexpected error in result: {result!r}'
        assert 'degraded' not in result, (
            f"'degraded' must NOT appear in happy-path response, got result={result!r}. "
            "Fault-only loudness: clean-path shape must be byte-for-byte unchanged."
        )
        assert 'failed_stores' not in result, (
            f"'failed_stores' must NOT appear in happy-path response, got result={result!r}."
        )

    @pytest.mark.asyncio
    async def test_plain_list_back_compat_does_not_add_degraded_keys(self):
        """When service.search returns a plain list (back-compat), response has no degraded keys."""
        mock_service = AsyncMock()
        mock_service.search = AsyncMock(return_value=[])
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'search',
            {'query': 'test query', 'project_id': _PROJECT_ID},
        )

        assert 'degraded' not in result, (
            f"'degraded' must NOT appear when search returns a plain list, got result={result!r}."
        )
