"""Behaviour tests for the get_entity MCP tool's degraded-fallback surfacing (task 2448).

step-3 (RED) / step-4 (GREEN):
  When memory_service.get_entity hits an embedding-provider quota/rate-limit
  error, the service-layer degraded fallback must surface through the MCP
  tool unchanged — {degraded: True, failed_stores: ['graphiti']} instead of a
  hard {error, error_type} dict. Happy-path response shape must be unchanged
  (no degraded keys when nothing failed) — fault-only loudness, matching the
  search tool's convention (test_search_tool.py).

Unlike test_search_tool.py (which mocks memory_service.search entirely), this
suite builds a REAL MemoryService(mock_config) with only graphiti mocked, so
the service-layer degraded classification (task 2448) is actually exercised
through the MCP tool boundary rather than assumed.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from _fm_helpers import _make_rate_limit_error

from fused_memory.server.tools import create_mcp_server
from fused_memory.services.memory_service import MemoryService

_PROJECT_ID = 'dark_factory'


def _parse_tool_result(result):
    """Extract the dict from a FastMCP TextContent result or pass-through dict."""
    if isinstance(result, list):
        content = result[0].text if hasattr(result[0], 'text') else str(result[0])
        return json.loads(content)
    return result


@pytest.fixture
def service(mock_config):
    """A REAL MemoryService with a mocked graphiti backend (no real DB needed)."""
    svc = MemoryService(mock_config)
    svc.graphiti = MagicMock()
    svc.graphiti.get_nodes_by_exact_name = AsyncMock(return_value=[])
    return svc


class TestGetEntityToolDegradedSurfacing:
    """get_entity MCP tool must surface the service's degraded fallback, not a hard error."""

    @pytest.mark.asyncio
    async def test_quota_error_returns_degraded_not_error(self, service):
        """A quota/rate-limit error from graphiti surfaces as a degraded dict,
        not a hard {error, error_type} dict."""
        service.graphiti.search_nodes = AsyncMock(return_value=[])
        service.graphiti.search = AsyncMock(side_effect=_make_rate_limit_error())
        server = create_mcp_server(service)

        result = await server._tool_manager.call_tool(
            'get_entity',
            {'name': 'X', 'project_id': _PROJECT_ID},
        )
        parsed = _parse_tool_result(result)

        assert 'error' not in parsed, f'Unexpected error in result: {parsed!r}'
        assert parsed.get('degraded') is True, (
            f"Expected result['degraded'] is True on quota exhaustion, got {parsed!r}. "
            'RED: get_entity does not yet degrade-fallback on quota errors.'
        )
        assert parsed.get('failed_stores') == ['graphiti'], (
            f"Expected result['failed_stores'] == ['graphiti'], "
            f'got {parsed.get("failed_stores")!r}.'
        )
        assert parsed.get('nodes') == []
        assert parsed.get('edges') == []

    @pytest.mark.asyncio
    async def test_clean_lookup_omits_degraded_keys(self, service):
        """Happy-path get_entity response has NO degraded/failed_stores keys
        (fault-only loudness — clean-path shape stays byte-for-byte unchanged)."""
        from _fm_helpers import MockEdge, MockNode

        mock_node = MockNode(name='Auth Service', uuid='node-uuid-1')
        mock_edge = MockEdge(fact='Auth service depends on Redis', uuid='edge-uuid-1')
        service.graphiti.search_nodes = AsyncMock(return_value=[mock_node])
        service.graphiti.search = AsyncMock(return_value=[mock_edge])
        server = create_mcp_server(service)

        result = await server._tool_manager.call_tool(
            'get_entity',
            {'name': 'Auth Service', 'project_id': _PROJECT_ID},
        )
        parsed = _parse_tool_result(result)

        assert 'error' not in parsed, f'Unexpected error in result: {parsed!r}'
        assert 'degraded' not in parsed, (
            f"'degraded' must NOT appear in happy-path response, got {parsed!r}."
        )
        assert 'failed_stores' not in parsed, (
            f"'failed_stores' must NOT appear in happy-path response, got {parsed!r}."
        )
        assert len(parsed['nodes']) == 1
        assert len(parsed['edges']) == 1


async def _hang(*args, **kwargs):
    await asyncio.Event().wait()


class TestGetEntityToolFuzzyReadTimeout:
    """A Graphiti read timeout in get_entity's fuzzy fallback is a hard error, not a miss.

    get_entity degrades only on a classified rate-limit/quota error; a timeout
    propagates.  `GraphitiBackend.search_nodes` used to swallow its own timeout
    into `[]`, which made the fuzzy branch answer a clean
    `{'nodes': [], 'edges': []}`, so the real backend must be in the path
    (task 5699).
    """

    @pytest.mark.asyncio
    async def test_search_nodes_timeout_returns_an_error_naming_it(
        self, mock_config, make_backend,
    ):
        mock_config.queue.backend_read_timeout_seconds = 0.05
        backend = make_backend(mock_config)
        backend.get_nodes_by_exact_name = AsyncMock(return_value=[])
        backend.client.search_ = AsyncMock(side_effect=_hang)
        backend.client.search = AsyncMock(return_value=[])
        service = MemoryService(mock_config)
        service.graphiti = backend

        result = await create_mcp_server(service)._tool_manager.call_tool(
            'get_entity',
            {'name': 'X', 'project_id': _PROJECT_ID},
        )
        parsed = _parse_tool_result(result)

        assert parsed.get('error_type') == 'TimeoutError', (
            f'a search_nodes timeout must surface as an error, not a clean miss; got {parsed!r}'
        )
        assert 'search_nodes' in parsed.get('error', ''), f'got {parsed!r}'
        assert 'nodes' not in parsed, f'got {parsed!r}'
