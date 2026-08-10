"""Tests for the delete_memory MCP tool's `id` alias for `memory_id`.

delete_memory's response (and search results) return the memory under the
key `id`, but the tool itself required `memory_id`. Agents already send
`{'id': ...}` back (see orchestrator test_agent_loop.py:990), so the tool
must accept `id` as an alias for `memory_id` — matching the
refresh_entity_summary `name`/`entity_name` alias pattern in
test_refresh_entity_summary.py (TestRefreshEntitySummaryMcpTool).
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from fused_memory.server.tools import create_mcp_server


def _parse_result(result):
    """Parse a FastMCP call_tool result (list of TextContent) into a dict."""
    if isinstance(result, list):
        content = result[0].text if hasattr(result[0], 'text') else str(result[0])
        return json.loads(content)
    return result


class TestDeleteMemoryIdAlias:
    """MCP tool delete_memory accepts `id` as an alias for `memory_id`."""

    @pytest.fixture
    def mock_service(self):
        """Mock MemoryService for tool registration."""
        svc = AsyncMock()
        svc.delete_memory = AsyncMock(
            return_value={'status': 'deleted', 'store': 'mem0', 'id': '00000000-0000-4000-8000-00000000000a'}
        )
        return svc

    @pytest.fixture
    def mcp_server(self, mock_service):
        """MCP server with mock memory service."""
        return create_mcp_server(mock_service)

    @pytest.mark.asyncio
    async def test_id_alias_delegates_with_memory_id(self, mcp_server, mock_service):
        """Calling with `id` delegates to memory_service.delete_memory(memory_id=...)."""
        await mcp_server._tool_manager.call_tool(
            'delete_memory',
            {'id': '00000000-0000-4000-8000-00000000000a', 'store': 'mem0', 'project_id': 'dark_factory'},
        )
        mock_service.delete_memory.assert_awaited_once()
        call_kwargs = mock_service.delete_memory.call_args[1]
        assert call_kwargs.get('memory_id') == '00000000-0000-4000-8000-00000000000a'
        assert call_kwargs.get('store') == 'mem0'
        assert call_kwargs.get('project_id') == 'dark_factory'

    @pytest.mark.asyncio
    async def test_canonical_memory_id_still_works(self, mcp_server, mock_service):
        """Backward compat: memory_id alone still calls the service correctly."""
        await mcp_server._tool_manager.call_tool(
            'delete_memory',
            {'memory_id': '00000000-0000-4000-8000-00000000000a', 'store': 'mem0', 'project_id': 'dark_factory'},
        )
        mock_service.delete_memory.assert_awaited_once()
        call_kwargs = mock_service.delete_memory.call_args[1]
        assert call_kwargs.get('memory_id') == '00000000-0000-4000-8000-00000000000a'

    @pytest.mark.asyncio
    async def test_neither_id_nor_memory_id_returns_validation_error(
        self, mcp_server, mock_service
    ):
        """Returns a ValidationError dict when neither memory_id nor id is provided."""
        result = await mcp_server._tool_manager.call_tool(
            'delete_memory',
            {'store': 'mem0', 'project_id': 'dark_factory'},
        )
        parsed = _parse_result(result)
        assert 'error' in parsed
        assert parsed.get('error_type') == 'ValidationError'
        mock_service.delete_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_conflicting_id_and_memory_id_returns_validation_error(
        self, mcp_server, mock_service
    ):
        """Returns a ValidationError dict when id and memory_id disagree."""
        result = await mcp_server._tool_manager.call_tool(
            'delete_memory',
            {'id': 'a', 'memory_id': 'b', 'store': 'mem0', 'project_id': 'dark_factory'},
        )
        parsed = _parse_result(result)
        assert 'error' in parsed
        assert parsed.get('error_type') == 'ValidationError'
        mock_service.delete_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_equal_id_and_memory_id_delegates_with_memory_id(
        self, mcp_server, mock_service
    ):
        """Supplying both id and memory_id with the SAME value is valid, not a conflict.

        The docstring's 'Exactly one must be provided; supplying both with
        conflicting values is an error' only rejects disagreement — matching
        values should resolve and delegate normally rather than error out.
        """
        await mcp_server._tool_manager.call_tool(
            'delete_memory',
            {'id': '00000000-0000-4000-8000-00000000000a', 'memory_id': '00000000-0000-4000-8000-00000000000a', 'store': 'mem0', 'project_id': 'dark_factory'},
        )
        mock_service.delete_memory.assert_awaited_once()
        call_kwargs = mock_service.delete_memory.call_args[1]
        assert call_kwargs.get('memory_id') == '00000000-0000-4000-8000-00000000000a'


# Full 36-char UUIDs, not short handles: the tool boundary validates id shape
# (`validate_full_uuid`) BEFORE it reaches the service, so a truncated 'm1'
# would be refused by the prologue and never exercise the cascade path.
_PARENT_ID = '00000000-0000-4000-8000-0000000000a1'
_CHILD_1 = '00000000-0000-4000-8000-0000000000c1'
_CHILD_2 = '00000000-0000-4000-8000-0000000000c2'


class TestDeleteMemoryCascadeTool:
    """The `cascade` opt-in and the child refusal at the MCP wire (task 3197).

    Pinning the safe default HERE and not only in the service signature
    matters: the wire is where an agent that never heard of this contract
    arrives.
    """

    @pytest.fixture
    def mock_service(self):
        svc = AsyncMock()
        svc.delete_memory = AsyncMock(
            return_value={'status': 'deleted', 'store': 'mem0', 'id': _PARENT_ID}
        )
        return svc

    @pytest.fixture
    def mcp_server(self, mock_service):
        return create_mcp_server(mock_service)

    @pytest.mark.asyncio
    async def test_cascade_defaults_to_false_on_the_wire(self, mcp_server, mock_service):
        await mcp_server._tool_manager.call_tool(
            'delete_memory',
            {'memory_id': _PARENT_ID, 'store': 'mem0', 'project_id': 'dark_factory'},
        )
        assert mock_service.delete_memory.call_args.kwargs['cascade'] is False

    @pytest.mark.asyncio
    async def test_cascade_true_is_forwarded(self, mcp_server, mock_service):
        await mcp_server._tool_manager.call_tool(
            'delete_memory',
            {
                'memory_id': _PARENT_ID, 'store': 'mem0',
                'project_id': 'dark_factory', 'cascade': True,
            },
        )
        assert mock_service.delete_memory.call_args.kwargs['cascade'] is True

    @pytest.mark.asyncio
    async def test_refusal_reaches_the_wire_with_every_child_id(
        self, mcp_server, mock_service
    ):
        """`@mcp_tool_errors` derives the shape generically, so the new
        exception class needs no registry edit — and the child ids ride the
        message, so an agent can act on the response alone without a second
        lookup."""
        from fused_memory.memory_metadata import ParentHasChildrenError

        mock_service.delete_memory = AsyncMock(
            side_effect=ParentHasChildrenError(
                parent_id=_PARENT_ID, child_ids=[_CHILD_1, _CHILD_2]
            )
        )

        parsed = _parse_result(await mcp_server._tool_manager.call_tool(
            'delete_memory',
            {'memory_id': _PARENT_ID, 'store': 'mem0', 'project_id': 'dark_factory'},
        ))

        assert parsed['error_type'] == 'ParentHasChildrenError'
        assert _CHILD_1 in parsed['error']
        assert _CHILD_2 in parsed['error']
        assert 'cascade' in parsed['error']

    @pytest.mark.asyncio
    async def test_prologue_still_runs_before_the_service_call(
        self, mcp_server, mock_service
    ):
        """The new parameter must not reorder the validation prologue that
        tests/test_mcp_boundary_canonicalization.py sweeps."""
        parsed = _parse_result(await mcp_server._tool_manager.call_tool(
            'delete_memory',
            {'memory_id': _PARENT_ID, 'store': 'not-a-store', 'project_id': 'dark_factory',
             'cascade': True},
        ))
        assert parsed['error_type'] == 'ValidationError'
        mock_service.delete_memory.assert_not_awaited()
