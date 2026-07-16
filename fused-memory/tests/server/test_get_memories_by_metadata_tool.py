"""Behaviour tests for the get_memories_by_metadata MCP tool (task-1659).

The tool is a read-only deterministic Qdrant payload-filter scroll — not semantic —
intended for exhaustive enumeration of memories by metadata fields such as
confirming all stage1_flag_markers are present (source + kind).

Mirrors test_count_by_metadata_tool.py exactly.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from fused_memory.backends.mem0_client import Mem0Backend
from fused_memory.server.tools import create_mcp_server
from fused_memory.services.memory_service import MemoryService

_PROJECT_ID = 'dark_factory'
_FILTERS = {'source': 'stage1_flag_marker'}

_SAMPLE_MEMORIES = [
    {'id': 'mem-1', 'created_at': '2026-01-01T00:00:00Z',
     'metadata': {'source': 'stage1_flag_marker', 'kind': 'stage1_flag_marker'}},
    {'id': 'mem-2', 'created_at': '2026-01-02T00:00:00Z',
     'metadata': {'source': 'stage1_flag_marker', 'kind': 'stage1_flag_marker'}},
]


class TestGetMemoriesByMetadataTool:
    """Behaviour tests for mcp__fused-memory__get_memories_by_metadata."""

    @pytest.mark.asyncio
    async def test_happy_path_returns_memories_and_forwards_args(self):
        """Happy path: service returns a list; tool returns {'memories': [...]} and
        calls service exactly once with the right project_id and filters."""
        mock_service = AsyncMock()
        mock_service.get_memories_by_metadata = AsyncMock(return_value=_SAMPLE_MEMORIES)
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'get_memories_by_metadata',
            {
                'project_id': _PROJECT_ID,
                'filters': _FILTERS,
            },
        )

        assert isinstance(result, dict), f'Expected dict, got {type(result)}: {result!r}'
        assert 'memories' in result, f"Expected 'memories' key, got: {result!r}"
        assert result['memories'] == _SAMPLE_MEMORIES, (
            f"Expected sample memories, got: {result!r}"
        )
        # Must not be an error response
        assert 'error' not in result, f"Unexpected error in result: {result!r}"

        # Service must have been called exactly once with the right arguments
        mock_service.get_memories_by_metadata.assert_called_once_with(
            project_id=_PROJECT_ID,
            filters=_FILTERS,
            limit=1000,
        )

    @pytest.mark.asyncio
    async def test_empty_result_returns_empty_list_not_error(self):
        """When service returns [], the tool returns {'memories': []} — not an error."""
        mock_service = AsyncMock()
        mock_service.get_memories_by_metadata = AsyncMock(return_value=[])
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'get_memories_by_metadata',
            {
                'project_id': _PROJECT_ID,
                'filters': {'source': 'stage1_flag_marker', 'kind': 'not_found'},
            },
        )

        assert isinstance(result, dict), f'Expected dict, got {type(result)}: {result!r}'
        assert result.get('memories') == [], (
            f"Expected empty memories list, got: {result!r}"
        )
        assert 'error' not in result, f"Unexpected error in result: {result!r}"

    @pytest.mark.asyncio
    async def test_invalid_project_id_returns_validation_error_without_calling_service(self):
        """Invalid project_id (contains unsafe chars) returns a validation error dict
        and does NOT call the service."""
        mock_service = AsyncMock()
        mock_service.get_memories_by_metadata = AsyncMock(return_value=[])
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'get_memories_by_metadata',
            {
                'project_id': 'bad id!',  # contains spaces and !
                'filters': _FILTERS,
            },
        )

        assert isinstance(result, dict), f'Expected dict, got {type(result)}: {result!r}'
        assert 'error' in result, f"Expected error key in result: {result!r}"
        assert result.get('error_type') == 'ValidationError', (
            f"Expected error_type='ValidationError', got: {result!r}"
        )
        # Service must NOT have been called
        mock_service.get_memories_by_metadata.assert_not_called()

    @pytest.mark.asyncio
    async def test_custom_limit_is_forwarded_to_service(self):
        """An explicit limit value is forwarded to the service call."""
        mock_service = AsyncMock()
        mock_service.get_memories_by_metadata = AsyncMock(return_value=_SAMPLE_MEMORIES)
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'get_memories_by_metadata',
            {
                'project_id': _PROJECT_ID,
                'filters': _FILTERS,
                'limit': 5000,
            },
        )

        assert 'memories' in result, f"Expected 'memories' key: {result!r}"
        # Service must have been called with the custom limit
        mock_service.get_memories_by_metadata.assert_called_once_with(
            project_id=_PROJECT_ID,
            filters=_FILTERS,
            limit=5000,
        )
        # limit is echoed back in the response
        assert result.get('limit') == 5000, f"Expected limit=5000 in response: {result!r}"

    @pytest.mark.asyncio
    async def test_service_exception_is_caught_and_returned_as_error_dict(self):
        """When the service raises, the tool catches it and returns
        {'error': ..., 'error_type': ...} without propagating."""
        mock_service = AsyncMock()
        mock_service.get_memories_by_metadata = AsyncMock(
            side_effect=ValueError('filters must not be empty')
        )
        server = create_mcp_server(mock_service)

        # Should not raise — exception must be caught
        result = await server._tool_manager.call_tool(
            'get_memories_by_metadata',
            {
                'project_id': _PROJECT_ID,
                'filters': {},
            },
        )

        assert isinstance(result, dict), f'Expected dict, got {type(result)}: {result!r}'
        assert 'error' in result, f"Expected error key in result: {result!r}"
        assert result.get('error_type') == 'ValueError', (
            f"Expected error_type='ValueError', got: {result!r}"
        )
        assert 'filters must not be empty' in result.get('error', ''), (
            f"Expected original error message in result: {result!r}"
        )

    @pytest.mark.asyncio
    async def test_scroll_read_timeout_surfaces_as_error_not_empty_list(self, mock_config):
        """A Qdrant scroll timeout surfaces as {'error', 'error_type': 'TimeoutError'} —
        it must NOT masquerade as an empty {'memories': []} result (no-silent-fail
        invariant; esc-2655-3).

        Drives a REAL MemoryService + REAL Mem0Backend (only the underlying async
        Qdrant client is patched) so this test genuinely exercises the whole
        scroll_by_metadata -> get_memories_by_metadata -> @mcp_tool_errors chain —
        a fully-mocked service would pass even against the pre-fix swallow-to-[]
        behaviour.
        """
        svc = MemoryService(mock_config)
        svc.mem0 = Mem0Backend(mock_config)
        mock_client = AsyncMock()
        mock_client.scroll = AsyncMock(side_effect=TimeoutError('qdrant scroll timed out'))
        server = create_mcp_server(svc)

        with patch.object(svc.mem0, '_get_async_qdrant', AsyncMock(return_value=mock_client)):
            result = await server._tool_manager.call_tool(
                'get_memories_by_metadata',
                {'project_id': _PROJECT_ID, 'filters': _FILTERS},
            )

        assert isinstance(result, dict), f'Expected dict, got {type(result)}: {result!r}'
        assert 'error' in result, f"Expected error key in result: {result!r}"
        assert result.get('error_type') == 'TimeoutError', (
            f"Expected error_type='TimeoutError', got: {result!r}"
        )
        assert result.get('memories') is None, (
            f"Timeout must not masquerade as an empty memories list: {result!r}"
        )


class TestGetMemoriesByMetadataTruncationDisclosure:
    """Truncation-disclosure tests (task 2562 item 2).

    The response now self-discloses truncation via ``truncated``/``total``
    instead of silently capping at ``limit`` with no way for the caller to
    detect it happened.
    """

    @pytest.mark.asyncio
    async def test_truncated_when_result_hits_limit_and_true_count_is_larger(self):
        """Exactly `limit` records returned + a larger true count -> truncated=True."""
        mock_service = AsyncMock()
        three_records = [{'id': f'mem-{i}'} for i in range(3)]
        mock_service.get_memories_by_metadata = AsyncMock(return_value=three_records)
        mock_service.count_memories_by_metadata = AsyncMock(return_value=10)
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'get_memories_by_metadata',
            {'project_id': _PROJECT_ID, 'filters': _FILTERS, 'limit': 3},
        )

        assert result['truncated'] is True, f'Expected truncated=True, got: {result!r}'
        assert result['total'] == 10, f"Expected total=10, got: {result!r}"
        mock_service.count_memories_by_metadata.assert_called_once_with(
            project_id=_PROJECT_ID,
            filters=_FILTERS,
        )

    @pytest.mark.asyncio
    async def test_not_truncated_when_below_limit_skips_count_call(self):
        """Fewer than `limit` records -> truncated=False, total=len(memories), and
        count_memories_by_metadata is never called (no extra Qdrant round-trip)."""
        mock_service = AsyncMock()
        mock_service.get_memories_by_metadata = AsyncMock(return_value=_SAMPLE_MEMORIES)
        mock_service.count_memories_by_metadata = AsyncMock(return_value=999)
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'get_memories_by_metadata',
            {'project_id': _PROJECT_ID, 'filters': _FILTERS},
        )

        assert result['truncated'] is False, f'Expected truncated=False, got: {result!r}'
        assert result['total'] == len(_SAMPLE_MEMORIES), f'Expected total==2, got: {result!r}'
        mock_service.count_memories_by_metadata.assert_not_called()

    @pytest.mark.asyncio
    async def test_boundary_exact_limit_and_matching_count_not_truncated(self):
        """Result length == limit AND true count == limit -> truncated=False."""
        mock_service = AsyncMock()
        mock_service.get_memories_by_metadata = AsyncMock(return_value=_SAMPLE_MEMORIES)
        mock_service.count_memories_by_metadata = AsyncMock(return_value=len(_SAMPLE_MEMORIES))
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'get_memories_by_metadata',
            {'project_id': _PROJECT_ID, 'filters': _FILTERS, 'limit': len(_SAMPLE_MEMORIES)},
        )

        assert result['truncated'] is False, f'Expected truncated=False, got: {result!r}'
        assert result['total'] == len(_SAMPLE_MEMORIES), f'Expected total==2, got: {result!r}'
        mock_service.count_memories_by_metadata.assert_called_once_with(
            project_id=_PROJECT_ID,
            filters=_FILTERS,
        )
