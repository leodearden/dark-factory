"""Behaviour tests for the get_memories_by_metadata MCP tool (task-1659).

The tool is a read-only deterministic Qdrant payload-filter scroll — not semantic —
intended for exhaustive enumeration of memories by metadata fields such as
confirming all stage1_flag_markers are present (source + kind).

Mirrors test_count_by_metadata_tool.py exactly.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from fused_memory.server.tools import create_mcp_server

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
