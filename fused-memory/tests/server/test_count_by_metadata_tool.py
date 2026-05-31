"""Behaviour tests for the count_memories_by_metadata MCP tool (task 1588).

The tool is a read-only exact Qdrant payload-filter count — not semantic —
intended for deterministic existence checks such as confirming whether a Stage 2
per-cycle summary is present for a given run_id.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from fused_memory.server.tools import create_mcp_server

_PROJECT_ID = 'dark_factory'
_FILTERS = {'kind': 'cycle_summary', 'run_id': '80a85eeb'}


class TestCountMemoriesByMetadataTool:
    """Behaviour tests for mcp__fused-memory__count_memories_by_metadata."""

    @pytest.mark.asyncio
    async def test_happy_path_returns_count_and_forwards_args(self):
        """Happy path: service returns 3; tool returns {'count': 3} and calls
        service exactly once with the right project_id and filters."""
        mock_service = AsyncMock()
        mock_service.count_memories_by_metadata = AsyncMock(return_value=3)
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'count_memories_by_metadata',
            {
                'project_id': _PROJECT_ID,
                'filters': _FILTERS,
            },
        )

        assert isinstance(result, dict), f'Expected dict, got {type(result)}: {result!r}'
        assert result.get('count') == 3, (
            f"Expected count=3, got: {result!r}"
        )
        # Must not be an error response
        assert 'error' not in result, f"Unexpected error in result: {result!r}"

        # Service must have been called exactly once with the right arguments
        mock_service.count_memories_by_metadata.assert_called_once_with(
            project_id=_PROJECT_ID,
            filters=_FILTERS,
        )

    @pytest.mark.asyncio
    async def test_zero_match_returns_count_zero_not_error(self):
        """When service returns 0, the tool returns {'count': 0} — not an error."""
        mock_service = AsyncMock()
        mock_service.count_memories_by_metadata = AsyncMock(return_value=0)
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'count_memories_by_metadata',
            {
                'project_id': _PROJECT_ID,
                'filters': {'kind': 'cycle_summary', 'run_id': 'notfound'},
            },
        )

        assert isinstance(result, dict), f'Expected dict, got {type(result)}: {result!r}'
        assert result.get('count') == 0, (
            f"Expected count=0, got: {result!r}"
        )
        assert 'error' not in result, f"Unexpected error in result: {result!r}"

        # Service must have been called with the exact project_id and filters
        mock_service.count_memories_by_metadata.assert_called_once_with(
            project_id=_PROJECT_ID,
            filters={'kind': 'cycle_summary', 'run_id': 'notfound'},
        )

    @pytest.mark.asyncio
    async def test_invalid_project_id_returns_validation_error_without_calling_service(self):
        """Invalid project_id (contains unsafe chars) returns a validation error dict
        and does NOT call the service."""
        mock_service = AsyncMock()
        mock_service.count_memories_by_metadata = AsyncMock(return_value=0)
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'count_memories_by_metadata',
            {
                'project_id': 'bad project id!',  # contains spaces and !
                'filters': _FILTERS,
            },
        )

        assert isinstance(result, dict), f'Expected dict, got {type(result)}: {result!r}'
        assert 'error' in result, f"Expected error key in result: {result!r}"
        assert result.get('error_type') == 'ValidationError', (
            f"Expected error_type='ValidationError', got: {result!r}"
        )
        # Service must NOT have been called
        mock_service.count_memories_by_metadata.assert_not_called()

    @pytest.mark.asyncio
    async def test_service_exception_is_caught_and_returned_as_error_dict(self):
        """When the service raises (e.g. ValueError on empty filters), the tool
        catches it and returns {'error': ..., 'error_type': ...} without propagating."""
        mock_service = AsyncMock()
        mock_service.count_memories_by_metadata = AsyncMock(
            side_effect=ValueError('filters must not be empty')
        )
        server = create_mcp_server(mock_service)

        # Should not raise — exception must be caught
        result = await server._tool_manager.call_tool(
            'count_memories_by_metadata',
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
        # The error message should contain the original exception message
        assert 'filters must not be empty' in result.get('error', ''), (
            f"Expected original error message in result: {result!r}"
        )
