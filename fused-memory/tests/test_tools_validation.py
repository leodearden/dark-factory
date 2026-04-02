"""Regression tests verifying that tools.py delegates validation to the shared validators.

These tests replace the original vacuous tests that either called shared validators
directly (not through tools.py at all) or checked hasattr() on closures that were
never module-level attributes.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

import fused_memory.server.tools as tools_module
from fused_memory.server.tools import create_mcp_server
from fused_memory.utils.validation import validate_project_id, validate_project_root


class TestToolsDelegateToSharedValidators:
    """Object identity checks proving tools.py imports and re-exports the shared validators.

    If tools.py ever re-introduces private closures for validation, these tests fail
    immediately — the module attribute would no longer be the same object as the
    shared validator imported from utils.validation.
    """

    def test_validate_project_id_is_shared_validator(self):
        """tools.validate_project_id must be the exact same object as utils.validation.validate_project_id.

        tools.py line 14: `from fused_memory.utils.validation import validate_project_id`
        This creates a module-level attribute that is identical (by identity) to the
        shared validator. Any re-implementation would break this check.
        """
        assert tools_module.validate_project_id is validate_project_id, (
            "tools.validate_project_id is not the shared validator from utils.validation. "
            "tools.py must delegate to the shared validator, not re-implement it."
        )

    def test_validate_project_root_is_shared_validator(self):
        """tools.validate_project_root must be the exact same object as utils.validation.validate_project_root."""
        assert tools_module.validate_project_root is validate_project_root, (
            "tools.validate_project_root is not the shared validator from utils.validation. "
            "tools.py must delegate to the shared validator, not re-implement it."
        )


class TestToolsValidationIntegration:
    """Integration tests that invoke MCP handlers through server._tool_manager.call_tool().

    These verify that validation errors propagate correctly through the MCP handler
    layer, not just that the validator functions themselves work. An error dict from
    the validator must flow back to the caller without being swallowed or transformed.
    """

    @pytest.mark.asyncio
    async def test_add_memory_rejects_whitespace_project_id(self):
        """add_memory handler rejects whitespace-only project_id before reaching the service.

        Whitespace-only was the specific bug that motivated the validator consolidation
        in task 253. The old private _validate_project_id used `if not project_id`
        (truthy check only), which passed whitespace-only inputs like '   '.
        This test verifies that the bugfix propagates through the MCP handler layer.
        """
        mock_service = AsyncMock()
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {'content': 'test content', 'project_id': '   '},
        )

        assert isinstance(result, dict), f'Expected dict, got {type(result)}: {result!r}'
        assert 'error' in result, f'Expected error key in result: {result!r}'
        assert 'non-empty' in result['error'] or 'whitespace' in result['error'].lower(), (
            f"Expected error message to mention 'non-empty' or 'whitespace', got: {result['error']!r}"
        )
        assert result['error_type'] == 'ValidationError', (
            f"Expected error_type='ValidationError', got: {result['error_type']!r}"
        )
        # Validation must short-circuit before reaching the service
        mock_service.add_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_tasks_rejects_whitespace_project_root(self):
        """get_tasks handler rejects whitespace-only project_root before reaching the task interceptor.

        Validates that the project_root consolidation bugfix propagates through the
        MCP handler layer for task tools, mirroring the project_id test for memory tools.
        """
        mock_service = AsyncMock()
        mock_task_interceptor = AsyncMock()
        server = create_mcp_server(mock_service, task_interceptor=mock_task_interceptor)

        result = await server._tool_manager.call_tool(
            'get_tasks',
            {'project_root': '   '},
        )

        assert isinstance(result, dict), f'Expected dict, got {type(result)}: {result!r}'
        assert 'error' in result, f'Expected error key in result: {result!r}'
        assert result['error_type'] == 'ValidationError', (
            f"Expected error_type='ValidationError', got: {result['error_type']!r}"
        )
        # Validation must short-circuit before reaching the task interceptor
        mock_task_interceptor.get_tasks.assert_not_called()


class TestSearchIncludePlannedMCPTool:
    """step-19: MCP search tool accepts and forwards include_planned parameter."""

    @pytest.mark.asyncio
    async def test_include_planned_forwarded_to_service(self):
        """search MCP tool passes include_planned=True to memory_service.search."""
        mock_service = AsyncMock()
        mock_service.search.return_value = []
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'search',
            {
                'query': 'test query',
                'project_id': 'test',
                'include_planned': True,
            },
        )

        assert isinstance(result, dict), f'Expected dict result, got {type(result)}: {result!r}'
        assert 'results' in result, f'Expected results key in response: {result!r}'
        mock_service.search.assert_called_once()
        call_kwargs = mock_service.search.call_args[1]
        assert call_kwargs.get('include_planned') is True, (
            'include_planned=True must be forwarded from MCP search tool to memory_service.search'
        )

    @pytest.mark.asyncio
    async def test_include_planned_defaults_to_false(self):
        """search MCP tool passes include_planned=False when not specified."""
        mock_service = AsyncMock()
        mock_service.search.return_value = []
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'search',
            {
                'query': 'test query',
                'project_id': 'test',
            },
        )

        assert isinstance(result, dict), f'Expected dict result, got {type(result)}: {result!r}'
        assert 'results' in result, f'Expected results key in response: {result!r}'
        mock_service.search.assert_called_once()
        call_kwargs = mock_service.search.call_args[1]
        assert call_kwargs.get('include_planned', False) is False, (
            'include_planned must default to False in MCP search tool'
        )

    @pytest.mark.asyncio
    async def test_include_planned_false_forwarded_to_service(self):
        """search MCP tool passes include_planned=False when explicitly set to False."""
        mock_service = AsyncMock()
        mock_service.search.return_value = []
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'search',
            {
                'query': 'test query',
                'project_id': 'test',
                'include_planned': False,
            },
        )

        assert isinstance(result, dict), f'Expected dict result, got {type(result)}: {result!r}'
        assert 'results' in result, f'Expected results key in response: {result!r}'
        mock_service.search.assert_called_once()
        call_kwargs = mock_service.search.call_args[1]
        assert call_kwargs.get('include_planned') is False, (
            'include_planned=False must be forwarded to memory_service.search'
        )
