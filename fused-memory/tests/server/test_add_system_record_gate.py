"""Integration tests for the recon-stage-only gate on add_system_record (task 2222 / W5-δ)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.server.tools import create_mcp_server

_PROJECT_ID = 'dark_factory'
_CONTENT = 'Stage 2 cycle summary for run r1'
_CATEGORY = 'observations_and_summaries'


class TestAddSystemRecordGate:
    """Write-gate: only recon-stage- agents may use the dedup-exempt system-write path."""

    @pytest.mark.asyncio
    async def test_rejects_non_recon_agent(self):
        """Non-recon-stage callers must be rejected with a structured error dict.

        The gate must return the diagnostic dict and must NOT call
        memory_service.add_system_record.
        """
        mock_service = AsyncMock()
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_system_record',
            {
                'content': _CONTENT,
                'project_id': _PROJECT_ID,
                'category': _CATEGORY,
                'agent_id': 'claude-interactive',
            },
        )

        assert isinstance(result, dict), f'Expected dict, got {type(result)}: {result!r}'
        assert result.get('error_type') == 'DedupExemptNotPermitted', (
            f"Expected error_type='DedupExemptNotPermitted', got: {result!r}"
        )
        assert result.get('agent_id') == 'claude-interactive', (
            f"Expected agent_id='claude-interactive', got: {result!r}"
        )
        assert isinstance(result.get('error'), str) and result['error'], (
            f'Expected a non-empty error string, got: {result!r}'
        )
        mock_service.add_system_record.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejects_non_recon_agent_before_category_validation(self):
        """The access-control gate must fire before category validation.

        A non-recon-stage caller with an invalid category must still get
        DedupExemptNotPermitted, not a category ValidationError — otherwise
        the gate ordering leaks (a rejected caller learns something about
        argument validity) and does needless work for a caller that will be
        denied regardless (task 2222 amendment).
        """
        mock_service = AsyncMock()
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_system_record',
            {
                'content': _CONTENT,
                'project_id': _PROJECT_ID,
                'category': 'not_a_real_category',
                'agent_id': 'claude-interactive',
            },
        )

        assert result.get('error_type') == 'DedupExemptNotPermitted', (
            f'Expected the recon-stage gate to fire before category validation, '
            f'got: {result!r}'
        )
        mock_service.add_system_record.assert_not_called()

    @pytest.mark.asyncio
    async def test_allows_recon_stage_agent(self):
        """A recon-stage- caller must pass the gate and reach the service."""
        mock_service = AsyncMock()
        _r = MagicMock()
        _r.model_dump.return_value = {'memory_ids': ['sys-1'], 'stores_written': ['mem0']}
        mock_service.add_system_record.return_value = _r
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_system_record',
            {
                'content': _CONTENT,
                'project_id': _PROJECT_ID,
                'category': _CATEGORY,
                'agent_id': 'recon-stage-task_knowledge_sync',
            },
        )

        assert result.get('error_type') != 'DedupExemptNotPermitted', (
            f'Gate must not fire for a recon-stage agent; got: {result!r}'
        )
        mock_service.add_system_record.assert_called_once()
        call_kwargs = mock_service.add_system_record.call_args.kwargs
        assert call_kwargs.get('content') == _CONTENT
        assert call_kwargs.get('project_id') == _PROJECT_ID
        assert call_kwargs.get('category') == _CATEGORY
        assert call_kwargs.get('agent_id') == 'recon-stage-task_knowledge_sync'
