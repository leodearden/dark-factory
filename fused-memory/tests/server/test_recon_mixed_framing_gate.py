"""Integration tests for the mixed-temporal-framing write-gate in add_memory
and add_episode (task 1950)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.server.tools import create_mcp_server

_MIXED_FRAMING_CONTENT = (
    'BillingService.run_billing previously used BillingConfig with static '
    'defaults; it now overlays governed own_use_rate_pence_per_kwh onto the '
    'BillingConfig'
)
_RESULTING_STATE_ONLY_CONTENT = (
    'BillingService.run_billing overlays governed own_use_rate_pence_per_kwh '
    'onto the BillingConfig'
)
_PROJECT_ID = 'dark_factory'


class TestAddMemoryMixedFramingGate:
    """Write-gate: recon-stage- agents must not write mixed-temporal-framing
    (was-X-now-Y) temporal_facts.
    """

    @pytest.mark.asyncio
    async def test_rejects_mixed_framing_from_recon_stage_agent(self):
        """Mixed-framing content + category=temporal_facts + recon-stage agent → rejected.

        The gate must return the standard error dict and must NOT call
        memory_service.add_memory.
        """
        mock_service = AsyncMock()
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _MIXED_FRAMING_CONTENT,
                'category': 'temporal_facts',
                'agent_id': 'recon-stage-memory_consolidator',
                'project_id': _PROJECT_ID,
            },
        )

        assert isinstance(result, dict), f'Expected dict, got {type(result)}: {result!r}'
        assert result.get('error') == 'mixed_temporal_framing_write_blocked', (
            f"Expected error='mixed_temporal_framing_write_blocked', got: {result!r}"
        )
        assert result.get('error_type') == 'ReconMixedFramingWriteRejected', (
            f"Expected error_type='ReconMixedFramingWriteRejected', got: {result!r}"
        )
        assert result.get('agent_id') == 'recon-stage-memory_consolidator', (
            f"Expected agent_id='recon-stage-memory_consolidator', got: {result!r}"
        )
        assert result.get('content_excerpt') == _MIXED_FRAMING_CONTENT[:200], (
            f'Expected content_excerpt=content[:200], got: {result!r}'
        )
        mock_service.add_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_allows_mixed_framing_from_non_recon_agent(self):
        """Same mixed-framing content+category but non-recon agent_id → allowed through.

        The gate must NOT fire for agent_ids that don't start with 'recon-stage-'.
        """
        mock_service = AsyncMock()
        _mem_result = MagicMock()
        _mem_result.model_dump.return_value = {'id': 'x'}
        mock_service.add_memory.return_value = _mem_result
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _MIXED_FRAMING_CONTENT,
                'category': 'temporal_facts',
                'agent_id': 'claude-interactive',
                'project_id': _PROJECT_ID,
            },
        )

        assert result.get('error') != 'mixed_temporal_framing_write_blocked', (
            f'Gate must not fire for non-recon agent; got: {result!r}'
        )
        mock_service.add_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_allows_resulting_state_only_from_recon_stage_agent(self):
        """Resulting-state-only content + temporal_facts + recon-stage agent → allowed through.

        The gate only fires when BOTH a prior-state marker and a resulting-state
        marker co-occur; resulting-state-only prose must not be blocked.
        """
        mock_service = AsyncMock()
        _mem_result = MagicMock()
        _mem_result.model_dump.return_value = {'id': 'y'}
        mock_service.add_memory.return_value = _mem_result
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _RESULTING_STATE_ONLY_CONTENT,
                'category': 'temporal_facts',
                'agent_id': 'recon-stage-memory_consolidator',
                'project_id': _PROJECT_ID,
            },
        )

        assert result.get('error') != 'mixed_temporal_framing_write_blocked', (
            f'Gate must not fire for resulting-state-only content; got: {result!r}'
        )
        mock_service.add_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_allows_mixed_framing_with_none_category_from_recon_stage_agent(self):
        """category=None + mixed-framing content + recon-stage agent → gate does NOT fire.

        The gate is intentionally scoped to category='temporal_facts' only, mirroring
        the count-snapshot gate's scoping. When category is None (auto-classification),
        the gate is bypassed.
        """
        mock_service = AsyncMock()
        _mem_result = MagicMock()
        _mem_result.model_dump.return_value = {'id': 'z'}
        mock_service.add_memory.return_value = _mem_result
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _MIXED_FRAMING_CONTENT,
                'category': None,  # auto-classification path
                'agent_id': 'recon-stage-memory_consolidator',
                'project_id': _PROJECT_ID,
            },
        )

        assert result.get('error') != 'mixed_temporal_framing_write_blocked', (
            f'Gate must not fire when category=None (auto-classify path); got: {result!r}'
        )
        mock_service.add_memory.assert_called_once()


class TestAddEpisodeMixedFramingGate:
    """Write-gate: recon-stage- agents must not write mixed-temporal-framing
    (was-X-now-Y) content via add_episode either — add_episode has no category
    param, so this guard is scoped to agent prefix + content only.
    """

    @pytest.mark.asyncio
    async def test_rejects_mixed_framing_from_recon_stage_agent(self):
        """Mixed-framing content + recon-stage agent → rejected.

        The gate must return the standard error dict and must NOT call
        memory_service.add_episode.
        """
        mock_service = AsyncMock()
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_episode',
            {
                'content': _MIXED_FRAMING_CONTENT,
                'agent_id': 'recon-stage-task_knowledge_sync',
                'project_id': _PROJECT_ID,
            },
        )

        assert isinstance(result, dict), f'Expected dict, got {type(result)}: {result!r}'
        assert result.get('error') == 'mixed_temporal_framing_write_blocked', (
            f"Expected error='mixed_temporal_framing_write_blocked', got: {result!r}"
        )
        assert result.get('error_type') == 'ReconMixedFramingWriteRejected', (
            f"Expected error_type='ReconMixedFramingWriteRejected', got: {result!r}"
        )
        assert result.get('agent_id') == 'recon-stage-task_knowledge_sync', (
            f"Expected agent_id='recon-stage-task_knowledge_sync', got: {result!r}"
        )
        assert result.get('content_excerpt') == _MIXED_FRAMING_CONTENT[:200], (
            f'Expected content_excerpt=content[:200], got: {result!r}'
        )
        mock_service.add_episode.assert_not_called()

    @pytest.mark.asyncio
    async def test_allows_mixed_framing_from_non_recon_agent(self):
        """Same mixed-framing content but non-recon agent_id → allowed through.

        The gate must NOT fire for agent_ids that don't start with 'recon-stage-'.
        """
        mock_service = AsyncMock()
        _episode_result = MagicMock()
        _episode_result.model_dump.return_value = {'id': 'x'}
        mock_service.add_episode.return_value = _episode_result
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_episode',
            {
                'content': _MIXED_FRAMING_CONTENT,
                'agent_id': 'claude-interactive',
                'project_id': _PROJECT_ID,
            },
        )

        assert result.get('error') != 'mixed_temporal_framing_write_blocked', (
            f'Gate must not fire for non-recon agent; got: {result!r}'
        )
        mock_service.add_episode.assert_called_once()

    @pytest.mark.asyncio
    async def test_allows_resulting_state_only_from_recon_stage_agent(self):
        """Resulting-state-only content + recon-stage agent → allowed through.

        The gate only fires when BOTH a prior-state marker and a resulting-state
        marker co-occur; resulting-state-only prose must not be blocked.
        """
        mock_service = AsyncMock()
        _episode_result = MagicMock()
        _episode_result.model_dump.return_value = {'id': 'y'}
        mock_service.add_episode.return_value = _episode_result
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_episode',
            {
                'content': _RESULTING_STATE_ONLY_CONTENT,
                'agent_id': 'recon-stage-task_knowledge_sync',
                'project_id': _PROJECT_ID,
            },
        )

        assert result.get('error') != 'mixed_temporal_framing_write_blocked', (
            f'Gate must not fire for resulting-state-only content; got: {result!r}'
        )
        mock_service.add_episode.assert_called_once()
