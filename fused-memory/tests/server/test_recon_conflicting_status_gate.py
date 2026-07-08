"""Integration tests for the conflicting-task-status-framing write-gate in
add_episode and add_memory (task 2276)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.server.tools import create_mcp_server

_CONFLICTING_CONTENT = (
    '...still-pending Task 4802 stale. Task 4802 had already landed as '
    'merge commit eb7f5d6437.'
)
_TERMINAL_ONLY_CONTENT = 'Task 4802 has landed as merge commit eb7f5d6437.'
_NON_TERMINAL_ONLY_CONTENT = 'Task 4802 is still pending.'
_PROJECT_ID = 'dark_factory'


class TestAddEpisodeConflictingStatusGate:
    """Write-gate: recon-stage- agents must not write content via add_episode that
    frames the SAME task_id as both non-terminal and terminal.
    """

    @pytest.mark.asyncio
    async def test_rejects_conflicting_status_from_recon_stage_agent(self):
        """Conflicting-status content + recon-stage agent → rejected.

        The gate must return the standard error dict and must NOT call
        memory_service.add_episode.
        """
        mock_service = AsyncMock()
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_episode',
            {
                'content': _CONFLICTING_CONTENT,
                'agent_id': 'recon-stage-task_knowledge_sync',
                'project_id': _PROJECT_ID,
            },
        )

        assert isinstance(result, dict), f'Expected dict, got {type(result)}: {result!r}'
        assert result.get('error') == 'conflicting_task_status_framing_write_blocked', (
            f"Expected error='conflicting_task_status_framing_write_blocked', got: {result!r}"
        )
        assert result.get('error_type') == 'ReconConflictingTaskStatusWriteRejected', (
            f"Expected error_type='ReconConflictingTaskStatusWriteRejected', got: {result!r}"
        )
        assert result.get('agent_id') == 'recon-stage-task_knowledge_sync', (
            f"Expected agent_id='recon-stage-task_knowledge_sync', got: {result!r}"
        )
        assert result.get('content_excerpt') == _CONFLICTING_CONTENT[:200], (
            f'Expected content_excerpt=content[:200], got: {result!r}'
        )
        assert result.get('conflicting_task_ids') == [4802], (
            f'Expected conflicting_task_ids=[4802], got: {result!r}'
        )
        mock_service.add_episode.assert_not_called()

    @pytest.mark.asyncio
    async def test_allows_conflicting_status_from_non_recon_agent(self):
        """Same conflicting-status content but non-recon agent_id → allowed through.

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
                'content': _CONFLICTING_CONTENT,
                'agent_id': 'claude-interactive',
                'project_id': _PROJECT_ID,
            },
        )

        assert result.get('error') != 'conflicting_task_status_framing_write_blocked', (
            f'Gate must not fire for non-recon agent; got: {result!r}'
        )
        mock_service.add_episode.assert_called_once()

    @pytest.mark.asyncio
    async def test_allows_single_framing_terminal_only_from_recon_stage_agent(self):
        """Terminal-only (single-framing) content + recon-stage agent → allowed through.

        The gate only fires when the SAME task_id is framed both non-terminal and
        terminal; terminal-only content must not be blocked.
        """
        mock_service = AsyncMock()
        _episode_result = MagicMock()
        _episode_result.model_dump.return_value = {'id': 'y'}
        mock_service.add_episode.return_value = _episode_result
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_episode',
            {
                'content': _TERMINAL_ONLY_CONTENT,
                'agent_id': 'recon-stage-task_knowledge_sync',
                'project_id': _PROJECT_ID,
            },
        )

        assert result.get('error') != 'conflicting_task_status_framing_write_blocked', (
            f'Gate must not fire for terminal-only content; got: {result!r}'
        )
        mock_service.add_episode.assert_called_once()


class TestAddMemoryConflictingStatusGate:
    """Write-gate: recon-stage- agents must not write temporal_facts content via
    add_memory that frames the SAME task_id as both non-terminal and terminal.
    """

    @pytest.mark.asyncio
    async def test_rejects_conflicting_status_from_recon_stage_agent(self):
        """Conflicting-status content + category=temporal_facts + recon-stage agent
        → rejected.

        The gate must return the standard error dict and must NOT call
        memory_service.add_memory.
        """
        mock_service = AsyncMock()
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _CONFLICTING_CONTENT,
                'category': 'temporal_facts',
                'agent_id': 'recon-stage-memory_consolidator',
                'project_id': _PROJECT_ID,
            },
        )

        assert isinstance(result, dict), f'Expected dict, got {type(result)}: {result!r}'
        assert result.get('error') == 'conflicting_task_status_framing_write_blocked', (
            f"Expected error='conflicting_task_status_framing_write_blocked', got: {result!r}"
        )
        assert result.get('error_type') == 'ReconConflictingTaskStatusWriteRejected', (
            f"Expected error_type='ReconConflictingTaskStatusWriteRejected', got: {result!r}"
        )
        assert result.get('conflicting_task_ids') == [4802], (
            f'Expected conflicting_task_ids=[4802], got: {result!r}'
        )
        mock_service.add_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_allows_conflicting_status_from_non_recon_agent(self):
        """Same conflicting-status content + temporal_facts but non-recon agent_id
        → allowed through.

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
                'content': _CONFLICTING_CONTENT,
                'category': 'temporal_facts',
                'agent_id': 'claude-interactive',
                'project_id': _PROJECT_ID,
            },
        )

        assert result.get('error') != 'conflicting_task_status_framing_write_blocked', (
            f'Gate must not fire for non-recon agent; got: {result!r}'
        )
        mock_service.add_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_allows_conflicting_status_with_none_category_from_recon_stage_agent(self):
        """category=None + conflicting-status content + recon-stage agent → gate
        does NOT fire.

        The gate is intentionally scoped to category='temporal_facts' only,
        mirroring the count-snapshot/mixed-framing gates. When category is None
        (auto-classification), the gate is bypassed.
        """
        mock_service = AsyncMock()
        _mem_result = MagicMock()
        _mem_result.model_dump.return_value = {'id': 'z'}
        mock_service.add_memory.return_value = _mem_result
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _CONFLICTING_CONTENT,
                'category': None,  # auto-classification path
                'agent_id': 'recon-stage-memory_consolidator',
                'project_id': _PROJECT_ID,
            },
        )

        assert result.get('error') != 'conflicting_task_status_framing_write_blocked', (
            f'Gate must not fire when category=None (auto-classify path); got: {result!r}'
        )
        mock_service.add_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_allows_single_framing_non_terminal_only_from_recon_stage_agent(self):
        """Non-terminal-only (single-framing) content + temporal_facts + recon-stage
        agent → allowed through.

        The gate only fires when the SAME task_id is framed both non-terminal and
        terminal; non-terminal-only content must not be blocked.
        """
        mock_service = AsyncMock()
        _mem_result = MagicMock()
        _mem_result.model_dump.return_value = {'id': 'w'}
        mock_service.add_memory.return_value = _mem_result
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _NON_TERMINAL_ONLY_CONTENT,
                'category': 'temporal_facts',
                'agent_id': 'recon-stage-memory_consolidator',
                'project_id': _PROJECT_ID,
            },
        )

        assert result.get('error') != 'conflicting_task_status_framing_write_blocked', (
            f'Gate must not fire for non-terminal-only content; got: {result!r}'
        )
        mock_service.add_memory.assert_called_once()
