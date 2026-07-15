"""Integration tests for the live-task-status current-fact framing write-gate
in add_memory and add_episode (task 2628)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.server.tools import create_mcp_server

_LIVE_STATUS_CONTENT = 'task 5207 status=in-progress, actively driven by run abc123'
_DURABLE_POINT_IN_TIME_CONTENT = (
    'A liveness check performed at 2026-07-14T20:31:00Z reported task 5207 '
    'status=in-progress.'
)
_PROJECT_ID = 'dark_factory'


class TestAddMemoryLiveStatusGate:
    """Write-gate: recon-stage- agents must not write decisions_and_rationale /
    observations_and_summaries / entities_and_relations / temporal_facts
    content via add_memory that frames a LIVE task-table field/value (status=,
    claimant_run_id=, heartbeat_at=, pid=, or a paraphrase) as a
    standing/current fact rather than a timestamped point-in-time check.
    """

    @pytest.mark.asyncio
    async def test_rejects_live_status_from_recon_stage_agent_decisions_and_rationale(self):
        """Live-status content + category=decisions_and_rationale + recon-stage
        agent → rejected.

        The gate must return the standard error dict and must NOT call
        memory_service.add_memory.
        """
        mock_service = AsyncMock()
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _LIVE_STATUS_CONTENT,
                'category': 'decisions_and_rationale',
                'agent_id': 'recon-stage-task_knowledge_sync',
                'project_id': _PROJECT_ID,
            },
        )

        assert isinstance(result, dict), f'Expected dict, got {type(result)}: {result!r}'
        assert result.get('error') == 'live_task_status_current_fact_write_blocked', (
            f"Expected error='live_task_status_current_fact_write_blocked', got: {result!r}"
        )
        assert result.get('error_type') == 'ReconLiveTaskStatusWriteRejected', (
            f"Expected error_type='ReconLiveTaskStatusWriteRejected', got: {result!r}"
        )
        assert result.get('agent_id') == 'recon-stage-task_knowledge_sync', (
            f'Expected agent_id echoed, got: {result!r}'
        )
        assert result.get('content_excerpt') == _LIVE_STATUS_CONTENT[:200], (
            f'Expected content_excerpt=content[:200], got: {result!r}'
        )
        assert result.get('hint'), f'Expected a non-empty hint, got: {result!r}'
        mock_service.add_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejects_live_status_from_recon_stage_agent_observations_and_summaries(self):
        """Same content + category=observations_and_summaries → also blocked
        (covers the 2nd target category named by the task).
        """
        mock_service = AsyncMock()
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _LIVE_STATUS_CONTENT,
                'category': 'observations_and_summaries',
                'agent_id': 'recon-stage-task_knowledge_sync',
                'project_id': _PROJECT_ID,
            },
        )

        assert result.get('error') == 'live_task_status_current_fact_write_blocked', (
            f"Expected error='live_task_status_current_fact_write_blocked', got: {result!r}"
        )
        assert result.get('error_type') == 'ReconLiveTaskStatusWriteRejected', (
            f"Expected error_type='ReconLiveTaskStatusWriteRejected', got: {result!r}"
        )
        mock_service.add_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_allows_live_status_from_non_recon_agent(self):
        """Same content but agent_id='claude-interactive' (non-recon) → allowed
        through. The gate must NOT fire for agent_ids that don't start with
        'recon-stage-'.
        """
        mock_service = AsyncMock()
        _mem_result = MagicMock()
        _mem_result.model_dump.return_value = {'id': 'x'}
        mock_service.add_memory.return_value = _mem_result
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _LIVE_STATUS_CONTENT,
                'category': 'decisions_and_rationale',
                'agent_id': 'claude-interactive',
                'project_id': _PROJECT_ID,
            },
        )

        assert result.get('error') != 'live_task_status_current_fact_write_blocked', (
            f'Gate must not fire for non-recon agent; got: {result!r}'
        )
        mock_service.add_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_allows_live_status_with_none_category(self):
        """Same content + category=None (auto-classify) → NOT blocked. Bypassing
        category=None mirrors the conflicting-status precedent.
        """
        mock_service = AsyncMock()
        _mem_result = MagicMock()
        _mem_result.model_dump.return_value = {'id': 'y'}
        mock_service.add_memory.return_value = _mem_result
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _LIVE_STATUS_CONTENT,
                'category': None,
                'agent_id': 'recon-stage-task_knowledge_sync',
                'project_id': _PROJECT_ID,
            },
        )

        assert result.get('error') != 'live_task_status_current_fact_write_blocked', (
            f'Gate must not fire when category=None (auto-classify path); got: {result!r}'
        )
        mock_service.add_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_allows_live_status_with_procedural_knowledge_category(self):
        """Same content + category=procedural_knowledge (non-gated category,
        where a norm ABOUT recording liveness legitimately lives) → NOT
        blocked.
        """
        mock_service = AsyncMock()
        _mem_result = MagicMock()
        _mem_result.model_dump.return_value = {'id': 'z'}
        mock_service.add_memory.return_value = _mem_result
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _LIVE_STATUS_CONTENT,
                'category': 'procedural_knowledge',
                'agent_id': 'recon-stage-task_knowledge_sync',
                'project_id': _PROJECT_ID,
            },
        )

        assert result.get('error') != 'live_task_status_current_fact_write_blocked', (
            f'Gate must not fire for non-gated category=procedural_knowledge; got: {result!r}'
        )
        mock_service.add_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_allows_durable_point_in_time_form_from_recon_stage_agent(self):
        """The durable timestamped point-in-time-check form + category=
        decisions_and_rationale + recon-stage agent → NOT blocked. Proves the
        guard rewards the correct form instead of blocking it.
        """
        mock_service = AsyncMock()
        _mem_result = MagicMock()
        _mem_result.model_dump.return_value = {'id': 'w'}
        mock_service.add_memory.return_value = _mem_result
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _DURABLE_POINT_IN_TIME_CONTENT,
                'category': 'decisions_and_rationale',
                'agent_id': 'recon-stage-task_knowledge_sync',
                'project_id': _PROJECT_ID,
            },
        )

        assert result.get('error') != 'live_task_status_current_fact_write_blocked', (
            f'Gate must not fire for the durable point-in-time-check form; got: {result!r}'
        )
        mock_service.add_memory.assert_called_once()


class TestAddEpisodeLiveStatusGate:
    """Write-gate: recon-stage- agents must not write content via add_episode that
    frames a LIVE task-table field/value as a standing/current fact rather than a
    timestamped point-in-time check. Episodes carry no `category`, so the gate
    fires on recon-stage + content alone (mirroring the conflicting-status /
    mixed-framing add_episode guards).
    """

    @pytest.mark.asyncio
    async def test_rejects_live_status_from_recon_stage_agent(self):
        """Live-status content + recon-stage agent → rejected.

        The gate must return the standard error dict and must NOT call
        memory_service.add_episode.
        """
        mock_service = AsyncMock()
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_episode',
            {
                'content': _LIVE_STATUS_CONTENT,
                'agent_id': 'recon-stage-task_knowledge_sync',
                'project_id': _PROJECT_ID,
            },
        )

        assert isinstance(result, dict), f'Expected dict, got {type(result)}: {result!r}'
        assert result.get('error') == 'live_task_status_current_fact_write_blocked', (
            f"Expected error='live_task_status_current_fact_write_blocked', got: {result!r}"
        )
        assert result.get('error_type') == 'ReconLiveTaskStatusWriteRejected', (
            f"Expected error_type='ReconLiveTaskStatusWriteRejected', got: {result!r}"
        )
        assert result.get('content_excerpt') == _LIVE_STATUS_CONTENT[:200], (
            f'Expected content_excerpt=content[:200], got: {result!r}'
        )
        assert result.get('hint'), f'Expected a non-empty hint, got: {result!r}'
        mock_service.add_episode.assert_not_called()

    @pytest.mark.asyncio
    async def test_allows_live_status_from_non_recon_agent(self):
        """Same content but non-recon agent_id → allowed through.

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
                'content': _LIVE_STATUS_CONTENT,
                'agent_id': 'claude-interactive',
                'project_id': _PROJECT_ID,
            },
        )

        assert result.get('error') != 'live_task_status_current_fact_write_blocked', (
            f'Gate must not fire for non-recon agent; got: {result!r}'
        )
        mock_service.add_episode.assert_called_once()

    @pytest.mark.asyncio
    async def test_allows_durable_point_in_time_form_from_recon_stage_agent(self):
        """The durable timestamped point-in-time-check form + recon-stage agent
        → NOT blocked. Proves the guard rewards the correct form instead of
        blocking it.
        """
        mock_service = AsyncMock()
        _episode_result = MagicMock()
        _episode_result.model_dump.return_value = {'id': 'y'}
        mock_service.add_episode.return_value = _episode_result
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_episode',
            {
                'content': _DURABLE_POINT_IN_TIME_CONTENT,
                'agent_id': 'recon-stage-task_knowledge_sync',
                'project_id': _PROJECT_ID,
            },
        )

        assert result.get('error') != 'live_task_status_current_fact_write_blocked', (
            f'Gate must not fire for the durable point-in-time-check form; got: {result!r}'
        )
        mock_service.add_episode.assert_called_once()
