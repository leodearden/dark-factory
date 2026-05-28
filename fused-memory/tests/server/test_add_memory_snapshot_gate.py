"""Integration tests for the count-snapshot write-gate in add_memory (task 1547)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.server.tools import create_mcp_server

_SNAPSHOT_CONTENT = (
    'As of 2026-05-28, project reify has 2 pending / 2 in-progress / '
    '0 blocked / 1 deferred / 1505 done / 148 cancelled tasks'
)
_LEGIT_CONTENT = 'Task 42 done via commit abc'
_PROJECT_ID = 'dark_factory'


class TestAddMemorySnapshotGate:
    """Write-gate: recon-stage- agents must not write count-snapshot temporal_facts."""

    @pytest.mark.asyncio
    async def test_rejects_snapshot_from_recon_stage_agent(self):
        """Snapshot content + category=temporal_facts + recon-stage agent → rejected.

        The gate must return the standard error dict and must NOT call
        memory_service.add_memory.
        """
        mock_service = AsyncMock()
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _SNAPSHOT_CONTENT,
                'category': 'temporal_facts',
                'agent_id': 'recon-stage-task_knowledge_sync',
                'project_id': _PROJECT_ID,
            },
        )

        assert isinstance(result, dict), f'Expected dict, got {type(result)}: {result!r}'
        assert result.get('error') == 'count_snapshot_write_blocked', (
            f"Expected error='count_snapshot_write_blocked', got: {result!r}"
        )
        assert result.get('error_type') == 'ReconSnapshotWriteRejected', (
            f"Expected error_type='ReconSnapshotWriteRejected', got: {result!r}"
        )
        assert result.get('agent_id') == 'recon-stage-task_knowledge_sync', (
            f"Expected agent_id='recon-stage-task_knowledge_sync', got: {result!r}"
        )
        assert result.get('content_excerpt') == _SNAPSHOT_CONTENT[:200], (
            f'Expected content_excerpt=content[:200], got: {result!r}'
        )
        mock_service.add_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_allows_snapshot_from_non_recon_agent(self):
        """Same snapshot content+category but non-recon agent_id → allowed through.

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
                'content': _SNAPSHOT_CONTENT,
                'category': 'temporal_facts',
                'agent_id': 'claude-interactive',
                'project_id': _PROJECT_ID,
            },
        )

        assert result.get('error') != 'count_snapshot_write_blocked', (
            f'Gate must not fire for non-recon agent; got: {result!r}'
        )
        mock_service.add_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_allows_snapshot_with_none_category_from_recon_stage_agent(self):
        """category=None + snapshot content + recon-stage agent → gate does NOT fire.

        The gate is intentionally scoped to category='temporal_facts' only.  When
        category is None (auto-classification), the gate is bypassed — recon stages
        are expected to supply category explicitly for the gate to take effect.
        This test pins that trade-off as intentional so the behaviour cannot regress
        silently.  See plan design decision: 'Restricting to temporal_facts keeps
        the gate scoped to durable Graphiti edges.'
        """
        mock_service = AsyncMock()
        _mem_result = MagicMock()
        _mem_result.model_dump.return_value = {'id': 'z'}
        mock_service.add_memory.return_value = _mem_result
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _SNAPSHOT_CONTENT,
                'category': None,  # auto-classification path
                'agent_id': 'recon-stage-memory_consolidator',
                'project_id': _PROJECT_ID,
            },
        )

        assert result.get('error') != 'count_snapshot_write_blocked', (
            f'Gate must not fire when category=None (auto-classify path); got: {result!r}'
        )
        mock_service.add_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_allows_legit_mention_from_recon_stage_agent(self):
        """Legit single-done mention with recon-stage agent → allowed through.

        The gate only fires for count-snapshot lines (>=2 digit+status tokens).
        A single incidental 'done' mention must not be blocked.
        """
        mock_service = AsyncMock()
        _mem_result = MagicMock()
        _mem_result.model_dump.return_value = {'id': 'y'}
        mock_service.add_memory.return_value = _mem_result
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _LEGIT_CONTENT,
                'category': 'temporal_facts',
                'agent_id': 'recon-stage-memory_consolidator',
                'project_id': _PROJECT_ID,
            },
        )

        assert result.get('error') != 'count_snapshot_write_blocked', (
            f'Gate must not fire for legit single-done mention; got: {result!r}'
        )
        mock_service.add_memory.assert_called_once()
