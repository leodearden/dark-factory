"""Integration tests for the stage1_flag_marker write-gate in add_memory (task 2596).

Mirrors tests/server/test_add_memory_snapshot_gate.py's harness. Task 2406
retired the deterministic Mem0 marker WRITE path — flag_dedup.dedup_flags
now UPSERTs stage1_flag_marker records only into the recon_ledger SQLite
table — so markers are code-managed and no legitimate add_memory call
should ever persist metadata.source/kind=='stage1_flag_marker' to Mem0.
This gate stops the only plausible remaining writer of that shape: an
off-script recon-stage-* LLM add_memory call (see stage1.py's corrected
completion-marker prose, task 2596 Prong C).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.server.tools import create_mcp_server

_PROJECT_ID = 'dark_factory'
_CONTENT = 'Stage 1 dedup completion note for task 2408.'


class TestReconFlagMarkerWriteGate:
    """Write-gate: recon-stage-* agents must not write stage1_flag_marker markers."""

    @pytest.mark.asyncio
    async def test_rejects_source_stage1_flag_marker_from_recon_stage_agent(self):
        """metadata.source=='stage1_flag_marker' + recon-stage agent -> rejected.

        The gate must return the standard error dict and must NOT call
        memory_service.add_memory.
        """
        mock_service = AsyncMock()
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _CONTENT,
                'category': 'observations_and_summaries',
                'agent_id': 'recon-stage-task_knowledge_sync',
                'project_id': _PROJECT_ID,
                'metadata': {'source': 'stage1_flag_marker', 'task_id': '2408'},
            },
        )

        assert isinstance(result, dict), f'Expected dict, got {type(result)}: {result!r}'
        assert result.get('error') == 'flag_marker_write_blocked', (
            f"Expected error='flag_marker_write_blocked', got: {result!r}"
        )
        assert result.get('error_type') == 'ReconFlagMarkerWriteRejected', (
            f"Expected error_type='ReconFlagMarkerWriteRejected', got: {result!r}"
        )
        assert result.get('agent_id') == 'recon-stage-task_knowledge_sync', (
            f"Expected agent_id echoed, got: {result!r}"
        )
        assert result.get('content_excerpt') == _CONTENT[:200], (
            f'Expected content_excerpt=content[:200], got: {result!r}'
        )
        mock_service.add_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejects_kind_stage1_flag_marker_from_recon_stage_agent(self):
        """metadata.kind=='stage1_flag_marker' (source absent) + recon-stage agent -> rejected."""
        mock_service = AsyncMock()
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _CONTENT,
                'category': 'observations_and_summaries',
                'agent_id': 'recon-stage-memory_consolidator',
                'project_id': _PROJECT_ID,
                'metadata': {'kind': 'stage1_flag_marker', 'task_id': '2408'},
            },
        )

        assert result.get('error') == 'flag_marker_write_blocked', (
            f"Expected error='flag_marker_write_blocked', got: {result!r}"
        )
        assert result.get('error_type') == 'ReconFlagMarkerWriteRejected', (
            f"Expected error_type='ReconFlagMarkerWriteRejected', got: {result!r}"
        )
        assert result.get('agent_id') == 'recon-stage-memory_consolidator', (
            f"Expected agent_id echoed, got: {result!r}"
        )
        mock_service.add_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_allows_stage1_flag_marker_metadata_from_non_recon_agent(self):
        """Same source/kind metadata but non-recon agent_id -> allowed through.

        The gate must NOT fire for agent_ids that don't start with
        'recon-stage-' (consistent with the sibling count_snapshot/
        mixed_framing guards — non-recon agents are unaffected).
        """
        mock_service = AsyncMock()
        _mem_result = MagicMock()
        _mem_result.model_dump.return_value = {'id': 'x'}
        mock_service.add_memory.return_value = _mem_result
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _CONTENT,
                'category': 'observations_and_summaries',
                'agent_id': 'claude-interactive',
                'project_id': _PROJECT_ID,
                'metadata': {'source': 'stage1_flag_marker', 'task_id': '2408'},
            },
        )

        assert result.get('error') != 'flag_marker_write_blocked', (
            f'Gate must not fire for non-recon agent; got: {result!r}'
        )
        mock_service.add_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_allows_ordinary_metadata_from_recon_stage_agent(self):
        """recon-stage agent writing ordinary (non-flag-marker) metadata -> allowed through."""
        mock_service = AsyncMock()
        _mem_result = MagicMock()
        _mem_result.model_dump.return_value = {'id': 'y'}
        mock_service.add_memory.return_value = _mem_result
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _CONTENT,
                'category': 'observations_and_summaries',
                'agent_id': 'recon-stage-task_knowledge_sync',
                'project_id': _PROJECT_ID,
                'metadata': {'source': 'stage2_persistence_marker', 'task_id': '2408'},
            },
        )

        assert result.get('error') != 'flag_marker_write_blocked', (
            f'Gate must not fire for ordinary metadata; got: {result!r}'
        )
        mock_service.add_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_allows_missing_metadata_from_recon_stage_agent(self):
        """recon-stage agent writing with metadata omitted -> allowed through (no crash)."""
        mock_service = AsyncMock()
        _mem_result = MagicMock()
        _mem_result.model_dump.return_value = {'id': 'z'}
        mock_service.add_memory.return_value = _mem_result
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _CONTENT,
                'category': 'observations_and_summaries',
                'agent_id': 'recon-stage-task_knowledge_sync',
                'project_id': _PROJECT_ID,
            },
        )

        assert result.get('error') != 'flag_marker_write_blocked', (
            f'Gate must not fire when metadata is omitted; got: {result!r}'
        )
        mock_service.add_memory.assert_called_once()
