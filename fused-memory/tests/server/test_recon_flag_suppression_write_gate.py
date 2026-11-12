"""Integration tests for the recon-stage flag-kind refusal on the Mem0 write tools (task 4863).

stage1_flag_suppression records are operator-managed recon_ledger rows: a
recon stage cannot create one, and a Mem0 record of that kind has no gate
effect. Both recon-stage write tools (add_memory, add_system_record) refuse
the kind, and add_system_record now also refuses stage1_flag_marker (the
task-2596 marker refusal previously covered add_memory only). The operator
path stays open at the tool layer. Harness mirrors
tests/server/test_recon_flag_marker_write_gate.py.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.server.tools import create_mcp_server

_PROJECT_ID = 'dark_factory'
_CONTENT = 'STAGE 1 FLAG SUPPRESSION task_id=6107'
_CATEGORY = 'observations_and_summaries'
_SUPPRESSION_METADATA = {
    'kind': 'stage1_flag_suppression',
    'task_id': '6107',
    'flag_types': ['task_memory_mismatch'],
}


def _service() -> AsyncMock:
    service = AsyncMock()
    add_memory_result = MagicMock()
    add_memory_result.model_dump.return_value = {'memory_ids': ['mem0-1']}
    service.add_memory.return_value = add_memory_result
    system_record_result = MagicMock()
    system_record_result.model_dump.return_value = {'memory_ids': ['sys-1']}
    service.add_system_record.return_value = system_record_result
    return service


def _assert_suppression_refusal(result: object, agent_id: str) -> None:
    assert isinstance(result, dict), f'Expected dict, got {type(result)}: {result!r}'
    assert result.get('error') == 'flag_suppression_write_blocked', result
    assert result.get('error_type') == 'ReconFlagSuppressionWriteRejected', result
    assert result.get('agent_id') == agent_id, result
    assert result.get('content_excerpt') == _CONTENT[:200], result
    assert result.get('hint'), result


class TestAddMemorySuppressionGate:
    @pytest.mark.asyncio
    async def test_rejects_suppression_from_recon_stage_agent(self):
        service = _service()
        server = create_mcp_server(service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _CONTENT,
                'category': _CATEGORY,
                'agent_id': 'recon-stage-memory_consolidator',
                'project_id': _PROJECT_ID,
                'metadata': dict(_SUPPRESSION_METADATA),
            },
        )

        _assert_suppression_refusal(result, 'recon-stage-memory_consolidator')
        service.add_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejects_suppression_on_the_auto_classified_path(self):
        service = _service()
        server = create_mcp_server(service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _CONTENT,
                'agent_id': 'recon-stage-memory_consolidator',
                'project_id': _PROJECT_ID,
                'metadata': dict(_SUPPRESSION_METADATA),
            },
        )

        _assert_suppression_refusal(result, 'recon-stage-memory_consolidator')
        service.add_memory.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'identity', [{'agent_id': 'claude-interactive'}, {}], ids=['interactive', 'omitted']
    )
    async def test_operator_path_is_open_at_the_tool(self, identity):
        service = _service()
        server = create_mcp_server(service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _CONTENT,
                'category': _CATEGORY,
                'project_id': _PROJECT_ID,
                'metadata': dict(_SUPPRESSION_METADATA),
                **identity,
            },
        )

        assert result.get('error') != 'flag_suppression_write_blocked', result
        service.add_memory.assert_called_once()


class TestAddSystemRecordFlagKindGate:
    @pytest.mark.asyncio
    async def test_rejects_suppression_from_recon_stage_agent(self):
        service = _service()
        server = create_mcp_server(service)

        result = await server._tool_manager.call_tool(
            'add_system_record',
            {
                'content': _CONTENT,
                'project_id': _PROJECT_ID,
                'category': _CATEGORY,
                'agent_id': 'recon-stage-task_knowledge_sync',
                'metadata': dict(_SUPPRESSION_METADATA),
            },
        )

        _assert_suppression_refusal(result, 'recon-stage-task_knowledge_sync')
        service.add_system_record.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejects_marker_from_recon_stage_agent(self):
        service = _service()
        server = create_mcp_server(service)

        result = await server._tool_manager.call_tool(
            'add_system_record',
            {
                'content': _CONTENT,
                'project_id': _PROJECT_ID,
                'category': _CATEGORY,
                'agent_id': 'recon-stage-task_knowledge_sync',
                'metadata': {'kind': 'stage1_flag_marker', 'task_id': '2408'},
            },
        )

        assert isinstance(result, dict), f'Expected dict, got {type(result)}: {result!r}'
        assert result.get('error') == 'flag_marker_write_blocked', result
        assert result.get('error_type') == 'ReconFlagMarkerWriteRejected', result
        service.add_system_record.assert_not_called()
