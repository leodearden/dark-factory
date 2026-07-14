"""Behaviour tests for the get_cycle_summary_presence MCP tool (task 2436, τ1).

Read-only presence check against the AUTHORITATIVE ReconLedgerStore
``cycle_summary`` row (plans/recon-reliability-prd.md §8.1), so Stage 3 (τ2)
can stop relying solely on the best-effort Mem0 mirror. Present/absent
correctness is proven against a REAL ReconLedgerStore (not a mocked store);
the wrapper-contract cases (invalid project_id, service exception) use a
mock service, mirroring test_count_by_metadata_tool.py's split.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from fused_memory.reconciliation.recon_ledger import ReconLedgerRecord, ReconLedgerStore
from fused_memory.server.tools import create_mcp_server
from fused_memory.services.memory_service import MemoryService

_PROJECT_ID = 'dark_factory'
_STAGE = 'task_knowledge_sync'


class TestGetCycleSummaryPresenceTool:
    """Behaviour tests for mcp__fused-memory__get_cycle_summary_presence."""

    @pytest.mark.asyncio
    async def test_present_then_absent_against_real_ledger_store(self, mock_config, tmp_path):
        """Real ReconLedgerStore: a written (project_id, run_id, stage) identity
        is reported present=True; a different, never-written run_id against the
        SAME store/server is reported present=False (anti-inversion: both
        directions asserted)."""
        service = MemoryService(mock_config)
        store = ReconLedgerStore(tmp_path / 'reconciliation.db')
        await store.initialize()
        service.set_recon_ledger(store)

        run_id = 'run-present-1'
        await store.upsert(
            ReconLedgerRecord(
                project_id=_PROJECT_ID,
                record_kind='cycle_summary',
                task_id='',
                flag_type=_STAGE,
                run_id=run_id,
                payload_json='{}',
                state='active',
                created_at='2026-07-01T00:00:00+00:00',
            )
        )

        try:
            server = create_mcp_server(service)

            present_result = await server._tool_manager.call_tool(
                'get_cycle_summary_presence',
                {
                    'project_id': _PROJECT_ID,
                    'run_id': run_id,
                    'stage': _STAGE,
                },
            )

            assert isinstance(present_result, dict), (
                f'Expected dict, got {type(present_result)}: {present_result!r}'
            )
            assert 'error' not in present_result, f'Unexpected error in result: {present_result!r}'
            assert present_result.get('present') is True, (
                f'Expected present=True, got: {present_result!r}'
            )
            assert present_result.get('ledger_available') is True, (
                f'Expected ledger_available=True, got: {present_result!r}'
            )
            assert present_result.get('project_id') == _PROJECT_ID
            assert present_result.get('run_id') == run_id
            assert present_result.get('stage') == _STAGE

            absent_result = await server._tool_manager.call_tool(
                'get_cycle_summary_presence',
                {
                    'project_id': _PROJECT_ID,
                    'run_id': 'absent-run',
                    'stage': _STAGE,
                },
            )

            assert isinstance(absent_result, dict), (
                f'Expected dict, got {type(absent_result)}: {absent_result!r}'
            )
            assert 'error' not in absent_result, f'Unexpected error in result: {absent_result!r}'
            assert absent_result.get('present') is False, (
                f'Expected present=False, got: {absent_result!r}'
            )
            assert absent_result.get('ledger_available') is True, (
                f'Expected ledger_available=True, got: {absent_result!r}'
            )
            assert absent_result.get('project_id') == _PROJECT_ID
            assert absent_result.get('run_id') == 'absent-run'
            assert absent_result.get('stage') == _STAGE
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_stage_disambiguates_rows_sharing_run_id(self, mock_config, tmp_path):
        """Stage 1 (memory_consolidator) and Stage 2 (task_knowledge_sync) can
        each write a cycle_summary row under the SAME run_id (identity also
        includes flag_type). A presence query for each stage must return only
        that stage's own row — the exact collision the stage->flag_type
        mapping exists to prevent — and a third, never-written stage under the
        same run_id must independently report present=False."""
        service = MemoryService(mock_config)
        store = ReconLedgerStore(tmp_path / 'reconciliation.db')
        await store.initialize()
        service.set_recon_ledger(store)

        run_id = 'run-shared-1'
        written_stages = ('memory_consolidator', 'task_knowledge_sync')
        for flag_type in written_stages:
            await store.upsert(
                ReconLedgerRecord(
                    project_id=_PROJECT_ID,
                    record_kind='cycle_summary',
                    task_id='',
                    flag_type=flag_type,
                    run_id=run_id,
                    payload_json='{}',
                    state='active',
                    created_at='2026-07-01T00:00:00+00:00',
                )
            )

        try:
            server = create_mcp_server(service)

            for flag_type in written_stages:
                result = await server._tool_manager.call_tool(
                    'get_cycle_summary_presence',
                    {
                        'project_id': _PROJECT_ID,
                        'run_id': run_id,
                        'stage': flag_type,
                    },
                )
                assert isinstance(result, dict), f'Expected dict, got {type(result)}: {result!r}'
                assert 'error' not in result, f'Unexpected error in result: {result!r}'
                assert result.get('present') is True, (
                    f'Expected present=True for stage={flag_type!r} (own row), got: {result!r}'
                )
                assert result.get('ledger_available') is True
                assert result.get('stage') == flag_type

            unwritten_result = await server._tool_manager.call_tool(
                'get_cycle_summary_presence',
                {
                    'project_id': _PROJECT_ID,
                    'run_id': run_id,
                    'stage': 'some_other_stage',
                },
            )
            assert 'error' not in unwritten_result, (
                f'Unexpected error in result: {unwritten_result!r}'
            )
            assert unwritten_result.get('present') is False, (
                f'Expected present=False for unwritten stage sharing run_id, '
                f'got: {unwritten_result!r}'
            )
            assert unwritten_result.get('ledger_available') is True
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_inconclusive_when_ledger_not_wired(self, mock_config):
        """A real MemoryService with NO recon_ledger wired reports a clean
        inconclusive dict (present=False, ledger_available=False) — NOT an
        @mcp_tool_errors error dict. Mirrors write_cycle_summary returning
        False when unwired: absence of the ledger is inconclusive, not a
        definitive absent."""
        service = MemoryService(mock_config)
        assert service.recon_ledger is None

        server = create_mcp_server(service)
        result = await server._tool_manager.call_tool(
            'get_cycle_summary_presence',
            {
                'project_id': _PROJECT_ID,
                'run_id': 'any-run',
                'stage': _STAGE,
            },
        )

        assert isinstance(result, dict), f'Expected dict, got {type(result)}: {result!r}'
        assert 'error' not in result, f'Unexpected error in result: {result!r}'
        assert result.get('present') is False, f'Expected present=False, got: {result!r}'
        assert result.get('ledger_available') is False, (
            f'Expected ledger_available=False, got: {result!r}'
        )
        assert result.get('project_id') == _PROJECT_ID
        assert result.get('run_id') == 'any-run'
        assert result.get('stage') == _STAGE

    @pytest.mark.asyncio
    async def test_invalid_project_id_returns_validation_error_without_calling_service(self):
        """Invalid project_id (contains unsafe chars) returns a validation error
        dict and does NOT call the service."""
        mock_service = AsyncMock()
        mock_service.get_cycle_summary_presence = AsyncMock(
            return_value={'present': False, 'ledger_available': True}
        )
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'get_cycle_summary_presence',
            {
                'project_id': 'bad project id!',  # contains spaces and !
                'run_id': 'run-1',
                'stage': _STAGE,
            },
        )

        assert isinstance(result, dict), f'Expected dict, got {type(result)}: {result!r}'
        assert 'error' in result, f'Expected error key in result: {result!r}'
        assert result.get('error_type') == 'ValidationError', (
            f"Expected error_type='ValidationError', got: {result!r}"
        )
        mock_service.get_cycle_summary_presence.assert_not_called()

    @pytest.mark.asyncio
    async def test_service_exception_is_caught_and_returned_as_error_dict(self):
        """When the service raises, the tool catches it and returns
        {'error': ..., 'error_type': ...} without propagating."""
        mock_service = AsyncMock()
        mock_service.get_cycle_summary_presence = AsyncMock(
            side_effect=ValueError('boom')
        )
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'get_cycle_summary_presence',
            {
                'project_id': _PROJECT_ID,
                'run_id': 'run-1',
                'stage': _STAGE,
            },
        )

        assert isinstance(result, dict), f'Expected dict, got {type(result)}: {result!r}'
        assert 'error' in result, f'Expected error key in result: {result!r}'
        assert result.get('error_type') == 'ValueError', (
            f"Expected error_type='ValueError', got: {result!r}"
        )
        assert 'boom' in result.get('error', ''), (
            f'Expected original error message in result: {result!r}'
        )
