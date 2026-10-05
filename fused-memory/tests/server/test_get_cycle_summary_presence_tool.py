"""Behaviour tests for the get_cycle_summary_presence MCP tool (task 2436, τ1).

Read-only presence check against the AUTHORITATIVE ReconLedgerStore
``cycle_summary`` row (plans/recon-reliability-prd.md §8.1), so Stage 3 (τ2)
can stop relying solely on the best-effort Mem0 mirror. Present/absent
correctness is proven against a REAL ReconLedgerStore (not a mocked store);
the wrapper-contract cases (invalid project_id, service exception) use a
mock service, mirroring test_count_by_metadata_tool.py's split.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from fused_memory.models.reconciliation import (
    ReconciliationRun,
    RunStatus,
    RunType,
    StageId,
    StageReport,
)
from fused_memory.reconciliation.journal import ReconciliationJournal
from fused_memory.reconciliation.recon_ledger import ReconLedgerRecord, ReconLedgerStore
from fused_memory.reconciliation.recon_self_model import MCP_CALL_SIGNATURES
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
    async def test_remediation_field_reflects_payload_marker(self, mock_config, tmp_path):
        """The `remediation` field (task 2652) surfaces the ledger row's
        payload_json `remediation` marker written by
        ``summary_pool.write_cycle_summary``: True/False for an explicit
        marker, None for a legacy row lacking the key entirely (payload
        '{}') or for a never-written run_id — so Stage 3 can disambiguate a
        Stage-2-only remediation run's expected missing Stage 1 summary from
        a genuine Stage 1 write failure."""
        service = MemoryService(mock_config)
        store = ReconLedgerStore(tmp_path / 'reconciliation.db')
        await store.initialize()
        service.set_recon_ledger(store)

        rows = {
            'run-remediation-true': '{"remediation": true}',
            'run-remediation-false': '{"remediation": false}',
            'run-legacy': '{}',
        }
        for run_id, payload_json in rows.items():
            await store.upsert(
                ReconLedgerRecord(
                    project_id=_PROJECT_ID,
                    record_kind='cycle_summary',
                    task_id='',
                    flag_type=_STAGE,
                    run_id=run_id,
                    payload_json=payload_json,
                    state='active',
                    created_at='2026-07-01T00:00:00+00:00',
                )
            )

        try:
            server = create_mcp_server(service)

            true_result = await server._tool_manager.call_tool(
                'get_cycle_summary_presence',
                {'project_id': _PROJECT_ID, 'run_id': 'run-remediation-true', 'stage': _STAGE},
            )
            assert 'error' not in true_result, f'Unexpected error in result: {true_result!r}'
            assert true_result.get('present') is True
            assert true_result.get('remediation') is True, (
                f'Expected remediation=True, got: {true_result!r}'
            )

            false_result = await server._tool_manager.call_tool(
                'get_cycle_summary_presence',
                {'project_id': _PROJECT_ID, 'run_id': 'run-remediation-false', 'stage': _STAGE},
            )
            assert 'error' not in false_result, f'Unexpected error in result: {false_result!r}'
            assert false_result.get('present') is True
            assert false_result.get('remediation') is False, (
                f'Expected remediation=False, got: {false_result!r}'
            )

            legacy_result = await server._tool_manager.call_tool(
                'get_cycle_summary_presence',
                {'project_id': _PROJECT_ID, 'run_id': 'run-legacy', 'stage': _STAGE},
            )
            assert 'error' not in legacy_result, f'Unexpected error in result: {legacy_result!r}'
            assert legacy_result.get('present') is True
            assert legacy_result.get('remediation') is None, (
                f'Expected remediation=None for a legacy row with no key, got: {legacy_result!r}'
            )

            absent_result = await server._tool_manager.call_tool(
                'get_cycle_summary_presence',
                {'project_id': _PROJECT_ID, 'run_id': 'never-written', 'stage': _STAGE},
            )
            assert 'error' not in absent_result, f'Unexpected error in result: {absent_result!r}'
            assert absent_result.get('present') is False
            assert absent_result.get('remediation') is None, (
                f'Expected remediation=None for an absent row, got: {absent_result!r}'
            )
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


class TestTypedAbsenceCrossesTheMcpBoundary:
    """The widened payload has to survive the wrapper, and the DOCUMENTED
    contract has to match it (task 3731).

    A payload change without matching consumer edits is a no-op: the stages
    read ``MCP_CALL_SIGNATURES`` rendered into their prompts, not the Python
    return annotation, so a stale entry there silently withholds the new keys
    from every consumer.
    """

    @staticmethod
    async def _server_with(mock_config, tmp_path, *, stage_ran):
        service = MemoryService(mock_config)
        store = ReconLedgerStore(tmp_path / 'reconciliation.db')
        await store.initialize()
        journal = ReconciliationJournal(tmp_path)
        await journal.initialize()
        service.set_recon_ledger(store)
        service.set_recon_journal(journal)

        now = datetime.now(UTC)
        run_id = f'run-{"ran" if stage_ran else "never-ran"}'
        await journal.start_run(
            ReconciliationRun(
                id=run_id,
                project_id=_PROJECT_ID,
                run_type=RunType.full,
                trigger_reason='test',
                started_at=now,
                status=RunStatus.running,
            )
        )
        if stage_ran:
            await journal.update_run_stage_reports(
                run_id,
                {
                    _STAGE: StageReport(
                        stage=StageId.task_knowledge_sync,
                        started_at=now,
                        completed_at=now,
                    )
                },
            )
        await journal.complete_run(run_id, 'interrupted')
        return create_mcp_server(service), store, journal, run_id

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('stage_ran', 'expected_reason', 'expected_gate'),
        [(False, 'stage_not_run', False), (True, 'missing', True)],
    )
    async def test_wrapper_forwards_the_discriminator(
        self, mock_config, tmp_path, stage_ran, expected_reason, expected_gate,
    ):
        server, store, journal, run_id = await self._server_with(
            mock_config, tmp_path, stage_ran=stage_ran,
        )
        try:
            result = await server._tool_manager.call_tool(
                'get_cycle_summary_presence',
                {'project_id': _PROJECT_ID, 'run_id': run_id, 'stage': _STAGE},
            )

            assert 'error' not in result, f'Unexpected error: {result!r}'
            assert result['present'] is False
            assert result['reason'] == expected_reason
            assert result['expected'] is expected_gate
            assert result['run_lookup_available'] is True
            assert result['run_status'] == 'interrupted'
        finally:
            await store.close()
            await journal.close()

    @pytest.mark.asyncio
    async def test_documented_signature_names_every_returned_key(
        self, mock_config, tmp_path,
    ):
        """DERIVED drift guard — the expectation comes from the live return, so
        it cannot rot the way a hardcoded list would.

        This already failed BEFORE task 3731's widening: the signature still
        omitted ``remediation``, which task 2652 added to the payload two
        cycles ago, so the stages have been reading a stale contract since.

        Compared KEY to KEY, not substring to blob: a ``'status'`` key would
        pass a plain ``in`` test against the documented ``'run_status'``, and
        ``'reason'`` against ``'reasons'``, so the drift this exists to catch
        could recur undetected.
        """
        server, store, journal, run_id = await self._server_with(
            mock_config, tmp_path, stage_ran=True,
        )
        try:
            result = await server._tool_manager.call_tool(
                'get_cycle_summary_presence',
                {'project_id': _PROJECT_ID, 'run_id': run_id, 'stage': _STAGE},
            )
            documented = MCP_CALL_SIGNATURES['get_cycle_summary_presence']
            # The signature renders each key as a quoted name followed by a
            # colon; the `reason` VALUE literals are quoted too but are
            # separated by `|`, so they are not picked up as keys.
            documented_keys = set(re.findall(r"'([a-z_]+)':", documented))

            undocumented = sorted(set(result) - documented_keys)

            assert not undocumented, (
                f'get_cycle_summary_presence returns {undocumented} but '
                f'MCP_CALL_SIGNATURES does not name them. The stages read the '
                f'rendered signature, not the Python return, so an unnamed key '
                f'is invisible to every consumer. Documented keys: '
                f'{sorted(documented_keys)}'
            )
        finally:
            await store.close()
            await journal.close()
