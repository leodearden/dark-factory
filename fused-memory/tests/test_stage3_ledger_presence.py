"""Tests for τ2 (task 2437): Stage 3 consults the ReconLedgerStore
authoritatively for cycle-summary presence, via τ1's (task 2436)
``get_cycle_summary_presence`` tool, falling back to the existing best-effort
Mem0 two-path check only when the ledger read is inconclusive.

``TestWriteThenReadLedgerSeam`` is a write→read boundary/seam test proving
the mechanical path the rewritten Stage-3 prompt now trusts is real, not
faked (PRD plans/stage3-ledger-presence-prd.md §12).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
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
from fused_memory.reconciliation.summary_pool import write_cycle_summary
from fused_memory.services.memory_service import MemoryService

_PROJECT_ID = 'dark_factory'
_STAGE = 'task_knowledge_sync'


class TestWriteThenReadLedgerSeam:
    """Write→read boundary/seam test (G2 integration signal + regression
    guard): a real MemoryService(mock_config) wired to a real
    ReconLedgerStore proves write_cycle_summary (τ1's write half) and
    get_cycle_summary_presence (τ1's read half) transact against the same
    ledger row. Passes on first write — τ1 (task 2436, merged) already
    delivered both ends; this is a characterization test for the seam the
    rewritten Stage-3 prompt now relies on, not a RED test with an impl
    counterpart in this task."""

    def _report(self, **overrides) -> StageReport:
        defaults: dict[str, Any] = dict(
            stage=StageId.task_knowledge_sync,
            started_at=datetime(2026, 7, 10, 11, 0, 0, tzinfo=UTC),
            completed_at=datetime(2026, 7, 10, 11, 5, 0, tzinfo=UTC),
            items_flagged=[{'description': 'a'}],
            stats={},
            llm_calls=1,
            tokens_used=10,
        )
        defaults.update(overrides)
        return StageReport(**defaults)

    @pytest.mark.asyncio
    async def test_written_summary_is_present_unwritten_run_id_is_absent(self, mock_config, tmp_path):
        service = MemoryService(mock_config)
        store = ReconLedgerStore(tmp_path / 'reconciliation.db')
        await store.initialize()
        service.set_recon_ledger(store)
        # Stub only the best-effort Mem0 paths — the ledger write+read runs fully real.
        service.add_system_record = AsyncMock(return_value=SimpleNamespace(memory_ids=['m1']))
        service.get_memories_by_metadata = AsyncMock(return_value=[])

        try:
            run_id = 'run-seam-present'
            await write_cycle_summary(
                service,
                _PROJECT_ID,
                self._report(),
                run_id,
                stage=_STAGE,
                recon_pool='stage2_cycle_summary',
                trim_source='stage2_cycle_summary_trim',
                cap=2,
            )

            present_result = await service.get_cycle_summary_presence(
                project_id=_PROJECT_ID, run_id=run_id, stage=_STAGE,
            )
            absent_result = await service.get_cycle_summary_presence(
                project_id=_PROJECT_ID, run_id='run-seam-never-written', stage=_STAGE,
            )

            # Anti-inversion: assert BOTH directions in the same test so the
            # seam cannot be wired backwards.
            assert present_result.get('present') is True, (
                f'Expected present=True for a written run_id, got: {present_result!r}'
            )
            assert present_result.get('ledger_available') is True
            assert absent_result.get('present') is False, (
                f'Expected present=False for an unwritten run_id, got: {absent_result!r}'
            )
            assert absent_result.get('ledger_available') is True
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_no_ledger_wired_is_inconclusive(self, mock_config):
        service = MemoryService(mock_config)
        assert service.recon_ledger is None

        result = await service.get_cycle_summary_presence(
            project_id=_PROJECT_ID, run_id='any-run', stage=_STAGE,
        )

        assert result.get('present') is False
        assert result.get('ledger_available') is False, (
            'ledger_available=False (inconclusive) must drive Stage 3 to the Mem0 fallback'
        )

    @pytest.mark.asyncio
    async def test_ledger_read_error_is_not_swallowed_as_definitive_absent(self, mock_config, tmp_path):
        """A wired ReconLedgerStore whose read raises (e.g. a transient DB
        error) must NOT be swallowed into a false present=False /
        ledger_available=True by the service layer — that shape is
        indistinguishable from a genuinely-absent row and would make Stage
        3's new PRIMARY rule report a false missing_knowledge. MemoryService
        has no try/except around the ledger read, so the exception
        propagates unmodified; it is only the @mcp_tool_errors decorator at
        the MCP tool boundary
        (tests/server/test_get_cycle_summary_presence_tool.py::
        test_service_exception_is_caught_and_returned_as_error_dict) that
        turns it into an {'error': ..., 'error_type': ...} dict — which
        carries neither 'present' nor 'ledger_available', so Stage 3's
        prompt correctly reads it as a tool error and falls through to the
        Mem0 fallback rather than concluding absence."""
        service = MemoryService(mock_config)
        store = ReconLedgerStore(tmp_path / 'reconciliation.db')
        await store.initialize()
        service.set_recon_ledger(store)
        store.get_by_identity = AsyncMock(side_effect=RuntimeError('ledger read boom'))

        try:
            with pytest.raises(RuntimeError, match='ledger read boom'):
                await service.get_cycle_summary_presence(
                    project_id=_PROJECT_ID, run_id='any-run', stage=_STAGE,
                )
        finally:
            await store.close()


class TestTypedAbsenceClassification:
    """``present=false`` conflates four unrelated situations (task 3731).

    A missing ``cycle_summary`` ledger row can mean the stage ran and lost its
    write (a real gap), the stage never ran at all, the row was reaped by the
    TTL ``gc()``, or nothing is wired to answer with. Only the first is a
    defect, but the old payload reported all four identically — which is why
    Stage 1 check A and Stage 3 filed false "missing stage 2 summary" findings
    against runs that never reached Stage 2.

    ``reason`` names which situation it is; ``expected`` is the gate consumers
    act on. Real ``ReconLedgerStore`` and real ``ReconciliationJournal``
    throughout, on the one shared SQLite file production uses.
    """

    @staticmethod
    async def _wire(mock_config, tmp_path, *, ledger=True, journal=True):
        """Real service + real stores, returning ``(service, store, journal)``.

        ``store``/``journal`` are returned even when not wired into the
        service, so the caller can still write rows through them and close
        them in a ``finally``.
        """
        service = MemoryService(mock_config)
        store = ReconLedgerStore(tmp_path / 'reconciliation.db')
        await store.initialize()
        recon_journal = ReconciliationJournal(tmp_path)
        await recon_journal.initialize()
        if ledger:
            service.set_recon_ledger(store)
        if journal:
            service.set_recon_journal(recon_journal)
        return service, store, recon_journal

    @staticmethod
    async def _start_run(journal, run_id, *, status='running', stage_ran=False):
        now = datetime.now(UTC)
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
        if status != 'running':
            await journal.complete_run(run_id, status)

    @staticmethod
    async def _write_row(store, run_id, *, remediation=False):
        await store.upsert(
            ReconLedgerRecord(
                project_id=_PROJECT_ID,
                record_kind='cycle_summary',
                payload_json=json.dumps({'remediation': remediation}),
                state='active',
                created_at=datetime.now(UTC).isoformat(),
                flag_type=_STAGE,
                run_id=run_id,
            )
        )

    @pytest.mark.asyncio
    async def test_present_row_needs_no_explanation(self, mock_config, tmp_path):
        service, store, journal = await self._wire(mock_config, tmp_path)
        try:
            await self._write_row(store, 'run-present')

            result = await service.get_cycle_summary_presence(
                project_id=_PROJECT_ID, run_id='run-present', stage=_STAGE,
            )

            assert result['present'] is True
            assert result['reason'] == 'present'
            assert result['expected'] is True
        finally:
            await store.close()
            await journal.close()

    @pytest.mark.asyncio
    async def test_unwired_ledger_is_typed_inconclusive(self, mock_config, tmp_path):
        service, store, journal = await self._wire(
            mock_config, tmp_path, ledger=False,
        )
        try:
            result = await service.get_cycle_summary_presence(
                project_id=_PROJECT_ID, run_id='run-any', stage=_STAGE,
            )

            assert result['present'] is False
            assert result['ledger_available'] is False
            assert result['reason'] == 'ledger_unavailable'
            assert result['expected'] is None
        finally:
            await store.close()
            await journal.close()

    @pytest.mark.asyncio
    async def test_unwired_journal_cannot_explain_an_absence(self, mock_config, tmp_path):
        service, store, journal = await self._wire(
            mock_config, tmp_path, journal=False,
        )
        try:
            result = await service.get_cycle_summary_presence(
                project_id=_PROJECT_ID, run_id='run-any', stage=_STAGE,
            )

            assert result['present'] is False
            assert result['ledger_available'] is True
            assert result['reason'] == 'run_unknown'
            assert result['expected'] is None
            assert result['run_lookup_available'] is False
            assert result['run_status'] is None
        finally:
            await store.close()
            await journal.close()

    @pytest.mark.asyncio
    async def test_no_runs_row_is_run_unknown_not_a_gap(self, mock_config, tmp_path):
        service, store, journal = await self._wire(mock_config, tmp_path)
        try:
            result = await service.get_cycle_summary_presence(
                project_id=_PROJECT_ID, run_id='run-never-recorded', stage=_STAGE,
            )

            assert result['reason'] == 'run_unknown'
            assert result['expected'] is None
            assert result['run_lookup_available'] is True
        finally:
            await store.close()
            await journal.close()

    @pytest.mark.asyncio
    async def test_stage_that_never_ran_is_not_a_gap(self, mock_config, tmp_path):
        """The 61-of-64 majority in live data, and the esc-3421-1 shape: an
        interrupted run whose stage_reports never names Stage 2 simply never
        reached it. Nothing was lost, so nothing should be flagged."""
        service, store, journal = await self._wire(mock_config, tmp_path)
        try:
            await self._start_run(
                journal, 'run-never-reached-stage2', status='interrupted',
            )

            result = await service.get_cycle_summary_presence(
                project_id=_PROJECT_ID, run_id='run-never-reached-stage2', stage=_STAGE,
            )

            assert result['present'] is False
            assert result['reason'] == 'stage_not_run'
            assert result['expected'] is False
            assert result['run_status'] == 'interrupted'
        finally:
            await store.close()
            await journal.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize('run_status', ['interrupted', 'failed'])
    async def test_stage_ran_but_row_absent_is_a_real_gap(
        self, mock_config, tmp_path, run_status,
    ):
        """Regression guard for the CRITICAL constraint: run_status must NOT
        gate the verdict. Three measured `failed` runs really did execute
        Stage 2 and lose the ledger write, so a status-gated implementation
        would suppress every one of them. Both statuses must flag."""
        service, store, journal = await self._wire(mock_config, tmp_path)
        try:
            run_id = f'run-lost-write-{run_status}'
            await self._start_run(journal, run_id, status=run_status, stage_ran=True)

            result = await service.get_cycle_summary_presence(
                project_id=_PROJECT_ID, run_id=run_id, stage=_STAGE,
            )

            assert result['present'] is False
            assert result['reason'] == 'missing'
            assert result['expected'] is True
            assert result['run_status'] == run_status
        finally:
            await store.close()
            await journal.close()

    @pytest.mark.asyncio
    async def test_raising_journal_degrades_loudly_without_propagating(
        self, mock_config, tmp_path, caplog,
    ):
        """The journal is a best-effort EXPLANATION of an absence the ledger
        already established. A broken runs lookup must never crash presence
        detection — but it must not be silent either, or a fault reads as an
        ordinary state."""
        service, store, journal = await self._wire(mock_config, tmp_path)
        journal.get_run_stage_execution = AsyncMock(
            side_effect=RuntimeError('runs read boom')
        )
        try:
            with caplog.at_level('WARNING'):
                result = await service.get_cycle_summary_presence(
                    project_id=_PROJECT_ID, run_id='run-any', stage=_STAGE,
                )

            assert result['reason'] == 'run_unknown'
            assert result['expected'] is None
            assert any(r.levelname == 'WARNING' for r in caplog.records)
        finally:
            await store.close()
            await journal.close()

    @pytest.mark.asyncio
    async def test_widening_is_strictly_additive(self, mock_config, tmp_path):
        """Every pre-existing key keeps its exact meaning — including
        `remediation`, which is orthogonal to the new discriminator and must
        still report on a present row."""
        service, store, journal = await self._wire(mock_config, tmp_path)
        try:
            await self._write_row(store, 'run-remediation', remediation=True)

            present = await service.get_cycle_summary_presence(
                project_id=_PROJECT_ID, run_id='run-remediation', stage=_STAGE,
            )
            absent = await service.get_cycle_summary_presence(
                project_id=_PROJECT_ID, run_id='run-absent', stage=_STAGE,
            )

            for result in (present, absent):
                assert set(result) >= {
                    'present', 'ledger_available', 'project_id', 'run_id',
                    'stage', 'remediation',
                }
                assert result['project_id'] == _PROJECT_ID
                assert result['stage'] == _STAGE
                assert result['ledger_available'] is True
            assert present['run_id'] == 'run-remediation'
            assert present['remediation'] is True
            assert present['reason'] == 'present'
            assert absent['remediation'] is None
        finally:
            await store.close()
            await journal.close()
