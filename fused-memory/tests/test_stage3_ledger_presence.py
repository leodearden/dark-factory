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
from datetime import UTC, datetime, timedelta
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
from fused_memory.reconciliation.recon_pool_map import CYCLE_SUMMARY_TTL_DAYS
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
    async def test_in_flight_run_is_inconclusive_not_stage_not_run(
        self, mock_config, tmp_path,
    ):
        """The current-cycle shape, and the one that matters most: Stage 3
        verifies the run it is running INSIDE, whose stage_reports the harness
        has not persisted yet (it writes the blob once, after the stage loop).
        The column therefore reads '{}' no matter how many stages have run, so
        reading an absent key as "the stage never ran" would type every
        in-flight run as a non-gap and silently suppress exactly the
        current-cycle ledger loss this check exists to catch. Must stay
        inconclusive so the consumer falls through to its Mem0 fallback."""
        service, store, journal = await self._wire(mock_config, tmp_path)
        try:
            await self._start_run(journal, 'run-in-flight', status='running')

            result = await service.get_cycle_summary_presence(
                project_id=_PROJECT_ID, run_id='run-in-flight', stage=_STAGE,
            )

            assert result['present'] is False
            assert result['reason'] == 'run_unknown'
            assert result['expected'] is None
            assert result['run_status'] == 'running'
        finally:
            await store.close()
            await journal.close()

    @pytest.mark.asyncio
    async def test_in_flight_run_with_present_row_still_reads_present(
        self, mock_config, tmp_path,
    ):
        """The in-flight guard must only widen the ABSENCE verdict: a row the
        current cycle's Stage 2 already wrote is still authoritative evidence
        of presence, so the healthy current-cycle case keeps its definitive
        answer rather than degrading to the fallback."""
        service, store, journal = await self._wire(mock_config, tmp_path)
        try:
            await self._start_run(journal, 'run-in-flight-present', status='running')
            await self._write_row(store, 'run-in-flight-present')

            result = await service.get_cycle_summary_presence(
                project_id=_PROJECT_ID, run_id='run-in-flight-present', stage=_STAGE,
            )

            assert result['present'] is True
            assert result['reason'] == 'present'
            assert result['expected'] is True
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


class TestRetentionCliff:
    """``expired``: the arm that stops reaped rows reading as data loss (task 3731).

    No longer prophylactic. As of 2026-09-14 the first reaping is long past:
    20287 runs older than the retention window hold no ``cycle_summary`` row,
    and 6875 of those carry ``stage_reports`` proving Stage 2 ran. Without this
    arm every one of them classifies as ``missing``/``expected=True`` and
    becomes a false "data loss" finding — making it the highest-volume arm in
    the ladder.

    Driven END TO END through the real writer and the real ``gc()`` rather than
    against a hardcoded 30, so this is a genuine writer/reader agreement seam:
    the ``expires_at`` stamp comes from ``write_cycle_summary`` and the
    classification from the presence reader, and a drift between the two breaks
    these tests.
    """

    _T0 = datetime(2026, 7, 1, 12, 0, 0, tzinfo=UTC)

    def _report(self) -> StageReport:
        return StageReport(
            stage=StageId.task_knowledge_sync,
            started_at=self._T0,
            completed_at=self._T0,
            stats={},
            llm_calls=1,
            tokens_used=10,
        )

    async def _wire(self, mock_config, tmp_path):
        service = MemoryService(mock_config)
        store = ReconLedgerStore(tmp_path / 'reconciliation.db')
        await store.initialize()
        journal = ReconciliationJournal(tmp_path)
        await journal.initialize()
        service.set_recon_ledger(store)
        service.set_recon_journal(journal)
        # Stub only the best-effort Mem0 paths — the ledger write runs fully real.
        service.add_system_record = AsyncMock(
            return_value=SimpleNamespace(memory_ids=['m1'])
        )
        service.get_memories_by_metadata = AsyncMock(return_value=[])
        return service, store, journal

    async def _write_summary(self, service, run_id):
        await write_cycle_summary(
            service,
            _PROJECT_ID,
            self._report(),
            run_id,
            stage=_STAGE,
            recon_pool='stage2_cycle_summary',
            trim_source='stage2_cycle_summary_trim',
            cap=2,
            now=self._T0,
        )

    async def _record_run(self, journal, run_id, *, stage_ran, started_at=None):
        started = started_at or self._T0
        await journal.start_run(
            ReconciliationRun(
                id=run_id,
                project_id=_PROJECT_ID,
                run_type=RunType.full,
                trigger_reason='test',
                started_at=started,
                status=RunStatus.running,
            )
        )
        if stage_ran:
            await journal.update_run_stage_reports(
                run_id, {_STAGE: self._report()}
            )
        # journal.complete_run stamps wall-clock time and takes no clock — it
        # is a WRITER, and this task does not touch writers (PRD §11) — so the
        # completion timestamp is stamped directly here instead. Otherwise the
        # run would age from real-now while the reader is handed an injected
        # `now` in the past, and the retention comparison would be meaningless.
        await journal._db.execute(
            "UPDATE runs SET status = 'completed', completed_at = ? WHERE id = ?",
            (started.isoformat(), run_id),
        )
        await journal._db.commit()

    @pytest.mark.asyncio
    async def test_reaped_row_is_expired_not_missing(self, mock_config, tmp_path):
        """The headline case. Anti-inversion: the same fixture is asserted live
        BEFORE gc, so the test cannot pass by never having written a row."""
        service, store, journal = await self._wire(mock_config, tmp_path)
        try:
            run_id = 'run-reaped'
            await self._write_summary(service, run_id)
            await self._record_run(journal, run_id, stage_ran=True)

            before = await service.get_cycle_summary_presence(
                project_id=_PROJECT_ID, run_id=run_id, stage=_STAGE, now=self._T0,
            )
            assert before['present'] is True
            assert before['reason'] == 'present'

            past_expiry = self._T0 + timedelta(days=CYCLE_SUMMARY_TTL_DAYS + 1)
            # gc()'s `now` is an ISO STRING, not a datetime.
            await store.gc(_PROJECT_ID, past_expiry.isoformat(), [])

            after = await service.get_cycle_summary_presence(
                project_id=_PROJECT_ID, run_id=run_id, stage=_STAGE, now=past_expiry,
            )

            assert after['present'] is False
            assert after['reason'] == 'expired'
            assert after['expected'] is None
        finally:
            await store.close()
            await journal.close()

    @pytest.mark.asyncio
    async def test_absence_inside_the_window_is_still_a_gap(self, mock_config, tmp_path):
        """The boundary must not smear into blanket suppression: a run still
        inside retention that ran the stage and has no row IS data loss."""
        service, store, journal = await self._wire(mock_config, tmp_path)
        try:
            run_id = 'run-inside-window'
            await self._record_run(journal, run_id, stage_ran=True)

            result = await service.get_cycle_summary_presence(
                project_id=_PROJECT_ID,
                run_id=run_id,
                stage=_STAGE,
                now=self._T0 + timedelta(days=CYCLE_SUMMARY_TTL_DAYS - 1),
            )

            assert result['reason'] == 'missing'
            assert result['expected'] is True
        finally:
            await store.close()
            await journal.close()

    @pytest.mark.asyncio
    async def test_stage_not_run_outranks_expired(self, mock_config, tmp_path):
        """esc-3421-1's run 745f2ffb-020c-4409-9543-e99980b9f1e9 exactly as it
        stands today: its ledger rows have been reaped, but its runs row
        survives with stage_reports={}. The durable positive fact outranks the
        destroyed-evidence fact — an expired-first ladder would report
        "evidence destroyed" when the evidence is intact and conclusive."""
        service, store, journal = await self._wire(mock_config, tmp_path)
        try:
            run_id = 'run-old-and-never-ran-stage2'
            await self._record_run(journal, run_id, stage_ran=False)

            result = await service.get_cycle_summary_presence(
                project_id=_PROJECT_ID,
                run_id=run_id,
                stage=_STAGE,
                now=self._T0 + timedelta(days=CYCLE_SUMMARY_TTL_DAYS + 1),
            )

            assert result['reason'] == 'stage_not_run'
            assert result['expected'] is False
        finally:
            await store.close()
            await journal.close()

    @pytest.mark.asyncio
    async def test_incomplete_run_ages_from_started_at(self, mock_config, tmp_path):
        """A still-running run has completed_at IS NULL — age falls back to
        started_at rather than raising."""
        service, store, journal = await self._wire(mock_config, tmp_path)
        try:
            run_id = 'run-never-completed'
            await journal.start_run(
                ReconciliationRun(
                    id=run_id,
                    project_id=_PROJECT_ID,
                    run_type=RunType.full,
                    trigger_reason='test',
                    started_at=self._T0,
                    status=RunStatus.running,
                )
            )
            await journal.update_run_stage_reports(run_id, {_STAGE: self._report()})

            result = await service.get_cycle_summary_presence(
                project_id=_PROJECT_ID,
                run_id=run_id,
                stage=_STAGE,
                now=self._T0 + timedelta(days=CYCLE_SUMMARY_TTL_DAYS + 1),
            )

            assert result['run_status'] == 'running'
            assert result['reason'] == 'expired'
            assert result['expected'] is None
        finally:
            await store.close()
            await journal.close()
