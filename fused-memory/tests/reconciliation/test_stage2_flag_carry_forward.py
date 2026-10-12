"""Stage-1 markers relayed by a run whose Stage 2 agent did not complete are
carried forward to the next Stage 2 until one completes (task 5984).

See plans/recon-stage-startup-wedge-rca-2026-09-27.md.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from fused_memory.models.reconciliation import StageAgentOutcome, StageId, StageReport
from fused_memory.reconciliation.recon_ledger import ReconLedgerRecord, ReconLedgerStore
from fused_memory.reconciliation.summary_pool import (
    trailing_incomplete_agent_run_ids,
    write_cycle_summary,
)

_STAGE2 = 'task_knowledge_sync'
_STAGE1 = 'memory_consolidator'
_T0 = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)


async def _seed_row(
    ledger: ReconLedgerStore,
    run_id: str,
    outcome: StageAgentOutcome | None,
    *,
    minute: int,
    project_id: str = 'dark_factory',
    stage: str = _STAGE2,
) -> None:
    memory_service = AsyncMock()
    memory_service.recon_ledger = ledger
    memory_service.add_system_record = AsyncMock(return_value=SimpleNamespace(memory_ids=['m1']))
    report = StageReport(
        stage=StageId(stage),
        started_at=_T0,
        completed_at=_T0,
        agent_outcome=outcome,
    )
    written = await write_cycle_summary(
        memory_service,
        project_id,
        report,
        run_id,
        stage=stage,
        recon_pool='stage2_cycle_summary',
        trim_source='stage2_cycle_summary_trim',
        cap=2,
        now=_T0 + timedelta(minutes=minute),
    )
    assert written is True


async def _seed_raw_row(
    ledger: ReconLedgerStore, run_id: str, payload: dict, *, minute: int,
) -> None:
    created = _T0 + timedelta(minutes=minute)
    await ledger.upsert(ReconLedgerRecord(
        project_id='dark_factory',
        record_kind='cycle_summary',
        task_id='',
        flag_type=_STAGE2,
        run_id=run_id,
        payload_json=json.dumps(payload),
        state='active',
        created_at=created.isoformat(),
        expires_at=(created + timedelta(days=30)).isoformat(),
    ))


async def _streak(ledger: ReconLedgerStore | None) -> frozenset[str]:
    return await trailing_incomplete_agent_run_ids(ledger, 'dark_factory', _STAGE2)


@pytest_asyncio.fixture
async def ledger_store(tmp_path):
    store = ReconLedgerStore(tmp_path / 'reconciliation.db')
    await store.initialize()
    yield store
    await store.close()


_NOT_COMPLETED = [o for o in StageAgentOutcome if o is not StageAgentOutcome.completed]


class TestTrailingIncompleteAgentRunIds:
    @pytest.mark.asyncio
    async def test_runs_after_the_last_completed_agent_are_owed(self, ledger_store):
        await _seed_row(ledger_store, 'R0', StageAgentOutcome.completed, minute=0)
        await _seed_row(ledger_store, 'R1', StageAgentOutcome.zero_output_timeout, minute=1)
        await _seed_row(ledger_store, 'R2', StageAgentOutcome.timed_out, minute=2)

        assert await _streak(ledger_store) == frozenset({'R1', 'R2'})

    @pytest.mark.asyncio
    @pytest.mark.parametrize('outcome', _NOT_COMPLETED, ids=lambda o: o.value)
    async def test_every_non_completed_outcome_is_owed(self, ledger_store, outcome):
        await _seed_row(ledger_store, 'R-only', outcome, minute=0)

        assert await _streak(ledger_store) == frozenset({'R-only'})

    @pytest.mark.asyncio
    async def test_a_newer_completed_agent_settles_every_older_run(self, ledger_store):
        await _seed_row(ledger_store, 'R0', StageAgentOutcome.failed, minute=0)
        await _seed_row(ledger_store, 'R1', StageAgentOutcome.zero_output_timeout, minute=1)
        await _seed_row(ledger_store, 'R2', StageAgentOutcome.completed, minute=2)

        assert await _streak(ledger_store) == frozenset()

    @pytest.mark.asyncio
    async def test_an_unrecorded_outcome_ends_the_streak(self, ledger_store):
        await _seed_row(ledger_store, 'R0', StageAgentOutcome.zero_output_timeout, minute=0)
        await _seed_row(ledger_store, 'R1', None, minute=1)
        await _seed_row(ledger_store, 'R2', StageAgentOutcome.failed, minute=2)

        assert await _streak(ledger_store) == frozenset({'R2'})

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'payload',
        [
            {'stage': _STAGE2, 'llm_calls': 0},
            {'stage': _STAGE2, 'agent_outcome': 'not-a-real-outcome'},
        ],
        ids=['legacy-without-key', 'corrupt-value'],
    )
    async def test_a_legacy_or_corrupt_row_ends_the_streak(self, ledger_store, payload):
        await _seed_row(ledger_store, 'R0', StageAgentOutcome.zero_output_timeout, minute=0)
        await _seed_raw_row(ledger_store, 'R1', payload, minute=1)
        await _seed_row(ledger_store, 'R2', StageAgentOutcome.timed_out, minute=2)

        assert await _streak(ledger_store) == frozenset({'R2'})

    @pytest.mark.asyncio
    async def test_other_stages_rows_neither_break_nor_join_the_streak(self, ledger_store):
        await _seed_row(ledger_store, 'R0', StageAgentOutcome.zero_output_timeout, minute=0)
        await _seed_row(ledger_store, 'R0', StageAgentOutcome.completed, minute=1, stage=_STAGE1)
        await _seed_row(ledger_store, 'R1', StageAgentOutcome.failed, minute=2)
        await _seed_row(
            ledger_store, 'R2', StageAgentOutcome.zero_output_timeout, minute=3, stage=_STAGE1,
        )

        assert await _streak(ledger_store) == frozenset({'R0', 'R1'})

    @pytest.mark.asyncio
    async def test_another_projects_rows_are_not_returned(self, ledger_store):
        await _seed_row(
            ledger_store, 'R-other', StageAgentOutcome.failed, minute=0, project_id='reify',
        )

        assert await _streak(ledger_store) == frozenset()

    @pytest.mark.asyncio
    async def test_no_ledger_owes_nothing(self):
        assert await _streak(None) == frozenset()

    @pytest.mark.asyncio
    async def test_a_failed_ledger_read_degrades_to_nothing_with_a_warning(
        self, tmp_path, caplog,
    ):
        never_initialized = ReconLedgerStore(tmp_path / 'reconciliation.db')

        with caplog.at_level(logging.WARNING, logger='fused_memory.reconciliation.summary_pool'):
            result = await _streak(never_initialized)

        assert result == frozenset()
        assert any(
            rec.levelno == logging.WARNING
            and rec.name == 'fused_memory.reconciliation.summary_pool'
            for rec in caplog.records
        )

    @pytest.mark.asyncio
    async def test_an_empty_ledger_owes_nothing(self, ledger_store):
        assert await _streak(ledger_store) == frozenset()
