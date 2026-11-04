"""Stage-1 markers relayed by a run whose Stage 2 agent did not complete are
carried forward to the next Stage 2 until one completes (task 5984).

See plans/recon-stage-startup-wedge-rca-2026-09-27.md.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from fused_memory.config.schema import ReconciliationConfig
from fused_memory.models.reconciliation import (
    StageAgentOutcome,
    StageId,
    StageReport,
    Watermark,
)
from fused_memory.models.scope import ProjectId, ProjectRoot, ProjectScope
from fused_memory.reconciliation.cli_stage_runner import StageResult
from fused_memory.reconciliation.recon_ledger import ReconLedgerRecord, ReconLedgerStore
from fused_memory.reconciliation.stages.task_knowledge_sync import TaskKnowledgeSync
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


_PROJECT = 'reify'
_T_RUN = datetime(2026, 10, 12, 10, 0, 0, tzinfo=UTC)
_BEFORE_RUN = (_T0 + timedelta(minutes=30)).isoformat()
_INSIDE_RUN_WINDOW = (_T_RUN + timedelta(seconds=1)).isoformat()


def _marker(marker_id: str, content: str, run_id: object, *, created_at: str = _BEFORE_RUN):
    return SimpleNamespace(
        id=marker_id,
        content=content,
        metadata={
            'flag_for_stage2': True,
            'task_id': '7',
            'flag_type': 'stale_status',
            'run_id': run_id,
        },
        created_at=created_at,
    )


def _count_metadata_writes(memory_service) -> None:
    """Answer ``count_memories_by_metadata`` from the metadata ``add_memory`` has written."""
    written: list[dict] = []

    async def add_memory(*_args, **kwargs):
        written.append(dict(kwargs.get('metadata') or {}))
        return {'memory_ids': []}

    async def count_memories_by_metadata(*, filters, **_scope):
        return sum(all(meta.get(k) == v for k, v in filters.items()) for meta in written)

    memory_service.add_memory = AsyncMock(side_effect=add_memory)
    memory_service.count_memories_by_metadata = AsyncMock(side_effect=count_memories_by_metadata)


class TestStage2CarriesForwardMarkers:
    @pytest.fixture
    def memory_service(self, ledger_store):
        memory_service = AsyncMock()
        memory_service.count_memories_by_metadata.return_value = 0
        memory_service.delete_memory = AsyncMock(return_value=None)
        memory_service.add_memory.return_value = {'memory_ids': []}
        memory_service.recon_ledger = ledger_store
        return memory_service

    async def _seed(self, ledger_store, run_id: str, outcome: StageAgentOutcome, *, minute: int):
        await _seed_row(ledger_store, run_id, outcome, minute=minute, project_id=_PROJECT)

    async def _run(
        self,
        memory_service,
        markers: list,
        *,
        run_id: str = 'run-now',
        outcome: StageAgentOutcome = StageAgentOutcome.completed,
    ) -> tuple[StageReport, str]:
        memory_service.search.return_value = markers
        taskmaster = AsyncMock()
        taskmaster.get_tasks.return_value = {'tasks': []}
        journal = AsyncMock()
        journal.get_run = AsyncMock(return_value=SimpleNamespace(started_at=_T_RUN))
        stage = TaskKnowledgeSync(
            StageId.task_knowledge_sync,
            memory_service,
            taskmaster,
            journal,
            ReconciliationConfig(enabled=True, explore_codebase_root='/tmp/test'),
            scope=ProjectScope(ProjectId(_PROJECT), ProjectRoot('/tmp/test')),
        )
        cli = AsyncMock(return_value=StageResult(
            success=outcome is StageAgentOutcome.completed,
            declared_outcome=outcome,
            report={'flagged_items': [], 'summary': 'ok', 'stats': {}},
            llm_calls=1, tokens_used=0, cost_usd=0.0,
            model='test-model',
        ))
        with patch('fused_memory.reconciliation.stages.base.run_stage_via_cli', new=cli):
            report = await stage.run(
                events=[], watermark=Watermark(project_id=_PROJECT),
                prior_reports=[], run_id=run_id,
            )
        return report, cli.call_args.kwargs['payload']

    @pytest.mark.asyncio
    async def test_marker_from_a_run_whose_agent_died_is_carried_forward(
        self, memory_service, ledger_store,
    ):
        await self._seed(ledger_store, 'run-ok', StageAgentOutcome.completed, minute=0)
        await self._seed(ledger_store, 'run-dead', StageAgentOutcome.zero_output_timeout, minute=1)

        report, payload = await self._run(memory_service, [
            _marker('m-dead', 'CARRIED-FLAG-from-run-dead', 'run-dead'),
        ])

        assert 'CARRIED-FLAG-from-run-dead' in payload
        assert report.stats['stage2_markers_carried_forward'] == 1
        row = await ledger_store.get_by_identity(
            _PROJECT, 'cycle_summary', flag_type=_STAGE2, run_id='run-now',
        )
        assert row is not None
        assert json.loads(row.payload_json)['stats']['stage2_markers_carried_forward'] == 1

    @pytest.mark.asyncio
    async def test_marker_from_a_completed_run_is_not_carried(self, memory_service, ledger_store):
        await self._seed(ledger_store, 'run-ok', StageAgentOutcome.completed, minute=0)

        report, payload = await self._run(memory_service, [
            _marker('m-ok', 'SETTLED-FLAG-from-run-ok', 'run-ok'),
        ])

        assert 'SETTLED-FLAG-from-run-ok' not in payload
        assert 'stage2_markers_carried_forward' in report.stats
        assert report.stats['stage2_markers_carried_forward'] == 0

    @pytest.mark.asyncio
    async def test_a_later_completed_agent_stops_the_carry(self, memory_service, ledger_store):
        await self._seed(ledger_store, 'run-dead', StageAgentOutcome.zero_output_timeout, minute=0)
        await self._seed(ledger_store, 'run-later', StageAgentOutcome.completed, minute=1)

        report, payload = await self._run(memory_service, [
            _marker('m-dead', 'ALREADY-SHOWN-FLAG-from-run-dead', 'run-dead'),
        ])

        assert 'ALREADY-SHOWN-FLAG-from-run-dead' not in payload
        assert report.stats['stage2_markers_carried_forward'] == 0

    @pytest.mark.asyncio
    async def test_marker_from_a_run_with_no_cycle_summary_is_not_carried(
        self, memory_service, ledger_store,
    ):
        await self._seed(ledger_store, 'run-dead', StageAgentOutcome.zero_output_timeout, minute=0)

        report, payload = await self._run(memory_service, [
            _marker('m-ghost', 'UNATTRIBUTED-FLAG-from-run-ghost', 'run-ghost'),
        ])

        assert 'UNATTRIBUTED-FLAG-from-run-ghost' not in payload
        assert report.stats['stage2_markers_carried_forward'] == 0

    @pytest.mark.asyncio
    async def test_a_non_string_run_id_is_neither_carried_nor_fatal(
        self, memory_service, ledger_store,
    ):
        await self._seed(ledger_store, 'run-dead', StageAgentOutcome.zero_output_timeout, minute=0)

        report, payload = await self._run(memory_service, [
            _marker('m-dead', 'CARRIED-FLAG-from-run-dead', 'run-dead'),
            _marker('m-list', 'LIST-RUN-ID-FLAG', ['run-dead']),
        ])

        assert 'CARRIED-FLAG-from-run-dead' in payload
        assert 'LIST-RUN-ID-FLAG' not in payload
        assert report.stats['stage2_markers_carried_forward'] == 1

    @pytest.mark.asyncio
    async def test_an_in_window_marker_counts_once_as_rescued(self, memory_service, ledger_store):
        await self._seed(ledger_store, 'run-dead', StageAgentOutcome.zero_output_timeout, minute=0)

        report, payload = await self._run(memory_service, [
            _marker(
                'm-in-window', 'IN-WINDOW-FLAG', 'run-dead', created_at=_INSIDE_RUN_WINDOW,
            ),
        ])

        assert 'IN-WINDOW-FLAG' in payload
        assert report.stats['rescued_in_window_markers'] == 1
        assert report.stats['stage2_markers_carried_forward'] == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize('run_id', [None, ''], ids=['none', 'empty'])
    async def test_a_marker_without_a_run_id_is_never_carried(
        self, memory_service, ledger_store, run_id,
    ):
        await self._seed(ledger_store, 'run-dead', StageAgentOutcome.zero_output_timeout, minute=0)

        report, payload = await self._run(memory_service, [
            _marker('m-no-run', 'NO-RUN-ID-FLAG', run_id, created_at=_T0.isoformat()),
        ])

        assert 'NO-RUN-ID-FLAG' not in payload
        assert report.stats['stale_missing_run_id_markers'] == 1
        assert report.stats['stage2_markers_carried_forward'] == 0

    @pytest.mark.asyncio
    async def test_a_marker_carried_through_dead_runs_escalates_on_its_third_rendering(
        self, memory_service,
    ):
        _count_metadata_writes(memory_service)
        marker = _marker('m-1', 'FLAG-relayed-by-run-1', 'run-1')
        dead = StageAgentOutcome.zero_output_timeout

        await self._run(memory_service, [marker], run_id='run-1', outcome=dead)
        report_2, payload_2 = await self._run(memory_service, [marker], run_id='run-2', outcome=dead)
        report_3, payload_3 = await self._run(memory_service, [marker], run_id='run-3')

        assert report_2.stats['stage2_markers_carried_forward'] == 1
        assert 'Stale Flags Requiring Escalation' not in payload_2
        assert report_3.stats['stage2_markers_carried_forward'] == 1
        assert "flag_id='m-1'" in payload_3
        assert 'cycle_count=3' in payload_3
