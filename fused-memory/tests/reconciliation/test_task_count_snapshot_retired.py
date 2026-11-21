"""Reconciliation no longer keeps a task-count snapshot, for any project.

Task 10081 retired the snapshot (plans/live-task-facts-at-recall-prd.md D5):
Stage 2 writes no ``kind='task_count_snapshot'`` Mem0 record and enumerates
none, and the harness no longer escalates a missing one. 'reify' is the
project under test because it used to receive the deterministic write.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest
from escalation.queue import EscalationQueue

from fused_memory.config.schema import ReconciliationConfig
from fused_memory.models.reconciliation import StageId, StageReport, Watermark
from fused_memory.models.scope import ProjectId, ProjectRoot, ProjectScope
from fused_memory.reconciliation.cli_stage_runner import StageResult
from fused_memory.reconciliation.stages.task_knowledge_sync import TaskKnowledgeSync
from reconciliation.harness_fixtures import make_harness_config, open_harness, stub_stage_runs

SNAPSHOT_KIND = 'task_count_snapshot'
PROJECT_ID = 'reify'
STALE_SNAPSHOT_CATEGORY = 'recon_stale_task_count_snapshot'
PRE_RETIREMENT_MISSED_SNAPSHOT_STATS = {'task_count_snapshot_mem0_written': 0}


def _memory_service() -> AsyncMock:
    memory_service = AsyncMock()
    memory_service.count_memories_by_metadata.return_value = 1
    memory_service.get_memories_by_metadata.return_value = []
    memory_service.delete_memory = AsyncMock(return_value=None)
    memory_service.search.return_value = []
    memory_service.add_memory.return_value = {'memory_ids': []}
    return memory_service


def _taskmaster() -> AsyncMock:
    taskmaster = AsyncMock()
    taskmaster.get_tasks.return_value = {
        'tasks': [
            {'id': 1, 'title': 'shipped', 'status': 'done'},
            {'id': 2, 'title': 'in flight', 'status': 'in-progress'},
        ],
    }
    return taskmaster


def _successful_cli_result() -> StageResult:
    return StageResult(
        success=True,
        report={'flagged_items': [], 'summary': 'ok', 'stats': {}},
        llm_calls=1, tokens_used=0, cost_usd=0.0,
        model='test-model',
    )


def _metadata_filters(call) -> dict:
    filters = call.kwargs.get('filters')
    if filters is None and len(call.args) > 1:
        filters = call.args[1]
    return filters or {}


@pytest.mark.asyncio
async def test_stage2_run_for_formerly_snapshotting_project_writes_no_snapshot():
    memory_service = _memory_service()
    stage = TaskKnowledgeSync(
        StageId.task_knowledge_sync,
        scope=ProjectScope(ProjectId(PROJECT_ID), ProjectRoot('/tmp/test')),
        memory_service=memory_service,
        taskmaster=_taskmaster(),
        journal=AsyncMock(),
        config=ReconciliationConfig(enabled=True, explore_codebase_root='/tmp/test'),
    )

    with patch(
        'fused_memory.reconciliation.stages.base.run_stage_via_cli',
        new=AsyncMock(return_value=_successful_cli_result()),
    ):
        report = await stage.run(
            events=[], watermark=Watermark(project_id=PROJECT_ID),
            prior_reports=[], run_id='run-retired',
        )

    snapshot_writes = [
        call for call in memory_service.add_memory.await_args_list
        if (call.kwargs.get('metadata') or {}).get('kind') == SNAPSHOT_KIND
    ]
    snapshot_reads = [
        call
        for method in (
            memory_service.get_memories_by_metadata,
            memory_service.count_memories_by_metadata,
        )
        for call in method.await_args_list
        if _metadata_filters(call).get('kind') == SNAPSHOT_KIND
    ]
    snapshot_stats = [key for key in report.stats if SNAPSHOT_KIND in key]

    assert snapshot_writes == []
    assert snapshot_reads == []
    assert snapshot_stats == []


async def _stage_report_shaped_before_retirement(
    self, events, watermark, prior_reports, run_id, model=None,
):
    now = datetime.now(UTC)
    stats = (
        dict(PRE_RETIREMENT_MISSED_SNAPSHOT_STATS)
        if self.stage_id == StageId.task_knowledge_sync
        else {}
    )
    return StageReport(stage=self.stage_id, started_at=now, completed_at=now, stats=stats)


@pytest.mark.asyncio
async def test_consecutive_full_cycles_raise_no_stale_snapshot_alarm(tmp_path, monkeypatch):
    escalation_queue = EscalationQueue(tmp_path / 'esc')
    stub_stage_runs(monkeypatch, _stage_report_shaped_before_retirement)
    taskmaster = AsyncMock()
    taskmaster.get_tasks.return_value = {'tasks': []}

    async with open_harness(
        tmp_path,
        taskmaster=taskmaster,
        config=make_harness_config(judge_enabled=False),
        known_projects={PROJECT_ID: '/tmp/test-project'},
        escalation_queue=escalation_queue,
    ) as harness:
        for _ in range(3):
            await harness.run_full_cycle(PROJECT_ID, 'test-trigger')

    alarm_categories = [
        esc.category for esc in escalation_queue.get_pending()
        if esc.category == STALE_SNAPSHOT_CATEGORY
    ]
    assert alarm_categories == []
