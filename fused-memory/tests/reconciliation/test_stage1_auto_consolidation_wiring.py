"""MemoryConsolidator.run calls the auto-consolidation post-step (task 5240, PRD C4).

Harness per ``tests/reconciliation/test_stage1.py::TestMemoryConsolidatorGateBacklogWiring``:
``BaseStage.run`` is patched out (it would spawn the LLM turn), the escalation
queue is a real ``EscalationQueue`` on ``tmp_path`` attached the way
``reconciliation/harness.py`` attaches it, and the memory service is the
fixture's bare AsyncMock with a real ``ConsolidationAutoConfig`` and a real
ledger set on it. The base report flags nothing, so the real ``dedup_flags``
has no flag to read or write.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from escalation.queue import EscalationQueue

from fused_memory.config.schema import ConsolidationAutoConfig
from fused_memory.models.reconciliation import StageId, StageReport, Watermark
from fused_memory.reconciliation.consolidation_auto_executor import (
    auto_consolidation_initial_stats,
)
from fused_memory.reconciliation.recon_ledger import ReconLedgerStore
from fused_memory.reconciliation.stages.base import BaseStage
from fused_memory.reconciliation.task_filter import FilteredTaskTree
from reconciliation.consolidator_fixtures import make_consolidator


def _tree() -> FilteredTaskTree:
    return FilteredTaskTree(
        active_tasks=[],
        done_tasks=[],
        cancelled_tasks=[],
        done_count=0,
        cancelled_count=0,
        other_count=0,
        total_count=0,
        max_task_id=0,
    )


def _base_report() -> StageReport:
    now = datetime.now(UTC)
    return StageReport(
        stage=StageId.memory_consolidator, started_at=now, completed_at=now, stats={},
    )


@pytest_asyncio.fixture
async def ledger(tmp_path):
    store = ReconLedgerStore(tmp_path / 'reconciliation.db')
    await store.initialize()
    yield store
    await store.close()


@pytest.fixture
def stage(tmp_path, ledger):
    stage = make_consolidator(project_root='/tmp/auto-consolidation-wiring')
    stage._escalation_queue = EscalationQueue(tmp_path / 'esc')
    stage.filtered_task_tree = _tree()
    stage.memory.config.consolidation_auto = ConsolidationAutoConfig(enabled_projects=[])
    stage.memory.recon_ledger = ledger
    stage.memory.get_memories_by_metadata = AsyncMock(return_value=[])
    return stage


async def _run(stage, run_id='run-wiring') -> StageReport:
    with patch.object(BaseStage, 'run', new=AsyncMock(return_value=_base_report())):
        return await stage.run(
            events=[],
            watermark=Watermark(project_id='test_project'),
            prior_reports=[],
            run_id=run_id,
        )


@pytest.mark.asyncio
async def test_a_full_cycle_runs_the_post_step(stage):
    report = await _run(stage)

    assert report.stats['auto_consolidations_skipped_project'] == 1


@pytest.mark.asyncio
async def test_a_remediation_pass_still_publishes_every_key(stage):
    stage.remediation_findings = []

    report = await _run(stage)

    for key, initial in auto_consolidation_initial_stats().items():
        assert report.stats[key] == initial, key


@pytest.mark.asyncio
async def test_the_config_is_read_live_each_cycle(stage):
    first = await _run(stage, 'run-1')
    stage.memory.config.consolidation_auto.enabled_projects = ['test_project']

    second = await _run(stage, 'run-2')

    assert first.stats['auto_consolidations_skipped_project'] == 1
    assert second.stats['auto_consolidations_skipped_project'] == 0
    assert second.stats['auto_consolidation_proposals_listed'] == 0
    assert second.stats['auto_consolidation_skipped_unavailable'] is None


@pytest.mark.asyncio
async def test_no_task_tree_fails_closed(stage):
    stage.memory.config.consolidation_auto.enabled_projects = ['test_project']
    stage.filtered_task_tree = None

    report = await _run(stage)

    assert report.stats['auto_consolidation_skipped_unavailable'] == 'task_tree'


@pytest.mark.asyncio
async def test_an_executor_failure_does_not_abort_stage_one(stage, ledger, monkeypatch):
    stage.memory.config.consolidation_auto.enabled_projects = ['test_project']
    monkeypatch.setattr(
        ledger, 'list_by_kind', AsyncMock(side_effect=RuntimeError('ledger down')),
    )

    report = await _run(stage)

    assert 'stage1_gate_backlog_stalled' in report.stats
    assert report.stats['auto_consolidations_skipped_project'] == 0
