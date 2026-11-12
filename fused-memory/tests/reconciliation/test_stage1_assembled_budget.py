"""Stage 1's budget accounting: AssembledPayload.budget_report() and the
report.stats['stage1_assembled_budget'] stat MemoryConsolidator.run() records from it.

The next cap re-evaluation reads that stat from runs.stage_reports
(plans/recon-evidence-caps-measurement-2026-10.md).
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from fused_memory.models.reconciliation import (
    AssembledPayload,
    ContextItem,
    EventSource,
    EventType,
    ReconciliationEvent,
    StageId,
    StageReport,
    Watermark,
)
from fused_memory.reconciliation.stages.base import BaseStage
from reconciliation.consolidator_fixtures import make_consolidator, make_scope


def _buffered_event(n: int) -> ReconciliationEvent:
    return ReconciliationEvent(
        id=f'ev{n}', type=EventType.memory_added, source=EventSource.agent,
        project_id='test_project', timestamp=datetime.now(UTC),
    )


def _budget_bound_payload() -> AssembledPayload:
    return AssembledPayload(
        events=[_buffered_event(0), _buffered_event(1)],
        context_items={f'c{n}': ContextItem(id=f'c{n}', source='mem0', formatted='x') for n in range(3)},
        total_tokens=8_960,
        events_remaining=5,
        token_budget=9_000,
    )


def test_budget_report_states_spend_breadth_and_deferral():
    assert _budget_bound_payload().budget_report() == {
        'estimated_tokens': 8960,
        'token_budget': 9000,
        'events': 2,
        'context_items': 3,
        'events_deferred': 5,
    }


async def _run_with_assembled_payload(assembled_payload: AssembledPayload | None) -> StageReport:
    stage = make_consolidator()
    stage.scope = make_scope('test_project', stage.scope.project_root)
    stage.assembled_payload = assembled_payload
    base_report = StageReport(
        stage=StageId.memory_consolidator,
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        items_flagged=[],
        stats={},
    )
    with patch.object(BaseStage, 'run', new=AsyncMock(return_value=base_report)):
        return await stage.run(
            events=[],
            watermark=Watermark(project_id='test_project'),
            prior_reports=[],
            run_id='run-assembled-budget',
        )


class TestRunRecordsAssembledBudget:
    @pytest.mark.asyncio
    async def test_the_assembled_path_records_its_budget_report(self):
        payload = _budget_bound_payload()

        report = await _run_with_assembled_payload(payload)

        assert report.stats['stage1_assembled_budget'] == payload.budget_report()

    @pytest.mark.asyncio
    async def test_the_time_windowed_path_records_none(self):
        report = await _run_with_assembled_payload(None)

        assert 'stage1_assembled_budget' in report.stats
        assert report.stats['stage1_assembled_budget'] is None
