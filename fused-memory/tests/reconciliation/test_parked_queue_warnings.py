"""A Graphiti write queue holding only parked rows raises both reconciliation WARNINGs.

The harness reads the queue once per cycle and warns with
``reconciliation.graphiti_queue_unhealthy``; Stage 1 re-surfaces the same health
record and warns with ``reconciliation.graphiti_queue_unhealthy_stage1``. Each
extra names the parked count, so a parked backlog with no dead letter is seen.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest
from _parked_queue_fixtures import queue_holding_parked_rows

from fused_memory.models.reconciliation import (
    EventSource,
    EventType,
    ReconciliationEvent,
    StageId,
    StageReport,
    Watermark,
)
from fused_memory.reconciliation.queue_health import summarize_graphiti_queue_health
from fused_memory.reconciliation.stages.base import BaseStage
from fused_memory.reconciliation.stages.memory_consolidator import MemoryConsolidator
from reconciliation.consolidator_fixtures import make_consolidator
from reconciliation.harness_fixtures import (
    make_harness,
    make_harness_config,
    make_memory_service,
    open_event_buffer,
    open_journal,
    stub_stage_runs,
)

PROJECT_ID = 'test-project'
PARKED_ROWS = 3
HARNESS_LOGGER = 'fused_memory.reconciliation.harness'
CONSOLIDATOR_LOGGER = 'fused_memory.reconciliation.stages.memory_consolidator'


def _event() -> ReconciliationEvent:
    return ReconciliationEvent(
        id='parked-queue-event',
        type=EventType.episode_added,
        source=EventSource.agent,
        project_id=PROJECT_ID,
        timestamp=datetime.now(UTC),
        payload={},
    )


def _empty_report(stage_id: StageId) -> StageReport:
    now = datetime.now(UTC)
    return StageReport(stage=stage_id, started_at=now, completed_at=now, stats={})


def _warnings_named(caplog: pytest.LogCaptureFixture, logger_name: str, message: str) -> list:
    return [
        r for r in caplog.records
        if r.name == logger_name and r.getMessage() == message
    ]


@pytest.mark.asyncio
async def test_the_harness_warns_on_a_queue_holding_only_parked_rows(tmp_path, caplog, monkeypatch):
    empty_project = AsyncMock()
    empty_project.get_tasks = AsyncMock(return_value={'tasks': []})
    empty_project.get_statuses_fresh = AsyncMock(return_value={})
    stage1_queue_health: list[dict | None] = []

    async def fake_run(self, events, watermark, prior_reports, run_id, model=None):
        if isinstance(self, MemoryConsolidator):
            stage1_queue_health.append(self.graphiti_queue_health)
        return _empty_report(self.stage_id)

    stub_stage_runs(monkeypatch, fake_run)
    async with (
        queue_holding_parked_rows(tmp_path, rows=PARKED_ROWS, group_id=PROJECT_ID) as queue,
        open_journal(tmp_path / 'journal') as journal,
        open_event_buffer(tmp_path / 'event_buffer.db') as event_buffer,
    ):
        memory = make_memory_service()
        memory.durable_queue = queue
        harness = make_harness(
            journal=journal,
            event_buffer=event_buffer,
            memory_service=memory,
            taskmaster=empty_project,
            config=make_harness_config(judge_enabled=False),
            known_projects={PROJECT_ID: '/tmp/test-project'},
        )
        with caplog.at_level(logging.WARNING, logger=HARNESS_LOGGER):
            await harness.run_full_cycle(PROJECT_ID, 'parked-queue-test', events=[_event()])

    unhealthy = _warnings_named(caplog, HARNESS_LOGGER, 'reconciliation.graphiti_queue_unhealthy')
    assert len(unhealthy) == 1, f'expected one unhealthy warning, got {unhealthy!r}'
    assert (unhealthy[0].parked_count, unhealthy[0].dead_count) == (PARKED_ROWS, 0)
    assert [h and h['parked_count'] for h in stage1_queue_health] == [PARKED_ROWS]


@pytest.mark.asyncio
async def test_stage1_warns_on_a_queue_holding_only_parked_rows(caplog):
    stage = make_consolidator(project_root='/tmp/reify')
    stage.graphiti_queue_health = summarize_graphiti_queue_health(
        {'counts': {'parked': PARKED_ROWS}, 'oldest_pending_age_seconds': 86400.0},
    )

    with (
        patch.object(
            BaseStage, 'run',
            new=AsyncMock(return_value=_empty_report(StageId.memory_consolidator)),
        ),
        patch('fused_memory.reconciliation.stage1_flag_chain.dedup_flags', new=AsyncMock(return_value=[])),
        caplog.at_level(logging.WARNING, logger=CONSOLIDATOR_LOGGER),
    ):
        await stage.run(
            events=[],
            watermark=Watermark(project_id='test_project'),
            prior_reports=[],
            run_id='run-parked-queue',
        )

    unhealthy = _warnings_named(
        caplog, CONSOLIDATOR_LOGGER, 'reconciliation.graphiti_queue_unhealthy_stage1',
    )
    assert len(unhealthy) == 1, f'expected one Stage 1 unhealthy warning, got {unhealthy!r}'
    assert (unhealthy[0].parked_count, unhealthy[0].dead_count) == (PARKED_ROWS, 0)
