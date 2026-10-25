"""Where a full cycle anchors the next cycle's episode/memory windows (task 4465).

Invariant: after a completed full cycle, the persisted
``last_episode_timestamp`` and ``last_memory_timestamp`` -- the lower bounds of
the NEXT cycle's "new since the last reconciliation" windows -- are at or
before the instant Stage 1 fetched its episodes and memories.

Why cycle completion is too late: Stage 1 fetches near the start of a cycle,
then the three stages run for minutes. An item created after that fetch but
before completion is shown to neither cycle: this one fetched before the item
existed, and the next one's window starts after it. That skip is silent loss.

What an earlier anchor costs: the next cycle's window covers everything created
from this cycle's start to its completion. Besides the external writes the old
anchor dropped, that includes two classes it hid:

- items created between the anchor and Stage 1's fetch, which this cycle
  already saw and the next one sees again;
- every record this cycle wrote itself after its fetch, e.g. memories added by
  consolidation, completion memories, and the ``cycle_summary`` mirror and
  ``task_count_snapshot`` records. No cycle was ever shown these before.

Both cost Stage 1 prompt tokens, and the second class may lead Stage 1 to
re-consolidate its predecessor's output. That cost is recoverable; a skipped
external item is not. Nothing here tests that re-presentation is harmless. If
the churn matters, drop reconciliation-sourced records from the "new" list by
their source metadata. Do not move the anchor back to completion: that reopens
the hole for external writes.

The anchor is ``run.started_at``. On a fresh run it is stamped before Stage 1.
On a resumed run (task sigma) it is the ORIGINAL start of the interrupted run,
whose Stage 1 may have fetched hours before the resuming invocation -- so a
clock read taken by the resuming invocation would recreate the hole.

Only the two data-window fields move. ``last_full_run_completed`` keeps its
completion-time meaning for its other readers (Stage 2's done-task audit and
the context assembler).
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from fused_memory.config.schema import FusedMemoryConfig, ReconciliationConfig
from fused_memory.models.reconciliation import (
    EventSource,
    EventType,
    ReconciliationEvent,
    ReconciliationRun,
    RunStatus,
    RunType,
    StageId,
    StageReport,
)
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.reconciliation.harness import ReconciliationHarness
from fused_memory.reconciliation.journal import ReconciliationJournal
from fused_memory.reconciliation.stages.memory_consolidator import MemoryConsolidator
from fused_memory.reconciliation.stages.task_knowledge_sync import (
    IntegrityCheck,
    TaskKnowledgeSync,
)

PROJECT_ID = 'test-project'
INTERRUPTED_RUN_START = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)


@pytest_asyncio.fixture
async def journal(tmp_path):
    j = ReconciliationJournal(tmp_path / 'window_anchor_journal')
    await j.initialize()
    yield j
    await j.close()


@pytest_asyncio.fixture
async def event_buffer(tmp_path):
    buf = EventBuffer(
        db_path=tmp_path / 'window_anchor_eb.db',
        buffer_size_threshold=2,
        max_staleness_seconds=3600,
    )
    await buf.initialize()
    yield buf
    await buf.close()


@pytest.fixture
def memory_service():
    svc = AsyncMock()
    svc.search = AsyncMock(return_value=[])
    svc.get_episodes = AsyncMock(return_value=[])
    svc.get_status = AsyncMock(
        return_value={
            'graphiti': {'connected': True},
            'mem0': {'connected': True},
            'projects': {},
        }
    )
    svc.get_entity = AsyncMock(return_value={'nodes': [], 'edges': []})
    svc.get_memories_by_metadata = AsyncMock(return_value=[])
    svc.mem0 = AsyncMock()
    svc.mem0.get_all = AsyncMock(return_value={'results': []})
    return svc


@pytest.fixture
def harness(journal, event_buffer, memory_service):
    config = FusedMemoryConfig(
        reconciliation=ReconciliationConfig(
            enabled=True,
            explore_codebase_root='/tmp/test',
            agent_llm_provider='anthropic',
            agent_llm_model='claude-sonnet-4-20250514',
            judge_enabled=False,
        )
    )
    return ReconciliationHarness(
        memory_service=memory_service,
        taskmaster=AsyncMock(),
        journal=journal,
        event_buffer=event_buffer,
        config=config,
        known_projects={PROJECT_ID: '/tmp/test-project'},
    )


@pytest.fixture
def stage_invoked_at():
    """Stub every stage's ``run`` and record when each stage was invoked."""
    invoked_at: dict[StageId, datetime] = {}

    async def fake_run(
        self, events, watermark, prior_reports, run_id, model=None, resume_session_id=None,
    ):
        now = datetime.now(UTC)
        invoked_at[self.stage_id] = now
        return StageReport(stage=self.stage_id, started_at=now, completed_at=now)

    with (
        patch.object(MemoryConsolidator, 'run', fake_run),
        patch.object(TaskKnowledgeSync, 'run', fake_run),
        patch.object(IntegrityCheck, 'run', fake_run),
    ):
        yield invoked_at


def _required(value: datetime | None) -> datetime:
    assert value is not None
    return value


@pytest.mark.asyncio
async def test_fresh_cycle_anchors_data_windows_at_run_start(
    harness, journal, stage_invoked_at,
):
    run = await harness.run_full_cycle(PROJECT_ID, 'test-trigger')
    watermark = await journal.get_watermark(PROJECT_ID)

    assert run.status == RunStatus.completed
    episode_anchor = _required(watermark.last_episode_timestamp)
    memory_anchor = _required(watermark.last_memory_timestamp)

    stage1_invoked_at = stage_invoked_at[StageId.memory_consolidator]
    assert episode_anchor <= stage1_invoked_at
    assert memory_anchor <= stage1_invoked_at
    assert episode_anchor == run.started_at
    assert memory_anchor == run.started_at

    stage3_invoked_at = stage_invoked_at[StageId.integrity_check]
    assert _required(watermark.last_full_run_completed) >= stage3_invoked_at
    assert watermark.last_full_run_id == run.id


@pytest.mark.asyncio
async def test_resumed_run_anchors_data_windows_at_original_start(
    harness, journal, event_buffer, stage_invoked_at,
):
    stage1_report = StageReport(
        stage=StageId.memory_consolidator,
        started_at=INTERRUPTED_RUN_START,
        completed_at=INTERRUPTED_RUN_START,
    )
    interrupted_run = ReconciliationRun(
        id='run-to-resume',
        project_id=PROJECT_ID,
        run_type=RunType.full,
        trigger_reason='test-trigger',
        started_at=INTERRUPTED_RUN_START,
        status=RunStatus.interrupted,
        stage_reports={StageId.memory_consolidator.value: stage1_report},
        session_id='S',
        stage_cursor=StageId.task_knowledge_sync.value,
        instance_id=event_buffer.instance_id,
    )
    await journal.start_run(interrupted_run)
    drained_event = ReconciliationEvent(
        id=str(uuid.uuid4()),
        type=EventType.episode_added,
        source=EventSource.agent,
        project_id=PROJECT_ID,
        timestamp=INTERRUPTED_RUN_START,
        payload={},
    )

    run = await harness.run_full_cycle(
        PROJECT_ID, 'resume_after_restart',
        resume_run=interrupted_run, events=[drained_event],
    )
    watermark = await journal.get_watermark(PROJECT_ID)

    assert run.status == RunStatus.completed
    assert StageId.memory_consolidator not in stage_invoked_at
    assert watermark.last_episode_timestamp == INTERRUPTED_RUN_START
    assert watermark.last_memory_timestamp == INTERRUPTED_RUN_START
    assert _required(watermark.last_full_run_completed) > INTERRUPTED_RUN_START
