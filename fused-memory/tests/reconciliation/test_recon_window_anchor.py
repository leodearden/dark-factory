"""Where a full cycle anchors the next cycle's episode, memory and task-change windows.

Invariant: after a completed full cycle, the persisted
``last_episode_timestamp`` and ``last_memory_timestamp`` -- the lower bounds of
the NEXT cycle's "new since the last reconciliation" windows -- are at or
before the instant Stage 1 fetched its episodes and memories (task 4465). The
persisted ``last_task_change_timestamp`` is at or before the cycle's task-tree
read, the read Stage 2's done-task audit renders (task 6070).

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
their source metadata. Measured immaterial on 2026-10-25 (task 6077):
plans/recon-window-anchor-payload-measurement-2026-10-25.md. Do not move the
anchor back to completion: that reopens the hole for external writes.

The anchor is ``run.started_at``. On a fresh run it is stamped before Stage 1.
On a resumed run (task sigma) it is the ORIGINAL start of the interrupted run,
whose Stage 1 may have fetched hours before the resuming invocation -- so a
clock read taken by the resuming invocation would recreate the hole. The
episode anchor alone is lowered further, to the start of the earliest Graphiti
write that was still executing at Stage 1's fetch, when that is earlier (task
4620). Why: ``fused_memory/reconciliation/episode_window.py``.

The task-change window moves too, and the resume argument above applies to it
unchanged. Its own cost: a task that reaches done between run start and the
tree read is audited by both cycles, and so is a done task whose ``updatedAt``
this cycle's own writes bumped. Stage 2's ``stage2_suppress`` count-gate bounds
both. ``last_full_run_completed`` keeps its completion-time meaning for its
display readers, ``memory_consolidator.py::_format_watermark`` and the
dashboard; Stage 2's done-task audit and the context assembler no longer read
it.
"""
from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from fused_memory.backends.sqlite_task_backend import task_timestamp_now
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
from fused_memory.reconciliation.episode_window import (
    UNFINISHED_WRITE_START_STAT,
    unfinished_write_start_stat,
)
from fused_memory.reconciliation.journal import ReconciliationJournal
from fused_memory.reconciliation.task_filter import (
    filter_task_tree,
    select_done_since_boundary,
)
from reconciliation.harness_fixtures import make_harness_config, open_harness, stub_stage_runs

PROJECT_ID = 'test-project'
INTERRUPTED_RUN_START = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
HARNESS_LOGGER = 'fused_memory.reconciliation.harness'


@pytest.fixture
def taskmaster():
    return AsyncMock()


@pytest_asyncio.fixture
async def harness(tmp_path, taskmaster):
    async with open_harness(
        tmp_path,
        taskmaster=taskmaster,
        config=make_harness_config(judge_enabled=False),
        known_projects={PROJECT_ID: '/tmp/test-project'},
    ) as h:
        yield h


def _stub_stage_runs(
    monkeypatch: pytest.MonkeyPatch,
    on_invoke: Callable[[StageId], None],
    stage1_stats: dict | None = None,
) -> None:
    """Replace every stage's ``run`` with a report, calling *on_invoke* first.

    Stage 1's report carries *stage1_stats*; every other report is empty.
    """

    async def fake_run(
        self, events, watermark, prior_reports, run_id, model=None, resume_session_id=None,
    ):
        on_invoke(self.stage_id)
        now = datetime.now(UTC)
        is_stage1 = self.stage_id == StageId.memory_consolidator
        stats = dict(stage1_stats or {}) if is_stage1 else {}
        return StageReport(stage=self.stage_id, started_at=now, completed_at=now, stats=stats)

    stub_stage_runs(monkeypatch, fake_run)


@pytest.fixture
def stage1_stats() -> dict:
    """The stats the stubbed Stage 1 reports; a test fills it before the cycle."""
    return {}


@pytest.fixture
def stage_invoked_at(monkeypatch, stage1_stats):
    """Stub every stage's ``run`` and record when each stage was invoked."""
    invoked_at: dict[StageId, datetime] = {}

    def record(stage_id: StageId) -> None:
        invoked_at[stage_id] = datetime.now(UTC)

    _stub_stage_runs(monkeypatch, record, stage1_stats)
    return invoked_at


def _required(value: datetime | None) -> datetime:
    assert value is not None
    return value


def _interrupted_run(instance_id: str, stage1_stats: dict) -> ReconciliationRun:
    """A run interrupted in Stage 2, after Stage 1 completed with *stage1_stats*."""
    stage1_report = StageReport(
        stage=StageId.memory_consolidator,
        started_at=INTERRUPTED_RUN_START,
        completed_at=INTERRUPTED_RUN_START,
        stats=stage1_stats,
    )
    return ReconciliationRun(
        id='run-to-resume',
        project_id=PROJECT_ID,
        run_type=RunType.full,
        trigger_reason='test-trigger',
        started_at=INTERRUPTED_RUN_START,
        status=RunStatus.interrupted,
        stage_reports={StageId.memory_consolidator.value: stage1_report},
        session_id='S',
        stage_cursor=StageId.task_knowledge_sync.value,
        instance_id=instance_id,
    )


async def _persisted_and_reloaded(
    journal: ReconciliationJournal, run: ReconciliationRun,
) -> ReconciliationRun:
    """Write *run* through the journal's own run writes, then read it back."""
    assert run.session_id is not None and run.stage_cursor is not None
    await journal.start_run(run)
    await journal.update_run_stage_reports(run.id, run.stage_reports)
    await journal.record_run_session(
        run.id, session_id=run.session_id, stage_cursor=run.stage_cursor,
    )
    reloaded = await journal.get_run(run.id)
    assert reloaded is not None
    return reloaded


def _drained_event() -> ReconciliationEvent:
    return ReconciliationEvent(
        id=str(uuid.uuid4()),
        type=EventType.episode_added,
        source=EventSource.agent,
        project_id=PROJECT_ID,
        timestamp=INTERRUPTED_RUN_START,
        payload={},
    )


@pytest.mark.asyncio
async def test_fresh_cycle_anchors_data_windows_at_run_start(harness, stage_invoked_at):
    run = await harness.run_full_cycle(PROJECT_ID, 'test-trigger')
    watermark = await harness.journal.get_watermark(PROJECT_ID)

    assert run.status == RunStatus.completed
    episode_anchor = _required(watermark.last_episode_timestamp)
    memory_anchor = _required(watermark.last_memory_timestamp)

    stage1_invoked_at = stage_invoked_at[StageId.memory_consolidator]
    assert episode_anchor <= stage1_invoked_at
    assert memory_anchor <= stage1_invoked_at
    assert episode_anchor == run.started_at
    assert memory_anchor == run.started_at
    assert _required(watermark.last_task_change_timestamp) == run.started_at

    stage3_invoked_at = stage_invoked_at[StageId.integrity_check]
    assert _required(watermark.last_full_run_completed) >= stage3_invoked_at
    assert watermark.last_full_run_id == run.id


@pytest.mark.asyncio
async def test_resumed_run_anchors_data_windows_at_original_start(harness, stage_invoked_at):
    interrupted_run = _interrupted_run(harness.buffer.instance_id, stage1_stats={})
    await harness.journal.start_run(interrupted_run)

    run = await harness.run_full_cycle(
        PROJECT_ID, 'resume_after_restart',
        resume_run=interrupted_run, events=[_drained_event()],
    )
    watermark = await harness.journal.get_watermark(PROJECT_ID)

    assert run.status == RunStatus.completed
    assert StageId.memory_consolidator not in stage_invoked_at
    assert watermark.last_episode_timestamp == INTERRUPTED_RUN_START
    assert watermark.last_memory_timestamp == INTERRUPTED_RUN_START
    assert watermark.last_task_change_timestamp == INTERRUPTED_RUN_START
    assert _required(watermark.last_full_run_completed) > INTERRUPTED_RUN_START


@pytest.mark.asyncio
async def test_task_done_mid_cycle_is_inside_next_cycles_task_change_window(
    harness, taskmaster, monkeypatch,
):
    """The harness half of the done-audit invariant: the persisted window contains task 7001.

    The Stage 2 half, that the audit windows by ``last_task_change_timestamp``, is
    tests/test_stages.py::TestTaskKnowledgeSyncDoneAuditSection::test_audit_window_starts_at_task_change_timestamp_not_completion.
    """
    before_cycle = '2026-01-01T00:00:00.000Z'
    tasks = [
        {'id': 7001, 'title': 'done mid-cycle', 'status': 'in-progress', 'updatedAt': before_cycle},
        {'id': 7002, 'title': 'done before the cycle', 'status': 'done', 'updatedAt': before_cycle},
    ]
    snapshots: list[list[dict]] = []

    async def get_tasks(*args, **kwargs):
        snapshot = [dict(t) for t in tasks]
        snapshots.append(snapshot)
        return {'tasks': snapshot}

    taskmaster.get_tasks.side_effect = get_tasks

    def finish_task_7001_during_stage3(stage_id: StageId) -> None:
        if stage_id == StageId.integrity_check:
            tasks[0].update(status='done', updatedAt=task_timestamp_now())

    _stub_stage_runs(monkeypatch, finish_task_7001_during_stage3)
    run = await harness.run_full_cycle(PROJECT_ID, 'test-trigger')
    watermark = await harness.journal.get_watermark(PROJECT_ID)

    assert run.status == RunStatus.completed
    assert snapshots
    assert all(
        t['status'] != 'done' for snapshot in snapshots for t in snapshot if t['id'] == 7001
    )
    next_cycle_audit = select_done_since_boundary(
        filter_task_tree({'tasks': tasks}).all_done_tasks,
        watermark.last_task_change_timestamp,
    )
    audited = {t['id'] for t in next_cycle_audit}
    assert 7001 in audited
    assert 7002 not in audited


@pytest.mark.asyncio
async def test_fresh_cycle_lowers_episode_anchor_below_unfinished_graphiti_write(
    harness, stage_invoked_at, stage1_stats, caplog,
):
    unfinished = datetime.now(UTC) - timedelta(minutes=10)
    stage1_stats[UNFINISHED_WRITE_START_STAT] = unfinished_write_start_stat(unfinished)

    with caplog.at_level(logging.INFO, logger=HARNESS_LOGGER):
        run = await harness.run_full_cycle(PROJECT_ID, 'test-trigger')
    watermark = await harness.journal.get_watermark(PROJECT_ID)

    assert run.status == RunStatus.completed
    assert watermark.last_episode_timestamp == unfinished
    assert watermark.last_memory_timestamp == run.started_at
    stage3_invoked_at = stage_invoked_at[StageId.integrity_check]
    assert _required(watermark.last_full_run_completed) >= stage3_invoked_at
    assert watermark.last_full_run_id == run.id

    anchored = [
        r for r in caplog.records
        if r.name == HARNESS_LOGGER and r.getMessage() == 'reconciliation.episode_window_anchored'
    ]
    assert len(anchored) == 1
    assert getattr(anchored[0], 'run_id', None) == run.id
    assert datetime.fromisoformat(getattr(anchored[0], 'anchor', '')) == unfinished


@pytest.mark.asyncio
async def test_resumed_run_anchors_on_persisted_stage1_unfinished_write_start(
    harness, stage_invoked_at,
):
    unfinished = INTERRUPTED_RUN_START - timedelta(minutes=5)
    interrupted_run = _interrupted_run(
        harness.buffer.instance_id,
        stage1_stats={UNFINISHED_WRITE_START_STAT: unfinished_write_start_stat(unfinished)},
    )
    reloaded = await _persisted_and_reloaded(harness.journal, interrupted_run)

    run = await harness.run_full_cycle(
        PROJECT_ID, 'resume_after_restart',
        resume_run=reloaded, events=[_drained_event()],
    )
    watermark = await harness.journal.get_watermark(PROJECT_ID)

    assert run.status == RunStatus.completed
    assert StageId.memory_consolidator not in stage_invoked_at
    assert watermark.last_episode_timestamp == unfinished
    assert watermark.last_memory_timestamp == INTERRUPTED_RUN_START
