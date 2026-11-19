"""Contract tests for ``reconciliation.harness_fixtures``, the shared harness factory (task 6075)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from fused_memory.models.reconciliation import RunStatus, StageId, StageReport
from reconciliation.harness_fixtures import (
    make_harness,
    make_harness_config,
    open_event_buffer,
    open_harness,
    open_journal,
    stub_stage_runs,
)


@pytest.mark.asyncio
async def test_open_harness_drives_a_full_cycle_through_stubbed_stages(tmp_path, monkeypatch):
    invoked: list[StageId] = []

    async def fake_run(
        self, events, watermark, prior_reports, run_id, model=None, resume_session_id=None,
    ):
        invoked.append(self.stage_id)
        now = datetime.now(UTC)
        return StageReport(stage=self.stage_id, started_at=now, completed_at=now)

    async with open_harness(
        tmp_path,
        taskmaster=AsyncMock(),
        config=make_harness_config(judge_enabled=False),
        known_projects={'test-project': '/tmp/test-project'},
    ) as harness:
        stub_stage_runs(monkeypatch, fake_run)
        run = await harness.run_full_cycle('test-project', 'test-trigger')

        assert run.status == RunStatus.completed
        assert invoked == [
            StageId.memory_consolidator,
            StageId.task_knowledge_sync,
            StageId.integrity_check,
        ]
        persisted = await harness.journal.get_run(run.id)
        assert persisted is not None
        assert persisted.status == 'completed'


@pytest.mark.asyncio
async def test_make_harness_builds_the_judge_its_config_asks_for(tmp_path):
    async with (
        open_journal(tmp_path / 'journal') as journal,
        open_event_buffer(tmp_path / 'events.db') as buffer,
    ):
        judgeless = make_harness(
            journal=journal,
            event_buffer=buffer,
            taskmaster=None,
            config=make_harness_config(judge_enabled=False),
        )
        default = make_harness(journal=journal, event_buffer=buffer, taskmaster=None)

        assert judgeless.judge is None
        assert default.judge is not None
