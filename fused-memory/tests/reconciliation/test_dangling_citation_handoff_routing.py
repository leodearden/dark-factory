"""The dangling-citation finding reaches Stage 1 through both channels (task 5749).

Stage 3's cross-run scan files a finding that cites nothing (the dangling id
must never back it) and carries its ids in a ``dangling_citation`` record.
These tests drive the real ``run_full_cycle`` and prove that record reaches
Stage 1, which holds ``repair_memory_citation``:

- (a) in the same cycle's remediation pass, as ``remediation_findings``;
- (b) in the next full cycle, as ``prior_s3_findings``.

Only public seams are used: a constructor-built harness from
``reconciliation.harness_fixtures``, and stage ``run`` methods stubbed at class
level.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from fused_memory.models.reconciliation import StageReport
from fused_memory.reconciliation.dangling_citation import (
    DanglingCitation,
    build_dangling_citation_finding,
    dangling_citation_from_finding,
)
from fused_memory.reconciliation.stages.memory_consolidator import MemoryConsolidator
from fused_memory.reconciliation.stages.task_knowledge_sync import IntegrityCheck
from reconciliation.harness_fixtures import make_harness_config, open_harness, stub_stage_runs

PROJECT = 'proj'
HARNESS_LOGGER = 'fused_memory.reconciliation.harness'
PLACEHOLDER_DROP = 'reconciliation.remediation_dropped_placeholder_finding'

RECORD = DanglingCitation(
    target_run_id='5f0c2d8e-3b1a-4e6f-9a7d-2c4b6e8f0a1b',
    finding_id='c7e9a1b3-5d2f-4a6c-8e0b-1d3f5a7c9e2b',
    memory_id='9a1b3c5d-7e9f-4b2d-a6c8-0e2f4a6c8e1d',
    store='mem0',
)


@pytest_asyncio.fixture
async def cycle_harness(tmp_path):
    taskmaster = AsyncMock()
    taskmaster.get_tasks = AsyncMock(return_value={'tasks': []})
    async with open_harness(
        tmp_path,
        taskmaster=taskmaster,
        config=make_harness_config(judge_enabled=False),
        known_projects={PROJECT: '/tmp/test-project'},
    ) as harness:
        yield harness


@pytest.fixture
def stage_1_calls(monkeypatch):
    """Stub every stage's run; Stage 3 files the dangling finding on its FIRST run only.

    Records ``(remediation_findings, prior_s3_findings)`` for every Stage-1 run.
    """
    calls: list[tuple[list[dict] | None, list[dict] | None]] = []
    integrity_runs: list[int] = []

    async def fake_run(self, events, watermark, prior_reports, run_id, model=None,
                       resume_session_id=None) -> StageReport:
        now = datetime.now(UTC)
        items: list[dict[str, Any]] = []
        if isinstance(self, MemoryConsolidator):
            calls.append((self.remediation_findings, self.prior_s3_findings))
        elif isinstance(self, IntegrityCheck):
            if not integrity_runs:
                items = [build_dangling_citation_finding(RECORD)]
            integrity_runs.append(1)
        return StageReport(
            stage=self.stage_id, started_at=now, completed_at=now, items_flagged=items,
        )

    stub_stage_runs(monkeypatch, fake_run)
    return calls


def _parses_to_record(findings: list[dict] | None) -> list[DanglingCitation | None]:
    return [dangling_citation_from_finding(f) for f in findings or []]


@pytest.mark.asyncio
async def test_remediation_pass_hands_the_dangling_finding_to_stage1(
    cycle_harness, stage_1_calls, caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger=HARNESS_LOGGER):
        await cycle_harness.run_full_cycle(PROJECT, 'test-trigger')

    assert len(stage_1_calls) == 2, 'Stage 1 must run for the cycle, then for remediation'
    remediation_findings, _prior = stage_1_calls[1]
    assert _parses_to_record(remediation_findings) == [RECORD]
    assert not [
        r for r in caplog.records
        if r.name == HARNESS_LOGGER and r.getMessage() == PLACEHOLDER_DROP
    ]


@pytest.mark.asyncio
async def test_next_full_cycle_hands_the_dangling_finding_to_stage1_as_prior_s3(
    cycle_harness, stage_1_calls,
) -> None:
    await cycle_harness.run_full_cycle(PROJECT, 'test-trigger')
    second_cycle_starts_at = len(stage_1_calls)
    await cycle_harness.run_full_cycle(PROJECT, 'test-trigger')

    _remediation, prior_s3_findings = stage_1_calls[second_cycle_starts_at]
    assert RECORD in _parses_to_record(prior_s3_findings)
