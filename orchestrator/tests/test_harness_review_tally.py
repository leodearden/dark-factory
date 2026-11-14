"""The harness tallies a cap-skipped review checkpoint as a skip, never as a run.

``HarnessReport`` is driven through its public surface. The two harness review
call sites are invoked directly on a bare ``Harness``: ``Harness.run()`` is
never driven in this suite, and per-tick methods are exercised by calling them
(see test_harness_service_restart.py).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.config import OrchestratorConfig
from orchestrator.harness import Harness, HarnessReport
from orchestrator.review_checkpoint import ReviewCheckpoint, ReviewOutcome, ReviewReport
from orchestrator.verify import VerifyResult


def _review(
    outcome: ReviewOutcome,
    *,
    review_id: str = 'r-1',
    findings_count: int = 0,
    tasks_created: tuple[str, ...] = (),
    cost_usd: float = 0.0,
) -> ReviewReport:
    return ReviewReport(
        review_id=review_id,
        mode='focused',
        changed_modules=[],
        phase1=VerifyResult(
            passed=True, summary='OK', test_output='', lint_output='', type_output='',
        ),
        outcome=outcome,
        briefing_missing=False,
        findings_count=findings_count,
        tasks_created=list(tasks_created),
        cost_usd=cost_usd,
    )


def _review_line(report: HarnessReport) -> str:
    return next(line for line in report.summary().splitlines() if 'Review checkpoints' in line)


class TestHarnessReportRecordsReviewOutcome:
    def test_a_review_that_ran_is_tallied_as_a_checkpoint(self) -> None:
        report = HarnessReport()

        report.record_review(_review(
            ReviewOutcome.RAN, findings_count=3, tasks_created=('t1',), cost_usd=1.5,
        ))

        assert report.review_checkpoints == 1
        assert report.review_findings == 3
        assert report.review_tasks_created == 1
        assert report.review_cost_usd == 1.5
        assert report.review_checkpoints_skipped == 0

    def test_a_skipped_review_is_tallied_as_a_skip_only(self) -> None:
        report = HarnessReport()

        report.record_review(_review(ReviewOutcome.SKIPPED_CAPPED))

        assert report.review_checkpoints == 0
        assert report.review_checkpoints_skipped == 1
        assert report.review_findings == 0
        assert report.review_tasks_created == 0
        assert report.review_cost_usd == 0.0
        assert report.reviews_attempted == 1

    def test_summary_reports_skips_when_no_review_ran(self) -> None:
        report = HarnessReport()
        report.record_review(_review(ReviewOutcome.SKIPPED_CAPPED))

        line = _review_line(report)

        assert re.search(r'\b0\s+ran\b', line)
        assert re.search(r'\b1\s+skipped\b', line)

    def test_to_dict_carries_the_skipped_count_beside_the_existing_keys(self) -> None:
        report = HarnessReport()
        report.record_review(_review(ReviewOutcome.SKIPPED_CAPPED))

        data = report.to_dict()

        assert data['review_checkpoints'] == 0
        assert data['review_checkpoints_skipped'] == 1
        for key in (
            'started_at', 'completed_at', 'total_tasks', 'completed', 'blocked',
            'escalated', 'total_cost_usd', 'paused_for_cap', 'cap_pause_duration_secs',
            'review_findings', 'review_tasks_created', 'review_cost_usd', 'task_reports',
        ):
            assert key in data


def _harness_with_review(tmp_path: Path, review: ReviewReport) -> Harness:
    harness = Harness(OrchestratorConfig(project_root=tmp_path))
    stand_in = MagicMock(spec_set=ReviewCheckpoint)
    stand_in.run_focused = AsyncMock(return_value=review)
    stand_in.run_full = AsyncMock(return_value=review)
    harness.review_checkpoint = cast(ReviewCheckpoint, stand_in)
    return harness


def _assert_skip_warned(caplog: pytest.LogCaptureFixture, review_id: str) -> None:
    assert any(
        record.levelno == logging.WARNING
        and review_id in record.getMessage()
        and ReviewOutcome.SKIPPED_CAPPED.value in record.getMessage()
        for record in caplog.records
    ), [record.getMessage() for record in caplog.records]


@pytest.mark.asyncio
class TestReviewCallSitesRecordOutcome:
    async def test_skipped_focused_checkpoint_is_not_counted_as_a_run(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        harness = _harness_with_review(
            tmp_path, _review(ReviewOutcome.SKIPPED_CAPPED, review_id='r-focused'),
        )
        caplog.set_level(logging.WARNING, logger='orchestrator.harness')

        await harness._run_review_checkpoint()

        assert harness.report.review_checkpoints == 0
        assert harness.report.review_checkpoints_skipped == 1
        _assert_skip_warned(caplog, 'r-focused')

    async def test_skipped_full_review_is_not_counted_as_a_run(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        harness = _harness_with_review(
            tmp_path, _review(ReviewOutcome.SKIPPED_CAPPED, review_id='r-full'),
        )
        caplog.set_level(logging.WARNING, logger='orchestrator.harness')

        await harness._run_full_review_and_tag()

        assert harness.report.review_checkpoints == 0
        assert harness.report.review_checkpoints_skipped == 1
        _assert_skip_warned(caplog, 'r-full')

    async def test_focused_checkpoint_that_ran_is_counted(self, tmp_path: Path) -> None:
        harness = _harness_with_review(tmp_path, _review(ReviewOutcome.RAN))

        await harness._run_review_checkpoint()

        assert harness.report.review_checkpoints == 1
