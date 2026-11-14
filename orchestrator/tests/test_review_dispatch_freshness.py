"""The dispatch-boundary half of the reviews/ freshness guard (task 3158; PRD
plans/task-meta-task-keying-prd.md M4): ``_execute_verify_review_loop`` clears
reviews/ before EXECUTE, so a dispatch never carries a prior dispatch's
reviews, even when it stops before REVIEW. The per-round half lives in
``test_reviewer_verdict_routing.py::TestReviewRoundAggregatesOnlyItsOwnRoster``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from _review_loop_harness import review_loop_workflow
from _review_payloads import blocking_review

from orchestrator.workflow import TaskWorkflow, WorkflowOutcome


def _seed_prior_dispatch_reviews(wf: TaskWorkflow) -> None:
    assert wf.artifacts is not None
    wf.artifacts.write_review('reviewer_comprehensive', blocking_review('reviewer_comprehensive'))
    wf.artifacts.write_review('merge', blocking_review('merge'))


@pytest.mark.asyncio
class TestDispatchCarriesNoPriorReviews:

    async def test_a_dispatch_that_stops_before_review_leaves_no_prior_reviews(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        wf = review_loop_workflow(tmp_path)
        _seed_prior_dispatch_reviews(wf)
        monkeypatch.setattr(
            wf, '_execute_iterations', AsyncMock(return_value=WorkflowOutcome.ESCALATED),
        )

        outcome = await wf._execute_verify_review_loop()

        assert outcome is WorkflowOutcome.ESCALATED
        assert wf.artifacts is not None
        assert wf.artifacts.read_reviews() == {}
        assert wf.artifacts.artifact_path('reviews').is_dir()

    async def test_a_dispatch_that_reaches_done_leaves_no_prior_reviews(
        self, tmp_path: Path,
    ):
        wf = review_loop_workflow(tmp_path)
        _seed_prior_dispatch_reviews(wf)

        outcome = await wf._execute_verify_review_loop()

        assert outcome is WorkflowOutcome.DONE
        assert wf.artifacts is not None
        assert wf.artifacts.read_reviews() == {}
