"""A real TaskWorkflow.run() declares review defects and applies reviewer rulings (task 7050, defect-tests δ2).

plans/defect-tests-prd.md B6, and the reviewer half of B8 and B23. The task's
defect records live in FakeScheduler.task_data, folded by
_defect_records_fake.FakeDefectRecords under shared.defects' transition rules.
"""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from _info_l0_router_fixtures import PASSED, TASK_ID, assignment, run_config
from _replanning_review_stub import ReplanningReviewStub
from _workflow_helpers import (
    PLAN,
    AgentStub,
    _build_workflow_with_escalation,
    _init_repo,
    _make_git_ops,
    _make_review,
    _make_status_setting_steward,
)
from shared.cli_invoke import AgentResult
from shared.defects import (
    DEFECTS_METADATA_KEY,
    DefectDeclaration,
    DefectOrigin,
    DefectStatus,
    IssueKind,
    mint_review_ref,
)

from orchestrator.agents.review_briefing import ReviewBriefing
from orchestrator.scheduler import TaskAssignment
from orchestrator.workflow import WorkflowOutcome

TASK_NUMBER = int(TASK_ID)
LOCATION = 'lib.py:4'
DESCRIPTION = 'farewell drops the trailing punctuation'
DEFECT_REF = mint_review_ref(TASK_NUMBER, LOCATION, DESCRIPTION)


def _blocking_issue(kind: IssueKind) -> dict[str, str]:
    return {
        'severity': 'blocking',
        'kind': kind.value,
        'location': LOCATION,
        'category': 'correctness',
        'description': DESCRIPTION,
        'suggested_fix': 'Keep the punctuation',
    }


class _AlwaysBlockingStub(AgentStub):
    """The comprehensive reviewer reports the same blocking issue every round."""

    def __init__(self, issue: dict[str, str]) -> None:
        super().__init__()
        self._issue = issue

    def _reviewer(self, role: str, output_schema: dict | None) -> AgentResult:
        review = _make_review(role, 'ISSUES_FOUND', [dict(self._issue)])
        return AgentResult(
            success=True, output=json.dumps(review), structured_output=review, cost_usd=0.10,
        )


def _dispatched() -> TaskAssignment:
    """The router fixtures' task, declaring the files PLAN changes so the merge-entry scope check passes."""
    task = assignment()
    files = list(PLAN['files'])
    return replace(
        task, task={**task.task, 'metadata': {**task.task['metadata'], 'files': files}}, modules=files,
    )


def _deferring_steward(run: _DefectRun) -> type:
    """A steward that settles the cap exit's escalation by deferring the task, a terminal decision that ends the run.

    Without one the ESCALATED wait times out, dismisses the escalation and
    resumes the loop, which reaches the spent cap again on every pass.
    """

    class _DeferringSteward:
        def __init__(self, wt_path: Path, cfg_dir: Path | None) -> None:
            steward = _make_status_setting_steward(run.queue, run.scheduler, TASK_ID, 'deferred')
            self._delegate = steward(wt_path, cfg_dir)

        def set_outcome_channel(self, channel) -> None:
            self._delegate.set_outcome_channel(channel)

        def set_wip_probe(self, probe) -> None:
            self._delegate.set_wip_probe(probe)

        async def start(self) -> None:
            await self._delegate.start()

        async def stop(self) -> None:
            await self._delegate.stop()

    return _DeferringSteward


class _DefectRun:
    """One TaskWorkflow.run() whose task is seeded into the FakeScheduler's store."""

    def __init__(
        self, tmp_path: Path, monkeypatch, stub: AgentStub, *,
        max_review_cycles: int, briefing: ReviewBriefing = ReviewBriefing.DIFF_ONLY,
    ) -> None:
        repo = tmp_path / 'repo'
        repo.mkdir()
        _init_repo(repo)
        config = run_config(repo, briefing=briefing, router_enabled=False)
        dispatched = _dispatched()
        self.workflow, self.scheduler, self.queue = _build_workflow_with_escalation(
            config, _make_git_ops(config.project_root), dispatched, stub, tmp_path,
            steward_factory=_deferring_steward(self),
        )
        self.workflow.config.max_review_cycles = max_review_cycles
        self.scheduler.task_data[TASK_ID] = copy.deepcopy(dispatched.task)
        monkeypatch.setattr('orchestrator.workflow.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification', AsyncMock(return_value=PASSED),
        )

    async def run(self) -> WorkflowOutcome:
        return (await self.workflow.run()).outcome

    async def run_to_the_review_cap(self) -> None:
        """Run until the cap exit's review_issues escalation, which the deferring steward settles."""
        assert await self.run() == WorkflowOutcome.BLOCKED
        categories = [esc.category for esc in self.queue.get_by_task(TASK_ID)]
        assert 'review_issues' in categories

    async def defects(self) -> list[DefectDeclaration]:
        task = await self.scheduler.get_task(TASK_ID)
        assert task is not None
        return [
            DefectDeclaration.model_validate(entry)
            for entry in task['metadata'].get(DEFECTS_METADATA_KEY, [])
        ]


@pytest.mark.asyncio
class TestReviewDefectsAreDeclared:
    async def test_a_final_cycle_defect_is_declared_before_the_cap_exit(self, tmp_path, monkeypatch):
        run = _DefectRun(
            tmp_path, monkeypatch, _AlwaysBlockingStub(_blocking_issue(IssueKind.DEFECT)),
            max_review_cycles=1,
        )

        await run.run_to_the_review_cap()

        (entry,) = await run.defects()
        assert entry.ref == DEFECT_REF
        assert entry.origin is DefectOrigin.REVIEW
        assert entry.status is DefectStatus.OPEN
        assert entry.sightings == 1

    async def test_a_defect_reported_again_next_cycle_is_a_second_sighting(self, tmp_path, monkeypatch):
        run = _DefectRun(
            tmp_path, monkeypatch, _AlwaysBlockingStub(_blocking_issue(IssueKind.DEFECT)),
            max_review_cycles=2,
        )

        await run.run_to_the_review_cap()

        (entry,) = await run.defects()
        assert entry.ref == DEFECT_REF
        assert entry.sightings == 2
        assert entry.status is DefectStatus.OPEN

    async def test_a_later_review_omitting_the_defect_leaves_it_open(self, tmp_path, monkeypatch):
        stub = ReplanningReviewStub(blocking_issue=_blocking_issue(IssueKind.DEFECT))
        run = _DefectRun(tmp_path, monkeypatch, stub, max_review_cycles=2)

        assert await run.run() == WorkflowOutcome.DONE

        (entry,) = await run.defects()
        assert entry.ref == DEFECT_REF
        assert entry.status is DefectStatus.OPEN

    async def test_a_blocking_issue_of_another_kind_declares_nothing(self, tmp_path, monkeypatch):
        run = _DefectRun(
            tmp_path, monkeypatch, _AlwaysBlockingStub(_blocking_issue(IssueKind.TEST_QUALITY)),
            max_review_cycles=1,
        )

        await run.run_to_the_review_cap()

        assert await run.defects() == []
