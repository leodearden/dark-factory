"""A real TaskWorkflow.run() declares review defects and applies reviewer rulings (task 7050, defect-tests δ2).

plans/defect-tests-prd.md B6, and the reviewer half of B8 and B23. The task's
defect records live in FakeScheduler.task_data, folded by
_defect_records_fake.FakeDefectRecords under shared.defects' transition rules.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from _defect_ruling_fixtures import RULED_REF, SECOND_RULED_REF, contract_ruling
from _info_l0_router_fixtures import PASSED, TASK_ID, assignment, run_config
from _replanning_review_stub import ReplanningReviewStub
from _workflow_helpers import (
    PLAN,
    AgentStub,
    FakeBriefing,
    _build_workflow_with_escalation,
    _init_repo,
    _make_git_ops,
    _make_review,
    _make_status_setting_steward,
)
from shared.cli_invoke import AgentResult
from shared.defects import (
    DEFECT_TEST_RETIREMENTS_METADATA_KEY,
    DEFECTS_METADATA_KEY,
    Declare,
    DeclareClosure,
    DeclareRetirement,
    DefectDeclaration,
    DefectOrigin,
    DefectStatus,
    Exemption,
    ExemptionReason,
    IssueKind,
    Retirement,
    RetirementCode,
    RetirementReason,
    RetirementState,
    RuledBy,
    Ruling,
    RulingSubject,
    apply_event,
    apply_retirement_event,
    mint_review_ref,
)
from shared.governed_exceptions import Debt, TicketRef

from orchestrator.agents.review_briefing import ReviewBriefing, ReviewSubject
from orchestrator.agents.roles import REVIEWER_COMPREHENSIVE
from orchestrator.escalation_rulings import Ruling as EscalationRuling
from orchestrator.review_defects import RECORD_DEFECT_OUTCOME_TOOL
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

    def seed(
        self, *, defects: Sequence[DefectDeclaration] = (), retirements: Sequence[Retirement] = (),
    ) -> None:
        """Store *defects* and *retirements* on the task, where the workflow reads them live."""
        metadata = self.scheduler.task_data[TASK_ID]['metadata']
        metadata[DEFECTS_METADATA_KEY] = [entry.model_dump(mode='json') for entry in defects]
        metadata[DEFECT_TEST_RETIREMENTS_METADATA_KEY] = [
            record.model_dump(mode='json') for record in retirements
        ]

    async def metadata(self) -> dict[str, Any]:
        task = await self.scheduler.get_task(TASK_ID)
        assert task is not None
        return task['metadata']

    async def defects(self) -> list[DefectDeclaration]:
        return [
            DefectDeclaration.model_validate(entry)
            for entry in (await self.metadata()).get(DEFECTS_METADATA_KEY, [])
        ]

    async def retirements(self) -> list[Retirement]:
        return [
            Retirement.model_validate(record)
            for record in (await self.metadata()).get(DEFECT_TEST_RETIREMENTS_METADATA_KEY, [])
        ]

    def outcome_tools_called(self) -> list[str]:
        return [name for name, _ in self.scheduler.defect_records.calls if name == RECORD_DEFECT_OUTCOME_TOOL]


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


SEEDED_AT = datetime(2026, 11, 28, 9, 0, tzinfo=UTC)
EXEMPTION = Exemption(
    reason=ExemptionReason.NEEDS_EXTERNAL_SYSTEM, owner=Debt(TicketRef('tkt_0RW782Z6RY3X8W8Q3NX8J1EKD0')),
)


def _open_defect(ref: str) -> DefectDeclaration:
    return apply_event(None, Declare(declaration=DefectDeclaration(
        ref=ref, origin=DefectOrigin.REVIEW, summary='x is wrong', declared_at=SEEDED_AT,
        declared_by=REVIEWER_COMPREHENSIVE.name, status=DefectStatus.OPEN,
    )))


def _exemption_awaiting_ruling(ref: str) -> DefectDeclaration:
    return apply_event(_open_defect(ref), DeclareClosure(ref=ref, closure=EXEMPTION))


def _retirement_awaiting_ruling(ref: str) -> Retirement:
    return apply_retirement_event(None, DeclareRetirement(retirement=Retirement(
        ref=ref, reason=RetirementReason(reason=RetirementCode.FEATURE_REMOVED),
    )))


def _reviewer_ruling(subject: RulingSubject, *, accepted: bool, verdict_cycle: int) -> Ruling:
    return Ruling(subject=subject, accepted=accepted, ruled_by=RuledBy.REVIEWER, verdict_cycle=verdict_cycle)


class _RulingReviewerStub(AgentStub):
    """The comprehensive reviewer's verdict carries *defect_rulings*, beside any blocking *issues*."""

    def __init__(
        self, rulings: Sequence[Mapping[str, object]], *, issues: Sequence[Mapping[str, str]] = (),
    ) -> None:
        super().__init__()
        self._rulings = [dict(ruling) for ruling in rulings]
        self._issues = [dict(issue) for issue in issues]

    def _reviewer(self, role: str, output_schema: dict | None) -> AgentResult:
        verdict = 'ISSUES_FOUND' if self._issues else 'PASS'
        review = _make_review(role, verdict, [dict(issue) for issue in self._issues])
        review['defect_rulings'] = [dict(ruling) for ruling in self._rulings]
        return AgentResult(
            success=True, output=json.dumps(review), structured_output=review, cost_usd=0.10,
        )


@pytest.mark.asyncio
class TestReviewerRulingsAreApplied:
    async def test_an_accepted_exemption_is_exempted_by_the_reviewer_in_cycle_one(self, tmp_path, monkeypatch):
        run = _DefectRun(tmp_path, monkeypatch, _RulingReviewerStub([contract_ruling()]), max_review_cycles=2)
        run.seed(defects=[_exemption_awaiting_ruling(RULED_REF)])

        assert await run.run() == WorkflowOutcome.DONE

        (entry,) = await run.defects()
        assert entry.status is DefectStatus.EXEMPTED
        assert entry.ruling == _reviewer_ruling(RulingSubject.CLOSURE, accepted=True, verdict_cycle=1)

    async def test_a_rejected_exemption_reopens_the_defect(self, tmp_path, monkeypatch):
        stub = _RulingReviewerStub([contract_ruling(accepted=False)])
        run = _DefectRun(tmp_path, monkeypatch, stub, max_review_cycles=2)
        run.seed(defects=[_exemption_awaiting_ruling(RULED_REF)])

        assert await run.run() == WorkflowOutcome.DONE

        (entry,) = await run.defects()
        assert entry.status is DefectStatus.OPEN
        assert entry.ruling == _reviewer_ruling(RulingSubject.CLOSURE, accepted=False, verdict_cycle=1)

    async def test_an_accepted_retirement_rules_its_record_and_leaves_the_defects_alone(
        self, tmp_path, monkeypatch,
    ):
        ruling = contract_ruling(ref=SECOND_RULED_REF, subject=RulingSubject.RETIREMENT.value)
        run = _DefectRun(tmp_path, monkeypatch, _RulingReviewerStub([ruling]), max_review_cycles=2)
        run.seed(defects=[_open_defect(RULED_REF)], retirements=[_retirement_awaiting_ruling(SECOND_RULED_REF)])
        seeded_defects = copy.deepcopy((await run.metadata())[DEFECTS_METADATA_KEY])

        assert await run.run() == WorkflowOutcome.DONE

        (record,) = await run.retirements()
        assert record.state is RetirementState.ACCEPTED
        assert record.ruling == _reviewer_ruling(RulingSubject.RETIREMENT, accepted=True, verdict_cycle=1)
        assert (await run.metadata())[DEFECTS_METADATA_KEY] == seeded_defects

    async def test_a_ruling_on_the_blocking_round_is_applied_before_the_cap_exit(self, tmp_path, monkeypatch):
        stub = _RulingReviewerStub([contract_ruling()], issues=[_blocking_issue(IssueKind.TEST_QUALITY)])
        run = _DefectRun(tmp_path, monkeypatch, stub, max_review_cycles=1)
        run.seed(defects=[_exemption_awaiting_ruling(RULED_REF)])

        await run.run_to_the_review_cap()

        (entry,) = await run.defects()
        assert entry.status is DefectStatus.EXEMPTED
        assert entry.ruling == _reviewer_ruling(RulingSubject.CLOSURE, accepted=True, verdict_cycle=1)

    async def test_a_panel_with_an_errored_reviewer_rules_nothing(self, tmp_path, monkeypatch):
        """The silent reviewer submits no verdict of its own, so the panel is incomplete.

        The stub answers every reviewer session as reviewer_comprehensive, so
        only that role's verdict is ever submitted.
        """
        silent = replace(REVIEWER_COMPREHENSIVE, name='reviewer_silent')
        monkeypatch.setattr(
            'orchestrator.workflow.reviewers_for', lambda briefing: [REVIEWER_COMPREHENSIVE, silent],
        )
        run = _DefectRun(tmp_path, monkeypatch, _RulingReviewerStub([contract_ruling()]), max_review_cycles=2)
        run.workflow.config.max_reviewer_retries = 0
        run.seed(defects=[_exemption_awaiting_ruling(RULED_REF)])

        assert await run.run() == WorkflowOutcome.BLOCKED

        (entry,) = await run.defects()
        assert entry.status is DefectStatus.PENDING
        assert run.outcome_tools_called() == []


class _TaskRecordingBriefing(FakeBriefing):
    """Records a copy of the task every reviewer is briefed with, then answers as FakeBriefing does.

    A copy, because the workflow later replaces its own task's metadata in place.
    """

    def __init__(self) -> None:
        self.briefed_tasks: list[Mapping[str, object] | None] = []

    async def build_reviewer_prompt(
        self, reviewer_type: str, diff: str, context: str | None = None,
        *, amendment_suggestions: list[dict] | None = None,
        task: dict | None = None,
        open_info_notes: Sequence | None = None,
        rulings: Sequence[EscalationRuling] = (),
    ) -> str:
        self.briefed_tasks.append(copy.deepcopy(task))
        return await super().build_reviewer_prompt(
            reviewer_type, diff, context, amendment_suggestions=amendment_suggestions,
            task=task, open_info_notes=open_info_notes, rulings=rulings,
        )

    async def build_context_complete_reviewer_prompt(
        self, reviewer_type: str, subject: ReviewSubject, context: str | None = None,
        *, amendment_suggestions: list[dict] | None = None,
        open_info_notes: Sequence | None = None,
        rulings: Sequence[EscalationRuling] = (),
    ) -> str:
        self.briefed_tasks.append(copy.deepcopy(dict(subject.task)))
        return await super().build_context_complete_reviewer_prompt(
            reviewer_type, subject, context, amendment_suggestions=amendment_suggestions,
            open_info_notes=open_info_notes, rulings=rulings,
        )


def _briefed_defects(task: Mapping[str, object] | None) -> list[DefectDeclaration]:
    assert task is not None
    metadata = task['metadata']
    assert isinstance(metadata, Mapping)
    return [DefectDeclaration.model_validate(entry) for entry in metadata.get(DEFECTS_METADATA_KEY, [])]


@pytest.mark.asyncio
class TestReviewerIsBriefedFromLiveRecords:
    """The store holds a closure awaiting a ruling that the dispatch task never carried."""

    def _run(self, tmp_path, monkeypatch, briefing: ReviewBriefing) -> tuple[_DefectRun, _TaskRecordingBriefing]:
        run = _DefectRun(tmp_path, monkeypatch, AgentStub(), max_review_cycles=2, briefing=briefing)
        recorder = _TaskRecordingBriefing()
        run.workflow.briefing = recorder
        run.seed(defects=[_exemption_awaiting_ruling(RULED_REF)])
        return run, recorder

    @pytest.mark.parametrize('briefing', [ReviewBriefing.DIFF_ONLY, ReviewBriefing.CONTEXT_COMPLETE])
    async def test_every_reviewer_sees_the_closure_beside_the_dispatch_fields(
        self, tmp_path, monkeypatch, briefing,
    ):
        run, recorder = self._run(tmp_path, monkeypatch, briefing)

        assert await run.run() == WorkflowOutcome.DONE

        assert recorder.briefed_tasks
        dispatched = _dispatched().task
        for task in recorder.briefed_tasks:
            assert task is not None
            assert task['title'] == dispatched['title']
            metadata = task['metadata']
            assert isinstance(metadata, Mapping)
            assert metadata['files'] == dispatched['metadata']['files']
            (entry,) = _briefed_defects(task)
            assert entry.ref == RULED_REF
            assert entry.status is DefectStatus.PENDING

    async def test_an_unreadable_store_briefs_the_dispatch_task(self, tmp_path, monkeypatch):
        run, recorder = self._run(tmp_path, monkeypatch, ReviewBriefing.DIFF_ONLY)
        outage = _StoreDownUntilBriefed(run, recorder)
        monkeypatch.setattr(run.scheduler, 'get_task', outage.get_task)
        monkeypatch.setattr('orchestrator.workflow.run_scoped_verification', outage.verify)

        assert await run.run() == WorkflowOutcome.DONE

        assert outage.refused_reads >= 1
        assert recorder.briefed_tasks
        dispatched = _dispatched().task
        for task in recorder.briefed_tasks:
            assert task is not None
            assert task['title'] == dispatched['title']
            assert _briefed_defects(task) == []


class _StoreDownUntilBriefed:
    """The task store refuses every read from the moment verification passes until a reviewer is briefed.

    The merge phase reads the task too and needs the store back, so the
    outage covers the review round's own read and nothing after it.
    """

    def __init__(self, run: _DefectRun, recorder: _TaskRecordingBriefing) -> None:
        self._read = run.scheduler.get_task
        self._recorder = recorder
        self._verified = False
        self.refused_reads = 0

    async def verify(self, *_args, **_kwargs):
        self._verified = True
        return PASSED

    async def get_task(self, task_id: str) -> dict | None:
        if self._verified and not self._recorder.briefed_tasks:
            self.refused_reads += 1
            raise OSError('store unreachable')
        return await self._read(task_id)
