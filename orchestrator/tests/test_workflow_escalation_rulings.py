"""TaskWorkflow briefs every fresh dispatch and every reviewer with the task's rulings.

Task 4764.  A ruling is an escalation on the task resolved with text.  Agents
that work from an existing plan see only rulings resolved after the plan was
last confirmed (``plan['_finalized_at']``); agents that author or replan the
plan, and both reviewer generations, see every ruling.  The full-run cases
drive the public ``TaskWorkflow.run()``; the per-site cases drive each
remaining dispatch site the way ``test_workflow_prompt_builder.py`` and
``test_plan_revalidation_skip.py`` do.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Generic, TypeVar
from unittest.mock import AsyncMock

import pytest
from _curator_fake import FakeCurator
from _info_l0_router_fixtures import PASSED, TASK_ID, run_config
from _info_l0_router_fixtures import assignment as _assignment
from _workflow_helpers import (
    PLAN,
    AgentStub,
    FakeBriefing,
    _build_workflow_with_escalation,
    _init_repo,
    _make_git_ops,
)
from escalation.models import Escalation
from escalation.note_to_file import file_note_to_file
from escalation.queue import EscalationQueue
from shared.cli_invoke import AgentResult
from test_plan_revalidation_skip import (
    _A,
    _beta_fixture,
    _confirm,
    _done,
    _pending,
    _scripted_architect,
    _wipe_to,
)
from test_workflow_e2e import ReplanningReviewStub
from test_workflow_prompt_builder import _PLAN_RENDERING_SITES, _TASK_ID, _render_calls

from orchestrator.agents.review_briefing import ReviewBriefing, ReviewSubject
from orchestrator.artifacts import TaskArtifacts
from orchestrator.escalation_rulings import Ruling
from orchestrator.event_store import EventStore, EventType
from orchestrator.scheduler import TaskAssignment
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome

RULING_TEXT = 'KEEP the solver wrap — operator ruling'
CONFIRMED_AFTER_THE_RULING = '2099-01-01T00:00:00+00:00'
CONFIRMED_BEFORE_THE_RULING = '2026-01-01T00:00:00+00:00'


def _resolve_ruling(queue: EscalationQueue, task_id: str) -> str:
    ruling = Escalation(
        id=queue.make_id(task_id), task_id=task_id, agent_role='implementer',
        severity='blocking', category='design_concern', level=2,
        summary='keep or drop the solver wrap?',
    )
    queue.submit(ruling)
    queue.resolve(ruling.id, RULING_TEXT, resolved_by='interactive')
    return ruling.id


class _RulingRecordingBriefing(FakeBriefing):
    """Canned prompts that also record which rulings each briefing carried."""

    def __init__(self) -> None:
        self.briefed: list[tuple[str, list[str]]] = []

    def rulings_for(self, builder: str) -> list[list[str]]:
        return [ids for name, ids in self.briefed if name == builder]

    def _record(self, builder: str, rulings: Sequence[Ruling]) -> None:
        self.briefed.append((builder, [r.escalation_id for r in rulings]))

    async def build_architect_prompt(
        self, task: dict, worktree=None, context: str | None = None,
        *, include_prior_proposals: bool = False,
        committed_work: list[dict] | None = None,
        rulings: Sequence[Ruling] = (),
    ) -> str:
        self._record('build_architect_prompt', rulings)
        return await super().build_architect_prompt(task, worktree, context)

    async def build_implementer_prompt(
        self, plan: dict, context: str | None = None,
        rebase_notice: dict | None = None, task_id: str | None = None,
        wip_notice: list[dict] | None = None,
        *, rulings: Sequence[Ruling] = (),
        scope_drift=None,
    ) -> str:
        self._record('build_implementer_prompt', rulings)
        return await super().build_implementer_prompt(plan, context)

    async def build_replan_prompt(
        self, task: dict, review_feedback: str, context: str | None = None,
        *, rulings: Sequence[Ruling] = (),
    ) -> str:
        self._record('build_replan_prompt', rulings)
        return await super().build_replan_prompt(task, review_feedback, context)

    async def build_simple_task_prompt(
        self, task: dict, worktree=None, context: str | None = None,
        *, rulings: Sequence[Ruling] = (),
    ) -> str:
        self._record('build_simple_task_prompt', rulings)
        return await super().build_simple_task_prompt(task, worktree, context)

    async def build_reviewer_prompt(
        self, reviewer_type: str, diff: str, context: str | None = None,
        *, amendment_suggestions: list[dict] | None = None,
        task: dict | None = None,
        open_info_notes: Sequence[Escalation] | None = None,
        rulings: Sequence[Ruling] = (),
    ) -> str:
        self._record('build_reviewer_prompt', rulings)
        return await super().build_reviewer_prompt(reviewer_type, diff, context)

    async def build_context_complete_reviewer_prompt(
        self, reviewer_type: str, subject: ReviewSubject, context: str | None = None,
        *, amendment_suggestions: list[dict] | None = None,
        open_info_notes: Sequence[Escalation] | None = None,
        rulings: Sequence[Ruling] = (),
    ) -> str:
        self._record('build_context_complete_reviewer_prompt', rulings)
        return await super().build_context_complete_reviewer_prompt(reviewer_type, subject, context)


class _FlakyReviewerStub(AgentStub):
    """The first reviewer attempt fails, so its review round retries it."""

    def __init__(self) -> None:
        super().__init__()
        self.reviewer_attempts = 0

    def _reviewer(self, role: str, output_schema: dict | None) -> AgentResult:
        self.reviewer_attempts += 1
        if self.reviewer_attempts == 1:
            return AgentResult(success=False, output='Connection refused', cost_usd=0.0)
        return super()._reviewer(role, output_schema)


class _NoteFilingImplementerStub(AgentStub):
    """The implementer files one note to file on its first turn, as a deviation note."""

    def __init__(self) -> None:
        super().__init__()
        self.queue: EscalationQueue | None = None
        self.notes: list[str] = []

    async def _implementer(self, cwd: Path) -> AgentResult:
        if not self.notes:
            assert self.queue is not None
            filed = file_note_to_file(self.queue, Escalation(
                id=self.queue.make_id(TASK_ID), task_id=TASK_ID, agent_role='implementer',
                severity='info', category='design_concern',
                summary='deviated from step-1: kept the helper private', note_to_file=True,
            ))
            self.notes.append(filed['id'])
        return await super()._implementer(cwd)


def _declared_simple(assignment: TaskAssignment) -> TaskAssignment:
    metadata = {**assignment.task['metadata'], 'complexity': 'simple'}
    return replace(assignment, task={**assignment.task, 'metadata': metadata})


_Recorder = TypeVar('_Recorder', bound=FakeBriefing)


class _WorkflowRun(Generic[_Recorder]):
    """A real ``TaskWorkflow.run()`` of *assignment* in a fresh repo: *stub* answers every agent, *recorder* every briefing."""

    def __init__(
        self, tmp_path: Path, monkeypatch, stub: AgentStub, recorder: _Recorder,
        assignment: TaskAssignment, *, briefing: ReviewBriefing, router_enabled: bool,
    ) -> None:
        repo = tmp_path / 'repo'
        repo.mkdir()
        _init_repo(repo)
        config = run_config(repo, briefing=briefing, router_enabled=router_enabled)
        self.workflow, _scheduler, self.queue = _build_workflow_with_escalation(
            config, _make_git_ops(config.project_root), assignment, stub, tmp_path,
        )
        self.briefing = recorder
        self.workflow.briefing = recorder
        self._monkeypatch = monkeypatch
        monkeypatch.setattr('orchestrator.workflow.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification', AsyncMock(return_value=PASSED),
        )
        self.event_store = EventStore(tmp_path / 'runs.db', run_id='run-test')
        self.workflow.event_store = self.event_store
        self.curator = FakeCurator()
        self.workflow.scheduler.dispatch_tool = self.curator

    async def run(self) -> WorkflowOutcome:
        return (await self.workflow.run()).outcome


class _FullRun(_WorkflowRun[_RulingRecordingBriefing]):
    def __init__(
        self, tmp_path: Path, monkeypatch, *,
        stub: AgentStub | None = None,
        assignment: TaskAssignment | None = None,
        briefing: ReviewBriefing = ReviewBriefing.DIFF_ONLY,
    ) -> None:
        super().__init__(
            tmp_path, monkeypatch, stub or AgentStub(), _RulingRecordingBriefing(),
            assignment or _assignment(), briefing=briefing, router_enabled=True,
        )

    def briefed_rows(self) -> list[dict]:
        return self.event_store.fetch_events_by_type(EventType.escalation_rulings_briefed)

    def start_from_a_plan_confirmed_after_the_ruling(self) -> None:
        self.workflow.initial_plan = {
            **PLAN, 'task_id': TASK_ID, '_finalized_at': CONFIRMED_AFTER_THE_RULING,
        }

    def fail_every_resolved_read(self) -> None:
        read = self.queue.get_by_task

        def get_by_task(*args, **kwargs):
            if kwargs.get('status') == 'resolved':
                raise OSError('disk unreadable')
            return read(*args, **kwargs)

        self._monkeypatch.setattr(self.queue, 'get_by_task', get_by_task)

    def record_resolved_reads(self) -> list[int]:
        """Each read of the resolved records, as the number of briefings built before it."""
        reads: list[int] = []
        read = self.queue.get_by_task

        def get_by_task(*args, **kwargs):
            if kwargs.get('status') == 'resolved':
                reads.append(len(self.briefing.briefed))
            return read(*args, **kwargs)

        self._monkeypatch.setattr(self.queue, 'get_by_task', get_by_task)
        return reads


@pytest.mark.asyncio
class TestAFullRunBriefsTheRulings:
    async def test_a_plan_authored_before_the_ruling_briefs_the_implementer_with_it(
        self, tmp_path, monkeypatch,
    ):
        run = _FullRun(tmp_path, monkeypatch)
        ruling = _resolve_ruling(run.queue, TASK_ID)

        assert await run.run() == WorkflowOutcome.DONE

        assert run.briefing.rulings_for('build_architect_prompt') == [[ruling]]
        implementer = run.briefing.rulings_for('build_implementer_prompt')
        assert implementer
        assert all(ids == [ruling] for ids in implementer)
        rows = [row for row in run.briefed_rows() if row['role'] == 'implementer']
        assert rows
        assert all(row['data']['escalation_ids'] == [ruling] for row in rows)

    async def test_the_diff_only_reviewer_is_briefed_with_every_ruling(
        self, tmp_path, monkeypatch,
    ):
        run = _FullRun(tmp_path, monkeypatch)
        ruling = _resolve_ruling(run.queue, TASK_ID)

        assert await run.run() == WorkflowOutcome.DONE

        reviewer = run.briefing.rulings_for('build_reviewer_prompt')
        assert reviewer
        assert all(ids == [ruling] for ids in reviewer)

    async def test_the_context_complete_reviewer_is_briefed_with_every_ruling(
        self, tmp_path, monkeypatch,
    ):
        run = _FullRun(tmp_path, monkeypatch, briefing=ReviewBriefing.CONTEXT_COMPLETE)
        ruling = _resolve_ruling(run.queue, TASK_ID)

        assert await run.run() == WorkflowOutcome.DONE

        reviewer = run.briefing.rulings_for('build_context_complete_reviewer_prompt')
        assert reviewer
        assert all(ids == [ruling] for ids in reviewer)

    async def test_a_plan_confirmed_after_the_ruling_hides_it_from_the_implementer_only(
        self, tmp_path, monkeypatch,
    ):
        run = _FullRun(tmp_path, monkeypatch)
        ruling = _resolve_ruling(run.queue, TASK_ID)
        run.start_from_a_plan_confirmed_after_the_ruling()

        assert await run.run() == WorkflowOutcome.DONE

        implementer = run.briefing.rulings_for('build_implementer_prompt')
        assert implementer
        assert all(ids == [] for ids in implementer)
        reviewer = run.briefing.rulings_for('build_reviewer_prompt')
        assert reviewer
        assert all(ids == [ruling] for ids in reviewer)

    async def test_dismissed_and_pending_escalations_are_not_rulings(
        self, tmp_path, monkeypatch,
    ):
        run = _FullRun(tmp_path, monkeypatch)
        dismissed = Escalation(
            id=run.queue.make_id(TASK_ID), task_id=TASK_ID, agent_role='implementer',
            severity='blocking', category='design_concern', summary='closed without a ruling',
        )
        run.queue.submit(dismissed)
        run.queue.resolve(dismissed.id, 'router close', dismiss=True)
        run.queue.submit(Escalation(
            id=run.queue.make_id(TASK_ID), task_id=TASK_ID, agent_role='implementer',
            severity='info', category='design_concern', summary='still open',
        ))

        assert await run.run() == WorkflowOutcome.DONE

        assert run.briefing.briefed
        assert all(ids == [] for _builder, ids in run.briefing.briefed)
        assert run.briefed_rows() == []

    async def test_an_implementer_s_note_to_file_is_never_briefed_as_a_ruling(
        self, tmp_path, monkeypatch,
    ):
        stub = _NoteFilingImplementerStub()
        run = _FullRun(tmp_path, monkeypatch, stub=stub)
        stub.queue = run.queue
        ruling = _resolve_ruling(run.queue, TASK_ID)

        assert await run.run() == WorkflowOutcome.DONE

        assert len(stub.notes) == 1
        note = run.queue.get(stub.notes[0])
        assert note is not None and note.status == 'resolved'
        reviewer = run.briefing.rulings_for('build_reviewer_prompt')
        assert reviewer
        assert all(ids == [ruling] for ids in reviewer)
        assert all(note.id not in row['data']['escalation_ids'] for row in run.briefed_rows())

    async def test_a_queue_fault_costs_only_the_rulings(self, tmp_path, monkeypatch):
        run = _FullRun(tmp_path, monkeypatch)
        _resolve_ruling(run.queue, TASK_ID)
        run.fail_every_resolved_read()

        assert await run.run() == WorkflowOutcome.DONE

        assert run.briefing.briefed
        assert all(ids == [] for _builder, ids in run.briefing.briefed)

    async def test_the_replan_is_briefed_with_every_ruling_whenever_the_plan_was_confirmed(
        self, tmp_path, monkeypatch,
    ):
        run = _FullRun(tmp_path, monkeypatch, stub=ReplanningReviewStub())
        ruling = _resolve_ruling(run.queue, TASK_ID)
        run.start_from_a_plan_confirmed_after_the_ruling()

        assert await run.run() == WorkflowOutcome.DONE

        assert run.briefing.rulings_for('build_replan_prompt') == [[ruling]]
        implementer = run.briefing.rulings_for('build_implementer_prompt')
        assert implementer
        assert all(ids == [] for ids in implementer)

    async def test_a_declared_simple_task_is_briefed_with_every_ruling(
        self, tmp_path, monkeypatch,
    ):
        run = _FullRun(tmp_path, monkeypatch, assignment=_declared_simple(_assignment()))
        ruling = _resolve_ruling(run.queue, TASK_ID)

        await run.run()

        assert run.briefing.rulings_for('build_simple_task_prompt') == [[ruling]]

    async def test_a_review_round_reads_the_rulings_once_for_every_reviewer_briefing(
        self, tmp_path, monkeypatch,
    ):
        run = _FullRun(tmp_path, monkeypatch, stub=_FlakyReviewerStub())
        ruling = _resolve_ruling(run.queue, TASK_ID)
        reads = run.record_resolved_reads()

        assert await run.run() == WorkflowOutcome.DONE

        reviewer_builds = [
            i for i, (builder, _ids) in enumerate(run.briefing.briefed)
            if builder == 'build_reviewer_prompt'
        ]
        assert len(reviewer_builds) == 2
        assert run.briefing.rulings_for('build_reviewer_prompt') == [[ruling], [ruling]]
        first, last = reviewer_builds[0], reviewer_builds[-1]
        assert reads.count(first) == 1
        assert not [snapshot for snapshot in reads if first < snapshot <= last]


_SITE_DRIVES = {
    name: (site.briefing_method, site.arrange, site.drive)
    for name, site in _PLAN_RENDERING_SITES.items()
    if name in {'completion', 'revalidation', 'debugger', 'amender'}
}


async def _briefed_rulings(
    tmp_path: Path, site: str, *, finalized_at: str | None,
) -> tuple[str, list[list[str]]]:
    queue = EscalationQueue(tmp_path / 'escalation_queue')
    ruling = _resolve_ruling(queue, _TASK_ID)
    briefing_method, site_arrange, drive = _SITE_DRIVES[site]

    def arrange(wf: TaskWorkflow) -> None:
        wf.escalation_queue = queue
        if finalized_at is not None:
            assert wf.artifacts is not None
            plan = wf.artifacts.read_plan()
            plan['_finalized_at'] = finalized_at
            wf.artifacts.write_plan(plan)
            wf.plan = wf.artifacts.read_plan()
        site_arrange(wf)

    calls = await _render_calls(
        tmp_path, briefing_method=briefing_method, arrange=arrange, drive=drive,
        between_builds=lambda _artifacts: None,
    )

    assert calls
    return ruling, [[r.escalation_id for r in call.kwargs['rulings']] for call in calls]


@pytest.mark.asyncio
class TestEachDispatchSiteBriefsTheRulings:
    @pytest.mark.parametrize('site', ['completion', 'revalidation', 'debugger', 'amender'])
    async def test_an_unconfirmed_plan_briefs_every_ruling(self, tmp_path, site):
        ruling, briefed = await _briefed_rulings(tmp_path, site, finalized_at=None)

        assert all(ids == [ruling] for ids in briefed)

    @pytest.mark.parametrize('site', ['revalidation', 'debugger', 'amender'])
    async def test_a_plan_confirmed_after_the_ruling_briefs_none(self, tmp_path, site):
        _ruling, briefed = await _briefed_rulings(
            tmp_path, site, finalized_at=CONFIRMED_AFTER_THE_RULING,
        )

        assert all(ids == [] for ids in briefed)

    async def test_the_integrity_remedy_pass_keeps_the_revalidated_plan_s_cutoff(
        self, tmp_path, monkeypatch,
    ):
        f = _beta_fixture(tmp_path, [_done('step-1', _A, 'test'), _pending('step-2', 'impl')])
        plan_path = f.artifacts.artifact_path('plan.json')
        plan = json.loads(plan_path.read_text())
        plan_path.write_text(json.dumps({**plan, '_finalized_at': CONFIRMED_BEFORE_THE_RULING}))
        queue = EscalationQueue(tmp_path / 'escalation_queue')
        ruling = _resolve_ruling(queue, f.wf.task_id)
        f.wf.escalation_queue = queue
        reconfirmed: list[str] = []

        def note_reconfirmation(a: TaskArtifacts) -> None:
            reconfirmed.append(a.read_plan()['_finalized_at'])

        wipe = [_wipe_to('step-1', 'step-2'), _confirm]
        _scripted_architect(f, monkeypatch, [[*wipe, note_reconfirmation], wipe])

        await f.wf._plan()

        record = queue.get(ruling)
        assert record is not None and record.resolved_at is not None
        assert datetime.fromisoformat(reconfirmed[0]) > datetime.fromisoformat(record.resolved_at)
        first, remedy = f.build_revalidation_prompt.call_args_list
        assert [r.escalation_id for r in first.kwargs['rulings']] == [ruling]
        assert [r.escalation_id for r in remedy.kwargs['rulings']] == [ruling]
