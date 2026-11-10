"""TaskWorkflow briefs every fresh dispatch and every reviewer with the task's rulings.

Task 4764.  A ruling is an escalation on the task resolved with text.  Agents
that work from an existing plan see only rulings resolved after the plan was
last confirmed (``plan['_finalized_at']``); agents that author the plan and
both reviewer generations see every ruling.  The full-run cases drive the
public ``TaskWorkflow.run()``; the per-site cases drive each dispatch site the
way ``test_workflow_prompt_builder.py`` does.
"""

from __future__ import annotations

import contextlib
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _curator_fake import FakeCurator
from _workflow_helpers import (
    PLAN,
    AgentStub,
    FakeBriefing,
    _build_workflow_with_escalation,
    _init_repo,
    _make_git_ops,
)
from escalation.models import Escalation
from escalation.queue import EscalationQueue
from test_reviewer_note_dispositions import PASSED, TASK_ID, _assignment, _run_config
from test_workflow_prompt_builder import (
    _PLAN_RENDERING_SITES,
    _TASK_ID,
    _BuiltTwice,
    _invoke_building_twice,
    _make_workflow,
    _write_two_step_plan,
)

from orchestrator.agents.review_briefing import ReviewBriefing, ReviewSubject
from orchestrator.escalation_rulings import Ruling
from orchestrator.event_store import EventStore, EventType
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome

RULING_TEXT = 'KEEP the solver wrap — operator ruling'
CONFIRMED_AFTER_THE_RULING = '2099-01-01T00:00:00+00:00'


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
    ) -> str:
        self._record('build_implementer_prompt', rulings)
        return await super().build_implementer_prompt(plan, context)

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


class _FullRun:
    def __init__(
        self, tmp_path: Path, monkeypatch, *,
        briefing: ReviewBriefing = ReviewBriefing.DIFF_ONLY,
    ) -> None:
        repo = tmp_path / 'repo'
        repo.mkdir()
        _init_repo(repo)
        config = _run_config(repo, briefing=briefing, router_enabled=True)
        stub = AgentStub()
        self.workflow, _scheduler, self.queue = _build_workflow_with_escalation(
            config, _make_git_ops(config.project_root), _assignment(), stub, tmp_path,
        )
        self.briefing = _RulingRecordingBriefing()
        self.workflow.briefing = self.briefing
        self._monkeypatch = monkeypatch
        monkeypatch.setattr('orchestrator.workflow.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification', AsyncMock(return_value=PASSED),
        )
        self.event_store = EventStore(tmp_path / 'runs.db', run_id='run-test')
        self.workflow.event_store = self.event_store
        self.workflow.scheduler.dispatch_tool = FakeCurator()

    async def run(self) -> WorkflowOutcome:
        return (await self.workflow.run()).outcome

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

    async def test_a_queue_fault_costs_only_the_rulings(self, tmp_path, monkeypatch):
        run = _FullRun(tmp_path, monkeypatch)
        _resolve_ruling(run.queue, TASK_ID)
        run.fail_every_resolved_read()

        assert await run.run() == WorkflowOutcome.DONE

        assert run.briefing.briefed
        assert all(ids == [] for _builder, ids in run.briefing.briefed)


Drive = Callable[[TaskWorkflow], Awaitable[object]]


def _replan(wf: TaskWorkflow) -> Awaitable[object]:
    return wf._replan(MagicMock(format_for_replan=MagicMock(return_value='FEEDBACK')))


def _simple_task(wf: TaskWorkflow) -> Awaitable[object]:
    return wf._run_simple_task()


_SITE_DRIVES: dict[str, tuple[str, Callable[[TaskWorkflow], None], Drive]] = {
    name: (site.briefing_method, site.arrange, site.drive)
    for name, site in _PLAN_RENDERING_SITES.items()
    if name in {'completion', 'revalidation', 'debugger', 'amender'}
}
_SITE_DRIVES['replan'] = ('build_replan_prompt', lambda wf: None, _replan)
_SITE_DRIVES['simple_task'] = ('build_simple_task_prompt', lambda wf: None, _simple_task)


async def _briefed_rulings(
    tmp_path: Path, site: str, *, finalized_at: str | None,
) -> tuple[str, list[list[str]]]:
    wf = _make_workflow(tmp_path)
    queue = EscalationQueue(tmp_path / 'escalation_queue')
    ruling = _resolve_ruling(queue, _TASK_ID)
    wf.escalation_queue = queue
    _write_two_step_plan(wf)
    if finalized_at is not None:
        assert wf.artifacts is not None
        plan = wf.artifacts.read_plan()
        plan['_finalized_at'] = finalized_at
        wf.artifacts.write_plan(plan)
        wf.plan = wf.artifacts.read_plan()
    briefing_method, arrange, drive = _SITE_DRIVES[site]
    arrange(wf)
    render = AsyncMock(return_value='PROMPT')
    setattr(wf.briefing, briefing_method, render)
    wf._invoke = _invoke_building_twice(wf, lambda _artifacts: None)  # type: ignore[method-assign]

    with contextlib.suppress(_BuiltTwice):
        await drive(wf)

    assert render.call_args_list
    return ruling, [
        [r.escalation_id for r in call.kwargs['rulings']] for call in render.call_args_list
    ]


@pytest.mark.asyncio
class TestEachDispatchSiteBriefsTheRulings:
    @pytest.mark.parametrize(
        'site', ['completion', 'revalidation', 'debugger', 'amender', 'replan', 'simple_task'],
    )
    async def test_an_unconfirmed_plan_briefs_every_ruling(self, tmp_path, site):
        ruling, briefed = await _briefed_rulings(tmp_path, site, finalized_at=None)

        assert all(ids == [ruling] for ids in briefed)

    @pytest.mark.parametrize('site', ['revalidation', 'debugger', 'amender', 'replan'])
    async def test_a_plan_confirmed_after_the_ruling_briefs_none(self, tmp_path, site):
        _ruling, briefed = await _briefed_rulings(
            tmp_path, site, finalized_at=CONFIRMED_AFTER_THE_RULING,
        )

        assert all(ids == [] for ids in briefed)

    async def test_the_simple_task_is_briefed_with_every_ruling_whatever_the_plan(self, tmp_path):
        ruling, briefed = await _briefed_rulings(
            tmp_path, 'simple_task', finalized_at=CONFIRMED_AFTER_THE_RULING,
        )

        assert all(ids == [ruling] for ids in briefed)
