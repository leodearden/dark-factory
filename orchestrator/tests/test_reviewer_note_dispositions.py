"""The reviewer leg of the info-L0 router, driven through the workflow.

plans/info-l0-disposition-router-prd.md D2 / D11 (task 5224): the reviewer is
briefed with its task's open info notes, and the review loop applies the
dispositions its verdict carries.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from _curator_fake import FakeCurator
from _workflow_helpers import (
    AgentStub,
    _build_workflow_with_escalation,
    _derive_meta_root_like_production,  # noqa: F401  autouse fixture, see its docstring
    _init_repo,
    _make_git_ops,
)
from escalation.models import Escalation
from escalation.queue import EscalationQueue
from shared.cli_invoke import AgentResult
from test_info_l0_router_workflow import _reaper_l1s_after_aging

from orchestrator.agents.review_briefing import ChangedFile, ReviewChange, ReviewSubject
from orchestrator.agents.roles import REVIEWER_COMPREHENSIVE
from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import (
    GitConfig,
    OrchestratorConfig,
    SandboxConfig,
    TimeoutsConfig,
    UnblockAutoConfig,
)
from orchestrator.escalation_router import REVIEWER_LEG_ID, read_hold
from orchestrator.event_store import EventStore, EventType
from orchestrator.mcp.verdict_tools import _envelope, _submit_review_verdict
from orchestrator.scheduler import TaskAssignment
from orchestrator.verify import VerifyResult
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome

TASK_ID = '42'
REVIEWER = REVIEWER_COMPREHENSIVE.name
STARVATION = 'orchestrator-starvation-watchdog'


def _config(project_root: Path, *, router_enabled: bool = True) -> OrchestratorConfig:
    return OrchestratorConfig(
        project_root=project_root,
        max_concurrent_tasks=1,
        steward_completion_timeout=0.25,
        timeouts=TimeoutsConfig(steward=0.25),
        unblock_auto=UnblockAutoConfig(enabled=False),
        info_l0_router_enabled=router_enabled,
        git=GitConfig(
            main_branch='main', branch_prefix='task/', remote='origin',
            worktree_dir='.worktrees',
        ),
    )


def _assignment() -> TaskAssignment:
    return TaskAssignment(
        task_id=TASK_ID,
        task={
            'id': TASK_ID, 'title': 'X', 'description': 'Y', 'status': 'pending',
            'metadata': {'files': ['lib']}, 'dependencies': [],
        },
        modules=['lib'],
    )


def _file_note(
    queue: EscalationQueue, *, task_id: str = TASK_ID, severity: str = 'info',
    agent_role: str = 'implementer',
) -> Escalation:
    note = Escalation(
        id=queue.make_id(task_id), task_id=task_id, agent_role=agent_role,
        severity=severity, category='design_concern',
        summary=f'{agent_role} {severity} note', detail='Seen while editing foo.py.',
        suggested_action='create_followup_task',
    )
    queue.submit(note)
    return note


def _subject() -> ReviewSubject:
    return ReviewSubject(
        task={'id': TASK_ID, 'title': 'X', 'description': 'Y'},
        plan=None,
        plan_absent_reason='no plan in this test',
        change=ReviewChange(
            base_sha='0123456789abcdef0123456789abcdef01234567',
            diff='+x = 1\n',
            files=(ChangedFile('lib/x.py', 1, 0),),
        ),
        conventions=(),
    )


class _Briefed:
    """A workflow wired for ``_run_reviewer`` whose briefing records what it was given."""

    def __init__(self, tmp_path: Path, *, router_enabled: bool = True) -> None:
        config = _config(tmp_path, router_enabled=router_enabled)
        workflow, _scheduler, queue = _build_workflow_with_escalation(
            config, _make_git_ops(config.project_root), _assignment(), AgentStub(), tmp_path,
            spawn_merge_worker=False,
        )
        worktree = tmp_path / 'wt'
        worktree.mkdir()
        artifacts = TaskArtifacts(worktree)
        artifacts.init(TASK_ID, 'X', 'Y')
        workflow.worktree, workflow.artifacts = worktree, artifacts
        briefing = workflow.briefing
        self.diff_only = AsyncMock(side_effect=briefing.build_reviewer_prompt)
        self.context_complete = AsyncMock(
            side_effect=briefing.build_context_complete_reviewer_prompt,
        )
        briefing.build_reviewer_prompt = self.diff_only
        briefing.build_context_complete_reviewer_prompt = self.context_complete
        workflow._invoke = AsyncMock(side_effect=self._invoke_passing_reviewer)
        self.workflow: TaskWorkflow = workflow
        self.queue = queue
        self.artifacts = artifacts

    async def _invoke_passing_reviewer(self, _role, build_prompt, *args, **kwargs) -> AgentResult:
        await build_prompt()
        _submit_review_verdict(self.artifacts, REVIEWER, 'sid', REVIEWER, 'PASS', [], 's')
        return AgentResult(success=True, output='')


def _briefed_ids(build: AsyncMock) -> list[str]:
    assert build.await_args is not None
    return [note.id for note in build.await_args.kwargs['open_info_notes']]


@pytest.mark.asyncio
class TestReviewerIsBriefedWithItsNotes:
    async def test_the_diff_only_reviewer_sees_exactly_the_eligible_notes(self, tmp_path):
        briefed = _Briefed(tmp_path)
        eligible = _file_note(briefed.queue)
        _file_note(briefed.queue, severity='blocking')
        _file_note(briefed.queue, agent_role=STARVATION)
        _file_note(briefed.queue, task_id='99')

        payload = await briefed.workflow._run_reviewer(REVIEWER_COMPREHENSIVE, 'diff')

        assert payload['verdict'] == 'PASS'
        assert _briefed_ids(briefed.diff_only) == [eligible.id]

    async def test_the_context_complete_reviewer_sees_the_same_notes(self, tmp_path):
        briefed = _Briefed(tmp_path)
        eligible = _file_note(briefed.queue)
        _file_note(briefed.queue, agent_role=STARVATION)

        await briefed.workflow._run_reviewer(REVIEWER_COMPREHENSIVE, 'diff', subject=_subject())

        assert _briefed_ids(briefed.context_complete) == [eligible.id]
        briefed.diff_only.assert_not_awaited()

    async def test_a_disabled_router_briefs_no_notes(self, tmp_path):
        briefed = _Briefed(tmp_path, router_enabled=False)
        _file_note(briefed.queue)

        await briefed.workflow._run_reviewer(REVIEWER_COMPREHENSIVE, 'diff')

        assert _briefed_ids(briefed.diff_only) == []

    async def test_a_queue_fault_costs_the_notes_not_the_review(
        self, tmp_path, monkeypatch, caplog,
    ):
        briefed = _Briefed(tmp_path)
        _file_note(briefed.queue)

        def get_by_task(*_args, **_kwargs):
            raise OSError('disk unreadable')

        monkeypatch.setattr(briefed.queue, 'get_by_task', get_by_task)
        with caplog.at_level(logging.WARNING, logger='orchestrator.workflow'):
            payload = await briefed.workflow._run_reviewer(REVIEWER_COMPREHENSIVE, 'diff')

        assert payload['verdict'] == 'PASS'
        assert _briefed_ids(briefed.diff_only) == []
        assert any(
            record.levelno == logging.WARNING and 'disk unreadable' in record.getMessage()
            for record in caplog.records
        )


# ---------------------------------------------------------------------------
# Full runs: the review loop applies the verdict's dispositions
# (boundary rows #1, #2 first half, #17)
# ---------------------------------------------------------------------------

UNKNOWN_ID = 'esc-42-999'


def _run_config(project_root: Path) -> OrchestratorConfig:
    return OrchestratorConfig(
        project_root=project_root,
        max_concurrent_tasks=1,
        max_execute_iterations=5,
        max_verify_attempts=3,
        max_review_cycles=2,
        judge_after_each_iteration=False,
        max_amendment_rounds=1,
        reviewer_stagger_secs=0.0,
        steward_completion_timeout=0.25,
        timeouts=TimeoutsConfig(steward=0.25),
        unblock_auto=UnblockAutoConfig(enabled=False),
        info_l0_router_enabled=True,
        git=GitConfig(
            main_branch='main', branch_prefix='task/', remote='origin',
            worktree_dir='.worktrees',
        ),
        sandbox=SandboxConfig(enabled=False),
    )


def _disposition(esc_id: str, disposition: str) -> dict:
    return {'escalation_id': esc_id, 'disposition': disposition, 'rationale': f'{disposition} rationale'}


class _DispositioningStub(AgentStub):
    """A reviewer whose PASS verdict carries *note_dispositions*."""

    def __init__(self) -> None:
        super().__init__()
        self.note_dispositions: list[dict] = []

    def _reviewer(self, role: str, output_schema: dict | None) -> AgentResult:
        result = super()._reviewer(role, output_schema)
        assert result.structured_output is not None
        result.structured_output['note_dispositions'] = self.note_dispositions
        return result


class _LostVerdictStub(_DispositioningStub):
    """A reviewer that writes an envelope the workflow discards (an off-contract verdict)."""

    async def invoke_agent(self, prompt: str, system_prompt: str, cwd: Path, **kwargs) -> AgentResult:
        role = self._detect_role(system_prompt)
        if not role.startswith('reviewer'):
            return await super().invoke_agent(prompt, system_prompt, cwd, **kwargs)
        self.calls.append(role)
        artifacts = TaskArtifacts(cwd, TaskArtifacts.meta_root_for(cwd.parent, cwd.name))
        artifacts.write_verdict(role, _envelope(role, 'sid', {
            'reviewer': role, 'verdict': 'MAYBE', 'issues': [], 'summary': 's',
            'note_dispositions': self.note_dispositions,
        }))
        return AgentResult(success=True, output='')


class _FullRun:
    def __init__(self, tmp_path: Path, monkeypatch, stub: _DispositioningStub) -> None:
        repo = tmp_path / 'repo'
        repo.mkdir()
        _init_repo(repo)
        config = _run_config(repo)
        self.workflow, _scheduler, self.queue = _build_workflow_with_escalation(
            config, _make_git_ops(config.project_root), _assignment(), stub, tmp_path,
        )
        monkeypatch.setattr('orchestrator.workflow.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(return_value=VerifyResult(
                passed=True, test_output='', lint_output='', type_output='',
                summary='All checks passed',
            )),
        )
        self.workflow.event_store = EventStore(tmp_path / 'runs.db', run_id='run-test')
        self.curator = FakeCurator()
        self.workflow.scheduler.dispatch_tool = self.curator

    async def run(self) -> WorkflowOutcome:
        return (await self.workflow.run()).outcome

    def current(self, note: Escalation) -> Escalation:
        current = self.queue.get(note.id)
        assert current is not None
        return current

    def rows(self, esc_id: str) -> list[dict]:
        store = self.workflow.event_store
        assert store is not None
        return [
            row['data'] for row in store.fetch_events_by_type(EventType.info_l0_dispositioned)
            if row['data']['escalation_id'] == esc_id
        ]


@pytest.mark.asyncio
class TestReviewLoopAppliesDispositions:
    @pytest.mark.parametrize(('disposition', 'resolution_class'), [
        ('addressed', 'addressed'),
        ('no_action', 'observation-consumed'),
    ])
    async def test_a_closing_disposition_closes_the_note_before_the_exit(
        self, tmp_path, monkeypatch, mock_orch_config, disposition, resolution_class,
    ):
        stub = _DispositioningStub()
        run = _FullRun(tmp_path, monkeypatch, stub)
        note = _file_note(run.queue)
        stub.note_dispositions = [_disposition(note.id, disposition)]

        assert await run.run() == WorkflowOutcome.DONE

        closed = run.current(note)
        assert (closed.resolution_class, closed.resolved_by) == (resolution_class, REVIEWER_LEG_ID)
        (row,) = run.rows(note.id)
        assert (row['by'], row['class'], row['exit_kind']) == (
            REVIEWER_LEG_ID, resolution_class, None,
        )
        assert run.curator.calls_named('submit_task') == []
        assert await _reaper_l1s_after_aging(mock_orch_config, run.queue) == []

    async def test_work_leaves_the_note_for_the_exit_router_to_convert(
        self, tmp_path, monkeypatch,
    ):
        stub = _DispositioningStub()
        run = _FullRun(tmp_path, monkeypatch, stub)
        note = _file_note(run.queue)
        stub.note_dispositions = [_disposition(note.id, 'work')]

        assert await run.run() == WorkflowOutcome.DONE

        reviewer_row, finalizer_row = run.rows(note.id)
        assert (reviewer_row['by'], reviewer_row['class'], reviewer_row['reason']) == (
            REVIEWER_LEG_ID, 'defer', 'work',
        )
        assert (finalizer_row['by'], finalizer_row['class'], finalizer_row['exit_kind']) == (
            'workflow-finalizer', 'convert-pending', 'done',
        )
        hold = read_hold(run.current(note))
        assert hold is not None and hold.ticket == 'tkt_1'
        assert len(run.curator.calls_named('submit_task')) == 1

    async def test_a_disposition_of_a_note_the_reviewer_was_not_shown_is_rejected(
        self, tmp_path, monkeypatch,
    ):
        stub = _DispositioningStub()
        run = _FullRun(tmp_path, monkeypatch, stub)
        foreign = _file_note(run.queue, task_id='99')
        stub.note_dispositions = [
            _disposition(UNKNOWN_ID, 'addressed'), _disposition(foreign.id, 'no_action'),
        ]

        assert await run.run() == WorkflowOutcome.DONE

        after = run.current(foreign)
        assert (after.status, after.triaged_by, after.resolution) == ('pending', None, None)
        for esc_id in (UNKNOWN_ID, foreign.id):
            (row,) = run.rows(esc_id)
            assert (row['by'], row['class'], row['task_id']) == (
                REVIEWER_LEG_ID, 'rejected_disposition', TASK_ID,
            )

    async def test_a_lost_verdict_carries_no_dispositions_and_the_exit_router_routes_the_note(
        self, tmp_path, monkeypatch,
    ):
        stub = _LostVerdictStub()
        run = _FullRun(tmp_path, monkeypatch, stub)
        note = _file_note(run.queue)
        stub.note_dispositions = [_disposition(note.id, 'addressed')]

        assert await run.run() == WorkflowOutcome.BLOCKED

        rows = run.rows(note.id)
        assert [row['by'] for row in rows] == ['workflow-finalizer']
        assert (rows[0]['class'], rows[0]['exit_kind']) == ('convert-pending', 'blocked')
        hold = read_hold(run.current(note))
        assert hold is not None and hold.ticket == 'tkt_1'
