"""The reviewer leg of the info-L0 router, driven through the workflow.

plans/info-l0-disposition-router-prd.md D2 / D11 (task 5224): the reviewer is
briefed with its task's open info notes, and the review loop applies the
dispositions its verdict carries.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from _curator_fake import FakeCurator
from _workflow_helpers import (
    AgentStub,
    FakeBriefing,
    _build_workflow_with_escalation,
    _init_repo,
    _make_git_ops,
    agent_side_artifacts,
)
from _workflow_helpers import (
    _derive_meta_root_like_production as _derive_meta_root_like_production,
)
from escalation.models import Escalation
from escalation.queue import EscalationQueue
from shared.cli_invoke import AgentResult

from orchestrator.agents.review_briefing import ReviewBriefing, ReviewSubject
from orchestrator.config import (
    GitConfig,
    OrchestratorConfig,
    ReviewConfig,
    SandboxConfig,
    TimeoutsConfig,
    UnblockAutoConfig,
)
from orchestrator.escalation_router import REVIEWER_LEG_ID, read_hold
from orchestrator.escalation_rulings import Ruling
from orchestrator.event_store import EventStore, EventType
from orchestrator.mcp.verdict_tools import _envelope
from orchestrator.scheduler import TaskAssignment
from orchestrator.verify import VerifyResult
from orchestrator.workflow import WorkflowOutcome

TASK_ID = '42'
STARVATION = 'orchestrator-starvation-watchdog'
UNKNOWN_ID = 'esc-42-999'
CORRUPTION = {
    'agent_role': 'orchestrator', 'category': 'infra_issue',
    'suggested_action': 'investigate_log_corruption',
}
PASSED = VerifyResult(
    passed=True, test_output='', lint_output='', type_output='', summary='All checks passed',
)


def _run_config(
    project_root: Path, *, briefing: ReviewBriefing, router_enabled: bool,
) -> OrchestratorConfig:
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
        info_l0_router_enabled=router_enabled,
        review=ReviewConfig(reviewer_briefing=briefing),
        git=GitConfig(
            main_branch='main', branch_prefix='task/', remote='origin',
            worktree_dir='.worktrees',
        ),
        sandbox=SandboxConfig(enabled=False),
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
    agent_role: str = 'implementer', category: str = 'design_concern',
    suggested_action: str = 'create_followup_task',
) -> Escalation:
    note = Escalation(
        id=queue.make_id(task_id), task_id=task_id, agent_role=agent_role,
        severity=severity, category=category,
        summary=f'{agent_role} {severity} note', detail='Seen while editing foo.py.',
        suggested_action=suggested_action,
    )
    queue.submit(note)
    return note


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
    """A reviewer whose verdict file is present but off-contract, so the workflow discards it."""

    async def invoke_agent(self, prompt: str, system_prompt: str, cwd: Path, **kwargs) -> AgentResult:
        role = self._detect_role(system_prompt)
        if not role.startswith('reviewer'):
            return await super().invoke_agent(prompt, system_prompt, cwd, **kwargs)
        self.calls.append(role)
        artifacts = agent_side_artifacts(cwd)
        artifacts.write_verdict(role, _envelope(role, 'sid', {
            'reviewer': role, 'verdict': 'MAYBE', 'issues': [], 'summary': 's',
            'note_dispositions': self.note_dispositions,
        }))
        return AgentResult(success=True, output='')


class _NoteRecordingBriefing(FakeBriefing):
    """Canned prompts that also record which notes each reviewer briefing listed."""

    def __init__(self) -> None:
        self.briefed: list[tuple[ReviewBriefing, list[str]]] = []

    async def build_reviewer_prompt(
        self, reviewer_type: str, diff: str, context: str | None = None,
        *, amendment_suggestions: list[dict] | None = None,
        task: dict | None = None,
        open_info_notes: Sequence[Escalation] | None = None,
        rulings: Sequence[Ruling] = (),
    ) -> str:
        self._record(ReviewBriefing.DIFF_ONLY, open_info_notes)
        return await super().build_reviewer_prompt(reviewer_type, diff, context)

    async def build_context_complete_reviewer_prompt(
        self, reviewer_type: str, subject: ReviewSubject, context: str | None = None,
        *, amendment_suggestions: list[dict] | None = None,
        open_info_notes: Sequence[Escalation] | None = None,
        rulings: Sequence[Ruling] = (),
    ) -> str:
        self._record(ReviewBriefing.CONTEXT_COMPLETE, open_info_notes)
        return await super().build_context_complete_reviewer_prompt(reviewer_type, subject, context)

    def _record(self, generation: ReviewBriefing, notes: Sequence[Escalation] | None) -> None:
        self.briefed.append((generation, [note.id for note in notes or ()]))


class _FullRun:
    def __init__(
        self, tmp_path: Path, monkeypatch, stub: _DispositioningStub, *,
        briefing: ReviewBriefing = ReviewBriefing.DIFF_ONLY, router_enabled: bool = True,
    ) -> None:
        repo = tmp_path / 'repo'
        repo.mkdir()
        _init_repo(repo)
        config = _run_config(repo, briefing=briefing, router_enabled=router_enabled)
        self.workflow, _scheduler, self.queue = _build_workflow_with_escalation(
            config, _make_git_ops(config.project_root), _assignment(), stub, tmp_path,
        )
        self.briefing = _NoteRecordingBriefing()
        self.workflow.briefing = self.briefing
        self._monkeypatch = monkeypatch
        monkeypatch.setattr('orchestrator.workflow.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification', AsyncMock(return_value=PASSED),
        )
        self.workflow.event_store = EventStore(tmp_path / 'runs.db', run_id='run-test')
        self.curator = FakeCurator()
        self.workflow.scheduler.dispatch_tool = self.curator

    def fail_the_first_queue_read_after_verification(self) -> None:
        """The reviewer's note read is the first queue read once verification passes."""
        armed = False
        read = self.queue.get_by_task

        def get_by_task(*args, **kwargs):
            nonlocal armed
            if armed:
                armed = False
                raise OSError('disk unreadable')
            return read(*args, **kwargs)

        async def verify(*_args, **_kwargs) -> VerifyResult:
            nonlocal armed
            armed = True
            return PASSED

        self._monkeypatch.setattr(self.queue, 'get_by_task', get_by_task)
        self._monkeypatch.setattr('orchestrator.workflow.run_scoped_verification', verify)

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
class TestReviewerIsBriefedWithItsNotes:
    async def test_the_diff_only_reviewer_sees_exactly_the_eligible_notes(
        self, tmp_path, monkeypatch,
    ):
        run = _FullRun(tmp_path, monkeypatch, _DispositioningStub())
        eligible = _file_note(run.queue)
        _file_note(run.queue, severity='blocking')
        _file_note(run.queue, agent_role=STARVATION)
        _file_note(run.queue, task_id='99')
        _file_note(run.queue, **CORRUPTION)

        assert await run.run() == WorkflowOutcome.DONE

        assert run.briefing.briefed == [(ReviewBriefing.DIFF_ONLY, [eligible.id])]

    async def test_the_context_complete_reviewer_sees_the_same_notes(
        self, tmp_path, monkeypatch,
    ):
        run = _FullRun(
            tmp_path, monkeypatch, _DispositioningStub(),
            briefing=ReviewBriefing.CONTEXT_COMPLETE,
        )
        eligible = _file_note(run.queue)
        _file_note(run.queue, agent_role=STARVATION)

        assert await run.run() == WorkflowOutcome.DONE

        assert run.briefing.briefed == [(ReviewBriefing.CONTEXT_COMPLETE, [eligible.id])]

    async def test_a_disabled_router_briefs_no_notes(self, tmp_path, monkeypatch):
        run = _FullRun(tmp_path, monkeypatch, _DispositioningStub(), router_enabled=False)
        _file_note(run.queue)

        assert await run.run() == WorkflowOutcome.DONE

        assert run.briefing.briefed == [(ReviewBriefing.DIFF_ONLY, [])]

    async def test_a_queue_fault_costs_the_notes_not_the_review(
        self, tmp_path, monkeypatch, caplog,
    ):
        run = _FullRun(tmp_path, monkeypatch, _DispositioningStub())
        _file_note(run.queue)
        run.fail_the_first_queue_read_after_verification()

        with caplog.at_level(logging.WARNING, logger='orchestrator.workflow'):
            assert await run.run() == WorkflowOutcome.DONE

        assert run.briefing.briefed == [(ReviewBriefing.DIFF_ONLY, [])]
        assert any(
            record.levelno == logging.WARNING and 'disk unreadable' in record.getMessage()
            for record in caplog.records
        )


# ---------------------------------------------------------------------------
# The review loop applies the verdict's dispositions
# (boundary rows #1, #2 first half, #17)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReviewLoopAppliesDispositions:
    @pytest.mark.parametrize(('disposition', 'resolution_class'), [
        ('addressed', 'addressed'),
        ('no_action', 'observation-consumed'),
    ])
    async def test_a_closing_disposition_closes_the_note_before_the_exit(
        self, tmp_path, monkeypatch, disposition, resolution_class,
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
        (submitted,) = run.curator.calls_named('submit_task')
        assert 'work rationale' in submitted['description']

    async def test_work_overrides_the_authors_no_action_declaration_at_the_exit(
        self, tmp_path, monkeypatch,
    ):
        stub = _DispositioningStub()
        run = _FullRun(tmp_path, monkeypatch, stub)
        note = _file_note(run.queue, suggested_action='no action needed for this task')
        stub.note_dispositions = [_disposition(note.id, 'work')]

        assert await run.run() == WorkflowOutcome.DONE

        _reviewer_row, finalizer_row = run.rows(note.id)
        assert (finalizer_row['class'], finalizer_row['exit_kind']) == ('convert-pending', 'done')
        assert run.current(note).status == 'pending'
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

        artifacts = run.workflow.artifacts
        assert artifacts is not None
        ((role, review),) = artifacts.read_reviews().items()
        assert review['verdict'] == 'ERROR'
        envelope = artifacts.read_verdict(role)
        assert envelope is not None
        assert envelope['verdict']['verdict'] == 'MAYBE'
        assert envelope['verdict']['note_dispositions'] == stub.note_dispositions

        rows = run.rows(note.id)
        assert [row['by'] for row in rows] == ['workflow-finalizer']
        assert (rows[0]['class'], rows[0]['exit_kind']) == ('convert-pending', 'blocked')
        hold = read_hold(run.current(note))
        assert hold is not None and hold.ticket == 'tkt_1'

    async def test_a_harness_notice_is_left_to_the_exit_router(self, tmp_path, monkeypatch):
        stub = _DispositioningStub()
        run = _FullRun(tmp_path, monkeypatch, stub)
        notice = _file_note(run.queue, **CORRUPTION)
        stub.note_dispositions = [_disposition(notice.id, 'no_action')]

        assert await run.run() == WorkflowOutcome.DONE

        reviewer_row, finalizer_row = run.rows(notice.id)
        assert (reviewer_row['by'], reviewer_row['class']) == (REVIEWER_LEG_ID, 'rejected_disposition')
        assert run.current(notice).resolution_class != 'observation-consumed'
        assert (finalizer_row['by'], finalizer_row['exit_kind']) == ('workflow-finalizer', 'done')
        hold = read_hold(run.current(notice))
        assert hold is not None and hold.ticket == 'tkt_1'
