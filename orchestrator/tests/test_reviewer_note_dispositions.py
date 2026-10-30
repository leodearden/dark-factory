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
from _workflow_helpers import AgentStub, _build_workflow_with_escalation, _make_git_ops
from escalation.models import Escalation
from escalation.queue import EscalationQueue
from shared.cli_invoke import AgentResult

from orchestrator.agents.review_briefing import ChangedFile, ReviewChange, ReviewSubject
from orchestrator.agents.roles import REVIEWER_COMPREHENSIVE
from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import GitConfig, OrchestratorConfig, TimeoutsConfig, UnblockAutoConfig
from orchestrator.mcp.verdict_tools import _submit_review_verdict
from orchestrator.scheduler import TaskAssignment
from orchestrator.workflow import TaskWorkflow

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
