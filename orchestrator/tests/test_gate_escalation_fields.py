"""The one gate-fields builder both born-at-L2 pure-gate producers share.

PRD plans/completion-driven-triggers-prd.md C4: a ``trigger_chain`` human
gate files ``review_due`` with a typed payload; every other pure gate files
``milestone_gate`` exactly as before.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from escalation.queue import EscalationQueue
from shared.task_read import TaskAbsent, TaskFound, TaskRead, TaskReadFailed
from test_deterministic_runner import _make_assignment

from orchestrator.deterministic_runner import (
    MILESTONE_GATE_CATEGORY,
    OPERATIONAL_LLM_GATE_MARKER_KEY,
    OPERATIONAL_LLM_NEEDS_LANE_TOKEN,
    REVIEW_DUE_CATEGORY,
    DeterministicRunner,
    GateEscalationFields,
    HumanGateFacts,
    build_gate_escalation_fields,
)
from orchestrator.review_due import CompletionGateUnread
from orchestrator.workflow import WorkflowOutcome

RUN_ID = 'review-dark_factory-20261004'
PROJECT_ID = 'dark_factory'
PROJECT_ROOT = '/home/leo/src/dark-factory'
ESC_ID = 'esc-42-1'
FIRED_AT = '2026-10-05T00:00:00+00:00'
VERDICT = {'verdict': 'pass', 'share': 0.75, 'by_priority': {'high': [3, 1, 0]}}


def _plain_gate(task_id: str = '42', deps: list | None = None) -> dict:
    return {
        'id': task_id,
        'title': 'Ship feature gate',
        'description': 'Leo signs off on the feature.',
        'dependencies': [7, 8] if deps is None else deps,
        'metadata': {
            'task_kind': 'deterministic',
            'always_escalates': True,
            'before_done': None,
            'gate_options': ['A: ship', 'B: hold'],
        },
    }


def _human_gate_task(
    task_id: str = '42',
    deps: list | None = None,
    skill: str = 'review',
    run_id: str = RUN_ID,
    role: str = 'human_gate',
) -> dict:
    task = _plain_gate(task_id, ['41'] if deps is None else deps)
    task['title'] = f'Run /review on {PROJECT_ID}'
    task['metadata']['trigger_chain'] = {
        'role': role, 'skill': skill, 'run_id': run_id, 'superseded_by': None,
    }
    return task


def _completion_gate(task_id: str = '41', verdict: object = None) -> dict:
    return {
        'id': task_id,
        'title': 'Run review-dark_factory-20261004 is complete',
        'status': 'done',
        'metadata': {
            'task_kind': 'deterministic',
            'trigger_chain': {
                'role': 'completion_gate', 'skill': 'review', 'run_id': RUN_ID,
                'superseded_by': None,
            },
            'predicate_verdict': dict(VERDICT) if verdict is None else verdict,
        },
    }


def _facts(completion_gate: dict | CompletionGateUnread | None) -> HumanGateFacts:
    return HumanGateFacts(
        project_id=PROJECT_ID, project_root=PROJECT_ROOT, completion_gate=completion_gate,
    )


def _build(task: dict, human_gate: HumanGateFacts | None) -> GateEscalationFields:
    return build_gate_escalation_fields(
        task, task['metadata'], human_gate, escalation_id=ESC_ID, fired_at=FIRED_AT,
    )


def _plain_detail(task: dict) -> str:
    deps = ', '.join(str(d) for d in task['dependencies'])
    return f"{task['description']}\n\nLanded dependencies: {deps}"


class TestPlainPureGate:
    def test_files_milestone_gate_with_todays_fields(self):
        task = _plain_gate()
        fields = _build(task, None)
        assert isinstance(fields, GateEscalationFields)
        assert fields.category == MILESTONE_GATE_CATEGORY == 'milestone_gate'
        assert fields.summary == task['title'][:200]
        assert fields.detail == 'Leo signs off on the feature.\n\nLanded dependencies: 7, 8'
        assert fields.options == ('A: ship', 'B: hold')
        assert fields.review_due is None

    def test_fields_are_frozen(self):
        fields = _build(_plain_gate(), None)
        with pytest.raises(dataclasses.FrozenInstanceError):
            fields.category = 'review_due'  # type: ignore[misc]

    def test_operational_llm_gate_keeps_its_token_prefix(self):
        task = _plain_gate()
        task['metadata'][OPERATIONAL_LLM_GATE_MARKER_KEY] = True
        fields = _build(task, None)
        assert fields.category == 'milestone_gate'
        assert fields.summary == f"{OPERATIONAL_LLM_NEEDS_LANE_TOKEN}: {task['title']}"[:200]
        assert fields.detail.startswith(f'[{OPERATIONAL_LLM_NEEDS_LANE_TOKEN}] ')
        assert fields.detail.endswith(_plain_detail(task))

    def test_completion_gate_role_builds_a_plain_milestone_gate(self):
        task = _human_gate_task(role='completion_gate')
        fields = _build(task, None)
        assert fields.category == 'milestone_gate'
        assert fields.summary == task['title'][:200]
        assert fields.detail == _plain_detail(task)
        assert fields.review_due is None


class TestHumanGate:
    def test_final_form_files_review_due_with_the_payload(self):
        task = _human_gate_task()
        fields = _build(task, _facts(_completion_gate()))
        assert fields.category == REVIEW_DUE_CATEGORY == 'review_due'
        assert fields.summary == 'Run /review on dark_factory (review-dark_factory-20261004 landed)'
        assert fields.review_due == {
            'skill': 'review',
            'project_id': PROJECT_ID,
            'project_root': PROJECT_ROOT,
            'run_id': RUN_ID,
            'human_gate_task_id': '42',
            'completion_gate_task_id': '41',
            'verdict': VERDICT,
            'fired_at': FIRED_AT,
            'launched_from': ESC_ID,
        }
        assert fields.review_due is not None
        assert not fields.review_due['skill'].startswith('/')
        assert fields.detail == _plain_detail(task)
        assert fields.options == ('A: ship', 'B: hold')

    def test_interim_form_has_no_completion_gate_and_no_extra_detail(self):
        task = _human_gate_task(deps=['30', '31'])
        fields = _build(task, _facts(None))
        assert fields.category == 'review_due'
        assert fields.review_due is not None
        assert fields.review_due['completion_gate_task_id'] is None
        assert fields.review_due['verdict'] is None
        assert fields.detail == _plain_detail(task)

    def test_unread_completion_gate_is_named_in_the_detail(self):
        task = _human_gate_task()
        fields = _build(
            task, _facts(CompletionGateUnread(task_id='41', reason='get_task error=boom')),
        )
        assert fields.category == 'review_due'
        assert fields.review_due is not None
        assert fields.review_due['verdict'] is None
        assert fields.review_due['completion_gate_task_id'] is None
        plain = _plain_detail(task)
        assert fields.detail.startswith(plain)
        extra = [line for line in fields.detail[len(plain):].splitlines() if line]
        assert len(extra) == 1
        assert '41' in extra[0]
        assert 'get_task error=boom' in extra[0]

    def test_non_dict_verdict_is_named_and_never_stored(self):
        task = _human_gate_task()
        fields = _build(task, _facts(_completion_gate(verdict='pass')))
        assert fields.category == 'review_due'
        assert fields.review_due is not None
        assert fields.review_due['verdict'] is None
        assert fields.review_due['completion_gate_task_id'] == '41'
        plain = _plain_detail(task)
        assert fields.detail.startswith(plain)
        extra = [line for line in fields.detail[len(plain):].splitlines() if line]
        assert len(extra) == 1
        assert 'str' in extra[0]

    def test_slash_spelled_skill_files_milestone_gate_naming_the_bare_form(self):
        task = _human_gate_task(skill='/review')
        fields = _build(task, _facts(_completion_gate()))
        assert fields.category == 'milestone_gate'
        assert fields.review_due is None
        assert "must be the bare skill name 'review'" in fields.detail
        assert fields.detail.startswith(_plain_detail(task))

    def test_valid_chain_without_facts_raises(self):
        task = _human_gate_task()
        with pytest.raises(ValueError, match='HumanGateFacts') as excinfo:
            _build(task, None)
        assert '42' in str(excinfo.value)


FIXED_NOW = datetime(2026, 10, 5, 12, 30, tzinfo=UTC)


def _chain_scheduler(task: dict, reads: Mapping[str, TaskRead | BaseException]) -> MagicMock:
    """A scheduler whose ``read_task`` answers each dependency id from *reads*."""

    async def read_task(task_id: str) -> TaskRead:
        answer = reads[str(task_id)]
        if isinstance(answer, BaseException):
            raise answer
        return answer

    scheduler = MagicMock()
    scheduler.config = SimpleNamespace(
        project_root=PROJECT_ROOT, fused_memory=SimpleNamespace(project_id=PROJECT_ID),
    )
    scheduler.read_task = AsyncMock(side_effect=read_task)
    scheduler.get_task = AsyncMock(return_value=task)
    scheduler.set_task_status = AsyncMock()
    scheduler.update_task = AsyncMock(return_value=True)
    return scheduler


async def _fire(
    tmp_path: Path, task: dict, reads: Mapping[str, TaskRead | BaseException],
) -> tuple[WorkflowOutcome, EscalationQueue, MagicMock]:
    queue = EscalationQueue(tmp_path)
    scheduler = _chain_scheduler(task, reads)
    runner = DeterministicRunner(
        scheduler=scheduler, escalation_queue=queue, clock=lambda: FIXED_NOW,
    )
    outcome = await runner.run(_make_assignment(task))
    return outcome, queue, scheduler


def _only_pending(queue: EscalationQueue, task_id: str):
    pending = queue.get_by_task(task_id, status='pending')
    assert len(pending) == 1
    return pending[0]


@pytest.mark.asyncio
class TestRunnerFilesTheHumanGate:
    async def test_final_form_files_review_due_with_the_completion_verdict(self, tmp_path: Path):
        task = _human_gate_task()
        outcome, queue, scheduler = await _fire(
            tmp_path, task, {'41': TaskFound(_completion_gate())},
        )

        assert outcome == WorkflowOutcome.BLOCKED
        esc = _only_pending(queue, '42')
        assert esc.category == 'review_due'
        assert esc.level == 2
        assert esc.severity == 'critical'
        assert esc.agent_role == 'orchestrator-deterministic'
        assert esc.review_due is not None
        verdict = esc.review_due['verdict']
        assert verdict is not None
        share = verdict['share']
        assert isinstance(share, float) and share >= 0.7
        assert esc.review_due['run_id'] == RUN_ID
        assert esc.review_due['completion_gate_task_id'] == '41'
        assert esc.review_due['launched_from'] == esc.id
        assert esc.review_due['fired_at'] == FIXED_NOW.isoformat()
        scheduler.update_task.assert_any_await(
            '42', {'gate_escalated_at': FIXED_NOW.isoformat()}, metadata_mode='merge',
        )
        scheduler.set_task_status.assert_awaited_once_with('42', 'blocked')

    async def test_interim_form_files_review_due_without_a_completion_gate(self, tmp_path: Path):
        task = _human_gate_task(deps=['30', '31'])
        outcome, queue, _ = await _fire(
            tmp_path, task,
            {'30': TaskFound({'id': '30', 'title': 'a run task', 'metadata': {}}), '31': TaskAbsent()},
        )

        assert outcome == WorkflowOutcome.BLOCKED
        esc = _only_pending(queue, '42')
        assert esc.category == 'review_due'
        assert esc.review_due is not None
        assert esc.review_due['completion_gate_task_id'] is None
        assert esc.review_due['verdict'] is None
        assert esc.detail == _plain_detail(task)

    @pytest.mark.parametrize('answer, error_text', [
        (TaskReadFailed('get_task error=boom'), 'get_task error=boom'),
        (RuntimeError('transport severed'), 'RuntimeError: transport severed'),
    ], ids=['read-failed', 'read-raised'])
    async def test_unreadable_dependency_is_named_in_the_detail(
        self, tmp_path: Path, answer: TaskRead | BaseException, error_text: str,
    ):
        task = _human_gate_task()
        outcome, queue, _ = await _fire(tmp_path, task, {'41': answer})

        assert outcome == WorkflowOutcome.BLOCKED
        esc = _only_pending(queue, '42')
        assert esc.category == 'review_due'
        assert esc.review_due is not None
        assert esc.review_due['verdict'] is None
        unread_lines = [line for line in esc.detail.splitlines() if error_text in line]
        assert len(unread_lines) == 1
        assert '41' in unread_lines[0]

    async def test_plain_gate_files_milestone_gate_without_reading_dependencies(self, tmp_path: Path):
        task = _plain_gate()
        outcome, queue, scheduler = await _fire(tmp_path, task, {})

        assert outcome == WorkflowOutcome.BLOCKED
        esc = _only_pending(queue, '42')
        assert esc.category == 'milestone_gate'
        assert esc.review_due is None
        scheduler.read_task.assert_not_awaited()
