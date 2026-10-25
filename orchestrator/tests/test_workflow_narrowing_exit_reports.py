"""What the merge-gate narrowing agent writes INSTEAD of narrowing the plan.

When the plan-files gate in ``TaskWorkflow._submit_to_merge_queue`` flags
declared-but-untouched files, it gives the architect one narrowing pass.  That
agent holds every ``report_*`` exit tool, so it may answer with an exit report
rather than a narrowed plan.  The gate honours the three verdicts on the TASK
(already_done, unactionable_task, false_premise) and discards the two
branch-scoped ones (ready_to_merge, blocking_dependency), so no report outlives
the pass.

Regression for task 4105 / esc-4105-7: a correct, first-hand-verified
already_done.json sat unread while the gate escalated to a human anyway.

Every private seam of the workflow is touched inside ``_arm_gate`` and
``_Gate.submit`` so the tests themselves observe outcomes through public
surfaces (the scheduler, the event store, the merge queue, the artifacts).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _recording_event_store import _RecordingEventStore
from test_workflow import (
    _PASSING_VERIFY_RESULT,
    _CheckSequence,
    _make_workflow,
    _merge_attempt_outcomes,
)

from orchestrator.agents.invoke import AgentResult
from orchestrator.artifacts import TaskArtifacts
from orchestrator.merge_lane import PLAN_FILES_NOT_TOUCHED_REASON_PREFIX
from orchestrator.merge_lane.gates import PlanFilesTouchedResult
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome, WorkflowState

_CLAIMED_COMMIT = '4e175c0732abc'
_TASK_ID = '2656'


@dataclass
class _Gate:
    wf: TaskWorkflow
    artifacts: TaskArtifacts
    events: _RecordingEventStore
    check_seq: _CheckSequence
    merge_queue: asyncio.Queue
    mark_blocked: AsyncMock
    mark_done: AsyncMock

    async def submit(self) -> WorkflowOutcome:
        return await self.wf._submit_to_merge_queue(_TASK_ID, merge_phase=True)


def _arm_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    stage_report: Callable[[TaskArtifacts], None],
    commit_on_main: bool = True,
    cherry: tuple[int, str] = (0, '- 1111111\n'),
) -> _Gate:
    """A workflow whose gate flags ``a.py`` and whose narrowing agent leaves
    the plan unchanged but stages one exit report via *stage_report*."""
    wf = _make_workflow(tmp_path=tmp_path, task_id=_TASK_ID)
    assert wf.artifacts is not None
    artifacts = wf.artifacts

    events = _RecordingEventStore()
    monkeypatch.setattr(wf, 'event_store', events)

    async def fake_run(cmd, **_kw):
        if cmd[:2] == ['git', 'cherry']:
            rc, out = cherry
            return rc, out, ''
        return 0, 'branch_head_sha\n', ''
    monkeypatch.setattr('orchestrator.workflow._run', fake_run)

    check_seq = _CheckSequence([PlanFilesTouchedResult(not_touched=['a.py'])])
    monkeypatch.setattr(
        'orchestrator.merge_lane.gates._check_plan_files_touched_in_branch',
        check_seq,
    )

    async def fake_invoke(role, prompt, cwd, **_kw):
        del role, prompt, cwd
        artifacts.write_plan({'files': ['a.py', 'b.py', 'c.py']})
        stage_report(artifacts)
        return AgentResult(success=True, output='', cost_usd=0.0, turns=1)
    monkeypatch.setattr(wf, '_invoke', fake_invoke)
    wf.briefing.build_plan_tightening_prompt = AsyncMock(return_value='PROMPT')

    merge_queue: asyncio.Queue = asyncio.Queue()
    wf.merge_queue = merge_queue

    # A corroborated landing claim: the reported commit touches the one file
    # the task declares and its effect survives at main.
    wf.task = {
        'id': _TASK_ID, 'title': 'T', 'description': 'd',
        'metadata': {'files': ['a.py']},
    }
    mark_done = AsyncMock()
    wf.scheduler.mark_done = mark_done
    wf.git_ops.get_main_sha = AsyncMock(return_value='mainsha0001')
    wf.git_ops.is_ancestor = AsyncMock(return_value=commit_on_main)
    wf.git_ops.get_merge_commit_diff_files = AsyncMock(return_value=(['a.py'], None))
    wf.git_ops.commit_effect_present_in_main = AsyncMock(return_value=True)
    monkeypatch.setattr(wf, '_reconcile_metadata_files_for_done', AsyncMock())

    mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
    monkeypatch.setattr(wf, '_mark_blocked', mark_blocked)

    return _Gate(
        wf=wf, artifacts=artifacts, events=events, check_seq=check_seq,
        merge_queue=merge_queue, mark_blocked=mark_blocked, mark_done=mark_done,
    )


def _stage_already_done(artifacts: TaskArtifacts) -> None:
    artifacts.write_already_done(
        commit=_CLAIMED_COMMIT, evidence='helpers on main; suite ran, 26 passed',
    )


@pytest.mark.asyncio
async def test_validated_already_done_closes_the_task_instead_of_escalating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    gate = _arm_gate(tmp_path, monkeypatch, stage_report=_stage_already_done)

    outcome = await gate.submit()

    assert outcome == WorkflowOutcome.DONE
    assert gate.wf.state == WorkflowState.DONE
    gate.mark_done.assert_awaited_once()
    args, kwargs = gate.mark_done.await_args
    assert args[0] == _TASK_ID
    assert kwargs['kind'] == 'found_on_main'
    assert kwargs['sha'] == _CLAIMED_COMMIT
    gate.mark_blocked.assert_not_awaited()
    assert 'plan_files_not_touched' not in _merge_attempt_outcomes(gate.events)
    assert gate.check_seq.calls == 1
    assert gate.merge_queue.qsize() == 0
    assert gate.artifacts.read_already_done() is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'cherry',
    [(0, '- 1111111\n+ 2222222\n'), (128, '')],
    ids=['branch_carries_an_unlanded_patch', 'git_cherry_error'],
)
async def test_already_done_is_discarded_unless_every_branch_patch_is_on_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cherry: tuple[int, str],
):
    """The handler attributes the reported commit by ANY one declared file, so
    it proves the task's work is on main, not that this branch's delivery is."""
    gate = _arm_gate(
        tmp_path, monkeypatch, stage_report=_stage_already_done, cherry=cherry,
    )

    outcome = await gate.submit()

    assert outcome == WorkflowOutcome.BLOCKED
    gate.mark_done.assert_not_awaited()
    gate.mark_blocked.assert_awaited_once()
    args, _kwargs = gate.mark_blocked.await_args
    assert args[0].startswith(PLAN_FILES_NOT_TOUCHED_REASON_PREFIX)
    assert 'plan_files_not_touched' in _merge_attempt_outcomes(gate.events)
    assert gate.merge_queue.qsize() == 0
    assert gate.artifacts.read_already_done() is None


@pytest.mark.asyncio
async def test_refuted_already_done_blocks_on_the_merge_phase_ladder_without_a_human(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    gate = _arm_gate(
        tmp_path, monkeypatch,
        stage_report=_stage_already_done, commit_on_main=False,
    )

    outcome = await gate.submit()

    assert outcome == WorkflowOutcome.BLOCKED
    gate.mark_blocked.assert_awaited_once()
    args, kwargs = gate.mark_blocked.await_args
    assert 'not reachable from main' in args[0]
    assert kwargs['merge_phase'] is True
    assert kwargs.get('escalate_to_human') is not True
    gate.mark_done.assert_not_awaited()
    assert 'plan_files_not_touched' not in _merge_attempt_outcomes(gate.events)
    assert gate.artifacts.read_already_done() is None


@dataclass(frozen=True)
class _VerdictCase:
    stage_report: Callable[[TaskArtifacts], None]
    read_report: Callable[[TaskArtifacts], dict | None]
    reason_prefix: str
    category: str | None


_TASK_VERDICTS = {
    'unactionable_task': _VerdictCase(
        stage_report=lambda a: a.write_unactionable_task(
            reason='spec contradicts main', evidence='e',
        ),
        read_report=TaskArtifacts.read_unactionable_task,
        reason_prefix='Architect reported task unactionable',
        category=None,
    ),
    'false_premise': _VerdictCase(
        stage_report=lambda a: a.write_false_premise(
            classification='exactness', premise='p', evidence='e',
            proposed_resolution='r',
        ),
        read_report=TaskArtifacts.read_false_premise,
        reason_prefix='Architect reported false RED-test premise',
        category='design_concern',
    ),
}


@pytest.mark.asyncio
@pytest.mark.parametrize('case', _TASK_VERDICTS.values(), ids=_TASK_VERDICTS.keys())
async def test_task_verdict_report_escalates_with_the_architect_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: _VerdictCase,
):
    gate = _arm_gate(tmp_path, monkeypatch, stage_report=case.stage_report)

    await gate.submit()

    gate.mark_blocked.assert_awaited_once()
    args, kwargs = gate.mark_blocked.await_args
    assert args[0].startswith(case.reason_prefix)
    assert not args[0].startswith(PLAN_FILES_NOT_TOUCHED_REASON_PREFIX)
    assert kwargs['escalate_to_human'] is True
    assert kwargs['merge_phase'] is True
    if case.category is not None:
        assert kwargs['category'] == case.category
    assert 'plan_files_not_touched' not in _merge_attempt_outcomes(gate.events)
    assert case.read_report(gate.artifacts) is None


_BRANCH_SCOPED = {
    'ready_to_merge': (
        lambda a: a.write_ready_to_merge(commit='branch_head_sha', evidence='e'),
        TaskArtifacts.read_ready_to_merge,
    ),
    'blocking_dependency': (
        lambda a: a.write_blocking_dependency('9999', 'r', 'mainsha0001'),
        TaskArtifacts.read_blocking_dependency,
    ),
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('stage_report', 'read_report'),
    _BRANCH_SCOPED.values(), ids=_BRANCH_SCOPED.keys(),
)
async def test_branch_scoped_reports_are_discarded_and_the_gate_still_decides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    stage_report: Callable[[TaskArtifacts], None],
    read_report: Callable[[TaskArtifacts], dict | None],
):
    gate = _arm_gate(tmp_path, monkeypatch, stage_report=stage_report)

    await gate.submit()

    gate.mark_blocked.assert_awaited_once()
    args, kwargs = gate.mark_blocked.await_args
    assert args[0].startswith(PLAN_FILES_NOT_TOUCHED_REASON_PREFIX)
    assert kwargs['escalate_to_human'] is True
    assert 'plan_files_not_touched' in _merge_attempt_outcomes(gate.events)
    assert gate.merge_queue.qsize() == 0
    assert read_report(gate.artifacts) is None


@pytest.mark.asyncio
async def test_merge_phase_returns_a_terminal_done_without_running_the_merge_success_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """A DONE from the merge attempt on an already-terminal machine is a
    closed task, not a landed merge: nothing is left to finalise."""
    gate = _arm_gate(tmp_path, monkeypatch, stage_report=_stage_already_done)
    wf = gate.wf

    async def close_without_a_merge(branch_name, **_kw):
        del branch_name
        wf.machine.transition(WorkflowState.DONE)
        return WorkflowOutcome.DONE
    monkeypatch.setattr(wf, '_submit_to_merge_queue', close_without_a_merge)
    monkeypatch.setattr(
        'orchestrator.workflow.run_scoped_verification',
        AsyncMock(return_value=_PASSING_VERIFY_RESULT),
    )
    wf.config.max_merge_retries = 1
    wf.config.max_pre_merge_retries = 1
    wf.git_ops.rebase_onto_main = AsyncMock(return_value=True)
    finalise_merged_done = AsyncMock(return_value=WorkflowOutcome.DONE)
    write_completion = AsyncMock()
    clear_retry_pending = AsyncMock()
    clear_phase_entered = AsyncMock()
    for name, stub in {
        '_maybe_defer_as_train_member': AsyncMock(return_value=None),
        '_check_escalations': MagicMock(return_value=[]),
        '_recover_before_merge': AsyncMock(return_value=None),
        '_finalise_merged_done': finalise_merged_done,
        '_write_completion_to_memory': write_completion,
        '_clear_merge_retry_pending': clear_retry_pending,
        '_clear_merge_phase_entered': clear_phase_entered,
        '_ensure_steward_started': AsyncMock(),
        '_await_steward_completion': AsyncMock(),
    }.items():
        monkeypatch.setattr(wf, name, stub)

    outcome = await asyncio.wait_for(wf._merge_and_finalise(_TASK_ID), 30)

    assert outcome == WorkflowOutcome.DONE
    finalise_merged_done.assert_not_awaited()
    write_completion.assert_not_awaited()
    # The liveness stamp and retry obligation must not outlive the closed task.
    clear_retry_pending.assert_awaited()
    clear_phase_entered.assert_awaited()
