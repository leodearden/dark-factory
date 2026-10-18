"""Info never gates a merge or short-circuits the escalated wait (task 5222).

PRD ``plans/info-l0-disposition-router-prd.md`` D3 — "Info never gates a
merge, vetoes recovery, or short-circuits the escalated wait" — and boundary
rows #7 (an info L2 at merge entry: the task merges, no ``blocked`` row) and
#8 (a blocking L2 at merge entry gates exactly as before).  Goal 3 extends both
to the wait's level-1 tail: an info L1 never hands off, and a hand-off says
"Steward re-escalated to human" only when the steward did (D13).

The workflow gate (``orchestrator.workflow._is_gating_escalation``) and the
recovery-pin classifier (``escalation.pins.classify_pins``) used to disagree on
an info-severity L2: the gate held the task at merge entry while the pin
classifier told the scheduler nothing held it, so the task looped block ->
re-pend -> re-dispatch (esc-4811-3, 2026-09-21).  The gate now asks
``escalation.pins`` whether a level>=2 record is a queue handoff, so the two
mechanisms agree by construction.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
from _workflow_helpers import (
    AgentStub,
    _build_workflow_with_escalation,
    _make_resolving_steward,
)
from escalation.models import Escalation
from escalation.pins import classify_pins
from escalation.queue import EscalationQueue
from test_workflow_merge_gating_strand import (
    _build_workflow,
    _make_advanced_worktree,
    _make_silent_steward,
    _stub_merge_internals,
    _submit_gating_l0,
    _time_the_wait,
    _wire_resolve_callback,
)
from test_workflow_merge_gating_strand import config as config
from test_workflow_merge_gating_strand import git_ops as git_ops
from test_workflow_merge_gating_strand import git_repo as git_repo
from test_workflow_merge_gating_strand import task_assignment as task_assignment

from orchestrator.agents.roles import STEWARD
from orchestrator.verify import VerifyResult
from orchestrator.workflow import _is_gating_escalation
from orchestrator.workflow_types import WorkflowOutcome


def _escalation(severity: str, level: int) -> Escalation:
    return Escalation(
        id=f'esc-42-{level}',
        task_id='42',
        agent_role='escalation-watcher-auto',
        severity=severity,
        category='design_concern',
        summary=f'{severity!r} at level {level}',
        level=level,
    )


def _submit_l2(queue: EscalationQueue, task_id: str, severity: str) -> Escalation:
    """Submit a pending level-2 record at *severity*."""
    esc = Escalation(
        id=queue.make_id(task_id),
        task_id=task_id,
        agent_role='escalation-watcher-auto',
        severity=severity,
        category='design_concern',
        summary=f'a {severity!r} L2 open at merge entry',
        detail='a cluster the watcher promoted for a human',
        level=2,
    )
    queue.submit(esc)
    return esc


def _submit_l1(
    queue: EscalationQueue, task_id: str, severity: str,
    agent_role: str = 'harness-orphan-reaper',
) -> Escalation:
    """Submit a pending level-1 record at *severity*, as an earlier incarnation left it."""
    esc = Escalation(
        id=queue.make_id(task_id),
        task_id=task_id,
        agent_role=agent_role,
        severity=severity,
        category='design_concern',
        summary=f'a {severity!r} L1 left by an earlier incarnation',
        level=1,
    )
    queue.submit(esc)
    return esc


def _make_handing_off_steward(
    queue: EscalationQueue, task_id: str, filer_role: str = STEWARD.name,
) -> type:
    """A steward that hands the gating L0 to a human: files a level-1 under
    *filer_role* and dismisses the L0.  With the steward's role, both producers
    look like this in the queue — ``TaskSteward``'s own give-up path and its
    agent's ``escalate_blocker(level=1)``.  It acts after a short delay, as the
    real steward's background loop does, so the wait has begun before it hands
    off."""

    class _FakeSteward:
        def __init__(self, wt_path, cfg_dir):
            self.give_up_task: asyncio.Task | None = None

        def set_outcome_channel(self, channel) -> None:
            pass

        def set_wip_probe(self, probe) -> None:
            pass

        async def _give_up(self) -> None:
            await asyncio.sleep(0.05)
            for esc in queue.get_by_task(task_id, status='pending', level=0):
                queue.submit(Escalation(
                    id=queue.make_id(task_id),
                    task_id=task_id,
                    agent_role=filer_role,
                    severity=esc.severity,
                    category=esc.category,
                    summary=f'Steward re-escalation: {esc.summary}',
                    level=1,
                ))
                queue.resolve(
                    esc.id, 'Auto-dismissed: re-escalated to level 1',
                    dismiss=True, resolved_by='steward',
                )

        async def start(self) -> None:
            self.give_up_task = asyncio.create_task(self._give_up())

        async def stop(self) -> None:
            if self.give_up_task is not None:
                await self.give_up_task

    return _FakeSteward


class TestGatingAgreesWithThePinClassifier:
    @pytest.mark.parametrize('level', [2, 3])
    @pytest.mark.parametrize(
        'severity',
        ['info', 'INFO', ' info ', 'blocking', 'critical', 'urgent', 'warn', ''],
    )
    def test_a_human_level_record_gates_iff_it_pins(self, severity, level):
        e = _escalation(severity, level)

        assert _is_gating_escalation(e) is classify_pins(
            '42', [e], live_claimant=True,
        ).pins

    def test_info_never_gates_and_an_unknown_severity_fails_safe(self):
        assert _is_gating_escalation(_escalation('info', 2)) is False
        assert _is_gating_escalation(_escalation('warn', 2)) is True


@pytest.mark.asyncio
class TestInfoL2DoesNotGateTheMergeEntry:
    async def test_only_an_info_l2_open_merges_in_slot(
        self, config, git_ops, task_assignment, monkeypatch, tmp_path,
    ):
        queue = EscalationQueue(tmp_path / 'queue')
        task_id = task_assignment.task_id
        worktree = await _make_advanced_worktree(git_ops, task_id)
        wf, scheduler = _build_workflow(
            config, git_ops, task_assignment, queue, worktree,
        )
        _wire_resolve_callback(queue, wf)
        submit = _stub_merge_internals(wf, monkeypatch)
        timing = _time_the_wait(wf)
        info_l2 = _submit_l2(queue, task_id, 'info')
        silent = _make_silent_steward()
        wf._steward_factory = silent

        outcome = await asyncio.wait_for(wf._run_merge_phase(f'task/{task_id}'), 20)

        assert outcome is None, f'an info L2 must not stop the merge; got {outcome}'
        assert submit.await_count == 1
        assert 'calls' not in timing, 'no hand-off may be raised for an info L2'
        written = scheduler.statuses.get(task_id, [])
        assert 'blocked' not in written, written
        assert 'pending' not in written, written
        still_pending = queue.get_by_task(task_id, status='pending', level=2)
        assert [e.id for e in still_pending] == [info_l2.id]
        assert silent.instances == []

    async def test_info_l2_does_not_short_circuit_the_wait_for_a_blocking_l0(
        self, config, git_ops, task_assignment, monkeypatch, tmp_path,
    ):
        queue = EscalationQueue(tmp_path / 'queue')
        task_id = task_assignment.task_id
        worktree = await _make_advanced_worktree(git_ops, task_id)
        wf, scheduler = _build_workflow(
            config, git_ops, task_assignment, queue, worktree,
        )
        _wire_resolve_callback(queue, wf)
        submit = _stub_merge_internals(wf, monkeypatch)
        timing = _time_the_wait(wf)
        _submit_gating_l0(queue, task_id)
        info_l2 = _submit_l2(queue, task_id, 'info')
        wf._steward_factory = _make_resolving_steward(queue, task_id)

        outcome = await asyncio.wait_for(wf._run_merge_phase(f'task/{task_id}'), 20)

        assert timing.get('calls') == 1, timing
        assert outcome is None, (
            f'the steward consumed the only gating record; got {outcome}'
        )
        assert submit.await_count == 1
        assert 'blocked' not in scheduler.statuses.get(task_id, [])
        still_pending = queue.get_by_task(task_id, status='pending', level=2)
        assert [e.id for e in still_pending] == [info_l2.id]


@pytest.mark.asyncio
class TestNonInfoL2StillGatesTheMergeEntry:
    @pytest.mark.parametrize('severity', ['blocking', 'warn'])
    async def test_non_info_l2_blocks_without_merging(
        self, config, git_ops, task_assignment, monkeypatch, tmp_path, severity,
    ):
        queue = EscalationQueue(tmp_path / 'queue')
        task_id = task_assignment.task_id
        worktree = await _make_advanced_worktree(git_ops, task_id)
        wf, scheduler = _build_workflow(
            config, git_ops, task_assignment, queue, worktree,
        )
        _wire_resolve_callback(queue, wf)
        submit = _stub_merge_internals(wf, monkeypatch)
        l2 = _submit_l2(queue, task_id, severity)
        silent = _make_silent_steward()
        wf._steward_factory = silent

        outcome = await asyncio.wait_for(wf._run_merge_phase(f'task/{task_id}'), 5)

        assert outcome is not None
        assert scheduler.statuses.get(task_id, [])[-1:] == ['blocked']
        assert submit.await_count == 0
        still_pending = queue.get_by_task(task_id, status='pending', level=2)
        assert [e.id for e in still_pending] == [l2.id]
        assert silent.instances == []


@pytest.mark.asyncio
class TestInfoL1DoesNotHandOffAtTheWaitTail:
    async def test_an_info_l1_left_open_merges_once_the_steward_resolves(
        self, config, git_ops, task_assignment, monkeypatch, tmp_path,
    ):
        queue = EscalationQueue(tmp_path / 'queue')
        task_id = task_assignment.task_id
        worktree = await _make_advanced_worktree(git_ops, task_id)
        wf, scheduler = _build_workflow(
            config, git_ops, task_assignment, queue, worktree,
        )
        _wire_resolve_callback(queue, wf)
        submit = _stub_merge_internals(wf, monkeypatch)
        info_l1 = _submit_l1(queue, task_id, 'info')
        _submit_gating_l0(queue, task_id)
        wf._steward_factory = _make_resolving_steward(queue, task_id)

        outcome = await asyncio.wait_for(wf._run_merge_phase(f'task/{task_id}'), 20)

        assert outcome is None, f'an info L1 must not hand off; got {outcome}'
        assert submit.await_count == 1
        assert 'blocked' not in scheduler.statuses.get(task_id, [])
        still_pending = queue.get_by_task(task_id, status='pending', level=1)
        assert [e.id for e in still_pending] == [info_l1.id]


@pytest.mark.asyncio
class TestTheWaitTailNamesWhoActed:
    """An escalated wait that ends with a non-info L1 open blocks either way;
    the reason says whether this wait's steward handed off (PRD D13)."""

    async def _run_escalated(
        self, config, git_ops, task_assignment, monkeypatch, tmp_path,
        *, make_steward, stale_l1_role: str | None = None,
    ):
        stub = AgentStub()
        workflow, _, queue = _build_workflow_with_escalation(
            config, git_ops, task_assignment, stub, tmp_path,
        )
        monkeypatch.setattr('orchestrator.workflow.invoke_agent', stub.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(return_value=VerifyResult(
                passed=True, test_output='', lint_output='',
                type_output='', summary='All checks passed',
            )),
        )
        task_id = task_assignment.task_id
        stale: list[Escalation] = []

        async def _escalate_once():
            if not stale and stale_l1_role is not None:
                stale.append(_submit_l1(queue, task_id, 'blocking', stale_l1_role))
            _submit_gating_l0(queue, task_id)
            return WorkflowOutcome.ESCALATED

        monkeypatch.setattr(workflow, '_execute_verify_review_loop', _escalate_once)
        workflow._steward_factory = make_steward(queue, task_id)

        report = await asyncio.wait_for(workflow.run(), 60)
        open_l1 = queue.get_by_task(task_id, status='pending', level=1)
        return report, open_l1, stale

    async def test_a_steward_hand_off_says_so(
        self, config, git_ops, task_assignment, monkeypatch, tmp_path,
    ):
        report, open_l1, _ = await self._run_escalated(
            config, git_ops, task_assignment, monkeypatch, tmp_path,
            make_steward=_make_handing_off_steward,
        )

        assert report.outcome == WorkflowOutcome.BLOCKED
        assert report.reason == 'Steward re-escalated to human'
        assert [e.agent_role for e in open_l1] == [STEWARD.name]

    async def test_an_l1_another_role_files_mid_wait_names_no_steward(
        self, config, git_ops, task_assignment, monkeypatch, tmp_path,
    ):
        report, open_l1, _ = await self._run_escalated(
            config, git_ops, task_assignment, monkeypatch, tmp_path,
            make_steward=lambda queue, task_id: _make_handing_off_steward(
                queue, task_id, filer_role='orchestrator',
            ),
        )

        assert report.outcome == WorkflowOutcome.BLOCKED
        assert report.reason == (
            f'Gated by open L1 {open_l1[0].id}; no steward applicable'
        )

    @pytest.mark.parametrize('stale_l1_role', ['harness-orphan-reaper', STEWARD.name])
    async def test_an_l1_open_before_the_wait_names_no_steward(
        self, config, git_ops, task_assignment, monkeypatch, tmp_path,
        stale_l1_role,
    ):
        report, _, stale = await self._run_escalated(
            config, git_ops, task_assignment, monkeypatch, tmp_path,
            make_steward=_make_resolving_steward, stale_l1_role=stale_l1_role,
        )

        assert report.outcome == WorkflowOutcome.BLOCKED
        assert report.reason == (
            f'Gated by open L1 {stale[0].id}; no steward applicable'
        )
        assert 'Steward re-escalated' not in report.detail
