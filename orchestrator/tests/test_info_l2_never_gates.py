"""Info never gates a merge or short-circuits the escalated wait (task 5222).

PRD ``plans/info-l0-disposition-router-prd.md`` D3 — "Info never gates a
merge, vetoes recovery, or short-circuits the escalated wait" — and boundary
rows #7 (an info L2 at merge entry: the task merges, no ``blocked`` row) and
#8 (a blocking L2 at merge entry gates exactly as before).

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

import pytest
from _workflow_helpers import _make_resolving_steward
from escalation.models import Escalation
from escalation.pins import classify_pins
from escalation.queue import EscalationQueue
from test_workflow_merge_gating_strand import (  # noqa: F401 -- fixtures used by name
    _build_workflow,
    _make_advanced_worktree,
    _make_silent_steward,
    _stub_merge_internals,
    _submit_gating_l0,
    _time_the_wait,
    _wire_resolve_callback,
    config,
    git_ops,
    git_repo,
    task_assignment,
)

from orchestrator.workflow import _is_gating_escalation


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
