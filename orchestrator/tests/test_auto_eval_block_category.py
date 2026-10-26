"""``_mark_blocked``'s escalation category reaches ``_maybe_auto_eval``'s trigger.

``TerminalReport.block_category`` carries the ``category=`` a block was filed
under, and ``Harness._run_slot`` maps it onto ``TaskReport.block_category`` —
so a worktree-conflict abort at setup, which blocks from ``plan``, is told
apart from a plan failure without matching the block reason.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from _workflow_helpers import _make_warmlane_workflow
from test_auto_eval import _make
from test_workflow_terminal_report import _run_stubbed_slot

from orchestrator.workflow_types import TerminalReport, WorkflowOutcome, WorkflowState


def test_block_category_round_trips_and_defaults_to_empty():
    report = TerminalReport(
        outcome=WorkflowOutcome.DONE, reason='', phase=WorkflowState.DONE,
        detail='', category=None,
    )
    assert report.block_category == ''
    assert TerminalReport(
        outcome=WorkflowOutcome.BLOCKED, reason='r', phase=WorkflowState.BLOCKED,
        detail='d', category=None, block_category='wip_conflict',
    ).block_category == 'wip_conflict'


@pytest.mark.asyncio
async def test_mark_blocked_records_its_category(tmp_path: Path):
    wf = _make_warmlane_workflow(tmp_path=tmp_path)
    wf.scheduler.update_task = AsyncMock()
    wf.state = WorkflowState.PLAN

    outcome = await wf._mark_blocked(
        'WIP-save aborted: unresolved conflict in worktree',
        category='wip_conflict',
        escalate_to_human=True,
    )

    assert outcome == WorkflowOutcome.BLOCKED
    report = wf._terminal_report
    assert report is not None
    assert report.blocked_from_phase == WorkflowState.PLAN
    assert report.block_category == 'wip_conflict'


@pytest.mark.asyncio
async def test_run_slot_maps_block_category_and_auto_eval_skips_it(tmp_path: Path):
    terminal = TerminalReport(
        outcome=WorkflowOutcome.BLOCKED,
        reason='WIP-save aborted: unresolved conflict in worktree',
        phase=WorkflowState.BLOCKED,
        detail='d',
        category=None,
        blocked_from_phase=WorkflowState.PLAN,
        block_category='wip_conflict',
    )
    task_report = await _run_stubbed_slot(tmp_path, terminal, task_id='81')

    assert task_report.block_phase == 'plan'
    assert task_report.block_category == 'wip_conflict'

    fixture = _make(project_root=tmp_path / 'auto-eval-proj')
    await fixture.harness._maybe_auto_eval(fixture.assignment, task_report)
    assert fixture.submit_calls == []
    fixture.git_ops_rename.assert_not_awaited()
