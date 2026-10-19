"""A merge-phase SUCCESS with no ``_merge_sha`` resolves honestly or blocks (task 3113).

``_finalise_merged_done`` used to exit DONE without writing the done row
whenever ``_merge_sha`` was unset.  The routes there are a merge queue
``already_merged`` result (a coalesced duplicate submission, as in reify
5367/5602, or a restart-replayed orphan, as in the 2026-10-17 specimens reify
9062, solar-challenge 748 and solar 846/270) and the steward-resolved "branch
on main" break.  The task never reached ``done``, and the theta exit-contract
gate (task 3546) waits on this exit being honest.

The DONE tail now runs the shared already-merged recovery ladder and stamps an
honest provenance; when nothing attributes the landing, it blocks with an L1
from the still-working machine.  Eval mode keeps its DONE-without-row exit.

Driven through ``_workflow_helpers._make``'s git stubs and a real git worktree
for ``git rev-parse HEAD``.  This file imports no merge-lane module on purpose.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from _workflow_helpers import _Fixture, _init_git_repo, _make
from escalation.queue import EscalationQueue

from orchestrator.delivered_checks import DeliveredChecksBlock
from orchestrator.workflow import WorkflowOutcome, WorkflowState

_GATE_TARGET = 'orchestrator.workflow.gate_mark_done_on_delivered_checks'


async def _fixture(tmp_path: Path, *, branch_on_main: bool) -> _Fixture:
    worktree = tmp_path / 'wt'
    await _init_git_repo(worktree)
    f = _make(
        worktree=worktree, project_root=tmp_path / 'proj',
        branch_on_main=branch_on_main,
    )
    f.wf._merge_sha = None
    return f


def _arm_escalation_queue(f: _Fixture, tmp_path: Path) -> EscalationQueue:
    """A real escalation queue, and no fire-and-forget dry-run investigation."""
    f.wf.escalation_queue = EscalationQueue(tmp_path / 'escalations')
    f.wf.config.unblock_auto.enabled = False
    return f.wf.escalation_queue


def _record_prior_work(f: _Fixture) -> None:
    """The implementer entry the real ``_has_prior_implementation`` reads as work."""
    f.artifacts.append_iteration_log({
        'agent': 'implementer', 'source': 'orchestrator',
        'steps_attempted': ['s1'], 'steps_completed': ['s1'],
        'commit': 'newhead',
    })


def _status_writes(f: _Fixture) -> list[str]:
    return [c.args[1] for c in f.set_task_status.await_args_list if len(c.args) >= 2]


@pytest.mark.asyncio
async def test_unresolvable_missing_merge_sha_blocks_and_files_l1(tmp_path: Path):
    f = await _fixture(tmp_path, branch_on_main=False)
    queue = _arm_escalation_queue(f, tmp_path)

    outcome = await f.wf._finalise_merged_done()

    assert outcome == WorkflowOutcome.BLOCKED
    assert f.wf.state == WorkflowState.BLOCKED
    f.mark_done.assert_not_awaited()
    assert 'blocked' in _status_writes(f)
    (esc,) = queue.get_by_task(f.wf.task_id, status='pending')
    assert esc.level == 1
    assert 'merge_sha' in esc.detail
    assert 'not found landed on main' in esc.detail


@pytest.mark.asyncio
async def test_eval_mode_missing_merge_sha_still_exits_done_without_row_write(
    tmp_path: Path,
):
    """Eval runs skip the merge by design, so they reach this tail sha-less."""
    f = await _fixture(tmp_path, branch_on_main=False)
    queue = _arm_escalation_queue(f, tmp_path)
    f.wf._worktree_external = True

    outcome = await f.wf._finalise_merged_done()

    assert outcome == WorkflowOutcome.DONE
    assert f.wf.state == WorkflowState.DONE
    f.mark_done.assert_not_awaited()
    assert 'blocked' not in _status_writes(f)
    assert queue.get_by_task(f.wf.task_id) == []


@pytest.mark.asyncio
async def test_landed_branch_without_merge_sha_recovers_found_on_main(
    tmp_path: Path,
):
    """An attributable landing stamps the task's own citation, never main's tip."""
    f = await _fixture(tmp_path, branch_on_main=True)
    queue = _arm_escalation_queue(f, tmp_path)
    _record_prior_work(f)

    outcome = await f.wf._finalise_merged_done()

    assert outcome == WorkflowOutcome.DONE
    f.mark_done.assert_awaited_once_with(
        f.wf.task_id, kind='found_on_main', sha='citationsha123',
        note='branch already on main at merge phase (post-MERGE recovery)',
    )
    assert f.wf._merge_recovery_basis == 'fallback'
    assert queue.get_by_task(f.wf.task_id) == []


@pytest.mark.asyncio
async def test_merge_sha_present_never_consults_the_ladder(tmp_path: Path):
    f = await _fixture(tmp_path, branch_on_main=True)
    f.wf._merge_sha = 'c' * 40
    f.wf.git_ops.get_merge_commit_diff_files = AsyncMock(return_value=([], None))

    outcome = await f.wf._finalise_merged_done()

    assert outcome == WorkflowOutcome.DONE
    f.mark_done.assert_awaited_once_with(f.wf.task_id, kind='merged', sha='c' * 40)
    f.is_ancestor.assert_not_awaited()
    citation = f.wf.git_ops.find_task_citation_commit
    assert isinstance(citation, AsyncMock)
    citation.assert_not_awaited()


@pytest.mark.asyncio
async def test_git_error_during_resolution_blocks_fail_safe(tmp_path: Path):
    f = await _fixture(tmp_path, branch_on_main=True)
    queue = _arm_escalation_queue(f, tmp_path)
    _record_prior_work(f)
    f.get_main_sha.side_effect = RuntimeError('main ref unreadable')

    outcome = await f.wf._finalise_merged_done()

    assert outcome == WorkflowOutcome.BLOCKED
    f.mark_done.assert_not_awaited()
    (esc,) = queue.get_by_task(f.wf.task_id, status='pending')
    assert esc.level == 1
    assert 'main ref unreadable' in esc.detail


@pytest.mark.asyncio
async def test_withheld_recovery_blocks_instead_of_done(tmp_path: Path):
    """The delivered-checks guard withholding the landing is not a DONE either."""
    f = await _fixture(tmp_path, branch_on_main=True)
    queue = _arm_escalation_queue(f, tmp_path)
    _record_prior_work(f)
    guard = AsyncMock(return_value=DeliveredChecksBlock(
        reason='failed', main_sha='mainsha123',
        failed_check={'name': 'cap-x', 'kind': 'grep', 'pattern': 'P', 'expect': 'present'},
    ))

    with patch(_GATE_TARGET, guard):
        outcome = await f.wf._finalise_merged_done()

    assert outcome == WorkflowOutcome.BLOCKED
    f.mark_done.assert_not_awaited()
    (esc,) = queue.get_by_task(f.wf.task_id, status='pending')
    assert esc.level == 1
