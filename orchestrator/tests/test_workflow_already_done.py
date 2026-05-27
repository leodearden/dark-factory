"""Tests for ``TaskWorkflow._handle_already_done_report``.

The helper processes the architect's ``.task/already_done.json`` artifact:
validates the cited commit is reachable from main, then sets task status
to ``done`` with provenance.  Validation failures route to ``_mark_blocked``
without escalating to a human — they signal an architect mistake, not an
unworkable task.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _orch_helpers import pydantic_spec

from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import OrchestratorConfig
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome, WorkflowState


@dataclass
class _Fixture:
    wf: TaskWorkflow
    artifacts: TaskArtifacts
    set_task_status: AsyncMock
    is_ancestor: AsyncMock
    get_main_sha: AsyncMock


def _make(
    *,
    worktree: Path,
    project_root: Path,
    task_id: str = '50',
    commit_on_main: bool = True,
    main_sha: str = 'mainsha123',
) -> _Fixture:
    assignment = MagicMock()
    assignment.task_id = task_id
    assignment.task = {'id': task_id, 'title': 'T', 'description': 'd'}
    assignment.modules = ['mod_a']

    _spec = pydantic_spec(OrchestratorConfig)
    config = MagicMock(spec_set=_spec)
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.max_review_cycles = 2
    config.max_amendment_rounds = 1
    config.lock_depth = 2
    config.steward_completion_timeout = 300.0
    config.project_root = project_root

    set_task_status = AsyncMock()
    scheduler = MagicMock()
    scheduler.set_task_status = set_task_status
    # Fix 1: workflow refreshes metadata.files via update_task before
    # set_task_status('done').  Stub as AsyncMock so the await succeeds.
    scheduler.update_task = AsyncMock(return_value=True)
    # _setup_worktree_and_artifacts now reads live status at dispatch before
    # claiming 'in-progress' (commit 57e25f1260); None → non-terminal → proceed.
    scheduler.get_status = AsyncMock(return_value=None)

    # Stage 1: workflow now calls scheduler.mark_done — wire the helper
    # to forward into set_task_status so existing assertions still observe
    # the (task_id, 'done', done_provenance=...) call shape.
    async def _fake_mark_done(tid, *, kind, sha, note=None):
        provenance: dict = {'kind': kind, 'commit': sha}
        if note is not None:
            provenance['note'] = note
        await set_task_status(tid, 'done', done_provenance=provenance)
    scheduler.mark_done = AsyncMock(side_effect=_fake_mark_done)

    is_ancestor = AsyncMock(return_value=commit_on_main)
    get_main_sha = AsyncMock(return_value=main_sha)
    git_ops = MagicMock()
    git_ops.is_ancestor = is_ancestor
    git_ops.get_main_sha = get_main_sha

    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,
        briefing=MagicMock(),
        mcp=MagicMock(),
    )

    worktree.mkdir(parents=True, exist_ok=True)
    artifacts = TaskArtifacts(worktree)
    artifacts.init(task_id, 'T', 'd', base_commit='oldbase')
    wf.artifacts = artifacts
    wf.worktree = worktree

    return _Fixture(
        wf=wf, artifacts=artifacts,
        set_task_status=set_task_status,
        is_ancestor=is_ancestor,
        get_main_sha=get_main_sha,
    )


@pytest.mark.asyncio
async def test_valid_commit_sets_done_with_provenance(tmp_path: Path):
    f = _make(
        worktree=tmp_path / 'wt',
        project_root=tmp_path / 'proj',
        commit_on_main=True,
    )
    f.artifacts.write_already_done(
        commit='abcdef1234567890',
        evidence='helpers.foo on main at task 42',
    )

    outcome = await f.wf._handle_already_done_report()

    assert outcome == WorkflowOutcome.DONE
    assert f.wf.state == WorkflowState.DONE
    f.set_task_status.assert_awaited_once()
    args, kwargs = f.set_task_status.await_args
    assert args[0] == '50'
    assert args[1] == 'done'
    provenance = kwargs['done_provenance']
    # 2026-04-27 hardening: kind discriminator added; verified_by/evidence
    # folded into the structured note field so the schema accepts the call.
    assert provenance['kind'] == 'found_on_main'
    assert provenance['commit'] == 'abcdef1234567890'
    assert 'architect' in provenance['note']
    assert 'helpers.foo on main at task 42' in provenance['note']
    # Artifact cleared so a re-run doesn't see a stale report.
    assert f.artifacts.read_already_done() is None
    # is_ancestor called with (commit, main_sha) — strictly main-membership check.
    f.is_ancestor.assert_awaited_once_with('abcdef1234567890', 'mainsha123')


@pytest.mark.asyncio
async def test_commit_not_on_main_blocks_without_l1(tmp_path: Path):
    f = _make(
        worktree=tmp_path / 'wt',
        project_root=tmp_path / 'proj',
        commit_on_main=False,
    )
    f.artifacts.write_already_done('feedfacecafebeef', 'evidence text')

    mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
    f.wf._mark_blocked = mark_blocked  # type: ignore[method-assign]

    outcome = await f.wf._handle_already_done_report()

    assert outcome == WorkflowOutcome.BLOCKED
    mark_blocked.assert_awaited_once()
    assert mark_blocked.await_args is not None
    args, kwargs = mark_blocked.await_args
    assert 'not reachable from main' in args[0]
    # Validation failure must NOT escalate to human — this is an architect bug.
    assert kwargs.get('escalate_to_human') is not True
    # Status NOT set to done.
    f.set_task_status.assert_not_called()
    # Artifact cleared even on validation failure (don't loop on stale report).
    assert f.artifacts.read_already_done() is None


@pytest.mark.asyncio
async def test_missing_commit_blocks(tmp_path: Path):
    f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj')
    # Write artifact directly without commit field.
    (f.artifacts.root / 'already_done.json').write_text(
        '{"evidence": "I forgot the commit"}\n'
    )
    mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
    f.wf._mark_blocked = mark_blocked  # type: ignore[method-assign]

    outcome = await f.wf._handle_already_done_report()

    assert outcome == WorkflowOutcome.BLOCKED
    mark_blocked.assert_awaited_once()
    assert mark_blocked.await_args is not None
    args, kwargs = mark_blocked.await_args
    assert 'malformed' in args[0].lower()
    assert kwargs.get('escalate_to_human') is not True
    f.set_task_status.assert_not_called()
    f.is_ancestor.assert_not_called()
    assert f.artifacts.read_already_done() is None


@pytest.mark.asyncio
async def test_empty_commit_string_blocks(tmp_path: Path):
    f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj')
    f.artifacts.write_already_done(commit='   ', evidence='blank')
    mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
    f.wf._mark_blocked = mark_blocked  # type: ignore[method-assign]

    outcome = await f.wf._handle_already_done_report()

    assert outcome == WorkflowOutcome.BLOCKED
    mark_blocked.assert_awaited_once()
    f.set_task_status.assert_not_called()
    f.is_ancestor.assert_not_called()


@pytest.mark.asyncio
async def test_run_short_circuits_on_architect_already_done(tmp_path: Path):
    """``run()`` returns DONE without enqueuing a merge when ``_plan`` returns DONE.

    Regression for the 2026-05-04 know-live incident: a deleted DONE
    early-return at workflow.run() let the workflow proceed to
    EXECUTE/VERIFY/MERGE after _handle_already_done_report set the task
    done.  The orphaned merge then halted the queue with no escalation
    owner.
    """
    import asyncio

    f = _make(
        worktree=tmp_path / 'wt',
        project_root=tmp_path / 'proj',
        commit_on_main=True,
    )

    # Stub _plan to return DONE — simulates _handle_already_done_report's
    # success path running inside _plan().
    f.wf._plan = AsyncMock(return_value=WorkflowOutcome.DONE)
    # If EXECUTE were entered, this would raise loudly.
    f.wf._execute_iterations = AsyncMock(  # type: ignore[method-assign]
        side_effect=AssertionError(
            'EXECUTE must not run when _plan returns DONE',
        ),
    )
    # Wire a merge queue and assert nothing gets enqueued.
    merge_queue: asyncio.Queue = asyncio.Queue()
    f.wf.merge_queue = merge_queue
    # _recover_if_already_merged returns None — let the run() proceed to PLAN.
    f.wf._recover_if_already_merged = AsyncMock(  # type: ignore[method-assign]
        return_value=None,
    )
    # _check_branch_on_main: bypass the post-PLAN ghost-loop guard.
    f.wf._check_branch_on_main = AsyncMock(  # type: ignore[method-assign]
        return_value=None,
    )

    outcome = await f.wf.run()

    assert outcome == WorkflowOutcome.DONE
    f.wf._execute_iterations.assert_not_called()
    assert merge_queue.empty(), (
        'No merge request must be enqueued when _plan returns DONE'
    )


@pytest.mark.asyncio
async def test_run_short_circuits_on_plan_escalated(tmp_path: Path):
    """``run()`` returns ESCALATED without entering the post-PLAN ghost-loop guard.

    Regression for the 2026-05-06 task 2911 incident: when ``_plan`` →
    ``_mark_blocked`` returns ESCALATED (architect blew its budget, steward
    timed out, L1 escalation filed), the post-_plan switch in run() didn't
    match ESCALATED and let control fall through to the ghost-loop guard,
    which mistook architect scratch in the worktree for "prior merge
    survived" and tried to flip the task to done.

    This test sets up the conditions that *would* cause the fall-through to
    misfire — branch already on main + uncommitted scratch in worktree —
    and proves the early-return in run() ignores both.

    Mirrors fix 9760ba74bf for the DONE case (same shape, different outcome).
    """
    import asyncio

    f = _make(
        worktree=tmp_path / 'wt',
        project_root=tmp_path / 'proj',
        commit_on_main=True,
    )

    # Stub _plan to return ESCALATED — simulates _mark_blocked's L1-handoff path.
    f.wf._plan = AsyncMock(return_value=WorkflowOutcome.ESCALATED)
    # If EXECUTE were entered, this would raise loudly.
    f.wf._execute_iterations = AsyncMock(  # type: ignore[method-assign]
        side_effect=AssertionError(
            'EXECUTE must not run when _plan returns ESCALATED',
        ),
    )
    # Wire a merge queue and assert nothing gets enqueued.
    merge_queue: asyncio.Queue = asyncio.Queue()
    f.wf.merge_queue = merge_queue
    # _recover_if_already_merged returns None — let run() proceed to PLAN.
    f.wf._recover_if_already_merged = AsyncMock(  # type: ignore[method-assign]
        return_value=None,
    )
    # Set up the ghost-loop guard's preconditions DELIBERATELY so it would
    # misfire if we reached it: branch HEAD genuinely on main + scratch in
    # the worktree.  Fix 1's job is to return ESCALATED before this code
    # runs at all — so we assert these stubs were NOT consulted.
    check_branch = AsyncMock(return_value=('wthead12', 'mainsha123'))
    f.wf._check_branch_on_main = check_branch  # type: ignore[method-assign]
    f.wf.git_ops.has_uncommitted_work = AsyncMock(return_value=True)

    outcome = await f.wf.run()

    assert outcome == WorkflowOutcome.ESCALATED
    f.wf._execute_iterations.assert_not_called()
    assert merge_queue.empty(), (
        'No merge request must be enqueued when _plan returns ESCALATED'
    )
    # Critical: the post-PLAN ghost-loop guard MUST NOT have been reached.
    # If it had, _check_branch_on_main would have been awaited (and the
    # truthy return would have set already_on_main=True; combined with
    # has_uncommitted_work=True, the buggy code would have logged "prior
    # merge survived" and tried to flip status to done).
    check_branch.assert_not_called()
    f.wf.git_ops.has_uncommitted_work.assert_not_called()
    # And status must NOT have been flipped to done.
    set_done_calls = [
        c for c in f.set_task_status.await_args_list
        if len(c.args) >= 2 and c.args[1] == 'done'
    ]
    assert not set_done_calls, (
        'Task status must not be flipped to done when _plan returns ESCALATED'
    )
