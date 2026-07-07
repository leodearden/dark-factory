"""Tests for the journal-first already-merged guard collapse (task 2245 / PRD α).

``MergeProvenance.lookup(task_id)`` (task 2153 / W1 α) is the single, authoritative
source consulted by every already-merged guard before falling back to the legacy
``_has_prior_implementation`` heuristic. See ``plans/workflow-state-machine-prd.md``
Contract §8 (MP-1: journal-first; MP-2: no recovery-DONE without a provenance basis).

``MergeProvenance._outbox`` is a process-global set via ``MergeProvenance.bind`` —
the autouse ``_reset_merge_provenance`` fixture resets it before and after every
test in this module so a bound outbox never leaks into another test.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _orch_helpers import pydantic_spec

from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import OrchestratorConfig
from orchestrator.landed_outbox import LandedOutbox, LandedRow, MergeProvenance
from orchestrator.scheduler import SetTaskStatusRejected
from orchestrator.workflow import (
    TaskWorkflow,
    WorkflowOutcome,
    WorkflowState,
    _PriorImplStatus,
    _RecoveryDecision,
)


@dataclass
class _Fixture:
    wf: TaskWorkflow
    artifacts: TaskArtifacts
    set_task_status: AsyncMock
    mark_done: AsyncMock
    update_task: AsyncMock
    is_ancestor: AsyncMock
    get_main_sha: AsyncMock


def _make(
    *,
    worktree: Path,
    project_root: Path,
    task_id: str = '50',
    branch_on_main: bool = True,
    main_sha: str = 'mainsha123',
) -> _Fixture:
    assignment = MagicMock()
    assignment.task_id = task_id
    assignment.task = {'id': task_id, 'title': 'T', 'description': 'd'}
    assignment.modules = []

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
    # Fix 1 (mirrors test_workflow_already_done.py): workflow refreshes
    # metadata.files via update_task before set_task_status('done').
    scheduler.update_task = AsyncMock(return_value=True)
    scheduler.get_status = AsyncMock(return_value=None)

    # Forward mark_done into set_task_status so assertions can observe the
    # (task_id, 'done', done_provenance=...) call shape either way.
    async def _fake_mark_done(tid, *, kind, sha, note=None):
        provenance: dict = {'kind': kind, 'commit': sha}
        if note is not None:
            provenance['note'] = note
        await set_task_status(tid, 'done', done_provenance=provenance)
    mark_done = AsyncMock(side_effect=_fake_mark_done)
    scheduler.mark_done = mark_done

    is_ancestor = AsyncMock(return_value=branch_on_main)
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
        mark_done=mark_done,
        update_task=scheduler.update_task,
        is_ancestor=is_ancestor,
        get_main_sha=get_main_sha,
    )


def _bind_landed_row(tmp_path: Path, *, task_id: str, advanced_sha: str) -> None:
    """Bind a real LandedOutbox (via MergeProvenance.bind) holding a row for *task_id*."""
    outbox = LandedOutbox(tmp_path / 'landed.json')
    outbox.record(LandedRow(
        task_id=task_id, branch_tip_sha='branchtip', advanced_sha=advanced_sha,
        landed_at=1.0,
    ))
    MergeProvenance.bind(outbox)


@pytest.fixture(autouse=True)
def _reset_merge_provenance():
    """MergeProvenance._outbox is a process-global — never leak a bound outbox."""
    MergeProvenance._outbox = None
    yield
    MergeProvenance._outbox = None


# ---------------------------------------------------------------------------
# Tests: TaskWorkflow._resolve_already_merged
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestResolveAlreadyMerged:
    """Unit tests for TaskWorkflow._resolve_already_merged() (PRD α, MP-1).

    Pure decision function: journal hit is authoritative and short-circuits
    before the legacy heuristic is ever consulted; a journal miss falls back
    to ``_has_prior_implementation``.
    """

    async def test_journal_hit_returns_done_without_consulting_fallback(
        self, tmp_path: Path,
    ):
        f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj')
        _bind_landed_row(tmp_path, task_id=f.wf.task_id, advanced_sha='advancedsha123')
        f.wf._has_prior_implementation = MagicMock(  # type: ignore[method-assign]
            side_effect=AssertionError(
                '_has_prior_implementation must not be called on a journal hit',
            ),
        )

        decision = f.wf._resolve_already_merged(wt_head='wthead123')

        assert decision == _RecoveryDecision(
            done=True, basis='journal', sha='advancedsha123',
        )

    async def test_journal_miss_with_prior_work_returns_fallback(
        self, tmp_path: Path,
    ):
        f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj')
        f.wf._has_prior_implementation = MagicMock(  # type: ignore[method-assign]
            return_value=_PriorImplStatus(has_work=True, entries=[], base_commit=None),
        )

        decision = f.wf._resolve_already_merged(wt_head='wthead123')

        assert decision == _RecoveryDecision(done=True, basis='fallback', sha=None)
        f.wf._has_prior_implementation.assert_called_once_with(wt_head='wthead123')

    async def test_journal_miss_with_no_prior_work_returns_no_recovery(
        self, tmp_path: Path,
    ):
        f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj')
        f.wf._has_prior_implementation = MagicMock(  # type: ignore[method-assign]
            return_value=_PriorImplStatus(has_work=False, entries=[], base_commit=None),
        )

        decision = f.wf._resolve_already_merged(wt_head='wthead123')

        assert decision == _RecoveryDecision(done=False, basis=None, sha=None)


# ---------------------------------------------------------------------------
# Tests: TaskWorkflow._finalise_recovery_done
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFinaliseRecoveryDone:
    """Unit tests for TaskWorkflow._finalise_recovery_done() (PRD α, MP-2 chokepoint).

    The sole writer of ``self._merge_recovery_basis`` — every already-merged
    guard's only route to a recovery-DONE goes through this method.
    """

    async def test_journal_basis_sets_marker_and_marks_done(self, tmp_path: Path):
        f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj')

        outcome = await f.wf._finalise_recovery_done(
            basis='journal', sha='advancedsha123', kind='merged',
            note='landed-outbox journal hit (pre-PLAN recovery)',
        )

        assert outcome == WorkflowOutcome.DONE
        assert f.wf._merge_recovery_basis == 'journal'
        assert f.wf.state == WorkflowState.DONE
        f.mark_done.assert_awaited_once_with(
            f.wf.task_id, kind='merged', sha='advancedsha123',
            note='landed-outbox journal hit (pre-PLAN recovery)',
        )
        f.set_task_status.assert_awaited_once()
        args, kwargs = f.set_task_status.await_args
        assert args[0] == f.wf.task_id
        assert args[1] == 'done'
        assert kwargs['done_provenance']['kind'] == 'merged'
        assert kwargs['done_provenance']['commit'] == 'advancedsha123'

    async def test_fallback_basis_sets_marker_and_marks_done(self, tmp_path: Path):
        f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj')

        outcome = await f.wf._finalise_recovery_done(
            basis='fallback', sha='mainsha123', kind='found_on_main',
            note='branch already on main at workflow start (pre-PLAN recovery)',
        )

        assert outcome == WorkflowOutcome.DONE
        assert f.wf._merge_recovery_basis == 'fallback'
        assert f.wf.state == WorkflowState.DONE
        f.mark_done.assert_awaited_once_with(
            f.wf.task_id, kind='found_on_main', sha='mainsha123',
            note='branch already on main at workflow start (pre-PLAN recovery)',
        )

    async def test_mark_done_rejection_routes_to_mark_blocked_not_phantom_done(
        self, tmp_path: Path,
    ):
        """A rejected mark_done must route to _mark_blocked, not report DONE."""
        f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj')
        rejection = SetTaskStatusRejected(
            task_id=f.wf.task_id, error_code='conflict', raw='row already terminal',
        )
        f.wf.scheduler.mark_done = AsyncMock(side_effect=rejection)
        mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
        f.wf._mark_blocked = mark_blocked  # type: ignore[method-assign]

        outcome = await f.wf._finalise_recovery_done(
            basis='fallback', sha='mainsha123', kind='found_on_main',
            note='fallback note',
        )

        assert outcome == WorkflowOutcome.BLOCKED
        mark_blocked.assert_awaited_once()
        assert mark_blocked.await_args is not None
        _args, kwargs = mark_blocked.await_args
        assert kwargs.get('escalate_to_human') is True


# ---------------------------------------------------------------------------
# Tests: TaskWorkflow._recover_if_already_merged (Guard 1, rewired journal-first)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRecoverIfAlreadyMerged:
    """Unit tests for the pre-PLAN guard rewired onto the journal-first lookup.

    ``_check_branch_on_main`` is stubbed directly (mirrors
    ``test_workflow.py``'s ``wf._check_branch_on_main = AsyncMock(...)``
    convention) so these stay unit tests of the guard's decision logic —
    no real git subprocess involved.
    """

    async def test_journal_hit_returns_done_without_consulting_git_or_fallback(
        self, tmp_path: Path,
    ):
        """A journal hit is authoritative — short-circuits before the
        git-layer probe and the legacy heuristic are ever consulted [row 1]."""
        f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj')
        _bind_landed_row(tmp_path, task_id=f.wf.task_id, advanced_sha='advancedsha123')
        f.wf._check_branch_on_main = AsyncMock(  # type: ignore[method-assign]
            side_effect=AssertionError(
                '_check_branch_on_main must not be called on a journal hit',
            ),
        )
        f.wf._has_prior_implementation = MagicMock(  # type: ignore[method-assign]
            side_effect=AssertionError(
                '_has_prior_implementation must not be called on a journal hit',
            ),
        )

        outcome = await f.wf._recover_if_already_merged()

        assert outcome == WorkflowOutcome.DONE
        assert f.wf._merge_recovery_basis == 'journal'
        f.mark_done.assert_awaited_once_with(
            f.wf.task_id, kind='merged', sha='advancedsha123',
            note='landed-outbox journal hit (pre-PLAN recovery)',
        )

    async def test_journal_miss_on_main_with_prior_work_falls_back_to_found_on_main(
        self, tmp_path: Path,
    ):
        """Journal miss + on-main + prior implementation work → fallback DONE."""
        f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj')
        f.wf._check_branch_on_main = AsyncMock(  # type: ignore[method-assign]
            return_value=('wthead123', 'mainsha123'),
        )
        f.wf._has_prior_implementation = MagicMock(  # type: ignore[method-assign]
            return_value=_PriorImplStatus(has_work=True, entries=[], base_commit=None),
        )

        outcome = await f.wf._recover_if_already_merged()

        assert outcome == WorkflowOutcome.DONE
        assert f.wf._merge_recovery_basis == 'fallback'
        f.mark_done.assert_awaited_once_with(
            f.wf.task_id, kind='found_on_main', sha='mainsha123',
            note='branch already on main at workflow start (pre-PLAN recovery)',
        )
        f.wf._has_prior_implementation.assert_called_once_with(wt_head='wthead123')

    async def test_journal_miss_on_main_with_no_prior_work_returns_none(
        self, tmp_path: Path,
    ):
        """Journal miss + on-main + no prior work → no recovery (rows 2-3:
        inherited .task/ contamination / rebased-HEAD==base must not false-done)."""
        f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj')
        f.wf._check_branch_on_main = AsyncMock(  # type: ignore[method-assign]
            return_value=('wthead123', 'mainsha123'),
        )
        f.wf._has_prior_implementation = MagicMock(  # type: ignore[method-assign]
            return_value=_PriorImplStatus(has_work=False, entries=[], base_commit=None),
        )

        outcome = await f.wf._recover_if_already_merged()

        assert outcome is None
        assert f.wf._merge_recovery_basis is None
        f.mark_done.assert_not_awaited()

    async def test_journal_miss_not_on_main_returns_none(self, tmp_path: Path):
        """Journal miss + branch not on main → no recovery, no fallback probe."""
        f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj')
        f.wf._check_branch_on_main = AsyncMock(return_value=None)  # type: ignore[method-assign]
        f.wf._has_prior_implementation = MagicMock(  # type: ignore[method-assign]
            side_effect=AssertionError(
                '_has_prior_implementation must not be called when not on main',
            ),
        )

        outcome = await f.wf._recover_if_already_merged()

        assert outcome is None
        assert f.wf._merge_recovery_basis is None
        f.mark_done.assert_not_awaited()
