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
