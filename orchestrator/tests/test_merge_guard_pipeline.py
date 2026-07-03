"""Tests for the shared guard pipeline ``classify_and_merge`` (MQ-refactor
kappa / task 1995).

Extracts the duplicated pre-merge guard+merge+drop-guard pipeline shared by
``MergeWorker._do_merge``, ``SpeculativeMergeWorker._merger_loop`` (inline
body), and ``SpeculativeMergeWorker._remerge`` into one module-level async
function, ``classify_and_merge(worker, req, base_sha, *, speculative,
started_monotonic) -> MergedOk | Decided``, and routes all three consumers
through it.

Steps covered (TDD order):
  prereq  —    re-anchor line numbers + this fixture scaffold (no prod change)
  step-0  RED   — MergedOk / Decided value-type tests
  step-2  GREEN — add MergedOk / Decided dataclasses + merge_queue re-export
  step-3  RED   — classify_and_merge guard matrix on SpeculativeMergeWorker
  step-4  GREEN — implement classify_and_merge
  step-5  GREEN — adopt classify_and_merge in _merger_loop
  step-6  RED   — classify_and_merge guard matrix on MergeWorker (serial)
  step-7  GREEN — capability-gate SpeculativeMergeWorker-only behaviour
  step-8  GREEN — adopt classify_and_merge in _do_merge
  step-9  RED   — parameterized path-equivalence (merger loop vs _remerge)
  step-10 GREEN — wire _remerge through classify_and_merge
  step-11 GREEN — finalize: docs + full-suite + frozen-surface check

This module reuses the standard per-file fixture quartet (git_repo/git_ops/
git_config/config) and helper functions (_make_request, _make_branch_with_file,
_make_event_store, _count_events) copied from test_merge_queue.py — there is
no shared conftest git_ops fixture; per-file duplication is the established
convention (see test_merge_queue_resource_audit.py's module docstring).
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from pathlib import Path
from typing import Literal

import pytest
from _orch_helpers import make_placeholder_future

from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.event_store import EventStore
from orchestrator.git_ops import GitOps, MergeResult, _run
from orchestrator.merge_queue import (
    DROPPED_PLAN_TARGETS_REASON_PREFIX,
    Decided,
    MergedOk,
    MergeOutcome,
    MergeRequest,
    SpeculativeMergeWorker,
)

# ---------------------------------------------------------------------------
# Fixtures (copied from test_merge_queue.py:62-101 — per-file duplication
# convention, no shared conftest git_ops fixture)
# ---------------------------------------------------------------------------


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Create a temporary git repository with an initial commit."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_setup_repo(repo))
    return repo


async def _setup_repo(repo: Path):
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def git_config() -> GitConfig:
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        # Tests use a tmp repo with no real remote; disabling the push avoids
        # per-test subprocess noise.
        push_after_advance=False,
    )


@pytest.fixture
def git_ops(git_config: GitConfig, git_repo: Path) -> GitOps:
    return GitOps(git_config, git_repo)


@pytest.fixture
def config(git_repo: Path, git_config: GitConfig) -> OrchestratorConfig:
    return OrchestratorConfig(project_root=git_repo, git=git_config)


# ---------------------------------------------------------------------------
# Helpers (copied from test_merge_queue.py: _make_request:104,
# _make_branch_with_file:2313, _make_event_store:11719, _count_events:11698)
# ---------------------------------------------------------------------------


def _make_request(
    task_id: str,
    branch: str,
    worktree: Path,
    config: OrchestratorConfig,
    pre_rebased: bool = False,
    lane: Literal['normal', 'high'] = 'normal',
    merge_first_enqueued_at: float | None = None,
    request_id: str | None = None,
) -> MergeRequest:
    try:
        future: asyncio.Future[MergeOutcome] = asyncio.get_running_loop().create_future()
    except RuntimeError:
        future = make_placeholder_future()
    kwargs: dict = dict(
        task_id=task_id,
        branch=branch,
        worktree=worktree,
        pre_rebased=pre_rebased,
        task_files=None,
        module_configs=[],
        config=config,
        result=future,
        lane=lane,
        merge_first_enqueued_at=merge_first_enqueued_at,
    )
    if request_id is not None:
        kwargs['request_id'] = request_id
    return MergeRequest(**kwargs)


async def _make_branch_with_file(
    git_ops: GitOps,
    branch_name: str,
    filename: str,
    content: str,
) -> Path:
    """Create a worktree branch with one committed file and return its path."""
    worktree = (await git_ops.create_worktree(branch_name)).path
    (worktree / filename).write_text(content)
    await git_ops.commit(worktree, f'Add {filename}')
    return worktree


def _make_event_store(tmp_path: Path) -> EventStore:
    """Create an EventStore backed by a tmp sqlite db.

    Module-level (rather than test_merge_queue.py's per-class-method copy)
    since this file has a single fixture scope — see module docstring.
    """
    db = tmp_path / 'guard_pipeline_events.db'
    return EventStore(db_path=db, run_id='guard-pipeline-test')


def _count_events(db_path, event_type: str) -> int:
    """Query the EventStore SQLite DB for a specific event type count."""
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            'SELECT COUNT(*) FROM events WHERE event_type = ?', (event_type,),
        ).fetchone()
        return row[0] if row else 0
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# step-0 (RED): MergedOk / Decided value types.
#
# MergedOk/Decided don't exist yet — imported LOCALLY inside each test
# (mirrors test_merge_queue_resource_audit.py / test_merge_request_ledger.py
# convention) so the not-yet-implemented symbols don't break collection of
# the rest of this file across the incremental RED/GREEN steps that follow.
# ---------------------------------------------------------------------------


class TestMergedOkDecidedTypes:
    """Value-type tests for MergedOk / Decided (step-0 / step-2, task 1995)."""

    def test_merged_ok_exposes_merge_result_wt_and_branch_tip(self):
        from orchestrator.merge_types import MergedOk

        mr = MergeResult(success=True, conflicts=False, details='ok')
        ok = MergedOk(merge_result=mr, merge_wt=Path('/x'), branch_tip='abc123')

        assert ok.merge_result is mr
        assert ok.merge_wt == Path('/x')
        assert ok.branch_tip == 'abc123'

    def test_decided_exposes_outcome_and_defaults_merge_result_to_none(self):
        from orchestrator.merge_types import Decided

        d = Decided(outcome=MergeOutcome('conflict', conflict_details='x'))

        assert d.outcome.status == 'conflict'
        assert d.outcome.conflict_details == 'x'
        assert d.merge_result is None

    def test_decided_carries_merge_result_when_explicitly_set(self):
        from orchestrator.merge_types import Decided

        mr = MergeResult(success=False, conflicts=True, details='c')
        d = Decided(outcome=MergeOutcome('blocked'), merge_result=mr)

        assert d.merge_result is mr

    def test_merged_ok_and_decided_reexported_from_merge_queue(self):
        """`from orchestrator.merge_queue import MergedOk, Decided` must work
        and resolve to the SAME objects defined in merge_types.py (shim, not
        a parallel redefinition) — mirrors the existing MergeOutcome/
        SpeculativeItem re-export pattern."""
        from orchestrator.merge_queue import Decided as Decided_via_queue
        from orchestrator.merge_queue import MergedOk as MergedOk_via_queue
        from orchestrator.merge_types import Decided as Decided_via_types
        from orchestrator.merge_types import MergedOk as MergedOk_via_types

        assert MergedOk_via_queue is MergedOk_via_types
        assert Decided_via_queue is Decided_via_types
