"""Tests for the frozen-prefix / verify-frontier invariant (task ε=1890).

Covers:
  prereq-1     — scaffolding: fixtures + helpers (this file, no test logic)
  step-01 RED  — frozen_prefix() + unfrozen_suffix() accessors
  step-03 RED  — frozen_prefix_tip(main_sha)
  step-05 RED  — check_frozen_prefix_invariant(main_sha) → list[str]
  step-07 RED  — snapshot() exposes additive 'frozen_prefix' key
  step-09 RED  — HEADLINE property: in-flight immutable under suffix reorder;
                 recompute excludes frozen rids (exclusion filter)
  step-11 RED  — _warn_if_verify_base_not_frozen_tip log-only guard

INVARIANT (§5.3): frozen prefix = {items currently verifying} ∪ {landed}.
An item once it enters verify is immutable (never reordered / re-based out
from under an in-flight verify).  A verify may only start against a base that
is the tip of the frozen prefix (no verify against a speculative-only base).
The unfrozen suffix may be reordered / inserted freely; any reorder recomputes
the merge-tree for the affected suffix ONLY.

Mapping:
  {verifying} = self._inflight entries with verify_task is not None
                + self._finalizing_head when phase in {'verifying',
                  'gate_reverify', 'finalizing'}
  {landed}    = main (its SHA reflects landed items)
  unfrozen    = self._lane_buffers (pick order high→normal, FIFO)
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, MergeResult, _run
from orchestrator.merge_queue import (
    EMPTY_SUFFIX_CONFLICT_GRAPH,
    InflightEntry,
    MergeRequest,
    SpeculativeItem,
    SpeculativeMergeWorker,
    SuffixConflictGraph,
)

# ── Module-level sentinel for verify_task in pure unit tests ─────────────────
#
# _frozen_inflight_entries() only checks `e.verify_task is not None`, so any
# non-None object doubles as a verifying-entry sentinel in tests that do not
# need a real asyncio.Task (no event-loop required for the accessor methods
# themselves).  The type annotation is unenforced at runtime.
_SENTINEL_VERIFY_TASK = object()  # noqa: PD901 (not a dataframe)


# ── Fixtures (mirrored from test_merge_queue_conflict_graph.py) ───────────────


async def _setup_repo(repo: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    (repo / 'shared.txt').write_text('line1\nline2\nline3\n')
    (repo / 'disjoint.txt').write_text('aaa\nbbb\nccc\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_setup_repo(repo))
    return repo


@pytest.fixture
def git_config() -> GitConfig:
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
    )


@pytest.fixture
def git_ops(git_config: GitConfig, git_repo: Path) -> GitOps:
    return GitOps(git_config, git_repo)


@pytest.fixture
def config(git_repo: Path, git_config: GitConfig) -> OrchestratorConfig:
    return OrchestratorConfig(project_root=git_repo, git=git_config)


# ── Module-level helpers ──────────────────────────────────────────────────────


def _make_req(
    task_id: str,
    branch: str,
    config: OrchestratorConfig,
    git_repo: Path,
    lane: str = 'normal',
) -> MergeRequest:
    """Build a minimal MergeRequest with a fresh event-loop future.

    Must be called from within an async context (asyncio.get_running_loop()).
    """
    return MergeRequest(
        task_id=task_id,
        branch=branch,
        worktree=git_repo,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=asyncio.get_running_loop().create_future(),
        lane=lane,
    )


def _make_worker(git_ops: GitOps) -> SpeculativeMergeWorker:
    """Build a bare SpeculativeMergeWorker for unit tests (no harness wiring)."""
    return SpeculativeMergeWorker(git_ops, asyncio.Queue())


async def _create_branch_editing(
    repo: Path,
    branch_name: str,
    filename: str,
    content: str,
) -> str:
    """Create a branch that edits filename with content; return the branch SHA."""
    await _run(['git', 'checkout', '-b', branch_name], cwd=repo)
    (repo / filename).write_text(content)
    await _run(['git', 'add', filename], cwd=repo)
    await _run(['git', 'commit', '-m', f'Edit {filename} in {branch_name}'], cwd=repo)
    _, sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=repo)
    await _run(['git', 'checkout', 'main'], cwd=repo)
    return sha.strip()


async def _make_merged_item(
    git_ops: GitOps,
    config: OrchestratorConfig,
    branch: str,
    filename: str,
    content: str,
    *,
    base_sha: str | None = None,
) -> tuple[MergeRequest, SpeculativeItem]:
    """Create a branch, merge it to main (or onto base_sha), return (req, item).

    Mirrors test_merge_queue_concurrent_verify.py _make_merged_item.
    base_sha allows speculative stacking: pass a prior item's merge_commit.
    """
    worktree = (await git_ops.create_worktree(branch)).path
    (worktree / filename).write_text(content)
    await git_ops.commit(worktree, f'Add {filename}')

    loop = asyncio.get_running_loop()
    req = MergeRequest(
        task_id=branch,
        branch=branch,
        worktree=worktree,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=loop.create_future(),
        lane='normal',
    )
    merge_result = await git_ops.merge_to_main(worktree, branch, base_sha)
    assert merge_result.success and merge_result.merge_commit, (
        f'merge_to_main failed for branch {branch!r}: {merge_result}'
    )
    item_base_sha = base_sha if base_sha is not None else await git_ops.get_main_sha()
    # Correct: for a non-speculative merge (base_sha=None), base_sha is main
    # BEFORE the merge.  merge_to_main does NOT advance main (advance_main
    # does that later in the real pipeline), so get_main_sha() here still
    # returns the pre-merge SHA.  For speculative merges, caller supplies
    # base_sha explicitly.
    item = SpeculativeItem(
        request=req,
        merge_result=merge_result,
        merge_wt=merge_result.merge_worktree,
        base_sha=item_base_sha,
        speculative=base_sha is not None,
        skip_verify=False,
    )
    return req, item


def _make_fake_item(
    task_id: str,
    *,
    base_sha: str = 'aaaa0000',
    merge_commit: str | None = 'bbbb1111',
    config: OrchestratorConfig,
    git_repo: Path,
) -> tuple[MergeRequest, SpeculativeItem]:
    """Build a (MergeRequest, SpeculativeItem) pair from fake SHAs.

    Does NOT create real git branches — suitable for pure-unit tests that
    only exercise the accessor methods (frozen_prefix, frozen_prefix_tip,
    check_frozen_prefix_invariant) without needing actual git operations.
    Must be called from an async context (for MergeRequest.result future).
    """
    req = _make_req(task_id, f'task/{task_id}', config, git_repo)
    fake_merge_result = MergeResult(
        success=True,
        merge_commit=merge_commit,
    ) if merge_commit is not None else None
    item = SpeculativeItem(
        request=req,
        merge_result=fake_merge_result,
        merge_wt=None,
        base_sha=base_sha,
        speculative=False,
        skip_verify=False,
    )
    return req, item


def _make_inflight_entry(
    item: SpeculativeItem,
    *,
    verifying: bool = True,
) -> InflightEntry:
    """Build an InflightEntry for unit tests.

    verifying=True  → verify_task set to _SENTINEL_VERIFY_TASK (any non-None
                       object; frozen_prefix() only checks `is not None`).
    verifying=False → verify_task=None (passthrough entry; excluded from the
                       frozen prefix by §5.3 definition).
    """
    return InflightEntry(
        item=item,
        lease=None,
        verify_task=_SENTINEL_VERIFY_TASK if verifying else None,  # type: ignore[arg-type]
        merge_wt=None,
        was_speculative=False,
        phase='verifying',
    )


# ── step-01 RED: frozen_prefix() + unfrozen_suffix() ─────────────────────────


@pytest.mark.asyncio
class TestFrozenAndUnfrozenAccessors:
    """frozen_prefix() and unfrozen_suffix() partition the pipeline state.

    RED until step-02 GREEN adds the methods to SpeculativeMergeWorker.
    """

    async def test_bare_worker_frozen_prefix_empty(
        self, git_ops: GitOps,
    ) -> None:
        """A bare worker has no in-flight entries → frozen_prefix() == ()."""
        worker = _make_worker(git_ops)
        assert worker.frozen_prefix() == ()

    async def test_bare_worker_unfrozen_suffix_empty(
        self, git_ops: GitOps,
    ) -> None:
        """A bare worker has no lane-buffer items → unfrozen_suffix() == ()."""
        worker = _make_worker(git_ops)
        assert worker.unfrozen_suffix() == ()

    async def test_two_verifying_inflight_in_frozen_prefix(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """Two verifying InflightEntry items → frozen_prefix() == (rid_a, rid_b)."""
        worker = _make_worker(git_ops)
        _, item_a = _make_fake_item('t-a', base_sha='aaa0', merge_commit='aaa1',
                                    config=config, git_repo=git_repo)
        _, item_b = _make_fake_item('t-b', base_sha='aaa1', merge_commit='aaa2',
                                    config=config, git_repo=git_repo)
        entry_a = _make_inflight_entry(item_a, verifying=True)
        entry_b = _make_inflight_entry(item_b, verifying=True)
        worker._inflight.append(entry_a)
        worker._inflight.append(entry_b)

        fp = worker.frozen_prefix()
        assert fp == (item_a.request.request_id, item_b.request.request_id)

    async def test_passthrough_excluded_from_frozen_prefix(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """A passthrough entry (verify_task=None) is EXCLUDED from frozen_prefix()."""
        worker = _make_worker(git_ops)
        _, item_a = _make_fake_item('t-a', base_sha='aaa0', merge_commit='aaa1',
                                    config=config, git_repo=git_repo)
        _, item_pass = _make_fake_item('t-pass', base_sha='aaa1', merge_commit=None,
                                       config=config, git_repo=git_repo)
        entry_a = _make_inflight_entry(item_a, verifying=True)
        entry_pass = _make_inflight_entry(item_pass, verifying=False)  # passthrough
        worker._inflight.append(entry_a)
        worker._inflight.append(entry_pass)

        fp = worker.frozen_prefix()
        assert item_a.request.request_id in fp
        assert item_pass.request.request_id not in fp

    async def test_lane_buffers_appear_in_unfrozen_suffix(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """Requests in _lane_buffers appear in unfrozen_suffix(), high before normal."""
        worker = _make_worker(git_ops)
        req_hi = _make_req('t-hi', 'task/t-hi', config, git_repo, lane='high')
        req_n1 = _make_req('t-n1', 'task/t-n1', config, git_repo)
        req_n2 = _make_req('t-n2', 'task/t-n2', config, git_repo)
        worker._lane_buffers['high'].append(req_hi)
        worker._lane_buffers['normal'].append(req_n1)
        worker._lane_buffers['normal'].append(req_n2)

        us = worker.unfrozen_suffix()
        # high lane before normal
        hi_idx = us.index(req_hi.request_id)
        n1_idx = us.index(req_n1.request_id)
        n2_idx = us.index(req_n2.request_id)
        assert hi_idx < n1_idx < n2_idx

    async def test_frozen_and_unfrozen_are_disjoint(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """frozen_prefix() and unfrozen_suffix() must be disjoint sets."""
        worker = _make_worker(git_ops)
        _, item_a = _make_fake_item('t-a', base_sha='aaa0', merge_commit='aaa1',
                                    config=config, git_repo=git_repo)
        _, item_b = _make_fake_item('t-b', config=config, git_repo=git_repo)
        entry_a = _make_inflight_entry(item_a, verifying=True)
        worker._inflight.append(entry_a)
        req_b = item_b.request
        worker._lane_buffers['normal'].append(req_b)

        fp = set(worker.frozen_prefix())
        us = set(worker.unfrozen_suffix())
        assert fp & us == set(), f'frozen ∩ unfrozen should be empty, got {fp & us}'
