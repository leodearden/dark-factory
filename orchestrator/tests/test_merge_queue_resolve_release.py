"""Tests for SpeculativeMergeWorker._resolve_and_release — the single
resolve-and-release chokepoint unifying the six _verifier_loop BaseException
handlers (task MQ-refactor zeta / 1991).

Steps covered:
  step-1  RED  — _resolve_and_release contract (SpeculativeItem input,
                 InflightEntry input, release_resources=False,
                 cancel_lease=True, chain_failed=False)
  step-2  GREEN — implement _resolve_and_release
  step-3  RED  — DISPATCH path fault injection (fill + blocking-get)
  step-4  GREEN — route dispatch handlers through the chokepoint
  step-5  RED  — PASSTHROUGH-FINALIZE path fault injection (fill + blocking-get)
  step-6  GREEN — route passthrough-finalize handlers through the chokepoint
  step-7  RED  — FINALIZE-HEAD path fault injection
  step-8  GREEN — route finalize-head handler through the chokepoint
  step-9  RED  — CASCADE path fault injection + release idempotency
  step-10 GREEN — route cascade handler through the chokepoint
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.merge_queue import (
    InflightEntry,
    MergeOutcome,
    MergeRequest,
    SpeculativeItem,
    SpeculativeMergeWorker,
)

# ---------------------------------------------------------------------------
# Fixtures + helpers (per-file duplication convention — see
# test_merge_queue_concurrent_verify.py / test_merge_queue_finalize_head_visibility.py)
# ---------------------------------------------------------------------------


async def _setup_repo(repo: Path) -> None:
    """Initialise a bare git repository with a single commit on main."""
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Create a temporary git repository with an initial commit."""
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
    """Single-host (no verify_runners) OrchestratorConfig."""
    return OrchestratorConfig(project_root=git_repo, git=git_config)


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


def _make_request(
    task_id: str,
    branch: str,
    worktree: Path,
    config: OrchestratorConfig,
) -> MergeRequest:
    """Build a MergeRequest with a fresh Future for the running event loop."""
    return MergeRequest(
        task_id=task_id,
        branch=branch,
        worktree=worktree,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=asyncio.get_running_loop().create_future(),
        lane='normal',
    )


async def _make_real_merged_item(
    git_ops: GitOps,
    config: OrchestratorConfig,
    branch: str,
    filename: str,
    content: str,
    *,
    speculative: bool,
) -> tuple[MergeRequest, SpeculativeItem]:
    """Build a real merged SpeculativeItem backed by a disk `_merge-*` worktree."""
    worktree = await _make_branch_with_file(git_ops, branch, filename, content)
    req = _make_request(branch, branch, worktree, config)
    merge_result = await git_ops.merge_to_main(worktree, branch)
    assert merge_result.success
    base_sha = await git_ops.get_main_sha()
    item = SpeculativeItem(
        request=req,
        merge_result=merge_result,
        merge_wt=merge_result.merge_worktree,
        base_sha=base_sha,
        speculative=speculative,
        skip_verify=False,
    )
    return req, item


# ---------------------------------------------------------------------------
# step-1 RED: _resolve_and_release contract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestResolveAndReleaseContract:
    """Unit-level contract tests for the (not-yet-existing) chokepoint coroutine.

    RED until step-2 GREEN adds SpeculativeMergeWorker._resolve_and_release.
    Every case below fails RED with AttributeError (method missing).
    """

    async def test_speculative_item_input_defaults_release_and_resolve(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """(a) SpeculativeItem input, defaults (release_resources=True,
        cancel_lease=False), chain_failed=True: resolves req.result, cleans +
        deregisters the owned merge worktree, releases the speculation slot
        exactly once, and sets _n_failed=True.
        """
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        req, item = await _make_real_merged_item(
            git_ops, config, 'task/rr-a', 'rr_a.py', 'a = 1\n', speculative=True,
        )
        worker._register_owned_merge_worktree(item.merge_wt)
        assert item.merge_wt in worker._owned_merge_worktrees
        assert item.merge_wt is not None and item.merge_wt.exists()

        depth0 = worker._speculation_slot._value
        outcome = MergeOutcome('blocked', reason='x')

        await worker._resolve_and_release(item, outcome, chain_failed=True)

        assert req.result.done()
        assert req.result.result() is outcome
        assert not item.merge_wt.exists(), 'merge worktree must be removed from disk'
        assert item.merge_wt not in worker._owned_merge_worktrees, (
            'merge worktree must be deregistered from the owned ledger'
        )
        assert worker._speculation_slot._value == depth0 + 1, (
            'speculation slot must be released exactly once for a speculative item'
        )
        assert worker._n_failed is True

    async def test_inflight_entry_release_resources_false_is_a_noop_release(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """(b) InflightEntry input with release_resources=False: resolves
        req.result and sets _n_failed, but performs NO release of the slot,
        worktree, or lease (they were already released by _finalize_inflight's
        finally clause at the real post-finalize call sites).
        """
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        req, item = await _make_real_merged_item(
            git_ops, config, 'task/rr-b', 'rr_b.py', 'b = 1\n', speculative=True,
        )
        worker._register_owned_merge_worktree(item.merge_wt)
        fake_lease = MagicMock()
        entry = InflightEntry(
            item=item,
            lease=fake_lease,
            verify_task=None,
            merge_wt=item.merge_wt,
            was_speculative=True,
            phase='finalizing',
        )
        depth0 = worker._speculation_slot._value
        outcome = MergeOutcome('blocked', reason='y')

        await worker._resolve_and_release(
            entry, outcome, chain_failed=True, release_resources=False,
        )

        assert req.result.done()
        assert req.result.result() is outcome
        assert worker._speculation_slot._value == depth0, 'slot must be untouched'
        assert item.merge_wt is not None and item.merge_wt.exists(), (
            'worktree must be untouched on disk'
        )
        assert item.merge_wt in worker._owned_merge_worktrees, (
            'worktree ledger entry must be untouched'
        )
        assert worker._n_failed is True

    async def test_release_resources_true_cancel_lease_true_uses_cancel_and_release(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """(c) release_resources=True, cancel_lease=True on an InflightEntry
        carrying a fake lease + a stubbed allocator: cancel_and_release(lease)
        is awaited; release(lease) is NOT.
        """
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        req, item = await _make_real_merged_item(
            git_ops, config, 'task/rr-c', 'rr_c.py', 'c = 1\n', speculative=False,
        )
        worker._register_owned_merge_worktree(item.merge_wt)
        fake_lease = MagicMock()
        allocator = MagicMock()
        allocator.cancel_and_release = AsyncMock()
        allocator.release = AsyncMock()
        worker._host_allocator = allocator
        entry = InflightEntry(
            item=item,
            lease=fake_lease,
            verify_task=None,
            merge_wt=item.merge_wt,
            was_speculative=False,
            phase='finalizing',
        )
        outcome = MergeOutcome('blocked', reason='z')

        await worker._resolve_and_release(
            entry, outcome, chain_failed=True,
            release_resources=True, cancel_lease=True,
        )

        allocator.cancel_and_release.assert_awaited_once_with(fake_lease)
        allocator.release.assert_not_awaited()

    async def test_chain_failed_false_leaves_n_failed_unchanged(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """(d) chain_failed=False must not reset _n_failed to False — the
        helper only ever sets it True (guarded `if chain_failed: ... = True`),
        never assigns the raw flag value.
        """
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        worker._n_failed = True  # simulate a prior chain failure
        req, item = await _make_real_merged_item(
            git_ops, config, 'task/rr-d', 'rr_d.py', 'd = 1\n', speculative=False,
        )
        worker._register_owned_merge_worktree(item.merge_wt)
        outcome = MergeOutcome('blocked', reason='w')

        await worker._resolve_and_release(item, outcome, chain_failed=False)

        assert req.result.done()
        assert worker._n_failed is True, (
            'chain_failed=False must not reset _n_failed back to False'
        )
