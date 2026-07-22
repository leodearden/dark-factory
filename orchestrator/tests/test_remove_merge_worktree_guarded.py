"""Real-git tests for GitOps.remove_merge_worktree_guarded.

PRD: docs/prds/merge-worktree-lifecycle-integrity.md, task alpha (C1).

remove_merge_worktree_guarded is the lease-enforced removal primitive that
closes the incident's 23s TOCTOU: a non-blocking try-acquire of the tree's
merge-verify flock is HELD across the ``git worktree remove`` (acquire-then-
remove, never check-then-remove). This module pins the full outcome
vocabulary — 'removed', 'skipped_lease_held', 'skipped_persistent',
'not_present', 'failed' — via real git worktrees (no mocking of git itself),
mirroring test_inflight_verify_merge_lease.py's fixture pattern. No test
bodies yet — this scaffolding adds only the shared real-git fixtures and
helpers each subsequent step's test function builds on.
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import pytest

from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.verify_cancel import (
    acquire_merge_verify_flock,
    lane_lock_path,
    release_merge_verify_flock,
    remove_lock_holder_pgid,
    write_lock_holder_pgid,
)

# ---------------------------------------------------------------------------
# Real-git fixtures (adapted from test_inflight_verify_merge_lease.py:39-76)
# ---------------------------------------------------------------------------


async def _setup_repo(repo: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


async def _head_sha(repo: Path) -> str:
    rc, out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=repo)
    assert rc == 0
    return out.strip()


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_setup_repo(repo))
    return repo


@pytest.fixture
def git_ops(git_repo: Path) -> GitOps:
    git_config = GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
    )
    return GitOps(git_config, git_repo)


async def _make_ephemeral_worktree(git_ops: GitOps) -> Path:
    """Build a real ephemeral ``_merge-<uuid>`` worktree at the repo's HEAD."""
    return await git_ops.create_throwaway_verify_worktree(await _head_sha(git_ops.project_root))
