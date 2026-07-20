"""Tests for eval worktree snapshot management (cleanup + setup env).

Hermetic: every test drives the snapshots helpers over ``tmp_path`` with the
git boundary either absent (cleanup's ``git worktree remove`` fails on a
non-git tmp dir and is caught) or unexercised (the pure ``read_python_pin`` /
``_eval_setup_env`` helpers). No real git repo, no ``uv sync`` run.
"""

from __future__ import annotations

import pytest

from orchestrator.artifacts import TaskArtifacts


@pytest.mark.asyncio
async def test_cleanup_removes_sibling_relocated_meta_root(tmp_path):
    # BUG 1 GREEN moved the architect plan to the RELOCATED .task-meta/<name>/
    # root — a SIBLING of the worktree, NOT nested under it. `git worktree
    # remove` therefore no longer deletes it, so cleanup_eval_worktree must
    # rmtree the sibling meta root itself or leak one dir per architect run.
    from orchestrator.evals.snapshots import cleanup_eval_worktree

    worktree = tmp_path / 'base' / 'run-abc'
    worktree.mkdir(parents=True)
    meta_root = TaskArtifacts.meta_root_for(worktree.parent, worktree.name)
    assert meta_root == tmp_path / 'base' / '.task-meta' / 'run-abc'
    meta_root.mkdir(parents=True)
    (meta_root / 'plan.json').write_text('{"steps": []}')

    # tmp_path is not a git repo, so the `git worktree remove` raises
    # RuntimeError which cleanup already catches+logs; the sibling meta-root
    # removal runs after that try/except, so no real git repo is needed here.
    await cleanup_eval_worktree(tmp_path, worktree)

    assert not meta_root.exists(), (
        'cleanup_eval_worktree left the sibling relocated meta root behind'
    )
