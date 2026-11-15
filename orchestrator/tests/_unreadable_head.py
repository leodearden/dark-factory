"""A real unreadable worktree HEAD, for tests pinning ``TaskWorkflow._get_head_commit``'s ``''`` sentinel."""

from __future__ import annotations

from pathlib import Path

from orchestrator.git_ops import _run


async def make_head_unreadable(worktree: Path) -> None:
    """Point *worktree*'s HEAD at an unborn branch, so ``git rev-parse HEAD`` exits 128.

    Asserts the setup itself succeeded: a HEAD left readable would let a
    fail-closed test pass without ever reaching the unreadable-HEAD path.
    """
    rc, _, err = await _run(
        ['git', 'symbolic-ref', 'HEAD', 'refs/heads/unborn-head'], cwd=worktree,
    )
    assert rc == 0, err
