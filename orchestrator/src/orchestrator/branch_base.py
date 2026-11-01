"""Where a task branch's own work starts, derived at use time.

A write-once base stored at worktree creation goes stale once main is merged
into the branch, so the base is re-derived from git whenever it is needed
(task 5695). This module depends only on ``shared.git_async`` and reads no
config.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from shared.git_async import run_git

_GIT_TIMEOUT_SECS = 120.0


async def own_work_base(checkout: Path, main_branch: str, stored_base: str | None) -> str:
    """The base B for which ``B...HEAD`` in *checkout* spans only the branch's own work."""
    fork_point, head = await asyncio.gather(
        _git_sha(checkout, ['git', 'merge-base', 'HEAD', main_branch]),
        _git_sha(checkout, ['git', 'rev-parse', 'HEAD']),
    )
    if fork_point and fork_point != head:
        return fork_point
    return stored_base or main_branch


async def _git_sha(checkout: Path, cmd: list[str]) -> str | None:
    result = await run_git(cmd, cwd=checkout, timeout=_GIT_TIMEOUT_SECS)
    return result.stdout if result.ok and result.stdout else None
