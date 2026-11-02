"""Where a task branch's own work starts, derived at use time.

A write-once base stored at worktree creation goes stale once main is merged
into the branch, so the base is re-derived from git whenever it is needed
(task 5695). This module depends only on ``shared.git_async`` and reads no
config.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from shared.git_async import GitResult, run_git

logger = logging.getLogger(__name__)

_GIT_TIMEOUT_SECS = 120.0


async def own_work_base(checkout: Path, main_branch: str, stored_base: str | None) -> str:
    """The base B for which ``B...HEAD`` in *checkout* spans only the branch's own work."""
    merge_base, head = await asyncio.gather(
        _git(checkout, 'merge-base', 'HEAD', main_branch),
        _git(checkout, 'rev-parse', 'HEAD'),
    )
    fork_point = _sha(merge_base)
    if fork_point and fork_point != _sha(head):
        return fork_point
    if fork_point is None:
        _warn_fork_point_unresolved(checkout, main_branch, merge_base, stored_base)
    return stored_base or main_branch


async def _git(checkout: Path, *args: str) -> GitResult:
    return await run_git(['git', *args], cwd=checkout, timeout=_GIT_TIMEOUT_SECS)


def _sha(result: GitResult) -> str | None:
    return result.stdout if result.ok and result.stdout else None


def _warn_fork_point_unresolved(
    checkout: Path, main_branch: str, merge_base: GitResult, stored_base: str | None,
) -> None:
    fallback = f'the stored base {stored_base}' if stored_base else f'the main branch name {main_branch}'
    logger.warning(
        'git merge-base HEAD %s failed in %s (returncode=%d, timed_out=%s, stderr=%r); '
        'the own-work base falls back to %s',
        main_branch, checkout, merge_base.returncode, merge_base.timed_out,
        merge_base.stderr, fallback,
    )
