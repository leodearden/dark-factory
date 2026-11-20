"""How a resumed lane's HEAD relates to its task branch, and attaching the lane
without moving that branch backward or sideways (task 10991).

A lane handed back to a task can sit in three places: on the task branch;
detached at its tip, or past it after an inter-iteration rebase; or on a
commit with nothing to do with it, because a previous occupant or a foreign
checkout left it there.  ``git checkout -B <branch>`` from the third place
resets the task branch to that foreign commit, which is how reify task/10089
lost its work on 2026-11-18.  :func:`head_relation` names the three places so
a caller can refuse the third.

Layering: this module sits BELOW ``orchestrator/src/orchestrator/git_ops.py``,
which imports it.  The git runner is injected, as in
``orchestrator/src/orchestrator/branch_stack.py``, and nothing here imports
git_ops.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from orchestrator.ref_ledger import GitRunner


class HeadRelation(Enum):
    """Where a lane's HEAD stands relative to the branch it is resuming."""

    ON_BRANCH = 'on_branch'            # attached to the branch itself
    EXTENDS_BRANCH = 'extends_branch'  # detached at its tip or a descendant
    FOREIGN = 'foreign'                # anywhere else, or unknown


def _branch_ref(full_branch: str) -> str:
    return f'refs/heads/{full_branch}'


async def branch_tip(run: GitRunner, worktree: Path, full_branch: str) -> str | None:
    """The commit *full_branch* names, or ``None`` when it does not exist."""
    rc, out, _ = await run(
        ['git', 'rev-parse', '--verify', '-q', f'{_branch_ref(full_branch)}^{{commit}}'],
        cwd=worktree,
    )
    return out.strip() if rc == 0 and out.strip() else None


async def describe_head(run: GitRunner, worktree: Path) -> str:
    """``<sha> (<name>)`` for *worktree*'s HEAD, for a message a reader acts on."""
    _, sha, _ = await run(['git', 'rev-parse', 'HEAD'], cwd=worktree)
    _, name, _ = await run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=worktree)
    return f'{sha.strip() or "unknown"} ({name.strip() or "unknown"})'


async def head_relation(
    run: GitRunner, worktree: Path, full_branch: str,
) -> HeadRelation:
    """How *worktree*'s HEAD relates to *full_branch*.

    A branch that does not exist yet is extended by any detached HEAD:
    creating it there drops nothing.  Every answer git could not give,
    ancestry included, is ``FOREIGN``, so the branch is never moved on a guess.
    """
    rc, out, _ = await run(['git', 'symbolic-ref', '-q', 'HEAD'], cwd=worktree)
    if rc == 0:
        if out.strip() == _branch_ref(full_branch):
            return HeadRelation.ON_BRANCH
        return HeadRelation.FOREIGN
    if rc != 1:
        return HeadRelation.FOREIGN
    tip = await branch_tip(run, worktree, full_branch)
    if tip is None:
        return HeadRelation.EXTENDS_BRANCH
    rc, _, _ = await run(['git', 'merge-base', '--is-ancestor', tip, 'HEAD'], cwd=worktree)
    return HeadRelation.EXTENDS_BRANCH if rc == 0 else HeadRelation.FOREIGN
