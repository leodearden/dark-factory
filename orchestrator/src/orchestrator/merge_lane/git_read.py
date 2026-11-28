"""Read-only git queries that keep git's own reason when they cannot answer.

``orchestrator/src/orchestrator/git_ops.py::GitOps`` answers a failed read with
``None`` or a message-only :class:`~orchestrator.git_ops.GitQueryFailed`.  The
merge content relation must hand git's stderr to its caller (PRD
``plans/one-merge-request-per-task-prd.md``, INV-2), so these reads raise
:class:`GitReadFailed`, which carries the command and the stderr as attributes.
"""

from __future__ import annotations

from collections.abc import Sequence

from orchestrator.git_ops import GitOps, GitQueryFailed, _run


class GitReadFailed(GitQueryFailed):
    """A read-only git command exited non-zero; its command and stderr are attributes."""

    def __init__(self, command: Sequence[str], stderr: str) -> None:
        super().__init__(f'{" ".join(command)} failed: {stderr.strip()}')
        self.command = tuple(command)
        self.stderr = stderr


async def git_read(
    git_ops: GitOps, command: list[str], *, input_text: str | None = None,
) -> str:
    """*command*'s stdout, run in the project root; a non-zero exit raises :class:`GitReadFailed`."""
    rc, out, err = await _run(command, cwd=git_ops.project_root, input_text=input_text)
    if rc != 0:
        raise GitReadFailed(command, err)
    return out


async def resolve_commit(git_ops: GitOps, rev: str) -> str:
    """The full sha of the commit *rev* names, read once."""
    out = await git_read(git_ops, ['git', 'rev-parse', '--verify', f'{rev}^{{commit}}'])
    return out.strip()


async def is_ancestor_or_raise(
    git_ops: GitOps, *, ancestor: str, descendant: str,
) -> bool:
    """Whether *ancestor* is an ancestor of *descendant*; a git error raises :class:`GitReadFailed`.

    The stderr-keeping twin of
    ``orchestrator/src/orchestrator/git_ops.py::GitOps.is_ancestor_checked``,
    for callers that must report why git could not answer.
    """
    command = ['git', 'merge-base', '--is-ancestor', ancestor, descendant]
    rc, _, err = await _run(command, cwd=git_ops.project_root)
    if rc in (0, 1):
        return rc == 0
    raise GitReadFailed(command, err)


async def second_parents(git_ops: GitOps, commits: Sequence[str]) -> tuple[str, ...]:
    """Each merge commit's second parent, in order, read by one ``git rev-parse``.

    No commits runs no git.  A commit with no second parent makes git exit
    non-zero, which raises :class:`GitReadFailed`, as does an answer whose
    line count is not one per commit.
    """
    if not commits:
        return ()
    command = ['git', 'rev-parse', *(f'{commit}^2' for commit in commits)]
    parents = tuple((await git_read(git_ops, command)).split())
    if len(parents) != len(commits):
        raise GitReadFailed(command, f'expected {len(commits)} parents, read {len(parents)}')
    return parents


__all__ = [
    'GitReadFailed',
    'git_read',
    'is_ancestor_or_raise',
    'resolve_commit',
    'second_parents',
]
