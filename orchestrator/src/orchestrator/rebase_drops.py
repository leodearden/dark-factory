"""Commits a rebase silently dropped, recorded under the branch they left (task 4669).

``git rebase`` drops a branch commit whose change the rebase target already
carries, and exits 0.  It says so only on stderr, in free text: "patch
contents already upstream" when replaying the commit leaves it empty, and
"skipped previously applied commit" when it is patch-identical to a target
commit.  Afterwards the branch simply no longer carries the commit, so a
later reader asking "did this branch change file X?" concludes it never did.

This module keeps the fact that it did.  :class:`RebaseDropLedger` holds one
git ref per dropped commit, ``<namespace><full_branch>/<sha>``, pointing at
the dropped commit itself.  A ref is atomic, is shared by every worktree,
survives restarts, and keeps the dropped commit reachable after the reflog
that once held it expires.  It is the same convention as
``orchestrator/src/orchestrator/branch_stack.py::StackBaseLedger``.

Layering: this module sits BELOW ``orchestrator/src/orchestrator/branch_stack.py``
and ``orchestrator/src/orchestrator/git_ops.py``, which both import it.  The
git runner is injected, exactly as
``orchestrator/src/orchestrator/rebase_recovery.py::guarded_abort`` receives
its runner, and nothing here imports git_ops or merge_lane.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from orchestrator.rebase_recovery import AbortRunner

logger = logging.getLogger(__name__)

REBASE_DROP_REF_NAMESPACE = 'refs/dark-factory/rebase-dropped/'
_BRANCH_REF_NAMESPACE = 'refs/heads/'

#: The injected ``(cmd, cwd=...) -> (rc, stdout, stderr)`` runner, the same
#: callable :func:`orchestrator.rebase_recovery.guarded_abort` is handed.
GitRunner = AbortRunner


@dataclass(frozen=True)
class DroppedCommit:
    """A commit a rebase dropped: its pre-rebase sha, its subject, and the
    paths it changed, sorted as git lists them."""

    sha: str
    subject: str
    paths: tuple[str, ...]


class _GitReadError(RuntimeError):
    """A git read this module needed failed."""

    def __init__(self, command: tuple[str, ...], stderr: str) -> None:
        super().__init__(f'{" ".join(command)} failed: {stderr.strip()}')


@dataclass(frozen=True)
class RebaseDropLedger:
    """One ref per dropped commit: ``<namespace><full_branch>/<sha>`` -> sha.

    *repo_root* may be any checkout of the repository: refs outside HEAD and
    the per-worktree namespaces are shared by every worktree.
    """

    repo_root: Path
    run: GitRunner

    def ref_for(self, full_branch: str, sha: str) -> str:
        return f'{REBASE_DROP_REF_NAMESPACE}{full_branch}/{sha}'

    async def record(self, full_branch: str, shas: Iterable[str]) -> None:
        """Record each of *shas* as dropped from *full_branch*.  Never raises."""
        for sha in dict.fromkeys(shas):
            rc, _, err = await self.run(
                ['git', 'update-ref', self.ref_for(full_branch, sha), sha],
                cwd=self.repo_root,
            )
            if rc != 0:
                logger.warning(
                    'Could not record rebase-dropped commit %s of %s: %s',
                    sha, full_branch, err.strip(),
                )

    async def dropped_of(self, full_branch: str) -> tuple[DroppedCommit, ...]:
        """Every commit recorded as dropped from *full_branch*, ordered by sha.

        A failed git read is logged and yields ().
        """
        try:
            shas = await _git_lines(
                self.run, self.repo_root,
                'for-each-ref', '--format=%(objectname)',
                f'{REBASE_DROP_REF_NAMESPACE}{full_branch}/',
            )
            return await _read_dropped_commits(self.run, self.repo_root, sorted(shas))
        except _GitReadError as exc:
            logger.warning(
                'Could not read rebase-dropped commits of %s: %s', full_branch, exc,
            )
            return ()

    async def prune_orphans(self) -> tuple[str, ...]:
        """Drop every record whose branch no longer exists; return those branches.

        Never raises: a failed listing prunes nothing.
        """
        rc, out, err = await self.run(
            [
                'git', 'for-each-ref', '--format=%(refname)',
                REBASE_DROP_REF_NAMESPACE, _BRANCH_REF_NAMESPACE,
            ],
            cwd=self.repo_root,
        )
        if rc != 0:
            logger.warning('Could not list rebase-drop records: %s', err.strip())
            return ()
        orphans = _orphaned_records(set(out.split()))
        for refname in orphans:
            await self._forget(refname)
        return tuple(sorted(set(orphans.values())))

    async def _forget(self, refname: str) -> None:
        rc, _, err = await self.run(['git', 'update-ref', '-d', refname], cwd=self.repo_root)
        if rc != 0:
            logger.debug('Could not clear rebase-drop record %s: %s', refname, err.strip())


def _orphaned_records(refnames: set[str]) -> dict[str, str]:
    """Each record ref whose branch is absent from *refnames*, mapped to that branch."""
    orphans: dict[str, str] = {}
    for refname in refnames:
        if not refname.startswith(REBASE_DROP_REF_NAMESPACE):
            continue
        full_branch = refname.removeprefix(REBASE_DROP_REF_NAMESPACE).rpartition('/')[0]
        if f'{_BRANCH_REF_NAMESPACE}{full_branch}' not in refnames:
            orphans[refname] = full_branch
    return orphans


async def _read_dropped_commits(
    run: GitRunner, cwd: Path, shas: Sequence[str],
) -> tuple[DroppedCommit, ...]:
    """Subject and changed paths of each of *shas*, in the order given.

    Raises :class:`_GitReadError` when any read fails.
    """
    return tuple([await _read_commit(run, cwd, sha) for sha in shas])


async def _read_commit(run: GitRunner, cwd: Path, sha: str) -> DroppedCommit:
    subject = await _git_out(run, cwd, 'show', '-s', '--format=%s', sha)
    paths = await _git_out(
        run, cwd, 'diff-tree', '--no-commit-id', '--name-only', '-r', '-z', '--root', sha,
    )
    return DroppedCommit(
        sha=sha,
        subject=subject.strip(),
        paths=tuple(path for path in paths.split('\x00') if path),
    )


async def _git_lines(run: GitRunner, cwd: Path, *args: str) -> list[str]:
    out = await _git_out(run, cwd, *args)
    return [line.strip() for line in out.splitlines() if line.strip()]


async def _git_out(run: GitRunner, cwd: Path, *args: str) -> str:
    command = ('git', *args)
    rc, out, err = await run(list(command), cwd=cwd)
    if rc != 0:
        raise _GitReadError(command, err)
    return out
