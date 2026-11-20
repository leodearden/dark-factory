"""Mechanics shared by the orchestrator's branch-keyed git-ref ledgers.

A ref ledger records facts about a branch as git refs under its own
namespace beneath ``refs/dark-factory/``, keyed by the branch's full name:
``orchestrator/src/orchestrator/branch_stack.py::StackBaseLedger`` keeps one
ref per branch, and ``orchestrator/src/orchestrator/rebase_drops.py::RebaseDropLedger``
one ref per dropped commit beneath the branch.  A ref is atomic, is shared by
every worktree, survives restarts, and keeps the commit it names reachable.

The lifetime rules live here, once, so both ledgers obey them alike: a record
lives no longer than its branch (:func:`prune_orphaned_records`), and a
record moves with its branch when the branch is renamed
(:func:`move_branch_records`).  A failed git command in either is logged,
not raised.

Layering: this module sits below both ledgers and imports neither.  The git
runner is injected, exactly as
``orchestrator/src/orchestrator/rebase_recovery.py::guarded_abort`` receives
its runner.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from orchestrator.rebase_recovery import AbortRunner

logger = logging.getLogger(__name__)

BRANCH_REF_NAMESPACE = 'refs/heads/'

#: The injected ``(cmd, cwd=...) -> (rc, stdout, stderr)`` runner.  It is the
#: same callable :func:`orchestrator.rebase_recovery.guarded_abort` is handed,
#: so one type names every use.
GitRunner = AbortRunner


class GitReadError(RuntimeError):
    """A git read a ledger, or a computation over its records, needed failed."""

    def __init__(self, command: tuple[str, ...], stderr: str) -> None:
        super().__init__(f'{" ".join(command)} failed: {stderr.strip()}')
        self.command = command
        self.stderr = stderr


async def git_out(run: GitRunner, cwd: Path, *args: str) -> str:
    """Stdout of ``git <args>`` in *cwd*; :class:`GitReadError` when it fails."""
    command = ('git', *args)
    rc, out, err = await run(list(command), cwd=cwd)
    if rc != 0:
        raise GitReadError(command, err)
    return out


async def git_lines(run: GitRunner, cwd: Path, *args: str) -> list[str]:
    """The non-blank, stripped stdout lines of ``git <args>`` in *cwd*."""
    out = await git_out(run, cwd, *args)
    return [line.strip() for line in out.splitlines() if line.strip()]


async def prune_orphaned_records(
    run: GitRunner,
    repo_root: Path,
    namespace: str,
    branch_of: Callable[[str], str],
) -> tuple[str, ...]:
    """Delete every record under *namespace* whose branch no longer exists.

    *branch_of* maps a record's ref name past *namespace* to the full branch
    name it is keyed on.  Returns the pruned branches, sorted.  A failed
    listing prunes nothing.
    """
    rc, out, err = await run(
        ['git', 'for-each-ref', '--format=%(refname)', namespace, BRANCH_REF_NAMESPACE],
        cwd=repo_root,
    )
    if rc != 0:
        logger.warning('Could not list ref-ledger records under %s: %s', namespace, err.strip())
        return ()
    orphans = _orphaned_records(set(out.split()), namespace, branch_of)
    for refname in sorted(orphans):
        await delete_ref(run, repo_root, refname)
    return tuple(sorted(set(orphans.values())))


def _orphaned_records(
    refnames: set[str], namespace: str, branch_of: Callable[[str], str],
) -> dict[str, str]:
    """Each record ref under *namespace* whose branch is absent from *refnames*, mapped to that branch."""
    orphans: dict[str, str] = {}
    for refname in refnames:
        if not refname.startswith(namespace):
            continue
        branch = branch_of(refname.removeprefix(namespace))
        if f'{BRANCH_REF_NAMESPACE}{branch}' not in refnames:
            orphans[refname] = branch
    return orphans


async def move_branch_records(
    run: GitRunner, repo_root: Path, old_prefix: str, new_prefix: str,
) -> tuple[str, ...]:
    """Re-key every record under *old_prefix* to *new_prefix*; return the moved refs' new names.

    Each record is written under its new name before its old name is deleted,
    so a failure never loses the commit a record names.  A record whose write
    fails stays where it was.
    """
    rc, out, err = await run(
        ['git', 'for-each-ref', '--format=%(refname) %(objectname)', old_prefix],
        cwd=repo_root,
    )
    if rc != 0:
        logger.warning('Could not list ref-ledger records under %s: %s', old_prefix, err.strip())
        return ()
    moved: list[str] = []
    for refname, sha in (line.split() for line in out.splitlines() if line.strip()):
        new_refname = f'{new_prefix}{refname.removeprefix(old_prefix)}'
        rc, _, err = await run(['git', 'update-ref', new_refname, sha], cwd=repo_root)
        if rc != 0:
            logger.warning(
                'Could not move ref-ledger record %s to %s: %s',
                refname, new_refname, err.strip(),
            )
            continue
        await delete_ref(run, repo_root, refname)
        moved.append(new_refname)
    return tuple(moved)


async def delete_ref(run: GitRunner, repo_root: Path, refname: str) -> None:
    """Delete *refname*, if present; a failure is logged at DEBUG."""
    rc, _, err = await run(['git', 'update-ref', '-d', refname], cwd=repo_root)
    if rc != 0:
        logger.debug('Could not delete ref-ledger record %s: %s', refname, err.strip())
