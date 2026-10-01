"""Branch-stack provenance: which off-main base a branch was stacked onto, and
which of its commits are therefore not its own (task 5618).

The train stacker rebases a member onto its predecessor's tip.  If that
predecessor later leaves without landing, the member still carries the
predecessor's commits below its own, and every later consumer would treat
``main..branch`` as the member's own work.  This module keeps the one fact
that makes those commits separable again — the base the branch was stacked
onto, recorded at the moment of stacking — and computes the cut between the
foreign prefix and the branch's own delta.

The record is one git ref per stacked branch under
:data:`STACK_BASE_REF_NAMESPACE`.  A ref is atomic, survives restarts, is
shared by every worktree, and keeps a never-landed base reachable.

Layering: this module sits BELOW ``orchestrator/src/orchestrator/git_ops.py``,
which imports it.  The git runner is therefore injected, exactly as
``orchestrator/src/orchestrator/rebase_recovery.py::guarded_abort`` receives
its runner, and nothing here imports git_ops or merge_queue.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from orchestrator.rebase_recovery import AbortRunner

logger = logging.getLogger(__name__)

STACK_BASE_REF_NAMESPACE = 'refs/dark-factory/stack-base/'

#: The injected ``(cmd, cwd=...) -> (rc, stdout, stderr)`` runner.  It is the
#: same callable :func:`orchestrator.rebase_recovery.guarded_abort` is handed,
#: so one type names both uses.
GitRunner = AbortRunner


class StackInspectionError(RuntimeError):
    """A git read needed to decide the foreign-commit cut failed."""

    def __init__(self, command: tuple[str, ...], stderr: str) -> None:
        super().__init__(f'{" ".join(command)} failed: {stderr.strip()}')
        self.command = command
        self.stderr = stderr


@dataclass(frozen=True)
class StackBaseLedger:
    """One ref per stacked branch: ``<namespace><full_branch>`` -> base SHA."""

    repo_root: Path
    run: GitRunner

    def ref_for(self, full_branch: str) -> str:
        return f'{STACK_BASE_REF_NAMESPACE}{full_branch}'

    async def record(self, full_branch: str, base_sha: str) -> None:
        """Record *base_sha* as *full_branch*'s stack base.  Never raises."""
        rc, _, err = await self.run(
            ['git', 'update-ref', self.ref_for(full_branch), base_sha],
            cwd=self.repo_root,
        )
        if rc != 0:
            logger.warning(
                'Could not record stack base %s for %s: %s',
                base_sha, full_branch, err.strip(),
            )

    async def base_of(self, full_branch: str) -> str | None:
        rc, out, _ = await self.run(
            [
                'git', 'rev-parse', '--verify', '--quiet',
                f'{self.ref_for(full_branch)}^{{commit}}',
            ],
            cwd=self.repo_root,
        )
        return out.strip() if rc == 0 else None

    async def forget(self, full_branch: str) -> None:
        """Drop *full_branch*'s record, if any.  Never raises."""
        rc, _, err = await self.run(
            ['git', 'update-ref', '-d', self.ref_for(full_branch)],
            cwd=self.repo_root,
        )
        if rc != 0:
            logger.debug(
                'Could not clear stack base of %s: %s', full_branch, err.strip(),
            )


async def foreign_commit_cut(
    run: GitRunner, repo_root: Path, *, tip: str, main_ref: str, base: str,
) -> str | None:
    """Newest commit of the contiguous FOREIGN prefix of ``main_ref..tip``.

    A commit is foreign when it is an ancestor of *base* (the stack is
    intact) or patch-equivalent to a commit of *base* (a later rebase onto
    main replayed the base's commits as new SHAs).  Stacking puts the base's
    commits strictly below the branch's own, so only a PREFIX can be foreign:
    a foreign-looking commit after the first own commit never extends the
    cut, because stripping past it would delete the branch's own work.

    Returns None when the oldest commit is already the branch's own.  Raises
    :class:`StackInspectionError` when any git read fails.
    """
    ordered = await _git_lines(
        run, repo_root, 'rev-list', '--reverse', '--topo-order', f'{main_ref}..{tip}',
    )
    foreign = set(await _git_lines(run, repo_root, 'rev-list', f'{main_ref}..{base}'))
    foreign |= await _patch_equivalent_to_base(
        run, repo_root, tip=tip, main_ref=main_ref, base=base,
    )
    return _end_of_prefix(ordered, foreign)


async def _patch_equivalent_to_base(
    run: GitRunner, repo_root: Path, *, tip: str, main_ref: str, base: str,
) -> set[str]:
    cherry = await _git_lines(run, repo_root, 'cherry', base, tip, main_ref)
    return {line.split()[1] for line in cherry if line.startswith('-')}


def _end_of_prefix(ordered: list[str], foreign: set[str]) -> str | None:
    cut: str | None = None
    for sha in ordered:
        if sha not in foreign:
            break
        cut = sha
    return cut


async def _git_lines(run: GitRunner, repo_root: Path, *args: str) -> list[str]:
    command = ('git', *args)
    rc, out, err = await run(list(command), cwd=repo_root)
    if rc != 0:
        raise StackInspectionError(command, err)
    return [line.strip() for line in out.splitlines() if line.strip()]


async def base_owners(run: GitRunner, repo_root: Path, base: str) -> tuple[str, ...]:
    """Local branches pointing exactly at *base*, sorted.  () on any error."""
    rc, out, _ = await run(
        [
            'git', 'for-each-ref', f'--points-at={base}',
            '--format=%(refname:short)', 'refs/heads/',
        ],
        cwd=repo_root,
    )
    if rc != 0:
        return ()
    return tuple(sorted(line.strip() for line in out.splitlines() if line.strip()))
