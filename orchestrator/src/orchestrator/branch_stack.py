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
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

from orchestrator.rebase_recovery import AbortRunner, guarded_abort

logger = logging.getLogger(__name__)

STACK_BASE_REF_NAMESPACE = 'refs/dark-factory/stack-base/'
_BRANCH_REF_NAMESPACE = 'refs/heads/'

STACKED_ON_UNLANDED_BASE_REASON_PREFIX = 'Branch is stacked on an unlanded base'

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

    async def prune_orphans(self) -> tuple[str, ...]:
        """Drop every record whose branch no longer exists; return those branches.

        A record lives no longer than its branch, however the branch ended.
        Never raises: a failed listing prunes nothing.
        """
        rc, out, err = await self.run(
            [
                'git', 'for-each-ref', '--format=%(refname)',
                STACK_BASE_REF_NAMESPACE, _BRANCH_REF_NAMESPACE,
            ],
            cwd=self.repo_root,
        )
        if rc != 0:
            logger.warning('Could not list stack-base records: %s', err.strip())
            return ()
        orphans = _orphaned_branches(set(out.split()))
        for full_branch in orphans:
            await self.forget(full_branch)
        return orphans


def _orphaned_branches(refnames: set[str]) -> tuple[str, ...]:
    recorded = (
        refname.removeprefix(STACK_BASE_REF_NAMESPACE)
        for refname in refnames
        if refname.startswith(STACK_BASE_REF_NAMESPACE)
    )
    return tuple(sorted(
        full_branch for full_branch in recorded
        if f'{_BRANCH_REF_NAMESPACE}{full_branch}' not in refnames
    ))


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


class UnstackOutcome(StrEnum):
    NOT_STACKED = 'not_stacked'
    UNSTACKED = 'unstacked'
    CONFLICT = 'conflict'
    BLOCKED = 'blocked'


@dataclass(frozen=True)
class UnstackResult:
    """Verdict of stripping an unlanded stack base from *branch*.

    ``base`` is set whenever a record was found, and ``cut`` whenever the
    foreign prefix was computed.  BLOCKED may lack a cut: an inspection
    error can stop the decision before the cut is known.  CONFLICT names at
    least one conflicted path; a rebase that fails without one is BLOCKED.
    """

    outcome: UnstackOutcome
    branch: str
    main_branch: str
    base: str | None = None
    cut: str | None = None
    base_owners: tuple[str, ...] = ()
    conflicted_paths: tuple[str, ...] = ()
    detail: str = ''

    def __post_init__(self) -> None:
        if self.outcome is not UnstackOutcome.NOT_STACKED and self.base is None:
            raise ValueError(f'{self.outcome} result for {self.branch} needs a base')
        needs_cut = self.outcome in (UnstackOutcome.UNSTACKED, UnstackOutcome.CONFLICT)
        if needs_cut and self.cut is None:
            raise ValueError(f'{self.outcome} result for {self.branch} needs a cut')
        if self.outcome is UnstackOutcome.CONFLICT and not self.conflicted_paths:
            raise ValueError(f'conflict result for {self.branch} names no conflicted path')

    @property
    def stops_merge(self) -> bool:
        return self.outcome in (UnstackOutcome.CONFLICT, UnstackOutcome.BLOCKED)

    def merge_block_reason(self) -> str:
        """The one attributed reason a stopped merge carries."""
        if not self.stops_merge:
            raise ValueError(f'a {self.outcome} result does not stop a merge')
        return ' '.join((
            f'{STACKED_ON_UNLANDED_BASE_REASON_PREFIX}:',
            f'{self.branch} carries commits of base {self._base_sha()[:12]}',
            f'({self._owner_clause()}), which never landed on {self.main_branch};',
            self._outcome_clause(),
            self._remedy_clause(),
        ))

    def _base_sha(self) -> str:
        if self.base is None:
            raise ValueError(f'{self.outcome} result for {self.branch} has no base')
        return self.base

    def _owner_clause(self) -> str:
        if not self.base_owners:
            return 'no live branch points at it any more'
        return f'branch {", ".join(self.base_owners)}'

    def _outcome_clause(self) -> str:
        if self.outcome is UnstackOutcome.CONFLICT:
            return (
                f'un-stacking its own delta onto {self.main_branch} conflicts in: '
                f'{", ".join(self.conflicted_paths)}. These conflicts are the '
                "branch's own; the base's commits are NOT part of this merge and "
                'must not be resolved into it.'
            )
        return f'it could not be un-stacked automatically: {self.detail}.'

    def _remedy_clause(self) -> str:
        if self.cut is None:
            return (
                f'Remedy: find the cut with git cherry -v {self._base_sha()} '
                f'{self.branch} {self.main_branch}'
            )
        return f'Remedy: git rebase --onto {self.main_branch} {self.cut} {self.branch}'


class OwnDeltaRebase(NamedTuple):
    ok: bool
    conflicted_paths: tuple[str, ...]
    stderr: str


async def rebase_own_delta(
    run: GitRunner, worktree: Path, *, onto: str, cut: str,
) -> OwnDeltaRebase:
    """Replay the commits after *cut* onto *onto*, in *worktree*.

    On failure the conflicted paths are collected BEFORE the guarded abort,
    because the abort erases them.
    """
    rc, _, err = await run(['git', 'rebase', '--onto', onto, cut], cwd=worktree)
    if rc == 0:
        return OwnDeltaRebase(ok=True, conflicted_paths=(), stderr='')
    conflicted = await _conflicted_paths(run, worktree)
    await guarded_abort('rebase', worktree, run)
    return OwnDeltaRebase(ok=False, conflicted_paths=conflicted, stderr=err)


async def _conflicted_paths(run: GitRunner, worktree: Path) -> tuple[str, ...]:
    _, out, _ = await run(
        [
            'git', '-c', 'core.quotePath=false', 'diff', '--name-only', '-z',
            '--diff-filter=U',
        ],
        cwd=worktree,
    )
    return tuple(path for path in out.split('\x00') if path)
