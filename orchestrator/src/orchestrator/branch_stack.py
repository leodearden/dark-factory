"""Branch-stack provenance: which off-main base a branch was stacked onto, and
which of its commits are therefore not its own (task 5618).

The train stacker rebases a member onto its predecessor's tip.  If that
predecessor later leaves without landing, the member still carries the
predecessor's commits below its own, and every later consumer would treat
``main..branch`` as the member's own work.  This module keeps the one fact
that makes those commits separable again — the base the branch was stacked
onto, recorded at the moment of stacking — and computes the cut between the
foreign prefix and the branch's own delta.  It also names the tip each
member had immediately before stacking (task 4953), so a train that does not
land can hand every member back on it.

The record is one git ref per stacked branch under
:data:`STACK_BASE_REF_NAMESPACE`.  A ref is atomic, survives restarts, is
shared by every worktree, and keeps a never-landed base reachable.

The ledger's record lifetime (pruning a deleted branch's records, and
carrying a record across a branch rename) is the shared
``orchestrator/src/orchestrator/ref_ledger.py`` mechanics.

Layering: this module sits BELOW ``orchestrator/src/orchestrator/git_ops.py``,
which imports it, and above ``orchestrator/src/orchestrator/rebase_drops.py``,
which it imports so the un-stack rebase reports the commits it drops.  The
git runner is therefore injected, exactly as
``orchestrator/src/orchestrator/rebase_recovery.py::guarded_abort`` receives
its runner, and nothing here imports git_ops or merge_queue.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

from orchestrator.rebase_drops import run_rebase_reporting_drops
from orchestrator.rebase_recovery import guarded_abort, operation_still_in_progress
from orchestrator.ref_ledger import (
    GitReadError,
    GitRunner,
    git_lines,
    move_branch_records,
    prune_orphaned_records,
)

logger = logging.getLogger(__name__)

STACK_BASE_REF_NAMESPACE = 'refs/dark-factory/stack-base/'

STACKED_ON_UNLANDED_BASE_REASON_PREFIX = 'Branch is stacked on an unlanded base'

#: A git read needed to decide the foreign-commit cut failed.
StackInspectionError = GitReadError

_FULL_SHA = re.compile(r'[0-9a-f]{40}(?:[0-9a-f]{24})?')


@dataclass(frozen=True)
class StackTarget:
    """Stack a branch on *commit*, a tip of the predecessor branch *branch*."""

    branch: str
    commit: str

    def __post_init__(self) -> None:
        if not self.branch:
            raise ValueError(f'a stack target needs a branch, got {self.branch!r}')
        if not _FULL_SHA.fullmatch(self.commit):
            raise ValueError(
                f'a stack target commit must be a full hex sha, got {self.commit!r}',
            )


class StackRecordState(StrEnum):
    ABSENT = 'absent'
    RECORDED = 'recorded'
    UNREADABLE = 'unreadable'


@dataclass(frozen=True)
class StackBaseRead:
    """What the ledger says of one branch.  UNREADABLE is a failed read, which
    is NOT "no record": the branch may be stacked.  ``detail`` says why."""

    state: StackRecordState
    base: str | None = None
    detail: str = ''

    def __post_init__(self) -> None:
        if (self.state is StackRecordState.RECORDED) != (self.base is not None):
            raise ValueError(f'a {self.state} stack-base read cannot have base {self.base}')

    def names(self, sha: str) -> bool:
        return self.base == sha


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

    async def ensure_record(self, full_branch: str, base_sha: str) -> StackBaseRead:
        """Make *full_branch*'s record name *base_sha*, writing only when it does
        not already, and return the read-back.  Never raises: the caller asks
        the answer's ``names(base_sha)``."""
        if not (await self.read(full_branch)).names(base_sha):
            await self.record(full_branch, base_sha)
        return await self.read(full_branch)

    async def read(self, full_branch: str) -> StackBaseRead:
        ref = self.ref_for(full_branch)
        rc, out, err = await self.run(
            ['git', 'for-each-ref', '--format=%(refname) %(objectname) %(objecttype)', ref],
            cwd=self.repo_root,
        )
        if rc != 0:
            return StackBaseRead(
                StackRecordState.UNREADABLE,
                detail=err.strip() or f'git for-each-ref {ref} exited {rc}',
            )
        rows = [line.split(' ') for line in out.splitlines()]
        exact = [row[1:] for row in rows if row[0] == ref]
        if not exact:
            return StackBaseRead(StackRecordState.ABSENT)
        sha, object_type = exact[0]
        if object_type != 'commit':
            return StackBaseRead(
                StackRecordState.UNREADABLE,
                detail=f'{ref} names {object_type} {sha}, not a commit',
            )
        return StackBaseRead(StackRecordState.RECORDED, sha)

    async def base_of(self, full_branch: str) -> str | None:
        """The recorded base; None both for no record and for a failed read,
        which :meth:`read` tells apart."""
        return (await self.read(full_branch)).base

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
        A failed git command is logged, not raised.
        """
        return await prune_orphaned_records(
            self.run, self.repo_root, STACK_BASE_REF_NAMESPACE, branch_of=lambda key: key,
        )

    async def move_branch(self, old_branch: str, new_branch: str) -> None:
        """Re-key *old_branch*'s record to *new_branch*, which it was renamed to.

        A failed git command is logged, not raised.
        """
        await move_branch_records(
            self.run, self.repo_root, self.ref_for(old_branch), self.ref_for(new_branch),
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
    ordered = await git_lines(
        run, repo_root, 'rev-list', '--reverse', '--topo-order', f'{main_ref}..{tip}',
    )
    foreign = set(await git_lines(run, repo_root, 'rev-list', f'{main_ref}..{base}'))
    foreign |= await _patch_equivalent_to_base(
        run, repo_root, tip=tip, main_ref=main_ref, base=base,
    )
    return _end_of_prefix(ordered, foreign)


async def _patch_equivalent_to_base(
    run: GitRunner, repo_root: Path, *, tip: str, main_ref: str, base: str,
) -> set[str]:
    cherry = await git_lines(run, repo_root, 'cherry', base, tip, main_ref)
    return {line.split()[1] for line in cherry if line.startswith('-')}


def _end_of_prefix(ordered: list[str], foreign: set[str]) -> str | None:
    cut: str | None = None
    for sha in ordered:
        if sha not in foreign:
            break
        cut = sha
    return cut


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


@dataclass(frozen=True)
class PreStackTip:
    """A train member's restore point: the tip it had immediately before a
    train stacked it (``sha``) and the tip stacking gave it (``stacked_sha``),
    which tells the train's own commits from any added since."""

    member_id: str
    sha: str
    stacked_sha: str


class TipRestoreOutcome(StrEnum):
    RESTORED = 'restored'
    UNCHANGED = 'unchanged'
    MISSING = 'missing'
    LANDED = 'landed'
    DIRTY = 'dirty'
    MOVED = 'moved'
    RESET_FAILED = 'reset_failed'


@dataclass(frozen=True)
class TipRestore:
    """Verdict of putting *branch* back on its pre-stack tip.

    ``found_sha`` is the tip the branch had when the restore looked (None
    when the branch is gone); ``worktree`` is the checkout holding it, if any;
    ``detail`` says why a refused restore moved nothing.
    """

    outcome: TipRestoreOutcome
    branch: str
    pre_stack_sha: str
    found_sha: str | None
    worktree: Path | None = None
    detail: str = ''


class OwnDeltaRebase(NamedTuple):
    ok: bool
    conflicted_paths: tuple[str, ...]
    stderr: str
    #: The guarded abort failed and left the rebase in progress.
    abort_failed: bool = False


async def rebase_own_delta(
    run: GitRunner, worktree: Path, *, onto: str, cut: str,
) -> OwnDeltaRebase:
    """Replay the commits after *cut* onto *onto*, in *worktree*.

    A commit the replay drops because *onto* already carries its change is
    logged and recorded (see ``orchestrator/src/orchestrator/rebase_drops.py``).
    On failure the conflicted paths are collected BEFORE the guarded abort,
    because the abort erases them, and ``abort_failed`` says whether the
    abort left the rebase in progress.
    """
    rebased = await run_rebase_reporting_drops(run, worktree, onto=onto, upstream=cut)
    if rebased.ok:
        return OwnDeltaRebase(ok=True, conflicted_paths=(), stderr='')
    conflicted = await _conflicted_paths(run, worktree)
    abort_rc, _, _ = await guarded_abort('rebase', worktree, run)
    abort_failed = abort_rc != 0 and await asyncio.to_thread(
        operation_still_in_progress, worktree,
    )
    return OwnDeltaRebase(
        ok=False, conflicted_paths=conflicted, stderr=rebased.stderr,
        abort_failed=abort_failed,
    )


async def _conflicted_paths(run: GitRunner, worktree: Path) -> tuple[str, ...]:
    _, out, _ = await run(
        [
            'git', '-c', 'core.quotePath=false', 'diff', '--name-only', '-z',
            '--diff-filter=U',
        ],
        cwd=worktree,
    )
    return tuple(path for path in out.split('\x00') if path)
