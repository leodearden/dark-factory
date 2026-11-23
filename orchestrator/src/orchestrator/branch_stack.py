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
land can hand every member back on it.  A dependent started early on a
predecessor's branch is re-stacked when that branch is rebuilt (task
10473): :func:`restack_own_delta` moves the dependent's own delta onto the
new tip and the record with it, and :func:`restack_trial` asks first whether
that would conflict.

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
from orchestrator.rebase_recovery import (
    RECOVERY_GIT,
    guarded_abort,
    operation_still_in_progress,
)
from orchestrator.ref_ledger import (
    GitReadError,
    GitRunner,
    git_lines,
    git_out,
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


class RestackOutcome(StrEnum):
    RESTACKED = 'restacked'
    CONFLICT = 'conflict'
    REFUSED = 'refused'
    ABORT_FAILED = 'abort_failed'
    RECORD_FAILED = 'record_failed'


_BLOCKING_RESTACK_OUTCOMES = frozenset({
    RestackOutcome.REFUSED, RestackOutcome.ABORT_FAILED, RestackOutcome.RECORD_FAILED,
})


@dataclass(frozen=True)
class RestackResult:
    """Verdict of re-stacking *branch*'s own delta, the commits after
    *upstream*, onto *onto* in *worktree*.

    ``old_tip`` and ``old_base`` are the branch tip and the recorded base
    before (``old_base`` is None when there was no record).  Only a REFUSED
    re-stack may lack ``old_tip`` or ``upstream``: it can stop before
    either is known.  RESTACKED carries
    ``new_tip``; CONFLICT names at least one conflicted path and left the
    branch, tree and record as they were.  ``detail`` says why a re-stack
    stopped, and for RECORD_FAILED whether the roll-back held.
    """

    outcome: RestackOutcome
    branch: str
    worktree: Path
    onto: str
    upstream: str | None
    old_tip: str | None
    old_base: str | None
    new_tip: str | None = None
    conflicted_paths: tuple[str, ...] = ()
    detail: str = ''

    def __post_init__(self) -> None:
        if (self.outcome is RestackOutcome.RESTACKED) != (self.new_tip is not None):
            raise ValueError(
                f'a {self.outcome} re-stack of {self.branch} cannot have new tip {self.new_tip}',
            )
        if self.outcome is RestackOutcome.CONFLICT and not self.conflicted_paths:
            raise ValueError(f'conflict re-stack of {self.branch} names no conflicted path')
        unknown_start = self.old_tip is None or self.upstream is None
        if unknown_start and self.outcome is not RestackOutcome.REFUSED:
            raise ValueError(
                f'a {self.outcome} re-stack of {self.branch} needs its old tip and upstream',
            )

    @property
    def requeues(self) -> bool:
        return self.outcome is RestackOutcome.CONFLICT

    @property
    def blocks(self) -> bool:
        return self.outcome in _BLOCKING_RESTACK_OUTCOMES

    def failure_message(self) -> str:
        """The one message a stopped re-stack carries."""
        if self.outcome is RestackOutcome.RESTACKED:
            raise ValueError(f'the re-stack of {self.branch} onto {self.onto} did not stop')
        clauses = [
            f'Re-stack of {self.branch} onto {self.onto} stopped ({self.outcome}):',
            f'the branch was on {self.old_tip or "an unresolved HEAD"}'
            f' with recorded base {self.old_base or "none"};',
        ]
        if self.conflicted_paths:
            clauses.append(f'conflicted paths: {", ".join(self.conflicted_paths)};')
        clauses.append(f'{self.detail}.' if self.detail else 'no further detail.')
        if self.outcome is RestackOutcome.ABORT_FAILED:
            clauses.append(
                'Recovery: python -m orchestrator.rebase_recovery preflight --worktree '
                f'{self.worktree}, then git -C {self.worktree} {" ".join(RECOVERY_GIT[1:])} '
                'rebase --abort',
            )
        return ' '.join(clauses)


class RestackFailed(Exception):
    """A re-stack stopped; ``result`` is its verdict."""

    def __init__(self, result: RestackResult) -> None:
        super().__init__(result.failure_message())
        self.result = result


class RestackConflict(RestackFailed):
    """The dependent's own delta conflicts with the new base: requeue it."""


class RestackBlocked(RestackFailed):
    """The re-stack could not be done or undone cleanly: block with the remedy."""


def restack_failure(result: RestackResult) -> RestackFailed:
    """THE mapping from a stopped re-stack's outcome to the exception it raises."""
    if result.outcome is RestackOutcome.RESTACKED:
        raise ValueError(f'the re-stack of {result.branch} onto {result.onto} did not stop')
    if result.requeues:
        return RestackConflict(result)
    return RestackBlocked(result)


async def restack_own_delta(
    run: GitRunner, worktree: Path, *,
    ledger: StackBaseLedger, branch: str, onto: str,
    record: StackBaseRead, upstream: str,
) -> RestackResult:
    """Replay *branch*'s commits after *upstream* onto *onto*, in *worktree*,
    then make the ledger name *onto*.

    *record* is the ledger read *upstream* was derived from.  It is not read
    again, so the replay starts from the cut its caller computed; an
    UNREADABLE record refuses.  The record is written only after the
    replay, so a replay that stops leaves the branch and its record as they
    were.  A record that does not read back rolls the branch back to its old
    tip, so the branch never sits on a base its record does not name.
    """
    stopped = RestackVerdict(branch, worktree, onto, upstream)
    head_read = await _head(run, worktree)
    head = head_read.sha
    if head is None:
        return stopped.refused(None, record.base, f'HEAD of {worktree} is {head_read.describe()}')
    if record.state is StackRecordState.UNREADABLE:
        return stopped.refused(
            head, None, f'the stack-base record of {branch} is unreadable: {record.detail}',
        )
    rebased = await rebase_own_delta(run, worktree, onto=onto, cut=upstream)
    if not rebased.ok:
        return await _stopped_replay(run, stopped, head, record.base, rebased)
    new_tip = (await git_out(run, worktree, 'rev-parse', '--verify', 'HEAD')).strip()
    after = await ledger.ensure_record(branch, onto)
    if not after.names(onto):
        detail = await _roll_back(run, ledger, stopped, head, record, after)
        return stopped.verdict(RestackOutcome.RECORD_FAILED, head, record.base, detail=detail)
    return stopped.verdict(RestackOutcome.RESTACKED, head, record.base, new_tip=new_tip)


@dataclass(frozen=True)
class RestackVerdict:
    """Builds every :class:`RestackResult` of one re-stack of *branch* onto
    *onto* in *worktree*.  *upstream* is None until it is known, and only a
    refusal may be built without it."""

    branch: str
    worktree: Path
    onto: str
    upstream: str | None = None

    def verdict(
        self, outcome: RestackOutcome, old_tip: str | None, old_base: str | None, *,
        new_tip: str | None = None, conflicted_paths: tuple[str, ...] = (), detail: str = '',
    ) -> RestackResult:
        return RestackResult(
            outcome=outcome, branch=self.branch, worktree=self.worktree, onto=self.onto,
            upstream=self.upstream, old_tip=old_tip, old_base=old_base, new_tip=new_tip,
            conflicted_paths=conflicted_paths, detail=detail,
        )

    def refused(self, old_tip: str | None, old_base: str | None, detail: str) -> RestackResult:
        return self.verdict(RestackOutcome.REFUSED, old_tip, old_base, detail=detail)


class _HeadRead(NamedTuple):
    sha: str | None
    err: str

    def describe(self) -> str:
        return self.sha or f'unresolved ({self.err})'


async def _head(run: GitRunner, worktree: Path) -> _HeadRead:
    rc, out, err = await run(['git', 'rev-parse', '--verify', 'HEAD'], cwd=worktree)
    return _HeadRead(out.strip() if rc == 0 else None, err.strip())


async def _stopped_replay(
    run: GitRunner, stopped: RestackVerdict, old_tip: str, old_base: str | None,
    rebased: OwnDeltaRebase,
) -> RestackResult:
    paths = rebased.conflicted_paths
    stderr = rebased.stderr.strip()
    if rebased.abort_failed:
        return stopped.verdict(
            RestackOutcome.ABORT_FAILED, old_tip, old_base, conflicted_paths=paths,
            detail=f'the abort left the rebase in progress after: {stderr}',
        )
    if not paths:
        return stopped.refused(old_tip, old_base, f'git refused the replay: {stderr}')
    head = await _head(run, stopped.worktree)
    if head.sha != old_tip:
        return stopped.verdict(
            RestackOutcome.ABORT_FAILED, old_tip, old_base, conflicted_paths=paths,
            detail=f'after the abort HEAD is {head.describe()}, not the old tip',
        )
    return stopped.verdict(RestackOutcome.CONFLICT, old_tip, old_base, conflicted_paths=paths)


async def _roll_back(
    run: GitRunner, ledger: StackBaseLedger, stopped: RestackVerdict, old_tip: str,
    before: StackBaseRead, after: StackBaseRead,
) -> str:
    """Put the branch back on *old_tip* and its record back on *before*;
    return the detail clause saying whether that held."""
    rc, _, err = await run(['git', 'reset', '--hard', old_tip], cwd=stopped.worktree)
    if after.base != before.base:
        if before.base is None:
            await ledger.forget(stopped.branch)
        else:
            await ledger.record(stopped.branch, before.base)
    restored = await ledger.read(stopped.branch)
    branch_clause = (
        'the branch is back on its old tip' if rc == 0
        else f'rolling the branch back FAILED: {err.strip()}'
    )
    record_clause = (
        'the record is as it was' if restored.base == before.base
        else f'the record now reads {restored.state} {restored.base}'
    )
    return (
        f'the record read back {after.state} {after.base} instead of {stopped.onto}; '
        f'{branch_clause}; {record_clause}'
    )


class RestackTrialOutcome(StrEnum):
    CLEAN = 'clean'
    CONFLICT = 'conflict'
    UNKNOWN = 'unknown'


@dataclass(frozen=True)
class RestackTrial:
    outcome: RestackTrialOutcome
    conflicted_paths: tuple[str, ...] = ()
    detail: str = ''


async def restack_trial(
    run: GitRunner, repo_root: Path, *, base: str, onto: str, tip: str,
) -> RestackTrial:
    """Would moving *tip*'s commits after *base* onto *onto* conflict?

    A three-way merge in the object store only: no ref, index or worktree
    moves.  git answers rc 1 both for a conflict and for an object it cannot
    merge, so only an rc 1 that wrote a tree is a CONFLICT.
    """
    rc, out, err = await run(
        [
            'git', 'merge-tree', '--write-tree', '--name-only', '--no-messages', '-z',
            '--merge-base', base, onto, tip,
        ],
        cwd=repo_root,
    )
    tree, *paths = out.split('\x00') if out else ('',)
    wrote_tree = _FULL_SHA.fullmatch(tree) is not None
    if rc == 0 and wrote_tree:
        return RestackTrial(RestackTrialOutcome.CLEAN)
    if rc == 1 and wrote_tree:
        conflicted = tuple(dict.fromkeys(path for path in paths if path))
        return RestackTrial(RestackTrialOutcome.CONFLICT, conflicted_paths=conflicted)
    return RestackTrial(
        RestackTrialOutcome.UNKNOWN, detail=err.strip() or f'git merge-tree exited {rc}',
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
