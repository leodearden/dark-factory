"""Says why a plan-files not-touched verdict did not measure the task's own work, when it did not."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from shared.git_async import GitResult, run_git

from orchestrator.git_ops import GitOps
from orchestrator.merge_lane.gates import PlanFilesTouchedResult, declaration_coverage
from orchestrator.recut_ruling import RecutSupersession

logger = logging.getLogger(__name__)

PLAN_DELIVERY_UNMEASURED_REASON_PREFIX = 'Plan-file delivery not measured'

_GIT_TIMEOUT_SECS = 120.0
_MAX_SIBLING_REFS = 5
_SHORT_SHA = 10


class UnmeasuredCause(StrEnum):
    TOUCHED_SET_UNREADABLE = 'touched_set_unreadable'
    OPERATION_IN_PROGRESS = 'operation_in_progress'
    UNCOMMITTED_EDITS = 'uncommitted_edits'
    WORK_ON_SIBLING_REF = 'work_on_sibling_ref'


@dataclass(frozen=True)
class _CausePolicy:
    suggested_action: str
    remedy: str


_POLICY: Mapping[UnmeasuredCause, _CausePolicy] = {
    UnmeasuredCause.TOUCHED_SET_UNREADABLE: _CausePolicy(
        'retry_merge',
        'Re-run the merge so the gate can read the branch history again.',
    ),
    UnmeasuredCause.OPERATION_IN_PROGRESS: _CausePolicy(
        'finish_or_abort_operation',
        'Finish (git <operation> --continue) or abort (git <operation> --abort) '
        'the stopped operation in the worktree, then re-submit.',
    ),
    UnmeasuredCause.UNCOMMITTED_EDITS: _CausePolicy(
        'commit_worktree_edits',
        'Commit the listed files in the worktree, then re-submit.',
    ),
    UnmeasuredCause.WORK_ON_SIBLING_REF: _CausePolicy(
        'recover_work_from_sibling_ref',
        'Re-point the task branch at the named ref, or cherry-pick its '
        'commits onto the task branch, then re-submit.',
    ),
}


@dataclass(frozen=True)
class UnmeasuredDelivery:
    cause: UnmeasuredCause
    entries: tuple[str, ...]
    """Declared entries the cause explains."""
    still_not_touched: tuple[str, ...]
    """The rest of the gate's not-touched entries, which no cause explains."""
    detail: str

    @property
    def suggested_action(self) -> str:
        return _POLICY[self.cause].suggested_action

    def block_reason(self) -> str:
        parts = [
            f'{PLAN_DELIVERY_UNMEASURED_REASON_PREFIX} for {", ".join(self.entries)}: '
            f'{self.cause.value} ({self.detail}).',
        ]
        if self.still_not_touched:
            parts.append(
                f'Also not touched by any commit: {", ".join(self.still_not_touched)}.',
            )
        parts.append(_POLICY[self.cause].remedy)
        parts.append(
            'Do not restart the task or narrow the plan: this delivery was never measured.',
        )
        return ' '.join(parts)


@dataclass(frozen=True)
class RebaseProgress:
    step: int | None
    total: int | None
    head_name: str | None
    onto: str | None

    def describe(self) -> str:
        at = (
            f'step {self.step} of {self.total}'
            if self.step is not None and self.total is not None
            else 'an unknown step'
        )
        return (
            f'a rebase of {self.head_name or "a detached HEAD"} onto '
            f'{self.onto or "an unknown commit"} stopped at {at}'
        )


_REBASE_STATE_DIRS = ('rebase-merge', 'rebase-apply')
_STOPPED_PICK_HEADS: Mapping[str, str] = {
    'MERGE_HEAD': 'merge',
    'CHERRY_PICK_HEAD': 'cherry-pick',
    'REVERT_HEAD': 'revert',
}


@dataclass(frozen=True)
class _SiblingRef:
    ref: str
    tip: str
    committed_at: datetime


@dataclass(frozen=True)
class _SiblingWork:
    sibling: _SiblingRef
    covered: frozenset[str]


async def diagnose_unmeasured_branch(
    check: PlanFilesTouchedResult, worktree: Path, *, task_id: str,
) -> UnmeasuredDelivery | None:
    """Why *check*'s not-touched entries were never measured, from the task's own worktree alone.

    Covers an unreadable touched set, a rebase in progress and uncommitted
    declared edits; reads no other ref. Never raises: a probe failure
    returns None, so the caller reports the not-touched verdict unchanged.
    """
    if not check.not_touched:
        return None
    return await _never_raising(
        'in-branch diagnosis', task_id, _diagnose_branch(check, Path(worktree)),
    )


async def find_work_on_sibling_ref(
    check: PlanFilesTouchedResult,
    git_ops: GitOps,
    *,
    branch: str,
    head: str,
    task_id: str,
    superseded: RecutSupersession | None,
) -> UnmeasuredDelivery | None:
    """The sibling ref ``<branch>-*`` holding *check*'s not-touched work, or None.

    *branch* is the prefixed task branch (``task/<id>``) and *head* the commit
    the gate measured. Only a *head* with no commit beyond main is explained
    by a sibling: a *head* carrying work of its own was measured, so its
    verdict stands. A ref *superseded* by an executed re-cut ruling is never
    offered. Never raises: a probe failure returns None, so the caller
    reports the not-touched verdict unchanged.
    """
    if not check.not_touched:
        return None
    return await _never_raising(
        'sibling-ref probe', task_id,
        _sibling_ref_finding(tuple(check.not_touched), git_ops, branch, head, superseded),
    )


async def _never_raising(
    probe: str, task_id: str, diagnosis: Awaitable[UnmeasuredDelivery | None],
) -> UnmeasuredDelivery | None:
    try:
        return await diagnosis
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.warning(
            'unmeasured-delivery: %s raised for task_id=%s — '
            'the not-touched verdict is reported unchanged',
            probe, task_id, exc_info=True,
        )
        return None


async def _diagnose_branch(
    check: PlanFilesTouchedResult, worktree: Path,
) -> UnmeasuredDelivery | None:
    not_touched = tuple(check.not_touched)
    if check.touched_set_error is not None:
        return UnmeasuredDelivery(
            UnmeasuredCause.TOUCHED_SET_UNREADABLE, not_touched, (),
            f'the branch history could not be read: {check.touched_set_error}',
        )
    stopped = await _operation_in_progress(worktree)
    if stopped is not None:
        return UnmeasuredDelivery(
            UnmeasuredCause.OPERATION_IN_PROGRESS, not_touched, (), stopped,
        )
    covered = _covered(not_touched, await _uncommitted_paths(not_touched, worktree))
    if covered:
        return _partial_finding(
            UnmeasuredCause.UNCOMMITTED_EDITS, not_touched, covered,
            f'staged, unstaged or untracked changes in the worktree {worktree}',
        )
    return None


def _covered(not_touched: tuple[str, ...], paths: Iterable[str]) -> frozenset[str]:
    undelivered = declaration_coverage(not_touched, paths).undelivered
    return frozenset(not_touched) - frozenset(undelivered)


def _partial_finding(
    cause: UnmeasuredCause, not_touched: tuple[str, ...], covered: frozenset[str], detail: str,
) -> UnmeasuredDelivery:
    return UnmeasuredDelivery(
        cause,
        tuple(entry for entry in not_touched if entry in covered),
        tuple(entry for entry in not_touched if entry not in covered),
        detail,
    )


async def _git(cwd: Path, *args: str) -> GitResult:
    return await run_git(['git', *args], cwd=cwd, timeout=_GIT_TIMEOUT_SECS)


def _lines(result: GitResult) -> list[str]:
    return [line for line in result.stdout.splitlines() if line.strip()]


async def _operation_in_progress(worktree: Path) -> str | None:
    """Describes the rebase, merge, cherry-pick or revert git stopped in *worktree*, if any."""
    names = (*_REBASE_STATE_DIRS, *_STOPPED_PICK_HEADS)
    paths = await _git(
        worktree, 'rev-parse', *(arg for name in names for arg in ('--git-path', name)),
    )
    if not paths.ok:
        logger.warning(
            'unmeasured-delivery: git rev-parse --git-path failed in %s '
            '(returncode=%d, stderr=%r); no in-progress operation read',
            worktree, paths.returncode, paths.stderr,
        )
        return None
    state = dict(zip(names, (worktree / line for line in _lines(paths)), strict=True))
    stopped = _describe_stopped(state)
    if stopped is None:
        return None
    unmerged = await _git(worktree, 'diff', '--name-only', '--diff-filter=U')
    return f'{stopped}; unmerged paths: {", ".join(_lines(unmerged)) or "none"}'


def _describe_stopped(state: Mapping[str, Path]) -> str | None:
    for name in _REBASE_STATE_DIRS:
        if state[name].is_dir():
            return _read_rebase_progress(state[name]).describe()
    for name, operation in _STOPPED_PICK_HEADS.items():
        if state[name].is_file():
            picked = _read_state(state[name])
            commit = picked.split()[0][:_SHORT_SHA] if picked else 'an unknown commit'
            return f'a {operation} of {commit} stopped'
    return None


def _read_rebase_progress(state_dir: Path) -> RebaseProgress:
    merge_backend = state_dir.name == 'rebase-merge'
    head_name = _read_state(state_dir / 'head-name')
    onto = _read_state(state_dir / 'onto')
    return RebaseProgress(
        step=_int_state(state_dir / ('msgnum' if merge_backend else 'next')),
        total=_int_state(state_dir / ('end' if merge_backend else 'last')),
        head_name=head_name.removeprefix('refs/heads/') if head_name else None,
        onto=onto[:_SHORT_SHA] if onto else None,
    )


def _read_state(path: Path) -> str | None:
    try:
        return path.read_text().strip() or None
    except OSError:
        return None


def _int_state(path: Path) -> int | None:
    text = _read_state(path)
    return int(text) if text and text.isdigit() else None


async def _uncommitted_paths(not_touched: tuple[str, ...], worktree: Path) -> list[str]:
    pathspecs = [entry for entry in not_touched if not Path(entry).is_absolute()]
    if not pathspecs:
        return []
    changed, untracked = await asyncio.gather(
        _git(worktree, '--literal-pathspecs', 'diff', '--name-only', '--no-renames',
             'HEAD', '--', *pathspecs),
        _git(worktree, '--literal-pathspecs', 'ls-files', '--others',
             '--exclude-standard', '--', *pathspecs),
    )
    failed = [result for result in (changed, untracked) if not result.ok]
    if failed:
        logger.warning(
            'unmeasured-delivery: could not list uncommitted changes in %s '
            '(stderr=%r); no uncommitted-edits finding',
            worktree, [result.stderr for result in failed],
        )
        return []
    return _lines(changed) + _lines(untracked)


async def _sibling_ref_finding(
    not_touched: tuple[str, ...],
    git_ops: GitOps,
    branch: str,
    head: str,
    superseded: RecutSupersession | None,
) -> UnmeasuredDelivery | None:
    if await git_ops.commits_beyond_main(head) != 0:
        return None
    works = [
        _SiblingWork(sibling, _covered(not_touched, await _sibling_own_work(sibling, git_ops)))
        for sibling in await _sibling_refs(git_ops, branch, superseded)
    ]
    best = max(works, key=lambda work: len(work.covered), default=None)
    if best is None or not best.covered:
        return None
    return _partial_finding(
        UnmeasuredCause.WORK_ON_SIBLING_REF, not_touched, best.covered,
        f'the declared work is on {best.sibling.ref} at '
        f'{best.sibling.tip[:_SHORT_SHA]}, not on the branch being merged',
    )


async def _sibling_refs(
    git_ops: GitOps, branch: str, superseded: RecutSupersession | None,
) -> list[_SiblingRef]:
    listed = await _git(
        Path(git_ops.project_root), 'for-each-ref', '--sort=-committerdate',
        '--format=%(objectname)%00%(committerdate:unix)%00%(refname)',
        f'refs/heads/{branch}-*',
    )
    if not listed.ok:
        logger.warning(
            'unmeasured-delivery: git for-each-ref refs/heads/%s-* failed '
            '(returncode=%d, stderr=%r); no sibling refs read',
            branch, listed.returncode, listed.stderr,
        )
        return []
    siblings = (_parse_sibling(line) for line in _lines(listed))
    live = [
        sibling for sibling in siblings
        if sibling is not None
        and not (
            superseded is not None
            and superseded.supersedes(sibling.ref, sibling.committed_at)
        )
    ]
    return live[:_MAX_SIBLING_REFS]


def _parse_sibling(line: str) -> _SiblingRef | None:
    try:
        tip, committed_unix, refname = line.split('\0')
        committed_at = datetime.fromtimestamp(int(committed_unix), UTC)
    except (ValueError, OverflowError, OSError):
        logger.warning(
            'unmeasured-delivery: unparseable for-each-ref line %r dropped', line,
        )
        return None
    return _SiblingRef(
        ref=refname.removeprefix('refs/heads/'), tip=tip, committed_at=committed_at,
    )


async def _sibling_own_work(sibling: _SiblingRef, git_ops: GitOps) -> list[str]:
    fork = await git_ops.merge_base_with_main(sibling.tip)
    if (
        fork is None
        or fork == sibling.tip
        or await git_ops.branch_content_in_main(sibling.tip)
    ):
        return []
    files, err = await git_ops.get_files_touched_in_branch(fork, sibling.tip)
    return [] if err is not None else files
