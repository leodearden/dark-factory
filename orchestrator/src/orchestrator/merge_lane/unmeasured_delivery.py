"""Says why a plan-files not-touched verdict did not measure the task's own work, when it did not."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from shared.git_async import GitResult, run_git

from orchestrator.git_ops import GitOps
from orchestrator.merge_lane.gates import PlanFilesTouchedResult, declaration_coverage

logger = logging.getLogger(__name__)

PLAN_DELIVERY_UNMEASURED_REASON_PREFIX = 'Plan-file delivery not measured'

_GIT_TIMEOUT_SECS = 120.0
_MAX_SIBLING_REFS = 5
_SHORT_SHA = 10


class UnmeasuredCause(StrEnum):
    TOUCHED_SET_UNREADABLE = 'touched_set_unreadable'
    MID_REBASE = 'mid_rebase'
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
    UnmeasuredCause.MID_REBASE: _CausePolicy(
        'finish_or_abort_rebase',
        'Finish (git rebase --continue) or abort (git rebase --abort) the '
        'rebase in the worktree, then re-submit.',
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
    unmerged: tuple[str, ...]

    def describe(self) -> str:
        at = (
            f'step {self.step} of {self.total}'
            if self.step is not None and self.total is not None
            else 'an unknown step'
        )
        return (
            f'a rebase of {self.head_name or "a detached HEAD"} onto '
            f'{self.onto or "an unknown commit"} stopped at {at}; '
            f'unmerged paths: {", ".join(self.unmerged) or "none"}'
        )


@dataclass(frozen=True)
class _SiblingRef:
    ref: str
    tip: str


@dataclass(frozen=True)
class _SiblingWork:
    sibling: _SiblingRef
    files: tuple[str, ...]
    covered: int


async def diagnose_unmeasured_delivery(
    check: PlanFilesTouchedResult,
    worktree: Path,
    git_ops: GitOps,
    *,
    branch: str,
    task_id: str,
) -> UnmeasuredDelivery | None:
    """Why *check*'s not-touched entries were never measured, or None when they were.

    *branch* is the prefixed task branch (``task/<id>``) whose sibling refs
    ``<branch>-*`` may hold the work. Never raises: a probe failure returns
    None, so the caller reports the not-touched verdict unchanged.
    """
    if not check.not_touched:
        return None
    try:
        return await _diagnose(check, Path(worktree), git_ops, branch)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.warning(
            'unmeasured-delivery: diagnosis raised for task_id=%s branch=%s — '
            'the not-touched verdict is reported unchanged',
            task_id, branch, exc_info=True,
        )
        return None


async def _diagnose(
    check: PlanFilesTouchedResult, worktree: Path, git_ops: GitOps, branch: str,
) -> UnmeasuredDelivery | None:
    not_touched = tuple(check.not_touched)
    if check.touched_set_error is not None:
        return UnmeasuredDelivery(
            UnmeasuredCause.TOUCHED_SET_UNREADABLE, not_touched, (),
            f'the branch history could not be read: {check.touched_set_error}',
        )
    rebase = await _rebase_in_progress(worktree)
    if rebase is not None:
        return UnmeasuredDelivery(
            UnmeasuredCause.MID_REBASE, not_touched, (), rebase.describe(),
        )
    dirty = await _uncommitted_paths(not_touched, worktree)
    if _covered(not_touched, dirty):
        return _partial_finding(
            UnmeasuredCause.UNCOMMITTED_EDITS, not_touched, dirty,
            f'staged, unstaged or untracked changes in the worktree {worktree}',
        )
    return await _sibling_ref_finding(not_touched, git_ops, branch)


def _covered(not_touched: tuple[str, ...], paths: Iterable[str]) -> set[str]:
    undelivered = declaration_coverage(not_touched, paths).undelivered
    return set(not_touched) - set(undelivered)


def _partial_finding(
    cause: UnmeasuredCause, not_touched: tuple[str, ...], paths: Iterable[str], detail: str,
) -> UnmeasuredDelivery:
    covered = _covered(not_touched, paths)
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


async def _rebase_in_progress(worktree: Path) -> RebaseProgress | None:
    paths = await _git(
        worktree, 'rev-parse', '--git-path', 'rebase-merge', '--git-path', 'rebase-apply',
    )
    if not paths.ok:
        logger.warning(
            'unmeasured-delivery: git rev-parse --git-path failed in %s '
            '(returncode=%d, stderr=%r); no rebase state read',
            worktree, paths.returncode, paths.stderr,
        )
        return None
    for state_dir in (worktree / line for line in _lines(paths)):
        if state_dir.is_dir():
            unmerged = await _git(worktree, 'diff', '--name-only', '--diff-filter=U')
            return _read_rebase_progress(state_dir, tuple(_lines(unmerged)))
    return None


def _read_rebase_progress(state_dir: Path, unmerged: tuple[str, ...]) -> RebaseProgress:
    merge_backend = state_dir.name == 'rebase-merge'
    head_name = _state_field(state_dir, 'head-name')
    onto = _state_field(state_dir, 'onto')
    return RebaseProgress(
        step=_int_field(state_dir, 'msgnum' if merge_backend else 'next'),
        total=_int_field(state_dir, 'end' if merge_backend else 'last'),
        head_name=head_name.removeprefix('refs/heads/') if head_name else None,
        onto=onto[:_SHORT_SHA] if onto else None,
        unmerged=unmerged,
    )


def _state_field(state_dir: Path, name: str) -> str | None:
    try:
        return (state_dir / name).read_text().strip() or None
    except OSError:
        return None


def _int_field(state_dir: Path, name: str) -> int | None:
    text = _state_field(state_dir, name)
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
    not_touched: tuple[str, ...], git_ops: GitOps, branch: str,
) -> UnmeasuredDelivery | None:
    works = []
    for sibling in await _sibling_refs(git_ops, branch):
        files = tuple(await _sibling_own_work(sibling, git_ops))
        works.append(_SiblingWork(sibling, files, len(_covered(not_touched, files))))
    best = max(works, key=lambda work: work.covered, default=None)
    if best is None or not best.covered:
        return None
    return _partial_finding(
        UnmeasuredCause.WORK_ON_SIBLING_REF, not_touched, best.files,
        f'the declared work is on {best.sibling.ref} at '
        f'{best.sibling.tip[:_SHORT_SHA]}, not on the branch being merged',
    )


async def _sibling_refs(git_ops: GitOps, branch: str) -> list[_SiblingRef]:
    listed = await _git(
        Path(git_ops.project_root), 'for-each-ref', '--sort=-committerdate',
        f'--count={_MAX_SIBLING_REFS}', '--format=%(objectname)%00%(refname)',
        f'refs/heads/{branch}-*',
    )
    if not listed.ok:
        logger.warning(
            'unmeasured-delivery: git for-each-ref refs/heads/%s-* failed '
            '(returncode=%d, stderr=%r); no sibling refs read',
            branch, listed.returncode, listed.stderr,
        )
        return []
    siblings = []
    for line in _lines(listed):
        tip, _, refname = line.partition('\0')
        siblings.append(_SiblingRef(ref=refname.removeprefix('refs/heads/'), tip=tip))
    return siblings


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
