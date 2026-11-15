"""Integrate current main into a resumed task branch that conflicts with it, before planning (task 7736).

The caller is ``orchestrator/src/orchestrator/workflow.py::TaskWorkflow._settle_conflicting_base``.
This module imports downward only, from ``orchestrator.git_ops``,
``orchestrator.merge_lane.outcome_facts`` and ``shared.git_async``; nothing it
imports imports it back.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from shared.git_async import GitResult, run_git

from orchestrator.git_ops import GitOps, GitQueryFailed
from orchestrator.merge_lane.outcome_facts import ConflictEvidence

_GIT_TIMEOUT_SECS = 120.0
_MERGE_TIMEOUT_SECS = 600.0
_BLOCK_REASON_LIMIT = 199
_PATHS_IN_REASON = 2


@dataclass(frozen=True)
class BaseConflict:
    """A resumed branch tip that does not merge cleanly onto main."""

    main_sha: str
    head_sha: str
    conflicted_paths: tuple[str, ...]

    def evidence(self) -> ConflictEvidence:
        """The conflict as 7302's evidence; ``()`` counterparts means against main itself."""
        return ConflictEvidence(self.conflicted_paths, counterpart_task_ids=())

    def to_event_payload(self) -> dict[str, Any]:
        return {
            'main_sha': self.main_sha,
            'head_sha': self.head_sha,
            **self.evidence().to_event_payload(),
        }


async def find_base_conflict(git_ops: GitOps, worktree: Path) -> BaseConflict | None:
    """The conflict *worktree*'s HEAD has with main, or None when it merges cleanly.

    The predicate is ``GitOps.merge_tree_conflicts``, the one the merge
    queue's suffix tracker bounces on. A git error raises; it is not a verdict.
    """
    main_sha = await git_ops.get_main_sha()
    head_sha = await worktree_head(worktree)
    probe = await git_ops.merge_tree_conflicts(main_sha, head_sha)
    if probe.clean:
        return None
    return BaseConflict(main_sha, head_sha, tuple(probe.conflicted_paths))


async def worktree_head(worktree: Path) -> str:
    """*worktree*'s HEAD as a sha, for GitOps methods that run at project_root."""
    result = await _git(worktree, 'rev-parse', 'HEAD')
    if not result.ok or not result.stdout:
        raise GitQueryFailed(f'cannot resolve HEAD in {worktree}: {result.stderr}')
    return result.stdout


@dataclass(frozen=True)
class MergerDisposition:
    resolved: bool
    reason: str


MergerCall = Callable[[str], Awaitable[MergerDisposition]]


class IntegrationRefusal(StrEnum):
    """Why a settlement did not leave the branch integrating current main."""

    BRANCH_NOT_CHECKED_OUT = 'branch_not_checked_out'
    MERGE_NOT_STARTED = 'merge_not_started'
    MERGER_DECLINED = 'merger_declined'
    MERGE_UNCONCLUDED = 'merge_unconcluded'
    HISTORY_REWRITTEN = 'history_rewritten'
    MAIN_NOT_INTEGRATED = 'main_not_integrated'


@dataclass(frozen=True)
class Settlement:
    conflict: BaseConflict
    branch: str
    refusals: tuple[IntegrationRefusal, ...]
    merger_reason: str = ''
    git_output: str = ''

    @property
    def settled(self) -> bool:
        return not self.refusals

    def block_reason(self) -> str:
        """One line for an escalation summary, at most 199 characters."""
        paths = self.conflict.conflicted_paths
        shown = ', '.join(paths[:_PATHS_IN_REASON])
        more = f' +{len(paths) - _PATHS_IN_REASON} more' if len(paths) > _PATHS_IN_REASON else ''
        line = (
            f'{self._refusal_values()}: main {self.conflict.main_sha[:8]} into {self.branch} '
            f'at {self.conflict.head_sha[:8]}, conflicts on {shown}{more}'
        )
        if self.merger_reason:
            line = f'{line}; merger: {self.merger_reason}'
        line = ' '.join(line.split())
        if len(line) <= _BLOCK_REASON_LIMIT:
            return line
        return line[:_BLOCK_REASON_LIMIT - 3] + '...'

    def block_detail(self) -> str:
        lines = [
            f'Merging main into {self.branch} at dispatch start was refused: {self._refusal_values()}',
            f'main: {self.conflict.main_sha}',
            f'pre-settlement tip: {self.conflict.head_sha}',
            'conflicted paths:',
            *(f'  {path}' for path in self.conflict.conflicted_paths),
        ]
        if self.merger_reason:
            lines.append(f'merger: {self.merger_reason}')
        if self.git_output:
            lines += ['git output:', self.git_output]
        lines.append(f'nothing was reset; the pre-settlement tip is {self.conflict.head_sha}')
        return '\n'.join(lines)

    def _refusal_values(self) -> str:
        return ', '.join(refusal.value for refusal in self.refusals)


async def settle_base_conflict(
    git_ops: GitOps,
    worktree: Path,
    conflict: BaseConflict,
    *,
    branch: str,
    merger: MergerCall,
) -> Settlement:
    """Merge main into *branch*, handing any conflict to *merger*, then check the result.

    Only an unconcluded merge is ever undone, by ``git merge --abort``;
    nothing is reset, so a refused settlement leaves its evidence in place.
    """
    if await _checked_out_branch(worktree) != branch:
        return Settlement(conflict, branch, (IntegrationRefusal.BRANCH_NOT_CHECKED_OUT,))
    merge = await run_git(
        ['git', 'merge', '--no-ff', '-m', f'Merge main into {branch}', conflict.main_sha],
        cwd=worktree, timeout=_MERGE_TIMEOUT_SECS,
    )
    if merge.ok:
        return await _checked_settlement(git_ops, worktree, conflict, branch)
    paths = await _unmerged_paths(git_ops, worktree) or ()
    if not paths:
        await _abort_if_merging(git_ops, worktree)
        output = '\n'.join(part for part in (merge.stdout, merge.stderr) if part)
        return Settlement(conflict, branch, (IntegrationRefusal.MERGE_NOT_STARTED,), git_output=output)
    details = await git_ops.get_conflict_details(worktree, paths)
    disposition = await merger(main_merge_brief(conflict, branch, details))
    if not disposition.resolved:
        await _abort_if_merging(git_ops, worktree)
        return Settlement(
            conflict, branch, (IntegrationRefusal.MERGER_DECLINED,), merger_reason=disposition.reason,
        )
    return await _checked_settlement(git_ops, worktree, conflict, branch, disposition.reason)


def main_merge_brief(conflict: BaseConflict, branch: str, details: str) -> str:
    """The ``conflicts`` text handed to the merger for a dispatch-start merge of main."""
    return f"""\
A merge of main {conflict.main_sha} into {branch} is IN PROGRESS in this worktree. \
It stopped on the conflicts listed below.

- Resolve every conflicted path, run the tests, and conclude the merge with \
`git commit --no-edit`, keeping both parents.
- Never abort the merge, rebase, reset, amend, or check out another ref.
- If a hook refuses the commit, do not re-commit with a single parent: call \
`submit_merge_disposition(blocked=true)` with the refusal output.

{details}"""


async def _checked_settlement(
    git_ops: GitOps, worktree: Path, conflict: BaseConflict, branch: str, merger_reason: str = '',
) -> Settlement:
    refusals = await _integration_refusals(git_ops, worktree, conflict)
    if IntegrationRefusal.MERGE_UNCONCLUDED in refusals:
        await _abort_if_merging(git_ops, worktree)
    return Settlement(conflict, branch, refusals, merger_reason=merger_reason)


async def _integration_refusals(
    git_ops: GitOps, worktree: Path, conflict: BaseConflict,
) -> tuple[IntegrationRefusal, ...]:
    """The invariant every settlement must meet: a concluded merge keeping the tip and adding main."""
    refusals: list[IntegrationRefusal] = []
    if await _merge_unconcluded(git_ops, worktree):
        refusals.append(IntegrationRefusal.MERGE_UNCONCLUDED)
    head = await _head_or_none(worktree)
    if head is None or not await git_ops.is_ancestor(conflict.head_sha, head):
        refusals.append(IntegrationRefusal.HISTORY_REWRITTEN)
    if head is None or not await git_ops.is_ancestor(conflict.main_sha, head):
        refusals.append(IntegrationRefusal.MAIN_NOT_INTEGRATED)
    return tuple(refusals)


async def _checked_out_branch(worktree: Path) -> str | None:
    result = await _git(worktree, 'symbolic-ref', '--short', 'HEAD')
    return result.stdout if result.ok else None


async def _merge_unconcluded(git_ops: GitOps, worktree: Path) -> bool:
    """A merge is still open, or unmerged paths remain or cannot be listed."""
    if await _merge_in_progress(worktree):
        return True
    paths = await _unmerged_paths(git_ops, worktree)
    return paths is None or bool(paths)


async def _merge_in_progress(worktree: Path) -> bool:
    return (await _git(worktree, 'rev-parse', '-q', '--verify', 'MERGE_HEAD')).ok


async def _unmerged_paths(git_ops: GitOps, worktree: Path) -> tuple[str, ...] | None:
    """The index's unmerged paths, or None when git cannot list them."""
    try:
        return await git_ops.unmerged_paths(worktree)
    except RuntimeError:
        return None


async def _abort_if_merging(git_ops: GitOps, worktree: Path) -> None:
    if await _merge_in_progress(worktree):
        await git_ops.abort_merge(worktree)


async def _head_or_none(worktree: Path) -> str | None:
    try:
        return await worktree_head(worktree)
    except GitQueryFailed:
        return None


async def _git(worktree: Path, *args: str) -> GitResult:
    return await run_git(['git', *args], cwd=worktree, timeout=_GIT_TIMEOUT_SECS)
