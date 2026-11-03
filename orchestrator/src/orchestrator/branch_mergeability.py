"""Whether a task branch gives the merge queue anything honest to land (task 7516).

The pre-enqueue rebind (``workflow.py::TaskWorkflow._submit_to_merge_queue``,
task 1923) moves the queued branch ref onto the worktree HEAD, so the queue
lands HEAD. A HEAD with no commits beyond main leaves the queue nothing honest
to land: with the ref on main too, the merge can only produce a no-op
"success" (and strands any uncommitted work); with the ref ahead, the rebind
would move it off its commits. :func:`read_branch_state` reads the facts
through ``GitOps``' public methods; :attr:`BranchState.mergeability` decides
from them alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from orchestrator.merge_lane.types import QueuedBranch

if TYPE_CHECKING:
    from orchestrator.git_ops import GitOps

_SHORT_SHA_LEN = 10
_REFUSING_SITE = (
    'orchestrator/src/orchestrator/workflow.py::TaskWorkflow._settle_unmergeable_branch'
)


class Mergeability(Enum):
    """The verdict on a branch; each value is the suffix of a refusal's root_cause."""

    MERGEABLE = 'mergeable'
    BRANCH_EMPTY = 'branch_empty'
    WORKTREE_OFF_BRANCH = 'worktree_off_branch'
    UNREADABLE = 'unreadable'


def _short(sha: str | None) -> str:
    return sha[:_SHORT_SHA_LEN] if sha else '(absent)'


def _ancestry(beyond_main: bool | None) -> str:
    return 'unreadable (git error)' if beyond_main is None else str(beyond_main)


@dataclass(frozen=True)
class BranchState:
    """What git says about a task branch and its worktree at one moment.

    ``branch_tip`` is None when the ref is absent. ``head_beyond_main`` and
    ``tip_beyond_main`` say whether ``worktree_head`` and ``branch_tip`` carry
    commits ``main_sha`` lacks, and are None when git could not read that
    ancestry; ``tip_beyond_main`` is False for an absent ref.
    """

    branch: QueuedBranch
    branch_tip: str | None
    worktree_head: str
    main_sha: str
    head_beyond_main: bool | None
    tip_beyond_main: bool | None
    worktree_dirty: bool

    @property
    def landing_anchor(self) -> str:
        """The commit the landing ladder tests for a BRANCH_EMPTY branch; not the merge source.

        At that verdict the ref's tip and HEAD are both on main. The tip is
        preferred because ``merge_lane/landing_evidence.py::validate_landing_evidence``
        pairs ``branch_tip_sha`` with the branch ref name.
        """
        return self.branch_tip or self.worktree_head

    @property
    def mergeability(self) -> Mergeability:
        """HEAD decides first; an unreadable tip with HEAD on main still refuses."""
        if not self.worktree_head or not self.main_sha or self.head_beyond_main is None:
            return Mergeability.UNREADABLE
        if self.head_beyond_main:
            return Mergeability.MERGEABLE
        if self.tip_beyond_main is not False:
            return Mergeability.WORKTREE_OFF_BRANCH
        return Mergeability.BRANCH_EMPTY

    def refusal_summary(self) -> str:
        """One line naming the refusal and the offending shas."""
        kind = self.mergeability
        if kind is Mergeability.BRANCH_EMPTY:
            dirty = '; its worktree holds uncommitted changes' if self.worktree_dirty else ''
            return (
                f'Merge refused: {self.branch.full_name} has no commits beyond main '
                f'(tip {_short(self.branch_tip)}, HEAD {_short(self.worktree_head)}, '
                f'main {_short(self.main_sha)}){dirty}'
            )
        if kind is Mergeability.WORKTREE_OFF_BRANCH:
            tip = _short(self.branch_tip)
            ref_state = (
                f'the ancestry of {self.branch.full_name} tip {tip} against main is unreadable'
                if self.tip_beyond_main is None
                else f'{self.branch.full_name} carries {tip}'
            )
            return (
                f'Merge refused: worktree HEAD {_short(self.worktree_head)} is on main '
                f'while {ref_state}'
            )
        raise ValueError(f'{kind.value} is not a refusal')

    def refusal_detail(
        self, *, task_id: str, worktree: Path, landing_verdict: str | None,
    ) -> str:
        """The escalation detail: every value the verdict read, plus recovery hints."""
        lines = [
            f'task: {task_id}',
            f'refusal: {self.mergeability.value}',
            f'branch: {self.branch.full_name}',
            f'branch_tip: {self.branch_tip or "(ref absent)"}',
            f'worktree: {worktree}',
            f'worktree_head: {self.worktree_head}',
            f'main_sha: {self.main_sha}',
            f'head_beyond_main: {_ancestry(self.head_beyond_main)}',
            f'tip_beyond_main: {_ancestry(self.tip_beyond_main)}',
            f'worktree_dirty: {self.worktree_dirty}',
        ]
        if landing_verdict is not None:
            lines.append(f'landing_verdict: {landing_verdict}')
        lines += [
            'The pre-enqueue rebind (workflow.py::TaskWorkflow._submit_to_merge_queue, '
            'task 1923) makes the queue land the worktree HEAD, and HEAD has no '
            'commits beyond main, so the queue has nothing honest to land.',
            f'recover: git reflog show {self.branch.full_name}',
            f'recover: git -C {worktree} diff --cached --stat',
            f'refused at: {_REFUSING_SITE}',
        ]
        return '\n'.join(lines)


async def read_branch_state(
    git_ops: GitOps,
    worktree: Path,
    branch: QueuedBranch,
    *,
    worktree_head: str,
    main_sha: str,
) -> BranchState:
    """Read *branch*'s tip and *worktree*'s state against *main_sha*.

    A failed git read comes back as a value, never an exception: an absent ref
    is None, an unreadable status is clean (``GitOps.has_uncommitted_work``),
    and an unreadable ancestry is None. For HEAD that means UNREADABLE, so the
    merge proceeds and the merge worker's own guards decide; for the tip with
    HEAD on main it means WORKTREE_OFF_BRANCH, a refusal that keeps the rebind
    from moving the ref onto main.
    """
    branch_tip = await git_ops.resolve_branch_sha(branch.full_name)
    head_beyond_main = await _beyond_main(git_ops, worktree_head, main_sha)
    if branch_tip is None:
        tip_beyond_main: bool | None = False
    elif branch_tip == worktree_head:
        tip_beyond_main = head_beyond_main
    else:
        tip_beyond_main = await _beyond_main(git_ops, branch_tip, main_sha)
    return BranchState(
        branch=branch,
        branch_tip=branch_tip,
        worktree_head=worktree_head,
        main_sha=main_sha,
        head_beyond_main=head_beyond_main,
        tip_beyond_main=tip_beyond_main,
        worktree_dirty=await git_ops.has_uncommitted_work(worktree),
    )


async def _beyond_main(git_ops: GitOps, sha: str, main_sha: str) -> bool | None:
    on_main = await git_ops.is_ancestor_checked(sha, main_sha)
    return None if on_main is None else not on_main
