"""Whether a task branch gives the merge queue anything honest to land (task 7516).

A merge lands the queued branch REF, never the worktree. So a branch with no
commits beyond main, or a worktree whose HEAD sits on main while the branch is
ahead, leaves the queue nothing honest to land: the first can only produce a
no-op "success" (and strands any uncommitted work), and the second is undone by
the pre-enqueue rebind of the ref to HEAD. :func:`read_branch_state` reads the
facts through ``GitOps``' public methods; :attr:`BranchState.mergeability`
decides from them alone.
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


@dataclass(frozen=True)
class BranchState:
    """What git says about a task branch and its worktree at one moment.

    ``branch_tip`` is None when the ref is absent. ``source_on_main`` and
    ``head_on_main`` say whether :attr:`merge_source` and ``worktree_head``
    are ancestors of ``main_sha``, i.e. carry no commits beyond it.
    """

    branch: QueuedBranch
    branch_tip: str | None
    worktree_head: str
    main_sha: str
    source_on_main: bool
    head_on_main: bool
    worktree_dirty: bool

    @property
    def merge_source(self) -> str:
        """The commit the queue would land: the ref's tip, else the worktree HEAD."""
        return self.branch_tip or self.worktree_head

    @property
    def mergeability(self) -> Mergeability:
        if not self.worktree_head or not self.main_sha:
            return Mergeability.UNREADABLE
        if self.source_on_main:
            return Mergeability.BRANCH_EMPTY
        if (
            self.branch_tip is not None
            and self.branch_tip != self.worktree_head
            and self.head_on_main
        ):
            return Mergeability.WORKTREE_OFF_BRANCH
        return Mergeability.MERGEABLE

    def refusal_summary(self) -> str:
        """One line naming the refusal and the offending shas."""
        kind = self.mergeability
        if kind is Mergeability.BRANCH_EMPTY:
            dirty = '; its worktree holds uncommitted changes' if self.worktree_dirty else ''
            return (
                f'Merge refused: {self.branch.full_name} has no commits beyond main '
                f'(tip {_short(self.merge_source)}, main {_short(self.main_sha)}){dirty}'
            )
        if kind is Mergeability.WORKTREE_OFF_BRANCH:
            return (
                f'Merge refused: worktree HEAD {_short(self.worktree_head)} is on main '
                f'while {self.branch.full_name} carries {_short(self.branch_tip)}'
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
            f'worktree_dirty: {self.worktree_dirty}',
        ]
        if landing_verdict is not None:
            lines.append(f'landing_verdict: {landing_verdict}')
        lines += [
            'A merge lands the queued branch ref, never the worktree, so this '
            'branch gives the queue nothing honest to land.',
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
    is None, and an unreadable ancestry reads as "beyond main", so the merge
    proceeds and the merge worker's own guards decide.
    """
    branch_tip = await git_ops.resolve_branch_sha(branch.full_name)
    merge_source = branch_tip or worktree_head
    source_on_main = await git_ops.is_ancestor(merge_source, main_sha)
    if merge_source == worktree_head:
        head_on_main = source_on_main
    else:
        head_on_main = await git_ops.is_ancestor(worktree_head, main_sha)
    return BranchState(
        branch=branch,
        branch_tip=branch_tip,
        worktree_head=worktree_head,
        main_sha=main_sha,
        source_on_main=source_on_main,
        head_on_main=head_on_main,
        worktree_dirty=await git_ops.has_uncommitted_work(worktree),
    )
