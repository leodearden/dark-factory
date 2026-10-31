"""What a warm-lane acquire does about a task branch already checked out somewhere.

git lets a branch be checked out in only one worktree, so an acquire that
would ``git checkout`` or ``git worktree add`` ``task/<id>`` while another
worktree holds it fails with "already used by worktree".  The acquire settles
that holder first: it reuses a lane the task still durably owns, detaches an
ownerless lane, and refuses everything else with
:class:`BranchCheckedOutElsewhere` before touching any lane.  A QUARANTINED
lane is refused too, never detached: recovery leaves it alone on purpose, and
its working tree may hold another incarnation's work that a detach would
commit onto this task's branch.

Pure: no git and no I/O.  The git-side orchestration lives on
``orchestrator/src/orchestrator/git_ops.py::GitOps._settle_branch_checkout``.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from orchestrator.lane_lifecycle import LaneRecord, LaneState


class BranchHolderAction(Enum):
    REUSE_OWN_LANE = 'reuse_own_lane'
    DETACH_OWNERLESS_LANE = 'detach_ownerless_lane'
    REFUSE_HELD_LANE = 'refuse_held_lane'
    REFUSE_QUARANTINED_LANE = 'refuse_quarantined_lane'
    REFUSE_FOREIGN_WORKTREE = 'refuse_foreign_worktree'


def decide_branch_holder(
    *,
    task_id: str,
    lane: Path | None,
    lane_held: bool,
    record: LaneRecord | None,
) -> BranchHolderAction:
    """Decide what to do about the worktree holding *task_id*'s branch.

    *lane* is the pool's key for the holder, or None when the holder is not a
    pool lane; *lane_held* says the pool currently holds that lane for an
    assignment; *record* is the lane's durable record.
    """
    if lane is None:
        return BranchHolderAction.REFUSE_FOREIGN_WORKTREE
    if lane_held:
        return BranchHolderAction.REFUSE_HELD_LANE
    if record is not None and record.state is LaneState.QUARANTINED:
        return BranchHolderAction.REFUSE_QUARANTINED_LANE
    if (
        record is not None
        and record.state in (LaneState.ASSIGNED, LaneState.IN_USE)
        and record.task_id == task_id
    ):
        return BranchHolderAction.REUSE_OWN_LANE
    return BranchHolderAction.DETACH_OWNERLESS_LANE


_REFUSAL_MESSAGES: dict[BranchHolderAction, str] = {
    BranchHolderAction.REFUSE_FOREIGN_WORKTREE: (
        '{branch} is checked out in {holder}, a worktree outside the warm-lane '
        'pool — no lane was touched. Run `git -C {holder} checkout --detach` '
        '(or remove that worktree), then unblock the task.'
    ),
    BranchHolderAction.REFUSE_HELD_LANE: (
        '{branch} is checked out in warm lane {holder}, which the pool holds '
        'for another assignment — no lane was touched. Once that assignment '
        'is released, run `git -C {holder} checkout --detach` if the branch '
        'is still checked out there, then unblock the task.'
    ),
    BranchHolderAction.REFUSE_QUARANTINED_LANE: (
        '{branch} is checked out in quarantined warm lane {holder} — no lane '
        'was touched. Inspect that lane (`git -C {holder} status`), then '
        'detach or remove it and unblock the task.'
    ),
    BranchHolderAction.DETACH_OWNERLESS_LANE: (
        '{branch} is checked out in ownerless warm lane {holder} and the detach '
        'of that lane failed (its WIP could not be committed) — the lane stays '
        'pinned to this task with its WIP retained. Resolve the lane '
        '(`git -C {holder} status`), run `git -C {holder} checkout --detach`, '
        'then unblock the task.'
    ),
}


class BranchCheckedOutElsewhere(RuntimeError):
    """A task branch is checked out in a worktree the acquire cannot settle."""

    def __init__(self, branch: str, holder: Path, action: BranchHolderAction) -> None:
        template = _REFUSAL_MESSAGES.get(action)
        if template is None:
            raise ValueError(f'{action.value} is not a refusal')
        super().__init__(template.format(branch=branch, holder=holder))
        self.branch = branch
        self.holder = holder
        self.action = action
