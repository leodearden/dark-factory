"""Hand a coalesce train's members back when the train does not land (task 4953).

Forming a coalesce train rebases each member onto its predecessor, keeping
each member's restore point
(``orchestrator/src/orchestrator/git_ops.py::GitOps.stack_train_branches``).
When the train does not land, :func:`restore_pre_stack_tips` puts each member
back on the tip it had before stacking, so the re-drive that follows hands a
re-dispatched agent only its own work.  A member is restored only while its
task status says no live agent can own its branch.
"""

from __future__ import annotations

import logging

from orchestrator.branch_stack import PreStackTip, TipRestore, TipRestoreOutcome
from orchestrator.git_ops import GitOps
from orchestrator.merge_lane.types import GroupMergeRequest

logger = logging.getLogger('orchestrator.merge_queue')

_RESTORABLE_STATUSES: frozenset[str] = frozenset({'merge-deferred', 'pending', 'blocked'})

_REFUSALS: frozenset[TipRestoreOutcome] = frozenset({
    TipRestoreOutcome.DIRTY, TipRestoreOutcome.MOVED, TipRestoreOutcome.RESET_FAILED,
})


async def restore_pre_stack_tips(git_ops: GitOps, train: GroupMergeRequest) -> None:
    """Put each stacked member of *train* back on its pre-stack tip.  Never raises.

    Leaves every ref alone when the train's tip is already on main: the train
    landed, and restoring would hide that from the re-drive's double-landing
    guard.  Each member's status is read just before its own restore, so a
    member the scheduler claims while another is being restored is left
    alone.  A claim landing after that read is still caught for commits, by
    the restore's compare-and-swap, but not for files its agent writes
    between the restore's dirty check and its worktree reset.
    """
    if not train.member_pre_stack_tips:
        return
    try:
        landed = await git_ops.is_ancestor(train.branch.full_name, git_ops.config.main_branch)
    except Exception:
        logger.exception(
            'Coalesce train %s: could not tell whether it landed — leaving '
            'members stacked', train.train_id,
        )
        return
    if landed:
        logger.info(
            'Coalesce train %s: tip %s is already on main — leaving every '
            'member branch as it is', train.train_id, train.branch.full_name,
        )
        return
    for tip in train.member_pre_stack_tips:
        await _restore_member(git_ops, train, tip)


async def _restore_member(git_ops: GitOps, train: GroupMergeRequest, tip: PreStackTip) -> None:
    try:
        status = (await train.status_check([tip.member_id])).get(tip.member_id)
        if status not in _RESTORABLE_STATUSES:
            logger.warning(
                'Coalesce train %s: member %s is %r, so a live agent may own its '
                'branch — leaving it stacked (pre-stack tip %s)',
                train.train_id, tip.member_id, status, tip.sha,
            )
            return
        restore = await git_ops.restore_pre_stack_tip(tip)
    except Exception:
        logger.exception(
            'Coalesce train %s: restoring member %s to pre-stack tip %s raised '
            '— continuing with remaining members', train.train_id, tip.member_id, tip.sha,
        )
        return
    _log_restore(train.train_id, tip.member_id, restore)


def _log_restore(train_id: str, member_id: str, restore: TipRestore) -> None:
    if restore.outcome in _REFUSALS:
        logger.warning(
            'Coalesce train %s: member %s left stacked (%s: %s); found %s, '
            'pre-stack %s. Remedy: %s',
            train_id, member_id, restore.outcome, restore.detail,
            restore.found_sha, restore.pre_stack_sha, _remedy(restore),
        )
    else:
        logger.info(
            'Coalesce train %s: member %s pre-stack restore %s (%s -> %s)',
            train_id, member_id, restore.outcome, restore.found_sha, restore.pre_stack_sha,
        )


def _remedy(restore: TipRestore) -> str:
    """The command that finishes a refused restore by hand, once a human has
    confirmed nothing on the branch or in its worktree is worth keeping."""
    if restore.worktree is None:
        return f'git update-ref refs/heads/{restore.branch} {restore.pre_stack_sha}'
    return f'git -C {restore.worktree} reset --hard {restore.pre_stack_sha}'
