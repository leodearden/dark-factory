"""Builds and files the halt-owning L1 for a merge outcome that halted the merge queue."""

from __future__ import annotations

import logging
import types
from collections.abc import Mapping
from typing import TYPE_CHECKING, NamedTuple, Protocol

if TYPE_CHECKING:
    from escalation.models import Escalation
    from escalation.queue import EscalationQueue

    from orchestrator.merge_lane.types import MergeOutcome

logger = logging.getLogger(__name__)

__all__ = [
    'MERGE_HALT_CATEGORY_BY_STATUS',
    'MERGE_HALT_STATUSES',
    'MergeHaltControl',
    'MergeHaltEscalationText',
    'build_merge_halt_escalation_text',
    'submit_halt_owning_escalation',
]

MERGE_HALT_CATEGORY_BY_STATUS: Mapping[str, str] = types.MappingProxyType({
    'wip_halted': 'wip_conflict',
    'done_wip_recovery': 'wip_conflict',
    'wip_recovery_no_advance': 'wip_conflict',
    'unmerged_state': 'unmerged_state',
    'stash_failed': 'stash_failed',
})

MERGE_HALT_STATUSES: frozenset[str] = frozenset(MERGE_HALT_CATEGORY_BY_STATUS)


class MergeHaltEscalationText(NamedTuple):
    category: str
    summary: str
    detail: str


class MergeHaltControl(Protocol):
    """The merge worker's halt-ownership surface; SpeculativeMergeWorker is one."""

    @property
    def is_wip_halted(self) -> bool: ...
    @property
    def halt_owner_esc_id(self) -> str | None: ...
    def set_halt_owner(self, esc_id: str) -> None: ...
    def unhalt_wip(self, reason: str | None = None) -> None: ...


def build_merge_halt_escalation_text(
    status: str,
    outcome: MergeOutcome,
    *,
    task_id: str,
    branch_name: str | None = None,
    train_id: str | None = None,
) -> MergeHaltEscalationText:
    """Category, summary and detail of the L1 for one halt *status*.

    *branch_name* labels a single-task merge and *train_id* a train merge; they
    are mutually exclusive.
    """
    category = MERGE_HALT_CATEGORY_BY_STATUS.get(status, 'wip_conflict')
    is_train = train_id is not None

    # Context label used in the detail intro sentence.
    if is_train:
        merge_ctx = f'Train merge for task {task_id} (train {train_id!r})'
    elif branch_name is not None:
        merge_ctx = f'Merge for task {task_id} (branch {branch_name})'
    else:
        merge_ctx = f'Merge for task {task_id}'

    if status == 'unmerged_state':
        if is_train:
            summary = (
                f'Train member {task_id} blocked: '
                f'project_root has unresolved (UU/AA/DD) markers'
            )
        else:
            summary = (
                'project_root has unresolved (UU/AA/DD) markers — '
                'merge queue halted'
            )
        detail = (
            f'{merge_ctx} was blocked because project_root already has unresolved '
            f'merge conflicts (UU/AA/DD markers) from a prior, unrelated event — '
            f'the merge queue refuses to stash/advance over a partially resolved tree.\n\n'
            f'Action required: inspect ``git status`` in project_root, '
            f'resolve the existing merge state (``git mergetool`` / edit '
            f'the conflicted files / ``git reset`` to abandon the prior '
            f'merge), then resolve this escalation to un-halt the merge queue.\n\n'
            f'Manual intervention required — do NOT let automated tooling '
            f'resolve this escalation.'
        )

    elif status == 'done_wip_recovery':
        recovery_branch = outcome.recovery_branch or '(unknown)'
        if is_train:
            summary = (
                f'Train member {task_id}: stash pop conflict — '
                f'WIP preserved on {recovery_branch}'
            )
            wip_clause = (
                'uncommitted WIP produced conflicts '
                '(split-brain: merge landed, members NOT flipped to done).'
            )
            # Task 1599 / Suggestion 4: the merge is already on main.
            # Instruct the human to flip all members done before resolving
            # so the re-dispatch does not re-attempt the already-landed merge.
            post_resolve = (
                'Note: the merge commit is already on main.  Before resolving this '
                'escalation, ensure all train members are marked done — the train '
                're-dispatches on unhalt and re-entering the merge queue with an '
                'already-merged tip may produce a confusing split result.\n\n'
                'Resolve this escalation to un-halt the merge queue.'
            )
        else:
            summary = f'Stash pop conflict — WIP preserved on {recovery_branch}'
            wip_clause = 'your uncommitted WIP produced conflicts.'
            post_resolve = 'Resolve this escalation to un-halt the merge queue.'
        detail = (
            f'{merge_ctx} landed on main successfully, but the stash pop of '
            f'{wip_clause}\n\n'
            f'Your WIP has been preserved on branch: {recovery_branch}\n\n'
            f'To recover:\n'
            f'  git checkout {recovery_branch}\n'
            f'  # Review and cherry-pick or reapply your changes\n\n'
            f'{post_resolve}'
        )

    elif status == 'wip_recovery_no_advance':
        recovery_branch = outcome.recovery_branch or '(unknown)'
        if is_train:
            summary = (
                f'Train member {task_id}: stash pop conflict (no advance) — '
                f'WIP on {recovery_branch}'
            )
            no_advance_intro = (
                f'{merge_ctx} did NOT advance main (CAS failure path). '
                f'A stash pop conflict occurred, leaving'
            )
        else:
            summary = (
                f'Stash pop conflict (merge did not advance) — '
                f'WIP on {recovery_branch}'
            )
            no_advance_intro = (
                f'{merge_ctx} did NOT advance main. '
                f'A stash pop conflict occurred after a CAS failure, leaving'
            )
        detail = (
            f'{no_advance_intro} the working tree in an unresolvable state.\n\n'
            f'Your WIP has been preserved on branch: {recovery_branch}\n\n'
            f'The merge queue has been halted. To recover:\n'
            f'  git checkout {recovery_branch}\n'
            f'  # Review and reapply your changes to a clean branch\n\n'
            f'Manual intervention required — do NOT let automated tooling '
            f'resolve this escalation. Resolve this escalation to un-halt '
            f'the merge queue.'
        )

    elif status == 'stash_failed':
        # SHARED main-checkout-hygiene fault (task 2758): advance_main could
        # not park project_root's dirty tracked WIP before advancing. Unlike
        # wip_halted (WIP overlapping the merge diff), these files need not
        # overlap the diff at all — project_root's tree is persistently
        # dirty, so EVERY task landing fails identically until a human
        # cleans it. The queue is halted so exactly ONE escalation is filed.
        dirty = outcome.dirty_files or []
        dirty_head = ', '.join(dirty[:5]) if dirty else '(unknown)'
        if is_train:
            summary = (
                f'Train member {task_id} blocked: project_root park failed '
                f'(dirty tracked tree) — {dirty_head}'
            )
        else:
            summary = (
                f'project_root park failed (dirty tracked tree) — '
                f'merge queue halted: {dirty_head}'
            )
        dirty_block = (
            '\n'.join(f'  - {f}' for f in dirty) if dirty
            else '  (dirty tracked file list unavailable)'
        )
        detail = (
            f'{merge_ctx} was blocked because advance_main could not park '
            f'the uncommitted WIP in project_root before advancing: '
            f'project_root has dirty TRACKED file(s) that could not be '
            f'stashed. This is a shared main-checkout infrastructure/hygiene '
            f'incident — it recurs identically for every task landing until '
            f'project_root is cleaned, so the merge queue has been halted '
            f'(one escalation, not one per stalled task).\n\n'
            f'Dirty tracked file(s):\n'
            f'{dirty_block}\n\n'
            f'Action required: inspect ``git status`` in project_root, then '
            f'commit or clean the persistently-dirty tracked file(s) '
            f'(``git commit`` / ``git checkout -- <path>`` / ``git stash``), '
            f'then resolve this escalation to un-halt the merge queue.\n\n'
            f'Manual intervention required — do NOT let automated tooling '
            f'resolve this escalation.'
        )

    else:
        # wip_halted (and any future halt-inducing status the probe catches)
        overlap = outcome.overlap_files or []
        if is_train:
            summary = (
                f'Train member {task_id}: WIP overlaps merge diff: '
                + ', '.join(overlap[:5])
            )
            subject_ctx = f'train member {task_id} (train {train_id!r})'
        else:
            summary = f'WIP overlaps merge diff: {", ".join(overlap[:5])}'
            branch_str = f' (branch {branch_name})' if branch_name else ''
            subject_ctx = f'task {task_id}{branch_str}'
        detail = (
            f'Merge for {subject_ctx} was blocked because uncommitted '
            f'work in project_root overlaps the merge diff.\n\n'
            f'Overlapping files:\n'
            + '\n'.join(f'  - {f}' for f in overlap)
            + '\n\nAction required: commit or stash the WIP, then resolve this '
            'escalation to un-halt the merge queue and retry.'
        )


    return MergeHaltEscalationText(category, summary, detail)


def submit_halt_owning_escalation(
    queue: EscalationQueue, merge_halt: MergeHaltControl | None, esc: Escalation,
) -> None:
    """Submit *esc*, then register it as the owner of *merge_halt*.

    Submit comes first: an owner with no pending record can never be resolved,
    so the halt would be permanent. If submit raises while the halt is engaged
    and ownerless, the halt is released before the exception propagates, so a
    failed submit never leaves the queue halted with nobody to unhalt it (the
    task 1765 orphan guard). A foreign owner's halt is never released.
    """
    try:
        queue.submit(esc)
    except BaseException:
        if (
            merge_halt is not None
            and merge_halt.is_wip_halted
            and merge_halt.halt_owner_esc_id is None
        ):
            try:
                merge_halt.unhalt_wip(reason='halt_escalation_submit_failed')
            except Exception:
                logger.exception(
                    'Task %s: unhalt_wip failed during submit-failure cleanup '
                    '(original submit exception will still propagate)',
                    esc.task_id,
                )
        raise
    if merge_halt is not None:
        merge_halt.set_halt_owner(esc.id)
