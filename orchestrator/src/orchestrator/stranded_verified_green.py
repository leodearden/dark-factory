"""Verified-green stranded-remediation detector (stranding-remediation-
scheduler-ergonomics-prd.md leaf α).

The stranded-blocked reaper (``Harness._reconcile_one_stranded``) normally
re-files an L1 for a task left ``blocked`` with no open escalation and no live
claimant; the auto-watcher then resolves it with ``resume`` → a
``blocked``→``pending`` re-pend.  The incident this module addresses (reify
5260) is that the re-pend lands in a *paused* scheduler that never
re-dispatches, so verified-green lane work sits stranded for hours.

The fix: before re-filing, detect the *verified-green shape* and, on a match,
submit the branch DIRECTLY to the merge queue (which runs even under a
scheduler pause) instead of re-pending.  This module houses the pure detector
so it is unit-testable against lightweight fakes / a tmp-git ``GitOps``
without a full ``Harness`` — mirroring the testable-free-function convention
of ``task_ground_truth.py`` / ``task_runtime.py`` / ``landing_evidence.py``.

The whole detector is wrapped fail-safe: any missing signal or any exception
yields *no match* (``None``), never a false positive — the "never bypasses"
safety posture (PRD §2.2).  The feature is naturally inert on non-pooled
projects (no ASSIGNED lane record → ``None`` → today's re-file path).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from orchestrator.artifacts import TaskArtifacts
from orchestrator.event_store import EventStore, EventType
from orchestrator.lane_lifecycle import LaneRecord, LaneState
from orchestrator.task_runtime import _derive_phase

if TYPE_CHECKING:
    from orchestrator.git_ops import GitOps

logger = logging.getLogger(__name__)

__all__ = [
    'VerifiedGreenMatch',
    'detect_verified_green',
    'last_verified_green_tip',
]


def last_verified_green_tip(
    event_store: EventStore | None, task_id: str | None,
) -> str | None:
    """Return the tip_sha of the task's LATEST passed ``workflow_verify`` row.

    Reads ``EventType.workflow_verify`` rows for *task_id* run-agnostically via
    :meth:`EventStore.fetch_events_by_type_all_runs` (rows ordered by id
    ascending) and returns the ``data.tip_sha`` of the latest row that both has
    a truthy ``data.passed`` AND carries a non-empty ``data.tip_sha`` string.

    Semantics (mirrors ``merge_disposition._branch_pre_merge_verify_green`` but
    returns the tip instead of a bool):

    - A later passed-with-tip row wins over an earlier one.
    - A later FAILED re-verify does not erase the latest passed-with-tip green.
    - A later passed row *lacking* a tip_sha does not shadow an earlier
      passed-WITH-tip row — the latest passed-WITH-tip wins.

    Returns ``None`` when *event_store* or *task_id* is None, when no passed
    row carries a tip_sha, when there are no rows, or on any read error
    (fail-safe: log and degrade, never raises).  Cross-run durability matters
    because the strand can span an orchestrator restart, so the green may live
    under a prior ``run_id``.
    """
    if event_store is None or task_id is None:
        return None
    try:
        rows = event_store.fetch_events_by_type_all_runs(
            EventType.workflow_verify, task_id=task_id,
        )
        for row in reversed(rows):
            data = row.get('data') or {}
            tip = data.get('tip_sha')
            if data.get('passed') and isinstance(tip, str) and tip:
                return tip
        return None
    except Exception:
        logger.warning(
            'last_verified_green_tip: event-store read failed for task_id=%s; '
            'degrading to None (fail-safe)',
            task_id,
            exc_info=True,
        )
        return None


@dataclass
class VerifiedGreenMatch:
    """A stranded task whose warm lane holds verified-green, mergeable work.

    - ``lane``: the warm-lane name (``all_records()`` key) — used to derive the
      ``.task-meta`` root for the plan read.
    - ``branch``: the lane branch (``record.branch`` or the configured
      ``task/<id>`` fallback) whose tip is the verified-green tip.
    - ``tip_sha``: the lane branch tip == the last passed ``workflow_verify``
      tip; carried as the MergeRequest ``snapshot_tip``.
    - ``worktree``: the resolved on-disk lane worktree — the MergeRequest
      worktree.
    """

    lane: str
    branch: str
    tip_sha: str
    worktree: Path


async def detect_verified_green(
    task_id: str,
    *,
    git_ops: GitOps,
    event_store: EventStore | None,
    worktree_resolver: Callable[[str], Path],
) -> VerifiedGreenMatch | None:
    """Detect the verified-green shape for a stranded ``blocked`` task (PRD §2.1).

    Returns a :class:`VerifiedGreenMatch` iff ALL three hold:

    (a) the task's warm lane has a durable ``LaneRecord`` at
        ``LaneState.ASSIGNED`` (NOT ``IN_USE`` — that means a live workflow
        holds the lane, which the reaper already treats as a live claimant);
    (b) the lane branch tip (``git_ops.resolve_branch_sha``) equals the task's
        last passed ``workflow_verify`` tip (:func:`last_verified_green_tip`,
        read cross-run so it survives the restart the strand may span); AND
    (c) the lane's ``plan.json`` has every step done
        (``_derive_phase(...) == 'DONE'``).

    Any miss — or ANY exception from a collaborator — yields ``None`` (fail-safe:
    never a false positive that would auto-submit a merge; PRD §2.2).  Naturally
    inert on non-pooled projects: no ASSIGNED lane record → ``None`` → the
    caller falls through to today's re-file path.
    """
    try:
        assigned: tuple[str, LaneRecord] | None = None
        for lane_name, record in git_ops._lane_lifecycle.all_records().items():
            if record.task_id == task_id and record.state == LaneState.ASSIGNED:
                assigned = (lane_name, record)
                break
        if assigned is None:
            return None
        lane_name, record = assigned

        branch = record.branch or f'{git_ops.config.branch_prefix}{task_id}'
        tip = await git_ops.resolve_branch_sha(branch)
        if tip is None:
            return None

        verified = last_verified_green_tip(event_store, task_id)
        if verified is None or tip != verified:
            return None

        worktree = worktree_resolver(task_id)
        meta_root = TaskArtifacts.meta_root_for(git_ops.worktree_base, lane_name)
        plan = TaskArtifacts(worktree, meta_root).read_plan()
        if _derive_phase(plan) != 'DONE':
            return None

        return VerifiedGreenMatch(
            lane=lane_name, branch=branch, tip_sha=tip, worktree=worktree,
        )
    except Exception:
        logger.warning(
            'detect_verified_green: fail-safe degrade to None for task_id=%s',
            task_id,
            exc_info=True,
        )
        return None
