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

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from orchestrator.artifacts import TaskArtifacts
from orchestrator.event_store import EventStore, EventType
from orchestrator.lane_lifecycle import LaneRecord, LaneState
from orchestrator.task_runtime import _derive_phase

if TYPE_CHECKING:
    from orchestrator.config import ModuleConfig, OrchestratorConfig
    from orchestrator.git_ops import GitOps
    from orchestrator.merge_types import MergeOutcome, MergeRequest, QueuedBranch

logger = logging.getLogger(__name__)

__all__ = [
    'DURABLE_MERGE_FAILURE_STATUSES',
    'MERGE_REQUEST_RESUBMIT_GRACE_S',
    'SUCCESS_TRANSIENT_MERGE_STATUSES',
    'VerifiedGreenMatch',
    'detect_verified_green',
    'last_verified_green_tip',
    'merge_request_marker_is_fresh',
    'submit_verified_green_merge_request',
]

DURABLE_MERGE_FAILURE_STATUSES: frozenset[str] = frozenset({
    'conflict', 'blocked', 'error', 'unknown_branch',
    'unmerged_state', 'stash_failed', 'wip_recovery_no_advance',
})
"""``MergeOutcome.status`` values that are a DURABLE merge/verify failure.

Together with :data:`SUCCESS_TRANSIENT_MERGE_STATUSES` this MUST exhaust
``MergeOutcome``'s status Literal — enforced at CI time by
``test_stranded_verified_green.test_status_sets_exhaust_merge_outcome_
vocabulary``.  Both auto-merge submit paths (the stranded reaper and the
architect-desync exit) classify their terminal outcome against these sets, and
BOTH treat a status found in NEITHER set LOUDLY as a durable failure, so a
new/unclassified outcome can never silently no-op into a stranded task with no
operator signal.
"""

SUCCESS_TRANSIENT_MERGE_STATUSES: frozenset[str] = frozenset({
    'done', 'already_merged', 'done_wip_recovery', 'superseded',
    'wip_halted',
})
"""``MergeOutcome.status`` values that are a success or a transient outcome.

Neither warrants a failure record: the happy landing is reported by the
caller's own success branch, and a transient/superseded outcome is re-driven
by a later sweep.  The branch + lane are preserved by omission.
"""

MERGE_REQUEST_RESUBMIT_GRACE_S: float = 30 * 60.0  # 30 minutes
"""How long a submit marker suppresses a re-submit of the same tip.

Shared by both verified-green merge-submit callers (the stranded reaper's
``metadata.stranded_merge_request`` and the architect desync exit's
``metadata.architect_merge_request``) — see
:func:`merge_request_marker_is_fresh`.
"""


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


# ---------------------------------------------------------------------------
# Shared merge-submit primitives.
#
# TWO callers submit a verified-green branch straight to the merge queue:
#
#   1. the stranded-blocked reaper (``Harness._maybe_submit_stranded_verified_
#      green``), when the periodic sweep finds a task stranded on green work; and
#   2. the architect's ``report_ready_to_merge`` desync exit
#      (``TaskWorkflow._handle_ready_to_merge_report``), when the architect
#      reports that only the physical merge is missing.
#
# They differ ONLY in the ``source`` event tag, the durable marker key, and the
# terminal done-callback — so the build→register-callback→enqueue→stamp
# plumbing and its dedup predicate live here once rather than in lockstep
# copies.  They live in THIS module (not harness.py) because workflow.py must
# call them too, and harness.py already runs TaskWorkflow — a workflow→harness
# import would be circular.  This module is a leaf.
# ---------------------------------------------------------------------------


def merge_request_marker_is_fresh(
    marker: Any,
    tip_sha: str,
    *,
    grace_s: float = MERGE_REQUEST_RESUBMIT_GRACE_S,
) -> bool:
    """Return True iff *marker* records a still-fresh submit for *tip_sha*.

    The submit race-guard shared by both verified-green merge callers: a
    durable ``{tip_sha, request_id, submitted_at}`` metadata marker suppresses
    a re-submit only when it is a well-formed dict whose ``tip_sha`` equals the
    tip about to be submitted AND whose ``submitted_at`` is within *grace_s* of
    now.

    Fail-safe: any malformed / non-dict / unparseable marker (or a mismatched
    tip / stale timestamp) returns False, so the caller falls through to a
    fresh submit rather than a wedged skip — a lost marker must never
    permanently strand the task.
    """
    if not isinstance(marker, dict):
        return False
    if marker.get('tip_sha') != tip_sha:
        return False
    submitted_at = marker.get('submitted_at')
    if not isinstance(submitted_at, str) or not submitted_at:
        return False
    try:
        ts = datetime.fromisoformat(submitted_at)
    except (ValueError, TypeError):
        return False
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    # A future timestamp (clock skew) yields a negative age < grace → treated
    # as fresh (skip), which is the safe side: never a duplicate submit.
    age_s = (datetime.now(UTC) - ts).total_seconds()
    return age_s < grace_s


async def submit_verified_green_merge_request(
    *,
    task_id: str,
    branch: QueuedBranch,
    worktree: Path,
    tip_sha: str,
    config: OrchestratorConfig,
    module_configs: list[ModuleConfig],
    merge_queue: asyncio.Queue,
    event_store: EventStore | None,
    source: str,
    marker_key: str,
    done_callback: Callable[[asyncio.Future[MergeOutcome]], None],
    update_task: Callable[[str, dict[str, Any]], Awaitable[Any]],
) -> MergeRequest:
    """Submit *branch* at *tip_sha* to the merge queue and stamp a dedup marker.

    Builds a ``MergeRequest`` (never pre-rebased, no task_files — the merge
    worker's own scoped re-verify is the sole gate; this helper NEVER advances
    main), registers *done_callback* on its future, enqueues it tagged with
    *source*, and best-effort stamps ``metadata[marker_key] =
    {tip_sha, request_id, submitted_at}`` via *update_task*.

    The marker stamp is deliberately best-effort: the request is already on the
    queue by then, and a lost marker only risks a benign re-submit on the next
    sweep, so a scheduler write failure must never abort the remediation.  It
    is written as a single key so merge-mode writes preserve siblings — and so
    the two callers' markers never clobber each other.

    Returns the submitted ``MergeRequest`` (callers read ``.request_id``).
    """
    from orchestrator.merge_queue import enqueue_merge_request  # noqa: PLC0415
    from orchestrator.merge_types import MergeRequest  # noqa: PLC0415

    future: asyncio.Future[MergeOutcome] = (
        asyncio.get_running_loop().create_future()
    )
    req = MergeRequest(
        task_id=task_id,
        branch=branch,
        worktree=worktree,
        pre_rebased=False,
        task_files=None,
        module_configs=module_configs,
        config=config,
        result=future,
        snapshot_tip=tip_sha,
    )
    req.result.add_done_callback(done_callback)
    await enqueue_merge_request(merge_queue, req, event_store, source=source)

    try:
        await update_task(
            task_id,
            {
                marker_key: {
                    'tip_sha': tip_sha,
                    'request_id': req.request_id,
                    'submitted_at': datetime.now(UTC).isoformat(),
                },
            },
        )
    except Exception:
        logger.warning(
            'submit_verified_green_merge_request: task %s — failed to stamp '
            '%s marker (benign; only risks a re-submit next sweep)',
            task_id, marker_key,
            exc_info=True,
        )

    return req
