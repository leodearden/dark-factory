"""merge_completion — the (b)+(c) run-scoped eligibility predicate for the
merge-stage completion mode (task 2633).

Lets an eligible, mechanically-fixable merge-verify/rebase failure be
autonomously completed by unblock-low-risk without ever shortcutting the
verify/review/amend pipeline. Completion is allowed only when the task's own
solution already cleared the full pipeline before landing jammed:

  (a) block_class == BlockClass.MERGE_VERIFY_RED — the CALLER's precondition.
      This module never sees block_class; ``_spawn_merge_verify_dry_run``
      (orchestrator/src/orchestrator/merge_queue.py) always stamps it on the
      post-merge-verify path, so the caller (``run_dry_run_unblock``) gates
      its call to ``merge_completion_eligible`` on it, not this module.
  (b) VERIFY passed — a ``workflow_verify`` event with truthy
      ``data['passed']``, emitted at the workflow's VERIFY->REVIEW edge
      (workflow.py ``_enter_phase``). Mirrors the read pattern of
      ``merge_disposition._branch_pre_merge_verify_green`` (I5).
  (c) REVIEW passed — a ``phase_enter`` event with ``phase == 'merge'``,
      emitted only when the workflow reaches ``WorkflowState.MERGE`` via the
      clean-review DONE path. Manual ``merge_request`` resubmissions
      (/unblock, /merge-queue, /do) and deterministic tasks never emit this
      event, so (c) cleanly separates a pipeline-completed review from a
      manual "looks green to me" vouch — the proof (b) alone cannot supply.

``merge_completion_eligible(event_store, task_id)`` covers (b)+(c) only, as a
single strict fail-closed boolean: any missing/failed condition, or any read
error, returns False — never raises. ``event_store`` is duck-typed (only
``fetch_events_by_type`` is used), and this module imports nothing beyond
``EventType`` + stdlib logging/typing, keeping it dependency-light like
``orchestrator.b3_gate``.

Run-scoped fail-closed semantics: ``fetch_events_by_type`` reads are filtered
to the CURRENT run_id (see ``EventStore.fetch_events_by_type``, run_id-scoped
by construction). A review that passed in a PRIOR run (e.g. after an
orchestrator restart) therefore reads as no-evidence here, not as a false
positive: ``merge_completion_eligible`` returns False, so the caller emits no
low-risk proposal and the task routes to a human via /unblock instead. This
run-scoped restart gap is a deliberate, owner-ratified tradeoff (task 2633);
a restart-durable ``review_approved`` marker was explicitly deferred as
out-of-scope hardening.
"""

from __future__ import annotations

import logging
from typing import Any

from orchestrator.event_store import EventType

logger = logging.getLogger(__name__)


def merge_completion_eligible(event_store: Any, task_id: str | None) -> bool:
    """Return True iff (b) VERIFY passed AND (c) REVIEW passed for *task_id*,
    both read from *event_store*'s current-run history.

    Strict fail-closed: returns False when *event_store* or *task_id* is
    None, and on any read error (never raises) — see module docstring for
    the run-scoped fail-closed rationale. *event_store* is duck-typed: only
    ``fetch_events_by_type(event_type) -> list[dict]`` is called, matching
    ``orchestrator.event_store.EventStore.fetch_events_by_type``'s shape
    (each row a dict with at least 'task_id', 'phase', 'data').

    (a) ``block_class == BlockClass.MERGE_VERIFY_RED`` is NOT checked here —
    it is the caller's precondition (see module docstring).
    """
    if event_store is None or task_id is None:
        return False
    try:
        verify_rows = event_store.fetch_events_by_type(EventType.workflow_verify)
        verify_passed = any(
            row.get('task_id') == task_id and bool((row.get('data') or {}).get('passed'))
            for row in verify_rows
        )
        if not verify_passed:
            return False

        phase_rows = event_store.fetch_events_by_type(EventType.phase_enter)
        return any(
            row.get('task_id') == task_id and row.get('phase') == 'merge'
            for row in phase_rows
        )
    except Exception:
        logger.warning(
            'merge_completion_eligible: event-store read failed for task_id=%s; '
            'degrading to False (evidence absent, fail-safe)',
            task_id,
            exc_info=True,
        )
        return False
