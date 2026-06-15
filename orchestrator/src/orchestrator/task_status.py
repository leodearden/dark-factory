"""Task status constants shared across workflow + scheduler.

Terminal-state enforcement (done/cancelled → non-terminal) is authoritative
on the server (fused-memory TaskInterceptor); the orchestrator no longer
guards transitions client-side. See ``reopen_reason`` on
``mcp__fused_memory__set_task_status`` for the audit-trail escape hatch.
"""

from __future__ import annotations

TERMINAL_STATUSES: frozenset[str] = frozenset({'done', 'cancelled'})

# Non-terminal (active) statuses — the complement of TERMINAL_STATUSES.
# Mirrors fused-memory's reconciliation/task_filter.ACTIVE_TASK_STATUSES;
# kept as a local constant to avoid a cross-package import at runtime.
# DRIFT RISK: if the server adds a new active status, update this set too —
# stale values would cause tasks in that status to be silently dropped from
# the active fetch and never dispatched.  The guard test
# ``test_active_task_statuses_matches_fused_memory`` in test_scheduler.py
# imports the canonical server-side set and asserts equality, so divergence
# fails CI rather than silently stranding tasks.
# Passed to get_tasks(statuses=...) on the scheduler's hot paths (acquire_next,
# _train_candidates, _build_train_state fallback) to shrink the payload.
ACTIVE_TASK_STATUSES: frozenset[str] = frozenset(
    {
        'pending',
        'in-progress',
        'blocked',
        'deferred',
        'review',
        'merge-deferred',
    }
)

# Statuses the workflow must NOT overwrite after the steward has set them.
# Superset of TERMINAL_STATUSES: also includes 'deferred', 'blocked', and
# 'merge-deferred'. 'deferred' / 'blocked' signal "leave this alone, human
# will sort it". 'merge-deferred' is the non-terminal holding state for
# atomic-train members: a train member sits here between own-verify-green
# and group-merge, and the workflow must not re-execute it (the train-
# group-merge worker owns the transition to 'done').
WORKFLOW_PRESERVE_STATUSES: frozenset[str] = frozenset(
    {'done', 'cancelled', 'deferred', 'blocked', 'merge-deferred'}
)
