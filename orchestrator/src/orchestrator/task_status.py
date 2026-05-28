"""Task status constants shared across workflow + scheduler.

Terminal-state enforcement (done/cancelled → non-terminal) is authoritative
on the server (fused-memory TaskInterceptor); the orchestrator no longer
guards transitions client-side. See ``reopen_reason`` on
``mcp__fused_memory__set_task_status`` for the audit-trail escape hatch.
"""

from __future__ import annotations

TERMINAL_STATUSES: frozenset[str] = frozenset({'done', 'cancelled'})

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
