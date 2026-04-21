"""Task status FSM: guards terminal state transitions.

Design: permissive — only terminal states are guarded. All other
transitions are allowed to avoid breaking legitimate flows like
blocked->done (manual fix) or in-progress->pending (crash recovery).
"""

from __future__ import annotations

TERMINAL_STATUSES: frozenset[str] = frozenset({'done', 'cancelled'})

# Statuses the workflow must NOT overwrite after the steward has set them.
# Superset of TERMINAL_STATUSES: also includes 'deferred' and 'blocked',
# which the steward uses to signal "leave this alone, human will sort it".
# TERMINAL_STATUSES remains the FSM guard for set_task_status transitions —
# this is a separate concern (workflow self-overwrite after steward).
WORKFLOW_PRESERVE_STATUSES: frozenset[str] = frozenset(
    {'done', 'cancelled', 'deferred', 'blocked'}
)


def is_valid_transition(from_status: str | None, to_status: str) -> bool:
    """Return True if the status transition is allowed.

    Rules:
    - None (unknown) -> anything: True (fail-open for first-time tasks)
    - terminal -> same (idempotent): True
    - terminal -> different: False (guard)
    - anything else: True (permissive for non-terminal states)
    """
    if from_status is None:
        return True
    if from_status == to_status:
        return True
    return from_status not in TERMINAL_STATUSES
