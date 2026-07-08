"""Workflow state-machine types (W9-β task).

Extracted verbatim from :mod:`orchestrator.workflow`: the ``WorkflowState``
phase enum and the ``WorkflowOutcome`` enum, following the types-module +
re-export-shim precedent established by :mod:`orchestrator.merge_types`
(extracted from ``orchestrator.merge_queue``).  ``orchestrator.workflow``
re-exports every name here through a top-level shim so existing importers
(``from orchestrator.workflow import WorkflowState``, etc.) keep working
unchanged.
"""

from __future__ import annotations

import enum

__all__ = [
    "IllegalTransition",
    "WorkflowOutcome",
    "WorkflowState",
]


class WorkflowState(enum.Enum):
    PLAN = 'plan'
    EXECUTE = 'execute'
    VERIFY = 'verify'
    REVIEW = 'review'
    MERGE = 'merge'
    MERGE_DEFERRED = 'merge-deferred'
    DONE = 'done'
    BLOCKED = 'blocked'
    ESCALATED = 'escalated'
    CANCELLED = 'cancelled'


class WorkflowOutcome(enum.Enum):
    DONE = 'done'
    PLANNED = 'planned'
    BLOCKED = 'blocked'
    REQUEUED = 'requeued'
    ESCALATED = 'escalated'
    CANCELLED = 'cancelled'
    MERGE_DEFERRED = 'merge-deferred'
    SOFT_CANCELLED = 'soft-cancelled'


class IllegalTransition(Exception):
    """Raised by :meth:`WorkflowStateMachine.transition` on an illegal move."""
