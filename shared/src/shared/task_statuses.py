"""Task status vocabulary — Table A of the task-status-authority contract.

See PRD ``plans/task-status-authority-prd.md`` (C1 / finding 6.4, decisions
D1/D3). ``TaskStatus`` is the union of the four status-set copies that were
previously duplicated across the codebase, plus the new ``infra-hold`` value
(D3):

- ``orchestrator/src/orchestrator/task_status.py`` (``TERMINAL_STATUSES``,
  ``ACTIVE_TASK_STATUSES``, ``WORKFLOW_PRESERVE_STATUSES``)
- ``fused-memory/src/fused_memory/middleware/task_interceptor.py``
  (``TERMINAL_STATUSES``, ``STATUS_TRIGGERS``)
- ``fused-memory/src/fused_memory/reconciliation/task_filter.py``
  (``ACTIVE_TASK_STATUSES``)
- ``fused-memory/src/fused_memory/server/tools.py`` (``_VALID_TASK_STATUSES``)

This module is intentionally PURE: it defines the vocabulary and its derived
partitions only. It does NOT rewire the call sites above to import from here
— that migration is owned by separate tasks (rho1a/omega2).

The module is intentionally NOT re-exported from ``shared/__init__.py``.
Consumers import via the fully-qualified path (``from shared.task_statuses
import TaskStatus, ...``), consistent with the ``mcp_envelope``/``neutral_cwd``/
``config_dir`` sub-module convention.
"""

from __future__ import annotations

import enum

__all__ = [
    "TaskStatus",
    "TERMINAL",
    "ACTIVE",
    "WORKFLOW_PRESERVE",
    "STATUS_TRIGGERS",
]


class TaskStatus(enum.StrEnum):
    """Closed vocabulary of task statuses used across the software factory.

    Members are genuine ``str`` instances (``enum.StrEnum``), so
    ``frozenset[TaskStatus]`` compares and hashes identically to the
    ``frozenset[str]`` copies this module supersedes — a drop-in substitute
    at every legacy call site.
    """

    PENDING = "pending"
    IN_PROGRESS = "in-progress"
    BLOCKED = "blocked"
    DEFERRED = "deferred"
    REVIEW = "review"
    MERGE_DEFERRED = "merge-deferred"
    INFRA_HOLD = "infra-hold"
    DONE = "done"
    CANCELLED = "cancelled"
