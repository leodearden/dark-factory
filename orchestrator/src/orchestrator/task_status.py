"""Task status constants shared across workflow + scheduler.

This module is a thin re-export shim over ``shared.task_statuses`` — the
single cross-package authority for the task status vocabulary (PRD
``plans/task-status-authority-prd.md``, contract C1). The local names below
are kept so existing importers (scheduler.py, harness.py, workflow.py, and
the test suite) are unaffected; the underlying sets are now defined once in
``shared`` and cannot drift.

Terminal-state enforcement (done/cancelled → non-terminal) is authoritative
on the server (fused-memory TaskInterceptor); the orchestrator no longer
guards transitions client-side. See ``reopen_reason`` on
``mcp__fused_memory__set_task_status`` for the audit-trail escape hatch.

``WORKFLOW_PRESERVE_STATUSES`` is the set of statuses the workflow must NOT
overwrite after the steward has set them — a superset of ``TERMINAL_STATUSES``
that also includes ``deferred``, ``blocked``, and ``merge-deferred``.
'deferred' / 'blocked' signal "leave this alone, human will sort it".
'merge-deferred' is the non-terminal holding state for atomic-train members:
a train member sits here between own-verify-green and group-merge, and the
workflow must not re-execute it (the train-group-merge worker owns the
transition to 'done').
"""

from __future__ import annotations

from shared.task_statuses import ACTIVE, TERMINAL, WORKFLOW_PRESERVE

# Re-exported as `frozenset[str]` (rather than the shared `frozenset[TaskStatus]`
# StrEnum type) so pyright's `x in TERMINAL_STATUSES`-style containment checks
# keep narrowing `str | None` down to `str` at the many existing call sites
# (workflow.py, harness.py, scheduler.py). `frozenset` is covariant, so this
# annotation is a pure static-typing restatement — same objects, same members,
# no behaviour change. TaskStatus is a StrEnum, so its members are genuine
# `str` instances either way.
TERMINAL_STATUSES: frozenset[str] = TERMINAL
ACTIVE_TASK_STATUSES: frozenset[str] = ACTIVE
WORKFLOW_PRESERVE_STATUSES: frozenset[str] = WORKFLOW_PRESERVE

__all__ = ['TERMINAL_STATUSES', 'ACTIVE_TASK_STATUSES', 'WORKFLOW_PRESERVE_STATUSES']
