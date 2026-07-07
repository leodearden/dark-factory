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

from shared.task_statuses import ACTIVE, TERMINAL, WORKFLOW_PRESERVE, TaskStatus

# Re-exported as `frozenset[str]` (rather than the shared `frozenset[TaskStatus]`
# StrEnum type) so pyright's `x in TERMINAL_STATUSES`-style containment checks
# keep narrowing `str | None` down to `str` at the many existing call sites
# (workflow.py, harness.py, scheduler.py). TaskStatus is a StrEnum, so its
# members are genuine `str` instances either way — this annotation is a pure
# static-typing restatement for TERMINAL_STATUSES/WORKFLOW_PRESERVE_STATUSES,
# which are byte-identical pure re-exports of shared.TERMINAL/WORKFLOW_PRESERVE.
#
# ACTIVE_TASK_STATUSES is DELIBERATELY NOT a raw re-export of shared.ACTIVE.
# shared.ACTIVE is the complement of TERMINAL and therefore includes the new
# `infra-hold` status (D3) — see the MIGRATION HAZARD note at
# shared/task_statuses.py:71-79. Handing that 7-member set straight to the
# scheduler's active-fetch consumers — acquire_next's status_map/age-anchor/
# dep-backfill seeding (scheduler.py:3350), _get_train_members' train-
# completion accounting (scheduler.py:1464), harness auto-eval discovery
# (harness.py:1772/1800/5361), and workflow's park-sibling scan
# (workflow.py:8122) — has NOT been audited: none of them has been confirmed
# to behave correctly if an infra-hold task shows up in their fetch results.
# Subtracting `TaskStatus.INFRA_HOLD` restores the exact 6-member set this
# constant held before this task, so the consolidation is genuinely
# behaviour-preserving today, while still sourcing from shared (no
# reintroduced string literals, so shared remains the single authority and
# drift stays impossible). Auditing each consumer above and enabling
# infra-hold in this fetch set is owned by the later infra-hold task (omega4,
# PRD C7/D3, migration sequence epsilon1/omega3/omega4).
TERMINAL_STATUSES: frozenset[str] = TERMINAL
ACTIVE_TASK_STATUSES: frozenset[str] = ACTIVE - {TaskStatus.INFRA_HOLD}
WORKFLOW_PRESERVE_STATUSES: frozenset[str] = WORKFLOW_PRESERVE

__all__ = ['TERMINAL_STATUSES', 'ACTIVE_TASK_STATUSES', 'WORKFLOW_PRESERVE_STATUSES']
