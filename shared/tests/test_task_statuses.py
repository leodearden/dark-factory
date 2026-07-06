"""Tests for shared.task_statuses — TaskStatus StrEnum + derived frozensets.

Table A of the task-status-authority contract (PRD plans/task-status-authority-prd.md,
C1 / finding 6.4, decision D1/D3). Pins exact parity between the new ``TaskStatus``
vocabulary and the four legacy status-set copies it supersedes, using literals
verified directly against the working tree on 2026-07-06.

``shared/tests/conftest.py`` only puts ``shared/src`` on ``sys.path`` — the
orchestrator and fused-memory packages are not importable from this test
environment, so the legacy sets are inlined below rather than imported. The
cross-package equality guards that import the rewired constants are added
later by rho1a/omega2 once those call sites switch to importing from shared.

TDD pair 1: TaskStatus enum + full vocabulary parity (GREEN on impl step-2).
TDD pair 2: derived frozensets TERMINAL/ACTIVE/WORKFLOW_PRESERVE/STATUS_TRIGGERS
(GREEN on impl step-4).
"""
from __future__ import annotations

import enum

from shared.task_statuses import TaskStatus

# ---------------------------------------------------------------------------
# Legacy literal sets — verified verbatim against the working tree 2026-07-06.
# ---------------------------------------------------------------------------

# orchestrator/src/orchestrator/task_status.py:11 TERMINAL_STATUSES
# fused-memory/src/fused_memory/middleware/task_interceptor.py:126 TERMINAL_STATUSES
_LEGACY_TERMINAL = frozenset({'done', 'cancelled'})

# orchestrator/src/orchestrator/task_status.py:24-33 ACTIVE_TASK_STATUSES
# fused-memory/src/fused_memory/reconciliation/task_filter.py:217-230 ACTIVE_TASK_STATUSES
_LEGACY_ACTIVE = frozenset(
    {'pending', 'in-progress', 'blocked', 'deferred', 'review', 'merge-deferred'}
)

# orchestrator/src/orchestrator/task_status.py:42-44 WORKFLOW_PRESERVE_STATUSES
_LEGACY_WORKFLOW_PRESERVE = frozenset(
    {'done', 'cancelled', 'deferred', 'blocked', 'merge-deferred'}
)

# fused-memory/src/fused_memory/middleware/task_interceptor.py:214 STATUS_TRIGGERS
_LEGACY_STATUS_TRIGGERS = frozenset({'done', 'blocked', 'cancelled', 'deferred'})

# fused-memory/src/fused_memory/server/tools.py:636 _VALID_TASK_STATUSES
_LEGACY_VALID = _LEGACY_ACTIVE | _LEGACY_TERMINAL


# ---------------------------------------------------------------------------
# Pair 1 — TaskStatus enum + full-vocabulary parity (step-1 RED / step-2 GREEN)
# ---------------------------------------------------------------------------


class TestTaskStatusEnum:
    def test_is_str_enum(self):
        assert issubclass(TaskStatus, enum.StrEnum)

    def test_members_are_str_equal(self):
        assert TaskStatus.DONE == 'done'
        assert isinstance(TaskStatus.PENDING, str)

    def test_infra_hold_exists(self):
        assert TaskStatus.INFRA_HOLD == 'infra-hold'

    def test_full_vocabulary_parity(self):
        # Union of the four legacy copies (8 members) plus the new 'infra-hold'
        # value (D3) == exactly the 9-member TaskStatus vocabulary.
        assert {s.value for s in TaskStatus} == _LEGACY_VALID | {'infra-hold'}
        assert len(TaskStatus) == 9
