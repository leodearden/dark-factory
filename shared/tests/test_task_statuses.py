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

import pytest

from shared.task_statuses import ACTIVE, STATUS_TRIGGERS, TERMINAL, WORKFLOW_PRESERVE, TaskStatus

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


# ---------------------------------------------------------------------------
# Pair 2 — derived frozensets (step-3 RED / step-4 GREEN)
# ---------------------------------------------------------------------------


class TestDerivedFrozensets:
    def test_terminal_parity(self):
        assert TERMINAL == _LEGACY_TERMINAL

    def test_active_parity(self):
        # ACTIVE is the complement of TERMINAL, so it picks up 'infra-hold' too.
        assert _LEGACY_ACTIVE | {'infra-hold'} == ACTIVE

    def test_workflow_preserve_parity(self):
        assert WORKFLOW_PRESERVE == _LEGACY_WORKFLOW_PRESERVE

    def test_status_triggers_parity(self):
        assert STATUS_TRIGGERS == _LEGACY_STATUS_TRIGGERS

    def test_all_four_are_frozensets(self):
        assert isinstance(TERMINAL, frozenset)
        assert isinstance(ACTIVE, frozenset)
        assert isinstance(WORKFLOW_PRESERVE, frozenset)
        assert isinstance(STATUS_TRIGGERS, frozenset)

    def test_terminal_active_partition(self):
        # TERMINAL and ACTIVE must exhaustively and disjointly partition the
        # full TaskStatus vocabulary.
        assert set(TaskStatus) == TERMINAL | ACTIVE
        assert TERMINAL.isdisjoint(ACTIVE)

    def test_infra_hold_is_active_non_terminal_non_dispatchable(self):
        # D3: infra-hold is active, non-terminal, and NOT the same as 'pending'
        # (dispatch legality itself lives in the scheduler, not here).
        assert TaskStatus.INFRA_HOLD in ACTIVE
        assert TaskStatus.INFRA_HOLD not in TERMINAL
        assert TaskStatus.INFRA_HOLD != TaskStatus.PENDING

    def test_workflow_preserve_and_status_triggers_are_subsets(self):
        assert set(TaskStatus) >= WORKFLOW_PRESERVE
        assert set(TaskStatus) >= STATUS_TRIGGERS


# ---------------------------------------------------------------------------
# Tracking placeholder — cross-package drift guard (reviewer finding, task
# 2163 amendment pass). NOT runnable yet: see skip reason.
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason=(
        'Placeholder for the cross-package drift guard that supersedes the '
        '_LEGACY_* inlined snapshots above. Those snapshots pin what the legacy '
        'sets looked like on 2026-07-06 but are never re-checked against the '
        'live source files, so a legacy set can drift before rho1a/omega2 '
        'rewire its call site to import from shared.task_statuses. Unskip once '
        'BOTH: (1) rho1a/omega2 have rewired the call sites below to import '
        'from shared.task_statuses (so the assertions compare a re-export to '
        "itself rather than two independently-maintained literals), and (2) "
        'orchestrator/fused_memory are importable from this test environment '
        '(shared/tests/conftest.py currently only puts shared/src on sys.path). '
        'Cross-package import is intentionally avoided elsewhere in this '
        'codebase for this exact problem — see the mirrored-not-imported '
        'precedent at orchestrator/tests/test_scheduler.py::'
        'test_active_task_statuses_matches_fused_memory — so item (2) may '
        'require deliberate test-env wiring rather than falling out for free.'
    )
)
class TestCrossPackageDriftGuardPlaceholder:
    """Shape of the future guard. Not runnable yet — see the skip reason."""

    def test_orchestrator_task_status_matches_shared(self):
        from orchestrator.task_status import (
            ACTIVE_TASK_STATUSES,
            TERMINAL_STATUSES,
            WORKFLOW_PRESERVE_STATUSES,
        )

        assert TERMINAL_STATUSES == TERMINAL
        assert ACTIVE_TASK_STATUSES == ACTIVE
        assert WORKFLOW_PRESERVE_STATUSES == WORKFLOW_PRESERVE

    def test_fused_memory_task_interceptor_matches_shared(self):
        from fused_memory.middleware.task_interceptor import TERMINAL_STATUSES, TaskInterceptor

        assert TERMINAL_STATUSES == TERMINAL
        assert TaskInterceptor.STATUS_TRIGGERS == STATUS_TRIGGERS

    def test_fused_memory_task_filter_matches_shared(self):
        from fused_memory.reconciliation.task_filter import ACTIVE_TASK_STATUSES

        assert ACTIVE_TASK_STATUSES == ACTIVE
