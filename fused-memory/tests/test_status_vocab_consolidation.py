"""Cross-module drift guards: fused_memory status-vocab copies vs shared.task_statuses.

PRD `plans/task-status-authority-prd.md` (C1/C2, finding 6.4) mandates that the
duplicated status-set copies across fused_memory are deleted and re-imported
from ``shared.task_statuses`` — the single source of truth. These tests pin
that re-export by object identity (``is``), not just value equality, so a
future edit that reintroduces a local literal (equal by accident) still fails
the guard.

This is the in-package equivalent of the skipped cross-package drift guard in
``shared/tests/test_task_statuses.py`` (that guard stays skipped pending the
omega2 orchestrator rewire and shared test-env sys.path wiring — both
fused_memory and shared are importable here today).
"""

from __future__ import annotations

from shared.task_statuses import ACTIVE, TERMINAL, TaskStatus

from fused_memory.middleware.task_interceptor import TERMINAL_STATUSES
from fused_memory.reconciliation.task_filter import (
    ACTIVE_TASK_STATUSES,
    filter_task_tree,
    summarize_statuses,
)


def test_task_interceptor_terminal_matches_shared():
    """task_interceptor.TERMINAL_STATUSES must be shared.TERMINAL, not an equal copy."""
    assert TERMINAL_STATUSES == TERMINAL
    assert TERMINAL_STATUSES is TERMINAL


def test_task_filter_active_matches_shared():
    """task_filter.ACTIVE_TASK_STATUSES must be shared.ACTIVE, not an equal copy."""
    assert ACTIVE_TASK_STATUSES == ACTIVE
    assert ACTIVE_TASK_STATUSES is ACTIVE
    assert 'infra-hold' in ACTIVE_TASK_STATUSES


def test_summarize_statuses_counts_infra_hold_as_active():
    """infra-hold is non-terminal — summarize_statuses must count it as active, not other."""
    result = summarize_statuses({'1': 'infra-hold'})
    assert result == {'total': 1, 'done': 0, 'cancelled': 0, 'active': 1, 'other': 0}


def test_filter_task_tree_classifies_infra_hold_as_active():
    """filter_task_tree must count an infra-hold task as active (not other_count)."""
    task_tree = {
        'tasks': [
            {'id': '7', 'status': 'infra-hold', 'title': 'Held', 'dependencies': []},
        ]
    }
    result = filter_task_tree(task_tree)
    assert len(result.active_tasks) == 1
    assert result.active_tasks[0]['id'] == '7'
    assert result.other_count == 0


def test_tools_valid_task_statuses_composition():
    """tools.py's validator composition (ACTIVE_TASK_STATUSES | TERMINAL_STATUSES,
    tools.py:636) must equal shared.ACTIVE | shared.TERMINAL == frozenset(TaskStatus),
    with 'infra-hold' included — auto-recomposed once both operands re-export shared.
    """
    composed = ACTIVE_TASK_STATUSES | TERMINAL_STATUSES
    assert composed == ACTIVE | TERMINAL
    assert composed == frozenset(TaskStatus)
    assert 'infra-hold' in composed
