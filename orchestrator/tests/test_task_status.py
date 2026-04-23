"""Unit tests for the task_status constants module.

The terminal-state FSM used to live in this file as ``is_valid_transition``;
enforcement has moved to the server (fused-memory TaskInterceptor), so the
client-side FSM is gone. What remains is the constant set that workflow.py
still uses to distinguish terminal outcomes after the steward runs.
"""

from orchestrator.task_status import TERMINAL_STATUSES, WORKFLOW_PRESERVE_STATUSES


class TestTerminalStatuses:
    def test_done_is_terminal(self):
        assert 'done' in TERMINAL_STATUSES

    def test_cancelled_is_terminal(self):
        assert 'cancelled' in TERMINAL_STATUSES

    def test_blocked_not_terminal(self):
        assert 'blocked' not in TERMINAL_STATUSES

    def test_pending_not_terminal(self):
        assert 'pending' not in TERMINAL_STATUSES

    def test_in_progress_not_terminal(self):
        assert 'in-progress' not in TERMINAL_STATUSES


class TestWorkflowPreserveStatuses:
    def test_superset_of_terminal(self):
        assert TERMINAL_STATUSES <= WORKFLOW_PRESERVE_STATUSES

    def test_includes_deferred(self):
        assert 'deferred' in WORKFLOW_PRESERVE_STATUSES

    def test_includes_blocked(self):
        assert 'blocked' in WORKFLOW_PRESERVE_STATUSES
