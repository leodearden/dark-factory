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

    def test_merge_deferred_not_terminal(self):
        """D1 invariant: merge-deferred is a non-terminal holding state.

        A merge-deferred train member transitions to 'done' (via the
        group-merge worker) or back to 'in-progress' (via re-dispatch
        after a sibling-driven failure). Promoting it to TERMINAL_STATUSES
        would silently break the train state machine (server-side terminal-
        exit FSM would reject the re-open).
        """
        assert 'merge-deferred' not in TERMINAL_STATUSES


class TestWorkflowPreserveStatuses:
    def test_superset_of_terminal(self):
        assert TERMINAL_STATUSES <= WORKFLOW_PRESERVE_STATUSES

    def test_includes_deferred(self):
        assert 'deferred' in WORKFLOW_PRESERVE_STATUSES

    def test_includes_blocked(self):
        assert 'blocked' in WORKFLOW_PRESERVE_STATUSES

    def test_includes_merge_deferred(self):
        """merge-deferred is workflow-preserved: the workflow does not
        re-execute a merge-deferred task on its own — the train-group-
        merge worker is the sole transition path from merge-deferred to
        done.
        """
        assert 'merge-deferred' in WORKFLOW_PRESERVE_STATUSES

    def test_workflow_preserve_decision_skips_merge_deferred(self):
        """Mirrors workflow.py:906 (`if current_status in
        WORKFLOW_PRESERVE_STATUSES:`) — when a task's current status is
        'merge-deferred' (e.g. steward set it during escalation
        resolution), the preserve-decision must fire so the workflow
        exits the resume loop with BLOCKED instead of re-invoking the
        implementer.
        """
        current_status = 'merge-deferred'
        # The exact conditional from workflow.py:906.
        should_preserve = current_status in WORKFLOW_PRESERVE_STATUSES
        assert should_preserve, (
            "'merge-deferred' must be in WORKFLOW_PRESERVE_STATUSES; "
            'workflow.py:906 reads this set to decide whether to '
            'preserve (return BLOCKED) or re-execute. Without the '
            'membership the workflow would re-invoke the implementer '
            'on a task in the train holding state, burning budget and '
            'corrupting train semantics.'
        )
