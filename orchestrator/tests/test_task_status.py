"""Unit tests for the task_status constants module.

The terminal-state FSM used to live in this file as ``is_valid_transition``;
enforcement has moved to the server (fused-memory TaskInterceptor), so the
client-side FSM is gone. What remains is the constant set that workflow.py
still uses to distinguish terminal outcomes after the steward runs.
"""

from shared.task_statuses import TaskStatus

from orchestrator.task_status import (
    TERMINAL_STATUSES,
    WORKFLOW_PRESERVE_STATUSES,
    is_infra_held,
)


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
        done.  The preserve-decision in workflow.py reads this set at
        runtime, so membership here is the authoritative signal.
        """
        assert 'merge-deferred' in WORKFLOW_PRESERVE_STATUSES


class TestIsInfraHeld:
    """is_infra_held is the single source of truth for the infra-hold

    exemption (PRD C7/D3): it keys on the first-class ``status`` field, not
    the retired ``metadata.infra_hold`` boolean, so the write/skip/resume
    sites cannot drift from one another.
    """

    def test_status_infra_hold_string_is_true(self):
        assert is_infra_held({'status': 'infra-hold'}) is True

    def test_status_infra_hold_enum_member_is_true(self):
        assert is_infra_held({'status': TaskStatus.INFRA_HOLD}) is True

    def test_status_in_progress_is_false(self):
        assert is_infra_held({'status': 'in-progress'}) is False

    def test_status_blocked_is_false(self):
        assert is_infra_held({'status': 'blocked'}) is False

    def test_missing_status_key_is_false(self):
        assert is_infra_held({}) is False

    def test_none_task_is_false(self):
        assert is_infra_held(None) is False

    def test_legacy_metadata_flag_without_status_is_false(self):
        """Proves the accessor keys on status, not the retired metadata flag.

        A task that still carries the legacy ``metadata.infra_hold`` boolean
        but whose status is something other than 'infra-hold' must NOT be
        reported as held — the metadata flag is retired and must not be
        consulted.
        """
        task = {'status': 'in-progress', 'metadata': {'infra_hold': True}}
        assert is_infra_held(task) is False
