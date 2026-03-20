"""Unit tests for the task_status FSM module."""

from orchestrator.task_status import TERMINAL_STATUSES, is_valid_transition


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


class TestIsValidTransition:
    """Test the is_valid_transition() function."""

    # --- Terminal source states are guarded ---

    def test_done_to_blocked_rejected(self):
        assert is_valid_transition('done', 'blocked') is False

    def test_done_to_pending_rejected(self):
        assert is_valid_transition('done', 'pending') is False

    def test_done_to_in_progress_rejected(self):
        assert is_valid_transition('done', 'in-progress') is False

    def test_cancelled_to_pending_rejected(self):
        assert is_valid_transition('cancelled', 'pending') is False

    def test_cancelled_to_blocked_rejected(self):
        assert is_valid_transition('cancelled', 'blocked') is False

    def test_cancelled_to_in_progress_rejected(self):
        assert is_valid_transition('cancelled', 'in-progress') is False

    # --- Idempotent transitions allowed ---

    def test_done_to_done_allowed(self):
        assert is_valid_transition('done', 'done') is True

    def test_cancelled_to_cancelled_allowed(self):
        assert is_valid_transition('cancelled', 'cancelled') is True

    # --- Normal non-terminal transitions allowed ---

    def test_in_progress_to_done_allowed(self):
        assert is_valid_transition('in-progress', 'done') is True

    def test_in_progress_to_blocked_allowed(self):
        assert is_valid_transition('in-progress', 'blocked') is True

    def test_pending_to_in_progress_allowed(self):
        assert is_valid_transition('pending', 'in-progress') is True

    def test_blocked_to_pending_allowed(self):
        assert is_valid_transition('blocked', 'pending') is True

    def test_blocked_to_done_allowed(self):
        """Manual fix: blocked->done must be allowed (legitimate recovery)."""
        assert is_valid_transition('blocked', 'done') is True

    # --- Fail-open when from_status is None (unknown) ---

    def test_none_to_pending_allowed(self):
        assert is_valid_transition(None, 'pending') is True

    def test_none_to_in_progress_allowed(self):
        assert is_valid_transition(None, 'in-progress') is True

    def test_none_to_done_allowed(self):
        assert is_valid_transition(None, 'done') is True

    def test_none_to_blocked_allowed(self):
        assert is_valid_transition(None, 'blocked') is True
