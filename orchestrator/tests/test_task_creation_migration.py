"""Tests for task-creation tool migration: add_task → submit_task + resolve_ticket.

Covers three scopes:
  1. Tool allowlists in STEWARD and DEEP_REVIEWER roles
  2. STEWARD.system_prompt instruction text
  3. DEEP_REVIEWER.system_prompt instruction text
  4. ReviewCheckpoint._build_prompt text
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from orchestrator.agents.roles import DEEP_REVIEWER, STEWARD
from orchestrator.review_checkpoint import ReviewCheckpoint
from orchestrator.verify import VerifyResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_checkpoint() -> ReviewCheckpoint:
    """Construct a minimal ReviewCheckpoint with mocked config (mirrors test_review_checkpoint_cap.py)."""
    config = MagicMock()
    config.project_root = Path('/tmp')
    config.models.deep_reviewer = 'opus'
    config.max_turns.deep_reviewer = 50
    config.budgets.deep_reviewer = 10.0
    config.effort.deep_reviewer = 'max'
    config.backends.deep_reviewer = 'claude'
    config.fused_memory.project_id = 'test-project'
    config.escalation.host = 'localhost'
    config.escalation.port = 9999

    mcp = MagicMock()
    mcp.mcp_config_json.return_value = {'mcpServers': {}}

    return ReviewCheckpoint(config, mcp=mcp, usage_gate=None)


_PHASE1_RESULT = VerifyResult(
    passed=True, summary='OK', test_output='', lint_output='', type_output=''
)


# ---------------------------------------------------------------------------
# Step 1 / Step 2: Tool allowlists
# ---------------------------------------------------------------------------

class TestAllowlists:
    """The tool allowlists in STEWARD and DEEP_REVIEWER must use the new two-step API."""

    def test_steward_has_submit_task(self):
        assert 'mcp__fused-memory__submit_task' in STEWARD.allowed_tools

    def test_steward_has_resolve_ticket(self):
        assert 'mcp__fused-memory__resolve_ticket' in STEWARD.allowed_tools

    def test_steward_does_not_have_add_task(self):
        assert 'mcp__fused-memory__add_task' not in STEWARD.allowed_tools

    def test_deep_reviewer_has_submit_task(self):
        assert 'mcp__fused-memory__submit_task' in DEEP_REVIEWER.allowed_tools

    def test_deep_reviewer_has_resolve_ticket(self):
        assert 'mcp__fused-memory__resolve_ticket' in DEEP_REVIEWER.allowed_tools

    def test_deep_reviewer_does_not_have_add_task(self):
        assert 'mcp__fused-memory__add_task' not in DEEP_REVIEWER.allowed_tools


# ---------------------------------------------------------------------------
# Step 3 / Step 4: STEWARD system_prompt migration
# ---------------------------------------------------------------------------

class TestStewardPromptMigration:
    """STEWARD.system_prompt must instruct the steward to use submit_task + resolve_ticket."""

    def test_references_submit_task(self):
        assert '`submit_task`' in STEWARD.system_prompt

    def test_references_resolve_ticket(self):
        assert '`resolve_ticket`' in STEWARD.system_prompt

    def test_references_timeout_seconds(self):
        assert 'timeout_seconds=60' in STEWARD.system_prompt

    def test_mentions_status_created(self):
        assert 'created' in STEWARD.system_prompt

    def test_mentions_status_combined(self):
        assert 'combined' in STEWARD.system_prompt

    def test_mentions_status_dropped(self):
        assert 'dropped' in STEWARD.system_prompt

    def test_does_not_reference_deprecated_add_task(self):
        assert '`add_task`' not in STEWARD.system_prompt


# ---------------------------------------------------------------------------
# Step 5 / Step 6: DEEP_REVIEWER system_prompt migration
# ---------------------------------------------------------------------------

class TestDeepReviewerPromptMigration:
    """DEEP_REVIEWER.system_prompt must instruct the reviewer to use submit_task + resolve_ticket."""

    def test_references_submit_task(self):
        assert '`submit_task`' in DEEP_REVIEWER.system_prompt

    def test_references_resolve_ticket(self):
        assert '`resolve_ticket`' in DEEP_REVIEWER.system_prompt

    def test_references_timeout_seconds(self):
        assert 'timeout_seconds=60' in DEEP_REVIEWER.system_prompt

    def test_mentions_status_created(self):
        assert 'created' in DEEP_REVIEWER.system_prompt

    def test_mentions_status_combined(self):
        assert 'combined' in DEEP_REVIEWER.system_prompt

    def test_mentions_status_dropped(self):
        assert 'dropped' in DEEP_REVIEWER.system_prompt

    def test_does_not_reference_deprecated_add_task(self):
        assert '`add_task`' not in DEEP_REVIEWER.system_prompt


# ---------------------------------------------------------------------------
# Step 7 / Step 8: ReviewCheckpoint._build_prompt migration
# ---------------------------------------------------------------------------

class TestReviewCheckpointPromptMigration:
    """ReviewCheckpoint._build_prompt must produce a prompt that uses submit_task + resolve_ticket."""

    def _get_prompt(self) -> str:
        cp = _make_checkpoint()
        return cp._build_prompt(
            mode='focused',
            phase1=_PHASE1_RESULT,
            modules=['orchestrator'],
            briefing_content='briefing',
            review_id='TEST',
        )

    def test_references_submit_task(self):
        assert '`submit_task`' in self._get_prompt()

    def test_references_resolve_ticket(self):
        assert '`resolve_ticket`' in self._get_prompt()

    def test_references_timeout_seconds(self):
        assert 'timeout_seconds=60' in self._get_prompt()

    def test_mentions_status_created(self):
        assert 'created' in self._get_prompt()

    def test_mentions_status_combined(self):
        assert 'combined' in self._get_prompt()

    def test_mentions_status_dropped(self):
        assert 'dropped' in self._get_prompt()

    def test_does_not_reference_deprecated_add_task_call(self):
        assert 'add_task(' not in self._get_prompt()
