"""Tests for task-creation tool migration: add_task → submit_task + resolve_ticket.

Covers two scopes:
  1. Tool allowlists in STEWARD and DEEP_REVIEWER roles
  2. Shared _submit_resolve_instructions helper fragment
"""

from __future__ import annotations

import textwrap

from orchestrator.agents.roles import DEEP_REVIEWER, STEWARD, _submit_resolve_instructions

# ---------------------------------------------------------------------------
# Helper contract tests
# ---------------------------------------------------------------------------


class TestSharedSubmitResolveFragment:
    """Contract tests for the _submit_resolve_instructions helper."""

    def test_helper_renders_shared_skeleton(self):
        result = _submit_resolve_instructions(
            '{"source": "X"}',
            outcome_target='test summary',
            step_prefix=('1', '2'),
            project_root_expr='"/tmp"',
        )

        # (a) 60s timeout rationale parenthetical
        assert '60 s is intentionally conservative' in result
        # (b) error-shape branch
        assert 'error_type' in result
        # (c) status branches
        assert 'created' in result
        assert 'combined' in result
        assert 'failed' in result
        # (d) supplied metadata_template
        assert '{"source": "X"}' in result
        # (e) supplied outcome_target
        assert 'test summary' in result
        # (f) step prefix labels
        assert '1.' in result
        assert '2.' in result
        # (g) supplied project_root_expr
        assert '"/tmp"' in result


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
