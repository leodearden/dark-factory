"""Tests for task-creation tool migration: add_task → submit_task + resolve_ticket.

Covers two scopes:
  1. Tool allowlists in STEWARD and DEEP_REVIEWER roles
  2. Shared submit_resolve_instructions helper fragment
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from orchestrator.agents.roles import DEEP_REVIEWER, STEWARD, submit_resolve_instructions
from orchestrator.review_checkpoint import ReviewCheckpoint
from orchestrator.verify import VerifyResult

# ---------------------------------------------------------------------------
# Helper contract tests
# ---------------------------------------------------------------------------


class TestSharedSubmitResolveFragment:
    """Contract tests for the submit_resolve_instructions helper."""

    def test_helper_renders_shared_skeleton(self):
        result = submit_resolve_instructions(
            '{"source": "X"}',
            outcome_target='SENTINEL_OUTCOME_XYZ',
            step_prefix=('1', '2'),
            project_root_expr='"/tmp"',
        )
        # The outcome_target sentinel must appear at least twice:
        # once in the error-shape sentence and once in the failed-branch sentence.
        assert result.count('SENTINEL_OUTCOME_XYZ') >= 2


# ---------------------------------------------------------------------------
# Site-wiring tests (minimal behavioral assertions)
# ---------------------------------------------------------------------------


class TestSiteWiringMinimal:
    """Minimal wiring checks: prompts must direct the agent through the two-step API."""

    def test_steward_prompt_directs_two_step_api(self):
        assert 'submit_task' in STEWARD.system_prompt
        assert 'resolve_ticket' in STEWARD.system_prompt

    def test_deep_reviewer_prompt_directs_two_step_api(self):
        assert 'submit_task' in DEEP_REVIEWER.system_prompt
        assert 'resolve_ticket' in DEEP_REVIEWER.system_prompt


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
# Site-wiring test: ReviewCheckpoint._build_prompt (site 4)
# ---------------------------------------------------------------------------


def _make_review_checkpoint() -> ReviewCheckpoint:
    """Construct a minimal ReviewCheckpoint with mocked config for site-4 tests."""
    config = MagicMock()
    config.project_root = Path('/tmp/pr')
    config.fused_memory.project_id = 'test-proj'
    config.escalation.host = 'localhost'
    config.escalation.port = 9999

    mcp = MagicMock()
    mcp.mcp_config_json.return_value = {'mcpServers': {}}

    return ReviewCheckpoint(config, mcp=mcp, usage_gate=None)


_PHASE1_RESULT = VerifyResult(
    passed=True, summary='OK', test_output='', lint_output='', type_output=''
)


# ---------------------------------------------------------------------------
# Rendering-alignment tests (step 12)
# ---------------------------------------------------------------------------


class TestHelperIndentationAlignment:
    """Verify that multiline metadata and extra_submit_guidance align properly with caller_indent."""

    def test_multiline_metadata_continuation_aligns_after_caller_indent(self):
        # Pass caller_indent directly — the helper applies it internally, removing
        # the need for a separate textwrap.indent() wrap at the call site.
        result = submit_resolve_instructions(
            '{"a": 1,\n"b": 2}',
            outcome_target='t',
            step_prefix=('a', 'b'),
            extra_submit_guidance='Line1\nLine2',
            caller_indent='   ',
        )
        # After 3-space caller_indent, "b": 2} continuation must sit at column 6
        # (same column as 'metadata=' which is at '   ' + '   metadata=').
        assert '\n      "b": 2}' in result, (
            f'Expected 6-space continuation not found.\nResult:\n{result!r}'
        )
        # extra_submit_guidance lines must also sit at column 6, subordinated
        # under step 'a.', not at the step-letter column (3 spaces).
        assert '\n      Line1\n      Line2' in result, (
            f'Expected 6-space extra guidance not found.\nResult:\n{result!r}'
        )


class TestReviewCheckpointSite:
    """ReviewCheckpoint._build_prompt must direct the agent through the two-step API for site 4."""

    def test_review_checkpoint_site_uses_helper(self):
        cp = _make_review_checkpoint()
        prompt = cp._build_prompt(
            mode='focused',
            phase1=_PHASE1_RESULT,
            modules=['orchestrator'],
            briefing_content='',
            review_id='REV-TEST',
        )
        # Verify the two-step API anchors are present in the rendered prompt.
        assert 'submit_task' in prompt
        assert 'resolve_ticket' in prompt
        # Verify context-specific values are interpolated into the prompt.
        assert 'REV-TEST' in prompt
        assert '/tmp/pr' in prompt
