"""Tests for task-creation tool migration: add_task → submit_task + resolve_ticket.

Covers two scopes:
  1. Tool allowlists in STEWARD and DEEP_REVIEWER roles
  2. Shared _submit_resolve_instructions helper fragment
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock

from orchestrator.agents.roles import DEEP_REVIEWER, STEWARD, _submit_resolve_instructions
from orchestrator.review_checkpoint import ReviewCheckpoint
from orchestrator.verify import VerifyResult

# ---------------------------------------------------------------------------
# Helper contract tests
# ---------------------------------------------------------------------------


class TestSharedSubmitResolveFragment:
    """Contract tests for the _submit_resolve_instructions helper."""

    def test_helper_renders_shared_skeleton(self):
        result = _submit_resolve_instructions(
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
    """Verify that multiline metadata and extra_submit_guidance align properly after caller indent."""

    def test_multiline_metadata_continuation_aligns_after_caller_indent(self):
        result = _submit_resolve_instructions(
            '{"a": 1,\n"b": 2}',
            outcome_target='t',
            step_prefix=('a', 'b'),
            extra_submit_guidance='Line1\nLine2',
        )
        wrapped = textwrap.indent(result, '   ')
        # After 3-space caller indent, "b": 2} continuation must sit at column 6
        # (same column as 'metadata=' which is at '   ' + '   metadata=').
        assert '\n      "b": 2}' in wrapped, (
            f'Expected 6-space continuation not found.\nWrapped:\n{wrapped!r}'
        )
        # extra_submit_guidance lines must also sit at column 6, subordinated
        # under step 'a.', not at the step-letter column (3 spaces).
        assert '\n      Line1\n      Line2' in wrapped, (
            f'Expected 6-space extra guidance not found.\nWrapped:\n{wrapped!r}'
        )

    def test_steward_prompt_aligns_metadata_and_extra_guidance(self):
        # Pre-triaged site: caller indent = '   ' (3 spaces), so sub-continuation
        # sits at 6 spaces (3 caller + 3 internal).
        assert '      "spawned_from": "<task_id under review>"' in STEWARD.system_prompt, (
            'Expected 6-space spawned_from in pre-triaged STEWARD prompt'
        )
        assert '      Populate `spawned_from`' in STEWARD.system_prompt, (
            'Expected 6-space Populate extra guidance in pre-triaged STEWARD prompt'
        )
        # Raw-format site: caller indent = '  ' (2 spaces), so sub-continuation
        # sits at 5 spaces (2 caller + 3 internal).
        assert '     "spawned_from": "<task_id under review>"' in STEWARD.system_prompt, (
            'Expected 5-space spawned_from in raw-format STEWARD prompt'
        )
        assert '     Include the code modules' in STEWARD.system_prompt, (
            'Expected 5-space Include extra guidance in raw-format STEWARD prompt'
        )


class TestReviewCheckpointSite:
    """ReviewCheckpoint._build_prompt must emit the shared helper-rendered block for site 4."""

    def test_review_checkpoint_site_uses_helper(self):
        cp = _make_review_checkpoint()
        prompt = cp._build_prompt(
            mode='focused',
            phase1=_PHASE1_RESULT,
            modules=['orchestrator'],
            briefing_content='',
            review_id='REV-TEST',
        )
        expected = textwrap.indent(
            _submit_resolve_instructions(
                '{"source": "review-cycle", "review_id": "REV-TEST", "modules": ["path/to/module", ...]}',
                outcome_target='finding description',
                project_root_expr='"/tmp/pr"',
                step_prefix=('1', '2'),
            ),
            '     ',
        )
        assert expected in prompt
