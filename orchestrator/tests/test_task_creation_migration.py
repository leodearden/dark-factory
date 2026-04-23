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
# Site-wiring tests (one assertion per site)
# ---------------------------------------------------------------------------

_STEWARD_PRETRIAGED_METADATA = (
    '{"source": "steward-triage", "spawn_context": "steward-triage",\n'
    '"spawned_from": "<task_id under review>", "modules": [...]}'
)
_STEWARD_PRETRIAGED_EXTRA = (
    'Populate `spawned_from` with the id of the task that produced the escalation\n'
    '(it is in the escalation detail under `task_id`). Use the file paths listed in\n'
    'the group for `modules`.'
)

_STEWARD_RAW_METADATA = (
    '{"source": "steward-triage", "spawn_context": "steward-triage",\n'
    '"spawned_from": "<task_id under review>", "modules": ["path/to/module", ...]}'
)
_STEWARD_RAW_EXTRA = (
    "Include the code modules (directory paths relative to project root) that this task\n"
    "will need to modify — these are used for concurrency locking. `spawned_from` lets\n"
    "the task curator spot duplicates against the original task's details."
)

_DEEP_REVIEWER_METADATA = '...'
_DEEP_REVIEWER_EXTRA = (
    'Always include:\n'
    '- `title`: concise description of the fix\n'
    '- `description`: what\'s wrong, where, and the suggested approach\n'
    '- `priority`: "high" for broken wiring/stubs, "medium" for consistency issues\n'
    '- `metadata`: `{"source": "review-cycle", "spawn_context": "review",\n'
    '  "review_id": "<from your prompt>", "modules": ["path/to/module", ...]}`\n'
    '  Include the code modules (directory paths relative to project root) that this task will need to modify.\n'
    '  These are used for concurrency locking — be specific and include both source and test directories.\n'
    '  `spawn_context` tells the task curator how to treat duplicates against the existing backlog.\n'
    '- `project_root`: use the value from your Agent Identity section'
)


class TestSiteWiring:
    """Each site must be wired to use the shared helper."""

    def test_steward_pretriaged_site_uses_helper(self):
        expected = textwrap.indent(
            _submit_resolve_instructions(
                _STEWARD_PRETRIAGED_METADATA,
                outcome_target='resolve_issue summary',
                step_prefix=('a', 'b'),
                extra_submit_guidance=_STEWARD_PRETRIAGED_EXTRA,
            ),
            '   ',
        )
        assert expected in STEWARD.system_prompt

    def test_steward_raw_site_uses_helper(self):
        expected = textwrap.indent(
            _submit_resolve_instructions(
                _STEWARD_RAW_METADATA,
                outcome_target='resolve_issue summary',
                step_prefix=('a', 'b'),
                extra_submit_guidance=_STEWARD_RAW_EXTRA,
            ),
            '  ',
        )
        assert expected in STEWARD.system_prompt

    def test_deep_reviewer_site_uses_helper(self):
        expected = _submit_resolve_instructions(
            _DEEP_REVIEWER_METADATA,
            outcome_target='review output',
            step_prefix=('1', '2'),
            extra_submit_guidance=_DEEP_REVIEWER_EXTRA,
        )
        assert expected in DEEP_REVIEWER.system_prompt


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
