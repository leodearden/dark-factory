"""Tests for task-creation tool migration: add_task → submit_task (+/- resolve_ticket).

Covers two scopes:
  1. Tool allowlists in STEWARD and DEEP_REVIEWER roles — neither role keeps
     ``resolve_ticket`` in its allowlist; the server-side ticket janitor
     surfaces curator failures asynchronously.
  2. Shared instruction-helper fragments: ``submit_resolve_instructions``
     (for callers that wait on the curator) and ``submit_only_instructions``
     (for steward / deep_reviewer, which fire-and-forget).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from _orch_helpers import pydantic_spec

from orchestrator.agents.roles import (
    ARCHITECT,
    DEEP_REVIEWER,
    IMPLEMENTER,
    STEWARD,
    submit_only_instructions,
    submit_resolve_instructions,
)
from orchestrator.config import OrchestratorConfig
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
    """Wiring checks: steward + deep_reviewer prompts must use the one-step submit API."""

    def test_steward_prompt_uses_submit_task(self):
        assert 'submit_task' in STEWARD.system_prompt

    def test_steward_prompt_does_not_direct_resolve_ticket(self):
        # Steward fires-and-forgets — the janitor surfaces curator failures.
        # Per-call resolve_ticket waits used to chain N×60s and burst the
        # outer steward-session timeout (e.g. know-live esc-70-201).
        # The prompt may *mention* resolve_ticket negatively ("Do NOT call …"),
        # but must not include any positive directive.
        assert 'Call `resolve_ticket`' not in STEWARD.system_prompt, (
            'Steward must not be told to call resolve_ticket; the ticket '
            'janitor reports failures asynchronously.'
        )
        assert 'Do NOT call `resolve_ticket`' in STEWARD.system_prompt, (
            'Steward prompt should explicitly tell the agent not to call '
            'resolve_ticket so an agent that still has the tool for some '
            'reason fires-and-forgets.'
        )

    def test_deep_reviewer_prompt_uses_submit_task(self):
        assert 'submit_task' in DEEP_REVIEWER.system_prompt

    def test_deep_reviewer_prompt_does_not_direct_resolve_ticket(self):
        assert 'Call `resolve_ticket`' not in DEEP_REVIEWER.system_prompt, (
            'Deep reviewer must not be told to call resolve_ticket; the '
            'janitor reports failures asynchronously.'
        )
        assert 'Do NOT call `resolve_ticket`' in DEEP_REVIEWER.system_prompt


# ---------------------------------------------------------------------------
# Step 1 / Step 2: Tool allowlists
# ---------------------------------------------------------------------------

class TestAllowlists:
    """The tool allowlists in STEWARD and DEEP_REVIEWER must use the one-step API.

    ``resolve_ticket`` is intentionally excluded from both roles so the prompt
    change (drop two-step → one-step) is enforceable at the harness level —
    even a misbehaving agent that ignores its instructions cannot block on
    the curator. Other callers (skills, humans) keep access to the underlying
    MCP tool directly.
    """

    def test_steward_has_submit_task(self):
        assert 'mcp__fused-memory__submit_task' in STEWARD.allowed_tools

    def test_steward_does_not_have_resolve_ticket(self):
        assert 'mcp__fused-memory__resolve_ticket' not in STEWARD.allowed_tools

    def test_steward_does_not_have_add_task(self):
        assert 'mcp__fused-memory__add_task' not in STEWARD.allowed_tools

    def test_deep_reviewer_has_submit_task(self):
        assert 'mcp__fused-memory__submit_task' in DEEP_REVIEWER.allowed_tools

    def test_deep_reviewer_does_not_have_resolve_ticket(self):
        assert 'mcp__fused-memory__resolve_ticket' not in DEEP_REVIEWER.allowed_tools

    def test_deep_reviewer_does_not_have_add_task(self):
        assert 'mcp__fused-memory__add_task' not in DEEP_REVIEWER.allowed_tools


# ---------------------------------------------------------------------------
# Task 2640 — Source 1: architect/implementer follow-up filing at the source
# ---------------------------------------------------------------------------


class TestFollowupFilingAllowlist:
    """ARCHITECT and IMPLEMENTER must be able to file follow-up task candidates.

    Both roles gain ``mcp__fused-memory__submit_task`` so an agent that
    discovers follow-up work OUTSIDE its task scope can file it at the source
    (fire-and-forget) instead of relying on the lossy orphan-reaper backstop.
    Both already declare the ``orchestrator`` MCP family, so this is a pure
    allowlist widening — no ``mcp_families`` change.
    """

    def test_architect_has_submit_task(self):
        assert 'mcp__fused-memory__submit_task' in ARCHITECT.allowed_tools

    def test_implementer_has_submit_task(self):
        assert 'mcp__fused-memory__submit_task' in IMPLEMENTER.allowed_tools


class TestFollowupFilingPrompts:
    """ARCHITECT/IMPLEMENTER prompts must direct fire-and-forget follow-up filing.

    Both prompts gain a follow-up-filing block (rendered via
    ``submit_only_instructions``) telling the agent to file work discovered
    OUTSIDE its task scope as a low-priority ``submit_task`` candidate, and to
    fire-and-forget (never wait on ``resolve_ticket``) — mirroring the
    steward/deep_reviewer contract asserted in ``TestSiteWiringMinimal``.
    Anchored on load-bearing tokens, not prose wording.
    """

    def test_architect_prompt_has_followup_filing_block(self):
        p = ARCHITECT.system_prompt
        assert 'submit_task' in p
        assert 'Filing follow-up work' in p
        assert 'OUTSIDE the scope' in p

    def test_implementer_prompt_has_followup_filing_block(self):
        p = IMPLEMENTER.system_prompt
        assert 'submit_task' in p
        assert 'Filing follow-up work' in p
        assert 'OUTSIDE the scope' in p

    def test_architect_followup_is_fire_and_forget(self):
        p = ARCHITECT.system_prompt
        assert 'Do NOT call `resolve_ticket`' in p
        assert 'Call `resolve_ticket`' not in p

    def test_implementer_followup_is_fire_and_forget(self):
        p = IMPLEMENTER.system_prompt
        assert 'Do NOT call `resolve_ticket`' in p
        assert 'Call `resolve_ticket`' not in p


class TestStewardSweepUp:
    """STEWARD prompt must carry the workflow-end info-note sweep-up (Source 3).

    Before exit the steward enumerates its own still-open info-notes via
    ``get_pending_escalations``, files each work-describing one as a task
    candidate (``submit_task``), and closes it via ``resolve_issue``.  Anchored
    on the distinctive section tokens, plus the filing/closing tool names.
    """

    def test_steward_prompt_has_workflow_end_sweepup_section(self):
        p = STEWARD.system_prompt
        assert 'Workflow-end sweep-up' in p
        assert 'info-note' in p

    def test_steward_sweepup_enumerates_and_files_and_closes(self):
        p = STEWARD.system_prompt
        # Enumerate own open notes, file candidates, close the swept note.
        assert 'get_pending_escalations' in p
        assert 'submit_task' in p
        assert 'resolve_issue' in p


# ---------------------------------------------------------------------------
# Site-wiring test: ReviewCheckpoint._build_prompt (site 4)
# ---------------------------------------------------------------------------


def _make_review_checkpoint() -> ReviewCheckpoint:
    """Construct a minimal ReviewCheckpoint with mocked config for site-4 tests."""
    config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
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
    """ReviewCheckpoint._build_prompt must direct the agent through the one-step API."""

    def test_review_checkpoint_site_uses_helper(self):
        cp = _make_review_checkpoint()
        prompt = cp._build_prompt(
            mode='focused',
            phase1=_PHASE1_RESULT,
            modules=['orchestrator'],
            briefing_content='',
            review_id='REV-TEST',
        )
        # One-step submit anchor present.
        assert 'submit_task' in prompt
        # The deep_reviewer no longer waits on the curator.
        assert 'Call `resolve_ticket`' not in prompt
        # Verify context-specific values are interpolated into the prompt.
        assert 'REV-TEST' in prompt
        assert '/tmp/pr' in prompt


class TestSubmitOnlyHelper:
    """Contract tests for submit_only_instructions (one-step submit_task fragment)."""

    def test_helper_renders_single_step(self):
        result = submit_only_instructions(
            '{"source": "X", "escalation_id": "esc-1", "suggestion_hash": "h"}',
            outcome_target='SENTINEL_OUTCOME_XYZ',
            step_label='1',
            project_root_expr='"/tmp"',
        )
        assert 'submit_task' in result
        # outcome_target should appear at least twice (error-shape + success path).
        assert result.count('SENTINEL_OUTCOME_XYZ') >= 2
        # Single step only — no positive directive to call resolve_ticket.
        assert 'Call `resolve_ticket`' not in result
        # Must explicitly forbid resolve_ticket so an agent that still has the
        # tool for other reasons fires-and-forgets.
        assert 'Do NOT call `resolve_ticket`' in result

    def test_helper_applies_caller_indent(self):
        result = submit_only_instructions(
            '{"source": "X"}',
            outcome_target='t',
            caller_indent='   ',
        )
        # Every non-empty line should start with the indent.
        for line in result.splitlines():
            if line:
                assert line.startswith('   '), (
                    f'Line missing caller_indent: {line!r}'
                )
