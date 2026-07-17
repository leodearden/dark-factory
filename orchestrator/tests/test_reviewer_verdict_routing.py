"""RED/GREEN tests for reviewer verdicts routed through the verdict-tools
MCP artifact instead of the structured-output/``json.loads`` cascade.

Task 2484 (PRD ``plans/mcp-verdict-servers-prd.md`` task δ): removes the
reviewer panel's ``output_schema``/``json.loads`` fallback in
``TaskWorkflow._run_reviewer`` in favor of reading
``TaskArtifacts.read_verdict(role.name)`` — the same
clear→invoke→read→defensive-extract→fail-safe shape task 2483 (PRD task γ)
established for the merger (see ``test_merger_disposition_verdict.py``) — and
removes the reviewer's now-inert ``mcp__jcodemunch__*`` grant (fold-in κ).
"""

from __future__ import annotations

from orchestrator.agents.roles import REVIEWER_COMPREHENSIVE


class TestReviewerGrantSurface:
    """Structural contract for the reviewer's tool grants (fold-in κ)."""

    def test_no_jcodemunch_grant(self):
        """The reviewer's inert jcodemunch grant has been removed."""
        assert 'mcp__jcodemunch__*' not in REVIEWER_COMPREHENSIVE.allowed_tools

    def test_has_verdict_tools_grant(self):
        """The verdict-tools grant (added by β/task 2482) is present."""
        assert 'mcp__verdict-tools__*' in REVIEWER_COMPREHENSIVE.allowed_tools

    def test_declares_verdict_tools_family(self):
        """The role declares the verdict_tools MCP family (added by β)."""
        assert 'verdict_tools' in REVIEWER_COMPREHENSIVE.mcp_families
