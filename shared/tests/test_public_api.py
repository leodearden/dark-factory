"""Tests verifying the public API contract of the dark-factory-shared package."""

from __future__ import annotations

from pathlib import Path


class TestTopLevelImports:
    """Verify that all public symbols are importable from the top-level shared namespace."""

    def test_top_level_imports(self):
        from shared import (
            AccountConfig,
            AccountState,
            AgentResult,
            SessionBudgetExhausted,
            UsageCapConfig,
            UsageGate,
            invoke_claude_agent,
            invoke_with_cap_retry,
        )

        assert AgentResult is not None
        assert invoke_claude_agent is not None
        assert invoke_with_cap_retry is not None
        assert UsageGate is not None
        assert AccountState is not None
        assert SessionBudgetExhausted is not None
        assert AccountConfig is not None
        assert UsageCapConfig is not None
