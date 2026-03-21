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


class TestModuleLevelAll:
    """Verify that each submodule defines __all__ with exactly the expected symbols."""

    def test_cli_invoke_all(self):
        from shared import cli_invoke

        assert hasattr(cli_invoke, '__all__'), 'cli_invoke must define __all__'
        assert set(cli_invoke.__all__) == {
            'AgentResult',
            'invoke_claude_agent',
            'invoke_with_cap_retry',
        }

    def test_usage_gate_all(self):
        from shared import usage_gate

        assert hasattr(usage_gate, '__all__'), 'usage_gate must define __all__'
        assert set(usage_gate.__all__) == {
            'UsageGate',
            'AccountState',
            'SessionBudgetExhausted',
        }

    def test_config_models_all(self):
        from shared import config_models

        assert hasattr(config_models, '__all__'), 'config_models must define __all__'
        assert set(config_models.__all__) == {
            'AccountConfig',
            'UsageCapConfig',
        }
