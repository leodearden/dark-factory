"""Regression-guard tests for mcp_lifecycle constants and helpers.

TestApplyMcpStartupEnv — MCP_STARTUP_TIMEOUT_MS constant and
apply_mcp_startup_env() helper (step-1/step-2).

TestJcodemunchLaunchPinned — JCODEMUNCH_COMMAND/JCODEMUNCH_ENV constants and
mcp_config_json() regression guard (step-3/step-4).

TestInvokeInjectsMcpStartupTimeout — _invoke_claude_with_sandbox injects
MCP_TIMEOUT into env_overrides (step-5/step-6).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Step-1/Step-2: apply_mcp_startup_env + MCP_STARTUP_TIMEOUT_MS
# ---------------------------------------------------------------------------


class TestApplyMcpStartupEnv:
    """apply_mcp_startup_env injects MCP_TIMEOUT, is copy-safe, and lets
    explicit caller values win."""

    def test_none_input_returns_dict_with_mcp_timeout(self):
        """apply_mcp_startup_env(None) returns dict containing MCP_TIMEOUT==MCP_STARTUP_TIMEOUT_MS."""
        from orchestrator.mcp_lifecycle import MCP_STARTUP_TIMEOUT_MS, apply_mcp_startup_env

        result = apply_mcp_startup_env(None)
        assert isinstance(result, dict)
        assert 'MCP_TIMEOUT' in result
        assert result['MCP_TIMEOUT'] == MCP_STARTUP_TIMEOUT_MS

    def test_existing_keys_are_preserved(self):
        """Passed keys survive; result also gains MCP_TIMEOUT."""
        from orchestrator.mcp_lifecycle import apply_mcp_startup_env

        result = apply_mcp_startup_env({'FOO': 'bar'})
        assert result['FOO'] == 'bar'
        assert 'MCP_TIMEOUT' in result

    def test_input_dict_is_not_mutated(self):
        """apply_mcp_startup_env returns a copy; the original dict is unchanged."""
        from orchestrator.mcp_lifecycle import apply_mcp_startup_env

        original = {'FOO': 'bar'}
        result = apply_mcp_startup_env(original)
        assert result is not original
        assert 'MCP_TIMEOUT' not in original

    def test_explicit_mcp_timeout_wins(self):
        """An explicit MCP_TIMEOUT value in env_overrides is not overwritten."""
        from orchestrator.mcp_lifecycle import apply_mcp_startup_env

        result = apply_mcp_startup_env({'MCP_TIMEOUT': '5'})
        assert result['MCP_TIMEOUT'] == '5'


# ---------------------------------------------------------------------------
# Step-3/Step-4: JCODEMUNCH_COMMAND / JCODEMUNCH_ENV + mcp_config_json
# ---------------------------------------------------------------------------


class TestJcodemunchLaunchPinned:
    """mcp_config_json() uses the named constants and is pinned to the prebuilt
    launcher (not uvx), guarding against regressions."""

    def test_command_is_prebuilt_launcher_not_uvx(self):
        """JCODEMUNCH_COMMAND is 'jcodemunch-mcp' (prebuilt, not 'uvx')."""
        from orchestrator.mcp_lifecycle import JCODEMUNCH_COMMAND

        assert JCODEMUNCH_COMMAND == 'jcodemunch-mcp'
        assert JCODEMUNCH_COMMAND != 'uvx'

    def test_env_contains_no_version_hint(self):
        """JCODEMUNCH_ENV contains JCODEMUNCH_NO_VERSION_HINT."""
        from orchestrator.mcp_lifecycle import JCODEMUNCH_ENV

        assert 'JCODEMUNCH_NO_VERSION_HINT' in JCODEMUNCH_ENV

    def test_mcp_config_json_uses_constants(self):
        """mcp_config_json() jcodemunch entry matches constants (regression guard)."""
        from orchestrator.mcp_lifecycle import JCODEMUNCH_COMMAND, JCODEMUNCH_ENV, McpLifecycle

        cfg = MagicMock()
        cfg.url = 'http://localhost:8000'
        lifecycle = McpLifecycle(cfg)
        out = lifecycle.mcp_config_json()

        jc = out['mcpServers']['jcodemunch']
        assert jc['command'] == JCODEMUNCH_COMMAND
        assert jc['env'] == JCODEMUNCH_ENV
        # Regression guard: 'uvx' must not appear in command or args
        assert 'uvx' not in jc.get('command', '')
        args = jc.get('args', [])
        assert not any('uvx' in str(a) for a in args)


# ---------------------------------------------------------------------------
# Step-5/Step-6: _invoke_claude_with_sandbox injects MCP_TIMEOUT
# ---------------------------------------------------------------------------


class TestInvokeInjectsMcpStartupTimeout:
    """_invoke_claude_with_sandbox transforms env_overrides before delegating.

    Both test cases use sandbox_modules=None, which exercises the non-sandbox
    delegate path (invoke_claude_agent).  The injection site is at the top of
    the function — before the branch — so it covers both paths by construction,
    but only the delegate path is directly observed here.
    """

    @pytest.mark.asyncio
    async def test_none_env_overrides_gets_mcp_timeout(self, tmp_path):
        """With env_overrides=None, the delegate receives MCP_TIMEOUT."""
        import orchestrator.agents.invoke as inv_mod
        from orchestrator.mcp_lifecycle import MCP_STARTUP_TIMEOUT_MS

        mock_agent = AsyncMock(return_value=MagicMock())
        monkeypatch_target = 'orchestrator.agents.invoke.invoke_claude_agent'

        import unittest.mock as um
        with um.patch(monkeypatch_target, mock_agent):
            await inv_mod._invoke_claude_with_sandbox(
                prompt='hello',
                system_prompt='be helpful',
                cwd=tmp_path,
                model='claude-opus-4-5',
                max_turns=1,
                max_budget_usd=0.01,
                allowed_tools=None,
                disallowed_tools=None,
                mcp_config=None,
                output_schema=None,
                permission_mode='bypassPermissions',
                sandbox_modules=None,
                effort=None,
                env_overrides=None,
            )

        mock_agent.assert_awaited_once()
        _, kwargs = mock_agent.call_args
        env = kwargs.get('env_overrides', {}) or {}
        assert env.get('MCP_TIMEOUT') == MCP_STARTUP_TIMEOUT_MS, (
            f"Expected MCP_TIMEOUT={MCP_STARTUP_TIMEOUT_MS!r} in env_overrides, got {env!r}"
        )

    @pytest.mark.asyncio
    async def test_existing_env_overrides_preserved_and_mcp_timeout_added(self, tmp_path):
        """With env_overrides={'FOO': 'bar'}, delegate gets FOO and MCP_TIMEOUT."""
        import orchestrator.agents.invoke as inv_mod
        from orchestrator.mcp_lifecycle import MCP_STARTUP_TIMEOUT_MS

        mock_agent = AsyncMock(return_value=MagicMock())

        import unittest.mock as um
        with um.patch('orchestrator.agents.invoke.invoke_claude_agent', mock_agent):
            await inv_mod._invoke_claude_with_sandbox(
                prompt='hello',
                system_prompt='be helpful',
                cwd=tmp_path,
                model='claude-opus-4-5',
                max_turns=1,
                max_budget_usd=0.01,
                allowed_tools=None,
                disallowed_tools=None,
                mcp_config=None,
                output_schema=None,
                permission_mode='bypassPermissions',
                sandbox_modules=None,
                effort=None,
                env_overrides={'FOO': 'bar'},
            )

        mock_agent.assert_awaited_once()
        _, kwargs = mock_agent.call_args
        env = kwargs.get('env_overrides', {}) or {}
        assert env.get('FOO') == 'bar', f"Expected FOO='bar' in env_overrides, got {env!r}"
        assert env.get('MCP_TIMEOUT') == MCP_STARTUP_TIMEOUT_MS, (
            f"Expected MCP_TIMEOUT={MCP_STARTUP_TIMEOUT_MS!r} in env_overrides, got {env!r}"
        )
