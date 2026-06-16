"""Regression-guard tests for mcp_lifecycle constants and helpers.

TestApplyMcpStartupEnv — MCP_STARTUP_TIMEOUT_MS constant and
apply_mcp_startup_env() helper (step-1/step-2).

TestJcodemunchLaunchPinned — JCODEMUNCH_COMMAND/JCODEMUNCH_ENV constants and
mcp_config_json() regression guard (step-3/step-4).

TestInvokeInjectsMcpStartupTimeout — _invoke_claude_with_sandbox injects
MCP_TIMEOUT into env_overrides (step-5/step-6).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.mcp_lifecycle import plan_tools_mcp_server

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


# ---------------------------------------------------------------------------
# Step-1/Step-2 (task 1775): _PLAN_TOOLS_FAST_START_FLAGS + plan_tools_mcp_server
# ---------------------------------------------------------------------------


class TestPlanToolsLaunchFastStart:
    """plan_tools_mcp_server() uses fast-start flags; regression guard against
    the bare cold-resolve form (reify esc-4415-240/esc-4437-123)."""

    @pytest.fixture
    def cfg(self):
        return plan_tools_mcp_server(orch_project_dir=Path('/orch'), worktree=Path('/wt'))

    def test_command_is_uv(self, cfg):
        """plan_tools_mcp_server() returns command == 'uv'."""
        assert cfg['command'] == 'uv'

    def test_args_contain_fast_start_flags(self, cfg):
        """args contains both --no-sync AND --frozen (regression: bare form has neither)."""
        args = cfg['args']
        assert '--no-sync' in args, f"Expected '--no-sync' in args, got {args}"
        assert '--frozen' in args, f"Expected '--frozen' in args, got {args}"

    def test_args_contain_project_then_orch_dir(self, cfg):
        """args contains '--project' immediately followed by '/orch'."""
        args = cfg['args']
        idx = args.index('--project')
        assert args[idx + 1] == '/orch'

    def test_args_contain_module_path_sequence(self, cfg):
        """args contains the sequence 'python', '-m', 'orchestrator.mcp.plan_tools'."""
        args = cfg['args']
        python_idx = args.index('python')
        assert args[python_idx + 1] == '-m'
        assert args[python_idx + 2] == 'orchestrator.mcp.plan_tools'

    def test_args_contain_worktree_then_path(self, cfg):
        """args contains '--worktree' immediately followed by '/wt'."""
        args = cfg['args']
        wt_idx = args.index('--worktree')
        assert args[wt_idx + 1] == '/wt'

    def test_run_level_flags_before_python(self, cfg):
        """All run-level flags (--project, --no-sync, --frozen) appear before 'python' token."""
        args = cfg['args']
        python_idx = args.index('python')
        for flag in ('--project', '--no-sync', '--frozen'):
            flag_idx = args.index(flag)
            assert flag_idx < python_idx, (
                f"Expected {flag!r} before 'python', "
                f"but {flag!r} at {flag_idx}, 'python' at {python_idx}"
            )


# ---------------------------------------------------------------------------
# Step-1/Step-2 (task 1776): plan_tools_mcp_server() direct-interpreter path
# ---------------------------------------------------------------------------


class TestPlanToolsLaunchDirectInterpreter:
    """plan_tools_mcp_server(python_executable=...) uses the direct-interpreter
    no-uv hot path; regression guard that eliminates the uv-futex wedge under
    load (reify esc-4415-240, esc-4437-123, task 1776)."""

    @pytest.fixture
    def cfg(self):
        return plan_tools_mcp_server(
            orch_project_dir=Path('/orch'),
            worktree=Path('/wt'),
            python_executable='/venv/bin/python',
        )

    def test_command_is_python_executable(self, cfg):
        """command IS the supplied interpreter, not 'uv'."""
        assert cfg['command'] == '/venv/bin/python'

    def test_args_exact_module_invocation_sequence(self, cfg):
        """args equals ['-m', 'orchestrator.mcp.plan_tools', '--worktree', '/wt'] exactly."""
        assert cfg['args'] == ['-m', 'orchestrator.mcp.plan_tools', '--worktree', '/wt']

    def test_no_uv_tokens_anywhere_in_full_command(self, cfg):
        """None of {'uv', 'run', '--no-sync', '--frozen', '--project'} appear
        in [command] + args (locks no-uv hot path against regression)."""
        full = [cfg['command'], *cfg['args']]
        forbidden = {'uv', 'run', '--no-sync', '--frozen', '--project'}
        found = forbidden & set(full)
        assert not found, (
            f"uv tokens {found!r} must not appear in direct-interpreter command: {full}"
        )

    def test_empty_string_python_executable_falls_back_to_uv(self):
        """python_executable='' is falsy → falls back to the uv path (command == 'uv').

        Documents the intended `if python_executable:` falsy-fallback semantics so a
        future refactor to `if python_executable is not None:` would surface a failing
        test rather than silently launching with command=''.
        """
        cfg = plan_tools_mcp_server(
            orch_project_dir=Path('/orch'),
            worktree=Path('/wt'),
            python_executable='',
        )
        assert cfg['command'] == 'uv', (
            "Empty string python_executable must fall back to uv path, "
            f"not launch with command={cfg['command']!r}"
        )
