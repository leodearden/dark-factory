"""Regression-guard tests for mcp_lifecycle constants and helpers.

TestApplyMcpStartupEnv — MCP_STARTUP_TIMEOUT_MS constant and
apply_mcp_startup_env() helper (step-1/step-2).

TestJcodemunchLaunchPinned — JCODEMUNCH_COMMAND/JCODEMUNCH_ENV constants and
mcp_config_json() regression guard (step-3/step-4).

TestInvokeInjectsMcpStartupTimeout — _invoke_claude_with_sandbox injects
MCP_TIMEOUT into env_overrides (step-5/step-6).

TestManagedRuntimeDataDirs — managed_runtime_data_dirs() pure helper: XDG
override, home fallback, and not-under-project-root (task 2439 step-1/step-2).

TestManagedSpawnEnvInjection — McpLifecycle.start()'s managed-spawn branch
injects QUEUE_DATA_DIR/RECONCILIATION_DATA_DIR out-of-tree, honoring an
operator-preset override (task 2439 step-3/step-4).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import pydantic_spec

from orchestrator.config import OrchestratorConfig
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

    def test_mcp_config_json_uses_constants(self, mock_orch_config):
        """mcp_config_json() jcodemunch entry matches constants (regression guard)."""
        from orchestrator.mcp_lifecycle import JCODEMUNCH_COMMAND, JCODEMUNCH_ENV, McpLifecycle

        mock_orch_config.fused_memory.url = 'http://localhost:8000'
        lifecycle = McpLifecycle(mock_orch_config)
        out = lifecycle.mcp_config_json()

        jc = out['mcpServers']['jcodemunch']
        assert jc['command'] == JCODEMUNCH_COMMAND
        assert jc['env'] == JCODEMUNCH_ENV
        # Regression guard: 'uvx' must not appear in command or args
        assert 'uvx' not in jc.get('command', '')
        args = jc.get('args', [])
        assert not any('uvx' in str(a) for a in args)


# ---------------------------------------------------------------------------
# Step-1/Step-2 (task 2042): mcp_config_json escalation_headers
# ---------------------------------------------------------------------------


class TestMcpConfigEscalationHeaders:
    """mcp_config_json() attaches escalation_headers to the escalation block
    only when BOTH escalation_url and escalation_headers are supplied; the
    default (no headers) output stays byte-for-byte unchanged so every other
    caller (steward, review-checkpoint, workflow, dry-run-unblock) keeps
    full-authority, header-less connections."""

    def test_headers_and_url_both_present_in_escalation_block(self, mock_orch_config):
        """escalation_headers + escalation_url -> escalation block carries both url and the exact headers dict."""
        from orchestrator.mcp_lifecycle import McpLifecycle

        mock_orch_config.fused_memory.url = 'http://localhost:8000'
        lifecycle = McpLifecycle(mock_orch_config)
        headers = {
            'X-Escalation-Levels': '0,1',
            'X-Escalation-Identity': 'orchestrator-escalation-watcher-auto',
        }
        out = lifecycle.mcp_config_json(
            escalation_url='http://h:9999/mcp',
            escalation_headers=headers,
        )

        escalation = out['mcpServers']['escalation']
        assert escalation['url'] == 'http://h:9999/mcp'
        assert escalation['headers'] == headers

    def test_no_headers_kwarg_leaves_escalation_block_byte_for_byte(self, mock_orch_config):
        """escalation_url alone (no escalation_headers) -> escalation block has NO 'headers' key."""
        from orchestrator.mcp_lifecycle import McpLifecycle

        mock_orch_config.fused_memory.url = 'http://localhost:8000'
        lifecycle = McpLifecycle(mock_orch_config)
        out = lifecycle.mcp_config_json(escalation_url='http://h:9999/mcp')

        escalation = out['mcpServers']['escalation']
        assert escalation == {'type': 'http', 'url': 'http://h:9999/mcp'}
        assert 'headers' not in escalation

    def test_headers_without_url_ignored_no_escalation_block(self, mock_orch_config):
        """No escalation_url -> no 'escalation' key at all, even if escalation_headers is passed (no crash)."""
        from orchestrator.mcp_lifecycle import McpLifecycle

        mock_orch_config.fused_memory.url = 'http://localhost:8000'
        lifecycle = McpLifecycle(mock_orch_config)

        out_bare = lifecycle.mcp_config_json()
        assert 'escalation' not in out_bare['mcpServers']

        out_headers_only = lifecycle.mcp_config_json(
            escalation_headers={'X-Escalation-Levels': '0,1'}
        )
        assert 'escalation' not in out_headers_only['mcpServers']

    def test_headers_do_not_alter_other_server_blocks(self, mock_orch_config):
        """escalation_headers only touches the escalation block; fused-memory/jcodemunch blocks unaffected."""
        from orchestrator.mcp_lifecycle import McpLifecycle

        mock_orch_config.fused_memory.url = 'http://localhost:8000'
        lifecycle = McpLifecycle(mock_orch_config)
        headers = {'X-Escalation-Levels': '0,1'}

        baseline = lifecycle.mcp_config_json(escalation_url='http://h:9999/mcp')
        with_headers = lifecycle.mcp_config_json(
            escalation_url='http://h:9999/mcp',
            escalation_headers=headers,
        )

        assert with_headers['mcpServers']['fused-memory'] == baseline['mcpServers']['fused-memory']
        assert with_headers['mcpServers']['jcodemunch'] == baseline['mcpServers']['jcodemunch']


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


# ---------------------------------------------------------------------------
# Step-1/Step-2 (task 2258, W11-ε1): plan_tools_mcp_server(meta_root=...)
# ---------------------------------------------------------------------------


class TestPlanToolsLaunchMetaRoot:
    """plan_tools_mcp_server(meta_root=...) appends '--meta-root <path>' to
    args for BOTH the uv-fallback and direct-interpreter forms; omitting
    meta_root (the default) leaves args byte-identical to before this
    parameter existed (worktree-lane-lifecycle PRD task ε1)."""

    def test_uv_fallback_contains_meta_root_flag(self):
        """uv-fallback form: '--meta-root' immediately followed by the path."""
        cfg = plan_tools_mcp_server(
            orch_project_dir=Path('/orch'),
            worktree=Path('/wt'),
            meta_root=Path('/base/.task-meta/wt'),
        )
        args = cfg['args']
        idx = args.index('--meta-root')
        assert args[idx + 1] == '/base/.task-meta/wt'

    def test_direct_interpreter_contains_meta_root_flag(self):
        """direct-interpreter form: '--meta-root' immediately followed by the path."""
        cfg = plan_tools_mcp_server(
            orch_project_dir=Path('/orch'),
            worktree=Path('/wt'),
            python_executable='/venv/bin/python',
            meta_root=Path('/base/.task-meta/wt'),
        )
        args = cfg['args']
        idx = args.index('--meta-root')
        assert args[idx + 1] == '/base/.task-meta/wt'

    def test_uv_fallback_omits_meta_root_flag_by_default(self):
        """Regression guard: meta_root omitted (None) → no '--meta-root' token
        anywhere in args, so existing exact-match tests stay valid."""
        cfg = plan_tools_mcp_server(orch_project_dir=Path('/orch'), worktree=Path('/wt'))
        assert '--meta-root' not in cfg['args']

    def test_direct_interpreter_omits_meta_root_flag_by_default(self):
        """Regression guard: meta_root omitted (None) → no '--meta-root' token
        anywhere in args for the direct-interpreter form either."""
        cfg = plan_tools_mcp_server(
            orch_project_dir=Path('/orch'),
            worktree=Path('/wt'),
            python_executable='/venv/bin/python',
        )
        assert '--meta-root' not in cfg['args']


# ---------------------------------------------------------------------------
# Step-1/Step-2 (task 2439): managed_runtime_data_dirs() pure helper
# ---------------------------------------------------------------------------


class TestManagedRuntimeDataDirs:
    """managed_runtime_data_dirs(project_id) — pure helper computing the
    out-of-tree queue/reconciliation runtime dirs for a managed fused-memory
    spawn, keeping its runtime SQLite state out of a watched project's
    tracked tree (task 2439)."""

    def test_xdg_state_home_set_returns_dark_factory_project_subdirs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With XDG_STATE_HOME set, both dirs nest under it as
        dark-factory/<project_id>/{queue,reconciliation}."""
        monkeypatch.setenv('XDG_STATE_HOME', str(tmp_path))
        from orchestrator.mcp_lifecycle import managed_runtime_data_dirs

        queue_dir, recon_dir = managed_runtime_data_dirs('reify')

        assert queue_dir == tmp_path / 'dark-factory' / 'reify' / 'queue'
        assert recon_dir == tmp_path / 'dark-factory' / 'reify' / 'reconciliation'

    def test_xdg_state_home_unset_falls_back_to_home_local_state(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With XDG_STATE_HOME unset, the base falls back to ~/.local/state."""
        monkeypatch.delenv('XDG_STATE_HOME', raising=False)
        from orchestrator.mcp_lifecycle import managed_runtime_data_dirs

        queue_dir, recon_dir = managed_runtime_data_dirs('reify')

        expected_root = Path.home() / '.local' / 'state' / 'dark-factory' / 'reify'
        assert queue_dir == expected_root / 'queue'
        assert recon_dir == expected_root / 'reconciliation'

    def test_neither_path_is_under_a_sample_project_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Regression guard: with the process cwd set to a watched
        project's root — exactly how McpLifecycle.start() spawns fused-memory
        (cwd=str(project_root)) — the returned dirs must still resolve under
        the XDG state root, not fall back to a CWD-relative path that would
        nest them inside the project (the pollution this task fixes).

        project_root and XDG_STATE_HOME are both siblings under tmp_path (not
        an unrelated hard-coded path), and we actually chdir into
        project_root, so a regression to a relative './data/...' default —
        which the OS would resolve against cwd — is caught instead of
        trivially passing."""
        xdg_home = tmp_path / 'xdg-state'
        project_root = tmp_path / 'watched-project'
        project_root.mkdir()
        monkeypatch.setenv('XDG_STATE_HOME', str(xdg_home))
        monkeypatch.chdir(project_root)
        from orchestrator.mcp_lifecycle import managed_runtime_data_dirs

        queue_dir, recon_dir = managed_runtime_data_dirs('reify')

        # Absolute — never susceptible to CWD-relative resolution (the
        # actual historical pollution vector: cwd=project_root + a relative
        # env var default).
        assert queue_dir.is_absolute()
        assert recon_dir.is_absolute()
        # Not nested under the project root the managed spawn's cwd is set to.
        assert not queue_dir.is_relative_to(project_root)
        assert not recon_dir.is_relative_to(project_root)
        # Positively pinned to the XDG root — not merely "some absolute
        # path elsewhere", closing the tautology gap.
        assert queue_dir.is_relative_to(xdg_home)
        assert recon_dir.is_relative_to(xdg_home)


# ---------------------------------------------------------------------------
# Step-3/Step-4 (task 2439): McpLifecycle.start() managed-spawn env injection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestManagedSpawnEnvInjection:
    """McpLifecycle.start()'s managed-spawn branch injects
    QUEUE_DATA_DIR/RECONCILIATION_DATA_DIR pointing out-of-tree, so the
    managed fused-memory's runtime SQLite state never dirties the watched
    project's tracked tree (task 2439). Mirrors
    TestMcpLifecycleProcessGroup's mock/patch idiom in test_mcp_retry.py."""

    @pytest.fixture
    def mock_config(self, tmp_path: Path) -> MagicMock:
        """Config-shaped mock for McpLifecycle: non-empty server_command
        (managed mode), a real project_id, and project_root=tmp_path."""
        cfg = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
        cfg.fused_memory.server_command = ['echo', 'ok']
        cfg.fused_memory.url = 'http://localhost:8002'
        cfg.fused_memory.config_path = 'fused-memory/config/config.yaml'
        cfg.fused_memory.project_id = 'reify'
        cfg.project_root = tmp_path
        return cfg

    async def test_start_injects_out_of_tree_queue_and_reconciliation_dirs(
        self, mock_config: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No operator override present -> env carries the computed
        managed_runtime_data_dirs() paths, neither of which is under
        project_root; cwd stays project_root (unchanged by this task)."""
        monkeypatch.delenv('QUEUE_DATA_DIR', raising=False)
        monkeypatch.delenv('RECONCILIATION_DATA_DIR', raising=False)

        from orchestrator.mcp_lifecycle import McpLifecycle, managed_runtime_data_dirs

        captured_kwargs: dict = {}

        async def fake_exec(*args: object, **kwargs: object) -> MagicMock:
            captured_kwargs.update(kwargs)
            proc = MagicMock()
            proc.returncode = None
            proc.stdout = MagicMock()
            proc.stderr = MagicMock()
            return proc

        with (
            patch(
                'orchestrator.mcp_lifecycle.asyncio.create_subprocess_exec',
                side_effect=fake_exec,
            ),
            patch.object(
                McpLifecycle, '_wait_for_health', new=AsyncMock(return_value=True)
            ),
            patch('orchestrator.mcp_lifecycle.McpSession') as mock_session_cls,
        ):
            mock_session_cls.return_value.initialize = AsyncMock()
            mcp = McpLifecycle(mock_config)
            await mcp.start()

        env = captured_kwargs.get('env')
        assert env is not None, 'create_subprocess_exec must be called with env='
        expected_queue, expected_recon = managed_runtime_data_dirs('reify')
        assert env['QUEUE_DATA_DIR'] == str(expected_queue)
        assert env['RECONCILIATION_DATA_DIR'] == str(expected_recon)
        assert not Path(env['QUEUE_DATA_DIR']).is_relative_to(tmp_path)
        assert not Path(env['RECONCILIATION_DATA_DIR']).is_relative_to(tmp_path)
        # cwd is unchanged by this task
        assert captured_kwargs.get('cwd') == str(tmp_path)

    async def test_start_honors_operator_preset_queue_data_dir(
        self, mock_config: MagicMock, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An operator-preset QUEUE_DATA_DIR in the environment survives —
        setdefault() must not clobber it."""
        monkeypatch.setenv('QUEUE_DATA_DIR', '/custom/q')
        monkeypatch.delenv('RECONCILIATION_DATA_DIR', raising=False)

        from orchestrator.mcp_lifecycle import McpLifecycle

        captured_kwargs: dict = {}

        async def fake_exec(*args: object, **kwargs: object) -> MagicMock:
            captured_kwargs.update(kwargs)
            proc = MagicMock()
            proc.returncode = None
            proc.stdout = MagicMock()
            proc.stderr = MagicMock()
            return proc

        with (
            patch(
                'orchestrator.mcp_lifecycle.asyncio.create_subprocess_exec',
                side_effect=fake_exec,
            ),
            patch.object(
                McpLifecycle, '_wait_for_health', new=AsyncMock(return_value=True)
            ),
            patch('orchestrator.mcp_lifecycle.McpSession') as mock_session_cls,
        ):
            mock_session_cls.return_value.initialize = AsyncMock()
            mcp = McpLifecycle(mock_config)
            await mcp.start()

        env = captured_kwargs.get('env')
        assert env is not None
        assert env['QUEUE_DATA_DIR'] == '/custom/q'

    async def test_start_honors_operator_preset_reconciliation_data_dir(
        self, mock_config: MagicMock, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An operator-preset RECONCILIATION_DATA_DIR in the environment
        survives — setdefault() must not clobber it. Mirrors
        test_start_honors_operator_preset_queue_data_dir: the two env vars
        are set via separate setdefault() calls, so each needs its own
        override assertion or a regression to one could go unnoticed."""
        monkeypatch.delenv('QUEUE_DATA_DIR', raising=False)
        monkeypatch.setenv('RECONCILIATION_DATA_DIR', '/custom/r')

        from orchestrator.mcp_lifecycle import McpLifecycle

        captured_kwargs: dict = {}

        async def fake_exec(*args: object, **kwargs: object) -> MagicMock:
            captured_kwargs.update(kwargs)
            proc = MagicMock()
            proc.returncode = None
            proc.stdout = MagicMock()
            proc.stderr = MagicMock()
            return proc

        with (
            patch(
                'orchestrator.mcp_lifecycle.asyncio.create_subprocess_exec',
                side_effect=fake_exec,
            ),
            patch.object(
                McpLifecycle, '_wait_for_health', new=AsyncMock(return_value=True)
            ),
            patch('orchestrator.mcp_lifecycle.McpSession') as mock_session_cls,
        ):
            mock_session_cls.return_value.initialize = AsyncMock()
            mcp = McpLifecycle(mock_config)
            await mcp.start()

        env = captured_kwargs.get('env')
        assert env is not None
        assert env['RECONCILIATION_DATA_DIR'] == '/custom/r'
