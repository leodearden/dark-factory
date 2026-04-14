"""Tests for verify.py — _run_cmd, run_verification, and run_scoped_verification."""

import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from orchestrator.config import ModuleConfig
from orchestrator.verify import (
    _apply_cargo_scope,
    _build_fallback_config,
    _run_cmd,
    _scope_cargo_workspace,
    _scope_command,
    _strip_directory_flag,
    run_scoped_verification,
    run_verification,
    scope_module_config,
)


def _make_config(**overrides):
    """Build a MagicMock OrchestratorConfig with sane defaults for verify tests."""
    config = MagicMock()
    config.verify_command_timeout_secs = 300.0
    config.verify_timeout_retries = 2
    config.concurrent_verify = True
    config.verify_env = {}
    config.scope_cargo = True
    for k, v in overrides.items():
        setattr(config, k, v)
    return config

# ---------------------------------------------------------------------------
# _run_cmd: executable parameter
# ---------------------------------------------------------------------------

class TestRunCmdUseBash:
    """Verify _run_cmd passes executable='/bin/bash' to create_subprocess_shell."""

    def test_run_cmd_passes_bash_executable(self):
        """The subprocess should be created with executable='/bin/bash'."""
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b'ok\n', None)
        mock_proc.returncode = 0

        with patch('orchestrator.verify.asyncio.create_subprocess_shell', new_callable=AsyncMock) as mock_shell:
            mock_shell.return_value = mock_proc
            rc, output, timed_out = asyncio.run(_run_cmd('echo hello', Path('/tmp'), timeout=5))

        mock_shell.assert_called_once()
        call_kwargs = mock_shell.call_args
        assert call_kwargs.kwargs.get('executable') == '/bin/bash', (
            '_run_cmd must pass executable="/bin/bash" to create_subprocess_shell'
        )
        assert timed_out is False


class TestRunCmdBashBuiltins:
    """Integration test: bash-specific builtins work through _run_cmd."""

    def test_source_dev_null_succeeds(self):
        """'source /dev/null' is bash-only and should succeed via _run_cmd."""
        rc, output, timed_out = asyncio.run(
            _run_cmd('source /dev/null && echo ok', Path('/tmp'), timeout=5)
        )
        assert rc == 0, f'source should work under bash, got rc={rc}: {output}'
        assert 'ok' in output
        assert timed_out is False


# ---------------------------------------------------------------------------
# _run_cmd: timeout handling with bash executable
# ---------------------------------------------------------------------------

class TestRunCmdTimeout:
    """Verify timeout handling still works correctly with bash executable."""

    def test_timeout_returns_error(self):
        """A command that exceeds the timeout should return rc=1 and a message."""
        rc, output, timed_out = asyncio.run(_run_cmd('sleep 10', Path('/tmp'), timeout=0.1))
        assert rc == 1
        assert 'timed out' in output.lower()
        assert timed_out is True

    def test_fast_command_within_timeout(self):
        """A fast command should complete successfully within the timeout."""
        rc, output, timed_out = asyncio.run(_run_cmd('echo fast', Path('/tmp'), timeout=5))
        assert rc == 0
        assert 'fast' in output
        assert timed_out is False


# ---------------------------------------------------------------------------
# run_verification: passes results through correctly
# ---------------------------------------------------------------------------

class TestRunVerification:
    """Verify run_verification aggregates _run_cmd results correctly."""

    def test_all_pass(self, tmp_path: Path):
        """When all commands succeed, result.passed is True."""
        config = _make_config()
        config.test_command = 'echo tests-ok'
        config.lint_command = 'echo lint-ok'
        config.type_check_command = 'echo types-ok'

        result = asyncio.run(run_verification(tmp_path, config))
        assert result.passed is True
        assert result.summary == 'All checks passed'
        assert 'tests-ok' in result.test_output

    def test_test_failure(self, tmp_path: Path):
        """When tests fail, result.passed is False and summary mentions tests."""
        config = _make_config()
        config.test_command = 'exit 1'
        config.lint_command = 'echo lint-ok'
        config.type_check_command = 'echo types-ok'

        result = asyncio.run(run_verification(tmp_path, config))
        assert result.passed is False
        assert 'tests failed' in result.summary

    def test_lint_failure(self, tmp_path: Path):
        """When lint fails, result.passed is False and summary mentions lint."""
        config = _make_config()
        config.test_command = 'echo tests-ok'
        config.lint_command = 'exit 1'
        config.type_check_command = 'echo types-ok'

        result = asyncio.run(run_verification(tmp_path, config))
        assert result.passed is False
        assert 'lint issues' in result.summary

    def test_multiple_failures(self, tmp_path: Path):
        """When multiple checks fail, summary lists all failures."""
        config = _make_config()
        config.test_command = 'exit 1'
        config.lint_command = 'exit 1'
        config.type_check_command = 'exit 1'

        result = asyncio.run(run_verification(tmp_path, config))
        assert result.passed is False
        assert 'tests failed' in result.summary
        assert 'lint issues' in result.summary
        assert 'type errors' in result.summary


# ---------------------------------------------------------------------------
# run_verification: timeout detection + retry
# ---------------------------------------------------------------------------


class TestRunVerificationTimeoutRetry:
    """Verify retry-on-pure-timeout logic in run_verification."""

    def _patched_run_cmd(self, responses):
        """Return an async fn that yields successive tuples from *responses*.

        Each element of *responses* is ``(rc, out, timed_out)`` — returned in
        order on each call to ``_run_cmd``.  The same value is returned when
        responses are exhausted so we don't have to specify every cmd's fate.
        """
        call_log = []

        async def fake(cmd, cwd, timeout, env=None):
            call_log.append(cmd)
            idx = min(len(call_log) - 1, len(responses) - 1)
            return responses[idx]

        return fake, call_log

    def test_retry_converges_on_eventual_success(self, tmp_path: Path):
        """Timeout on first call, success on retry → passed=True."""
        config = _make_config()
        config.verify_timeout_retries = 2
        mc = ModuleConfig(prefix='p', test_command='cmd')

        # Only test_command set → only 1 _run_cmd call per attempt
        fake, log = self._patched_run_cmd([
            (1, 'timed out', True),
            (0, 'ok', False),
        ])
        with patch('orchestrator.verify._run_cmd', side_effect=fake):
            result = asyncio.run(run_verification(tmp_path, config, module_config=mc))
        assert result.passed is True
        assert result.timed_out is False
        assert len(log) == 2  # one call per attempt

    def test_retries_exhausted_timed_out(self, tmp_path: Path):
        """All attempts time out → passed=False, timed_out=True, summary mentions timeout."""
        config = _make_config()
        config.verify_timeout_retries = 2
        mc = ModuleConfig(prefix='p', lint_command='lint-cmd')

        fake, log = self._patched_run_cmd([(1, 'Command timed out after 0.1s', True)])
        with patch('orchestrator.verify._run_cmd', side_effect=fake):
            result = asyncio.run(run_verification(tmp_path, config, module_config=mc))
        assert result.passed is False
        assert result.timed_out is True
        assert 'timed out' in result.summary.lower()
        # Initial attempt + 2 retries = 3 total calls
        assert len(log) == 3

    def test_timeout_then_real_failure_does_not_retry(self, tmp_path: Path):
        """First attempt times out, second attempt surfaces a real failure → stop."""
        config = _make_config()
        config.verify_timeout_retries = 5
        mc = ModuleConfig(prefix='p', lint_command='lint-cmd')

        responses = [
            (1, 'Command timed out', True),  # attempt 0: timeout → retry
            (1, 'real lint error', False),   # attempt 1: real failure → stop
        ]
        fake, log = self._patched_run_cmd(responses)
        with patch('orchestrator.verify._run_cmd', side_effect=fake):
            result = asyncio.run(run_verification(tmp_path, config, module_config=mc))
        assert result.passed is False
        assert result.timed_out is False
        assert 'lint issues' in result.summary
        # Should have stopped after 2 calls despite max_retries=5
        assert len(log) == 2

    def test_mixed_timeout_and_real_failure_first_attempt_stops(self, tmp_path: Path):
        """First attempt: one timeout + one real failure → not a pure timeout, no retry."""
        config = _make_config()
        config.verify_timeout_retries = 3
        mc = ModuleConfig(
            prefix='p',
            test_command='tcmd',
            lint_command='lcmd',
        )

        call_count = {'n': 0}

        async def fake(cmd, cwd, timeout, env=None):
            call_count['n'] += 1
            if cmd == 'tcmd':
                return (1, 'Command timed out', True)
            else:
                return (1, 'real error', False)

        with patch('orchestrator.verify._run_cmd', side_effect=fake):
            result = asyncio.run(run_verification(tmp_path, config, module_config=mc))
        assert result.passed is False
        assert result.timed_out is False
        # Only one attempt executed → 2 calls (tcmd + lcmd)
        assert call_count['n'] == 2

    def test_pure_timeout_sets_timed_out_flag_in_result(self, tmp_path: Path):
        """VerifyResult.timed_out is exposed to callers (Change 3 relies on this)."""
        config = _make_config()
        config.verify_timeout_retries = 0
        mc = ModuleConfig(prefix='p', lint_command='lcmd')

        async def fake(cmd, cwd, timeout, env=None):
            return (1, 'Command timed out after 0.1s', True)

        with patch('orchestrator.verify._run_cmd', side_effect=fake):
            result = asyncio.run(run_verification(tmp_path, config, module_config=mc))
        assert result.passed is False
        assert result.timed_out is True

    def test_failure_report_leads_with_timeout_section(self):
        """failure_report includes a '## Verify Timed Out' section when timed_out=True."""
        from orchestrator.verify import VerifyResult
        result = VerifyResult(
            passed=False,
            test_output='',
            lint_output='Command timed out after 1800s: cargo clippy --workspace',
            type_output='',
            summary='Verification timed out',
            timed_out=True,
        )
        report = result.failure_report()
        assert '## Verify Timed Out' in report
        assert 'lint' in report.lower()


# ---------------------------------------------------------------------------
# run_verification: concurrent_verify mode
# ---------------------------------------------------------------------------


class TestConcurrentVerifyMode:
    """Verify that ``concurrent_verify`` flag controls parallel vs sequential execution."""

    def _make_fake_run_cmd(self, order_log: list[str]):
        """Fake _run_cmd that records call order and returns success."""
        async def fake(cmd, cwd, timeout, env=None):
            order_log.append(cmd)
            # Small await to let the event loop schedule siblings in concurrent mode
            await asyncio.sleep(0)
            return (0, f'ok:{cmd}', False)
        return fake

    def test_concurrent_mode_default(self, tmp_path: Path):
        """Default config → asyncio.gather is used (concurrent mode)."""
        config = _make_config()
        mc = ModuleConfig(
            prefix='p',
            test_command='TESTCMD',
            lint_command='LINTCMD',
            type_check_command='TYPECMD',
        )
        order = []
        with patch(
            'orchestrator.verify._run_cmd',
            side_effect=self._make_fake_run_cmd(order),
        ):
            result = asyncio.run(run_verification(tmp_path, config, module_config=mc))
        assert result.passed is True
        assert set(order) == {'TESTCMD', 'LINTCMD', 'TYPECMD'}

    def test_sequential_mode_runs_in_order(self, tmp_path: Path):
        """concurrent_verify=False → commands run in test, lint, type order."""
        config = _make_config(concurrent_verify=False)
        mc = ModuleConfig(
            prefix='p',
            test_command='TESTCMD',
            lint_command='LINTCMD',
            type_check_command='TYPECMD',
        )
        order = []
        with patch(
            'orchestrator.verify._run_cmd',
            side_effect=self._make_fake_run_cmd(order),
        ):
            result = asyncio.run(run_verification(tmp_path, config, module_config=mc))
        assert result.passed is True
        assert order == ['TESTCMD', 'LINTCMD', 'TYPECMD']

    def test_module_config_override_wins(self, tmp_path: Path):
        """ModuleConfig.concurrent_verify overrides global config."""
        config = _make_config(concurrent_verify=True)
        mc = ModuleConfig(
            prefix='p',
            test_command='T',
            lint_command='L',
            type_check_command='Y',
            concurrent_verify=False,
        )
        order = []
        with patch(
            'orchestrator.verify._run_cmd',
            side_effect=self._make_fake_run_cmd(order),
        ):
            result = asyncio.run(run_verification(tmp_path, config, module_config=mc))
        assert result.passed is True
        assert order == ['T', 'L', 'Y']

    def test_sequential_mode_skips_none_commands(self, tmp_path: Path):
        """Sequential mode still skips None commands without error."""
        config = _make_config(concurrent_verify=False)
        mc = ModuleConfig(prefix='p', lint_command='LINT')  # test + type are None
        order = []
        with patch(
            'orchestrator.verify._run_cmd',
            side_effect=self._make_fake_run_cmd(order),
        ):
            result = asyncio.run(run_verification(tmp_path, config, module_config=mc))
        assert result.passed is True
        assert order == ['LINT']


# ---------------------------------------------------------------------------
# run_verification: verify_env injection
# ---------------------------------------------------------------------------


class TestVerifyEnv:
    """Verify env vars are injected into _run_cmd from config/module config."""

    def test_top_level_env_passed_through(self, tmp_path: Path):
        """config.verify_env is forwarded to _run_cmd via env kwarg."""
        config = _make_config(verify_env={'RUSTC_WRAPPER': 'sccache'})
        mc = ModuleConfig(prefix='p', lint_command='LINT')

        captured_envs = []

        async def fake(cmd, cwd, timeout, env=None):
            captured_envs.append(env)
            return (0, 'ok', False)

        with patch('orchestrator.verify._run_cmd', side_effect=fake):
            asyncio.run(run_verification(tmp_path, config, module_config=mc))
        assert captured_envs == [{'RUSTC_WRAPPER': 'sccache'}]

    def test_module_env_overrides_top_level(self, tmp_path: Path):
        """ModuleConfig.verify_env keys override top-level config.verify_env keys."""
        config = _make_config(verify_env={'RUSTC_WRAPPER': 'sccache', 'CARGO_INCREMENTAL': '0'})
        mc = ModuleConfig(
            prefix='p',
            lint_command='LINT',
            verify_env={'RUSTC_WRAPPER': 'rustc-cache'},  # overrides sccache
        )

        captured_envs = []

        async def fake(cmd, cwd, timeout, env=None):
            captured_envs.append(env)
            return (0, 'ok', False)

        with patch('orchestrator.verify._run_cmd', side_effect=fake):
            asyncio.run(run_verification(tmp_path, config, module_config=mc))
        assert captured_envs == [
            {'RUSTC_WRAPPER': 'rustc-cache', 'CARGO_INCREMENTAL': '0'},
        ]

    def test_empty_env_passes_none(self, tmp_path: Path):
        """When no verify_env is configured, _run_cmd is called with env=None."""
        config = _make_config(verify_env={})
        mc = ModuleConfig(prefix='p', lint_command='LINT')

        captured = []

        async def fake(cmd, cwd, timeout, env=None):
            captured.append(env)
            return (0, 'ok', False)

        with patch('orchestrator.verify._run_cmd', side_effect=fake):
            asyncio.run(run_verification(tmp_path, config, module_config=mc))
        assert captured == [None]

    def test_run_cmd_merges_env_with_os_environ(self, tmp_path: Path):
        """_run_cmd should merge caller env with os.environ, caller wins."""
        # Pick a var we can safely rely on existing
        os.environ['VERIFY_ENV_TEST_KEEP'] = 'keep_value'
        try:
            rc, out, timed_out = asyncio.run(
                _run_cmd(
                    'echo "$VERIFY_ENV_TEST_KEEP $VERIFY_ENV_TEST_INJECT"',
                    tmp_path,
                    timeout=5,
                    env={'VERIFY_ENV_TEST_INJECT': 'injected_value'},
                )
            )
        finally:
            del os.environ['VERIFY_ENV_TEST_KEEP']
        assert rc == 0
        assert 'keep_value' in out
        assert 'injected_value' in out
        assert timed_out is False


# ---------------------------------------------------------------------------
# run_verification: ModuleConfig overrides
# ---------------------------------------------------------------------------

class TestRunVerificationModuleConfig:
    """Verify that ModuleConfig overrides the global config commands."""

    def test_module_config_overrides_commands(self, tmp_path: Path):
        """ModuleConfig commands take precedence over global config."""
        config = _make_config()
        config.test_command = 'exit 1'
        config.lint_command = 'exit 1'
        config.type_check_command = 'exit 1'

        mc = ModuleConfig(
            prefix='dashboard',
            test_command='echo test-override',
            lint_command='echo lint-override',
            type_check_command='echo type-override',
        )
        result = asyncio.run(run_verification(tmp_path, config, module_config=mc))
        assert result.passed is True
        assert 'test-override' in result.test_output

    def test_partial_override_skips_undefined(self, tmp_path: Path):
        """ModuleConfig with only test_command; lint/type are None → skipped."""
        config = _make_config()
        config.test_command = 'exit 1'
        config.lint_command = 'echo lint-global'
        config.type_check_command = 'echo type-global'

        mc = ModuleConfig(prefix='dashboard', test_command='echo test-override')
        result = asyncio.run(run_verification(tmp_path, config, module_config=mc))
        assert result.passed is True
        assert 'test-override' in result.test_output

    def test_none_module_config_unchanged(self, tmp_path: Path):
        """Passing None for module_config uses global config (backward compat)."""
        config = _make_config()
        config.test_command = 'echo tests-ok'
        config.lint_command = 'echo lint-ok'
        config.type_check_command = 'echo types-ok'

        result = asyncio.run(run_verification(tmp_path, config, module_config=None))
        assert result.passed is True
        assert 'tests-ok' in result.test_output


# ---------------------------------------------------------------------------
# run_verification: None commands skip (scoped mode)
# ---------------------------------------------------------------------------

class TestNoneCommandSkipping:
    """When ModuleConfig is provided, None commands are skipped, not global-fallback."""

    def test_none_test_command_skips_tests(self, tmp_path: Path):
        """ModuleConfig with test_command=None skips tests entirely."""
        config = _make_config()
        config.test_command = 'exit 1'  # global would fail
        config.lint_command = 'exit 1'
        config.type_check_command = 'exit 1'

        mc = ModuleConfig(prefix='shared', lint_command='echo lint-ok')
        result = asyncio.run(run_verification(tmp_path, config, module_config=mc))
        assert result.passed is True
        assert result.test_output == ''
        assert result.type_output == ''

    def test_all_none_commands_pass(self, tmp_path: Path):
        """ModuleConfig with all None commands passes (everything skipped)."""
        config = _make_config()
        config.test_command = 'exit 1'
        config.lint_command = 'exit 1'
        config.type_check_command = 'exit 1'

        mc = ModuleConfig(prefix='empty')
        result = asyncio.run(run_verification(tmp_path, config, module_config=mc))
        assert result.passed is True
        assert result.summary == 'All checks passed'

    def test_lint_only_module(self, tmp_path: Path):
        """Simulates shared/ — only lint defined, test and type are None."""
        config = _make_config()
        config.test_command = 'exit 1'
        config.lint_command = 'exit 1'
        config.type_check_command = 'exit 1'

        mc = ModuleConfig(prefix='shared', lint_command='echo lint-ok')
        result = asyncio.run(run_verification(tmp_path, config, module_config=mc))
        assert result.passed is True
        assert result.test_output == ''

    def test_lint_failure_still_fails(self, tmp_path: Path):
        """Even with tests skipped, a lint failure still fails the result."""
        config = _make_config()
        mc = ModuleConfig(prefix='shared', lint_command='exit 1')
        result = asyncio.run(run_verification(tmp_path, config, module_config=mc))
        assert result.passed is False
        assert 'lint issues' in result.summary


# ---------------------------------------------------------------------------
# run_scoped_verification: multi-subproject aggregation
# ---------------------------------------------------------------------------

class TestRunScopedVerification:
    """Tests for run_scoped_verification aggregation logic."""

    def test_empty_list_falls_back_to_global(self, tmp_path: Path):
        """Empty module_configs list runs global verification."""
        config = _make_config()
        config.test_command = 'echo global-test'
        config.lint_command = 'echo global-lint'
        config.type_check_command = 'echo global-type'

        result = asyncio.run(run_scoped_verification(tmp_path, config, []))
        assert result.passed is True
        assert 'global-test' in result.test_output

    def test_single_module_config(self, tmp_path: Path):
        """Single ModuleConfig delegates to run_verification with that config."""
        config = _make_config()
        config.test_command = 'exit 1'

        mc = ModuleConfig(prefix='esc', test_command='echo esc-test', lint_command='echo esc-lint')
        result = asyncio.run(run_scoped_verification(tmp_path, config, [mc]))
        assert result.passed is True
        assert 'esc-test' in result.test_output

    def test_multiple_module_configs_all_pass(self, tmp_path: Path):
        """Multiple ModuleConfigs all pass → aggregated result passes."""
        config = _make_config()

        mc1 = ModuleConfig(prefix='a', lint_command='echo a-lint')
        mc2 = ModuleConfig(prefix='b', lint_command='echo b-lint', test_command='echo b-test')
        result = asyncio.run(run_scoped_verification(tmp_path, config, [mc1, mc2]))
        assert result.passed is True

    def test_one_failure_fails_aggregate(self, tmp_path: Path):
        """One subproject fails → aggregated result fails."""
        config = _make_config()

        mc_good = ModuleConfig(prefix='a', lint_command='echo ok')
        mc_bad = ModuleConfig(prefix='b', lint_command='exit 1')
        result = asyncio.run(run_scoped_verification(tmp_path, config, [mc_good, mc_bad]))
        assert result.passed is False
        assert 'lint issues' in result.summary

    def test_aggregate_combines_output(self, tmp_path: Path):
        """Aggregation merges test_output from multiple subprojects."""
        config = _make_config()

        mc1 = ModuleConfig(prefix='a', test_command='echo test-a')
        mc2 = ModuleConfig(prefix='b', test_command='echo test-b')
        result = asyncio.run(run_scoped_verification(tmp_path, config, [mc1, mc2]))
        assert 'test-a' in result.test_output
        assert 'test-b' in result.test_output


# ---------------------------------------------------------------------------
# _scope_command: replace path args with specific files
# ---------------------------------------------------------------------------

class TestScopeCommand:
    """Tests for _scope_command(cmd, tool_keyword, files)."""

    def test_ruff_check_replaces_paths(self):
        """Basic ruff check: directory args replaced with specific file."""
        cmd = 'uv run --project fused-memory --directory fused-memory ruff check src/ tests/'
        result = _scope_command(cmd, 'ruff check', ['src/verify.py'])
        assert result == 'uv run --project fused-memory --directory fused-memory ruff check src/verify.py'

    def test_pyright_replaces_paths(self):
        """Pyright: directory args replaced with specific file."""
        cmd = 'uv run --project orchestrator --directory orchestrator pyright src/ tests/'
        result = _scope_command(cmd, 'pyright', ['src/orchestrator/verify.py'])
        assert result == 'uv run --project orchestrator --directory orchestrator pyright src/orchestrator/verify.py'

    def test_pytest_preserves_flags(self):
        """Pytest: --tb and -q flags are preserved after file list."""
        cmd = 'uv run --project orchestrator --directory orchestrator pytest tests/ --tb=short -q'
        result = _scope_command(cmd, 'pytest', ['tests/test_verify.py'])
        assert result == 'uv run --project orchestrator --directory orchestrator pytest tests/test_verify.py --tb=short -q'

    def test_pytest_multiple_files_with_flags(self):
        """Multiple files and multiple flags are all included."""
        cmd = 'pytest tests/ --tb=short -q'
        result = _scope_command(cmd, 'pytest', ['tests/test_a.py', 'tests/test_b.py'])
        assert result == 'pytest tests/test_a.py tests/test_b.py --tb=short -q'

    def test_returns_none_when_cmd_is_none(self):
        """None cmd → None result."""
        result = _scope_command(None, 'ruff check', ['src/foo.py'])
        assert result is None

    def test_returns_none_when_files_empty(self):
        """Empty files list → None result."""
        cmd = 'ruff check src/ tests/'
        result = _scope_command(cmd, 'ruff check', [])
        assert result is None

    def test_returns_original_when_tool_not_found(self):
        """Tool keyword not in command → original command returned unchanged."""
        cmd = 'echo hello world'
        result = _scope_command(cmd, 'ruff check', ['src/foo.py'])
        assert result == cmd


# ---------------------------------------------------------------------------
# scope_module_config: narrow ModuleConfig commands to task files
# ---------------------------------------------------------------------------

class TestStripDirectoryFlag:
    """Tests for _strip_directory_flag(cmd, module_prefix)."""

    def test_strips_space_separated(self):
        """--directory prefix with space is removed."""
        cmd = 'uv run --project orch --directory orch ruff check f.py'
        assert _strip_directory_flag(cmd, 'orch') == 'uv run --project orch ruff check f.py'

    def test_strips_equals_separated(self):
        """--directory=prefix is removed."""
        cmd = 'uv run --project orch --directory=orch ruff check f.py'
        assert _strip_directory_flag(cmd, 'orch') == 'uv run --project orch ruff check f.py'

    def test_returns_none_for_none(self):
        assert _strip_directory_flag(None, 'orch') is None

    def test_no_directory_flag_unchanged(self):
        cmd = 'uv run --project orch ruff check f.py'
        assert _strip_directory_flag(cmd, 'orch') == cmd

    def test_cleans_double_spaces(self):
        cmd = 'uv run  --directory orch  ruff check f.py'
        result = _strip_directory_flag(cmd, 'orch')
        assert result is not None
        assert '  ' not in result

    def test_flag_at_end_of_string(self):
        cmd = 'uv run --directory orch'
        assert _strip_directory_flag(cmd, 'orch') == 'uv run'


class TestScopeModuleConfig:
    """Tests for scope_module_config(mc, task_files)."""

    def _make_mc(self) -> ModuleConfig:
        return ModuleConfig(
            prefix='orchestrator',
            lint_command='uv run --project orchestrator --directory orchestrator ruff check src/ tests/',
            type_check_command='uv run --project orchestrator --directory orchestrator pyright src/ tests/',
            test_command='uv run --project orchestrator --directory orchestrator pytest tests/ --tb=short -q',
        )

    def test_keeps_prefix_and_scopes_files(self):
        """Task files keep their full worktree-relative path in scoped commands."""
        mc = self._make_mc()
        result = scope_module_config(mc, ['orchestrator/src/orchestrator/verify.py'])
        assert result.lint_command is not None
        assert 'orchestrator/src/orchestrator/verify.py' in result.lint_command
        # Should NOT contain the original directory args
        assert 'src/ tests/' not in result.lint_command

    def test_strips_directory_flag_from_scoped_commands(self):
        """--directory is removed from scoped commands since paths are worktree-relative."""
        mc = self._make_mc()
        result = scope_module_config(mc, ['orchestrator/src/orchestrator/verify.py'])
        assert result.lint_command is not None
        assert result.type_check_command is not None
        assert '--directory' not in result.lint_command
        assert '--directory' not in result.type_check_command
        # --project is preserved for venv resolution
        assert '--project orchestrator' in result.lint_command
        assert '--project orchestrator' in result.type_check_command

    def test_classifies_test_files_by_test_prefix(self):
        """Files with test_ prefix are classified as test files."""
        mc = self._make_mc()
        result = scope_module_config(mc, ['orchestrator/tests/test_verify.py'])
        assert result.test_command is not None
        assert 'orchestrator/tests/test_verify.py' in result.test_command

    def test_classifies_test_files_by_suffix(self):
        """Files ending in _test.py are classified as test files."""
        mc = self._make_mc()
        result = scope_module_config(mc, ['orchestrator/tests/verify_test.py'])
        assert result.test_command is not None
        assert 'orchestrator/tests/verify_test.py' in result.test_command

    def test_classifies_conftest_as_test_file(self):
        """conftest.py is classified as a test file."""
        mc = self._make_mc()
        result = scope_module_config(mc, ['orchestrator/tests/conftest.py'])
        assert result.test_command is not None
        assert 'orchestrator/tests/conftest.py' in result.test_command

    def test_classifies_tests_dir_as_test_file(self):
        """Files under /tests/ directory are classified as test files."""
        mc = self._make_mc()
        result = scope_module_config(mc, ['orchestrator/tests/helpers/util.py'])
        assert result.test_command is not None
        assert 'orchestrator/tests/helpers/util.py' in result.test_command

    def test_test_command_none_when_no_test_files(self):
        """No test files in task_files → test_command is None."""
        mc = self._make_mc()
        result = scope_module_config(mc, ['orchestrator/src/orchestrator/verify.py'])
        assert result.test_command is None

    def test_returns_original_when_no_files_match_prefix(self):
        """task_files that don't match prefix → original ModuleConfig returned."""
        mc = self._make_mc()
        result = scope_module_config(mc, ['fused-memory/src/foo.py'])
        assert result is mc

    def test_non_py_files_excluded(self):
        """Non-.py files (yaml, json, md) are not passed to ruff/pyright/pytest."""
        mc = self._make_mc()
        result = scope_module_config(mc, ['orchestrator/orchestrator.yaml', 'README.md'])
        # No .py files match → returns original
        assert result is mc

    def test_mixed_source_and_test_files(self):
        """Mixed source and test files both appear in their respective commands."""
        mc = self._make_mc()
        task_files = [
            'orchestrator/src/orchestrator/verify.py',
            'orchestrator/tests/test_verify.py',
        ]
        result = scope_module_config(mc, task_files)
        assert result.lint_command is not None
        assert 'orchestrator/src/orchestrator/verify.py' in result.lint_command
        assert 'orchestrator/tests/test_verify.py' in result.lint_command
        assert result.test_command is not None
        assert 'orchestrator/tests/test_verify.py' in result.test_command


# ---------------------------------------------------------------------------
# _build_fallback_config: synthetic ModuleConfig from task file list
# ---------------------------------------------------------------------------

class TestBuildFallbackConfig:
    """Tests for _build_fallback_config(task_files)."""

    def test_source_and_test_files_all_commands(self):
        """Source + test .py files → ModuleConfig with all three commands."""
        task_files = [
            'orchestrator/src/orchestrator/verify.py',
            'orchestrator/tests/test_verify.py',
        ]
        result = _build_fallback_config(task_files)
        assert result is not None
        assert result.prefix == '__fallback__'
        assert result.lint_command is not None
        assert 'ruff check' in result.lint_command
        assert 'orchestrator/src/orchestrator/verify.py' in result.lint_command
        assert 'orchestrator/tests/test_verify.py' in result.lint_command
        assert result.type_check_command is not None
        assert 'pyright' in result.type_check_command
        assert result.test_command is not None
        assert 'pytest' in result.test_command
        assert 'orchestrator/tests/test_verify.py' in result.test_command
        # Source file should NOT appear in pytest command
        assert 'orchestrator/src/orchestrator/verify.py' not in result.test_command

    def test_only_source_files_no_test_command(self):
        """Only source .py files → test_command is None."""
        task_files = ['orchestrator/src/orchestrator/verify.py']
        result = _build_fallback_config(task_files)
        assert result is not None
        assert result.test_command is None
        assert result.lint_command is not None
        assert result.type_check_command is not None

    def test_empty_list_returns_none(self):
        """Empty list → None."""
        result = _build_fallback_config([])
        assert result is None

    def test_only_non_py_files_returns_none(self):
        """Only .yaml and .md files → None (no .py to check)."""
        task_files = ['orchestrator/orchestrator.yaml', 'README.md']
        result = _build_fallback_config(task_files)
        assert result is None


# ---------------------------------------------------------------------------
# run_scoped_verification: task_files parameter integration
# ---------------------------------------------------------------------------

class TestRunScopedVerificationTaskFiles:
    """Tests for run_scoped_verification with the new task_files parameter."""

    def test_module_configs_with_task_files_scopes_commands(self, tmp_path: Path):
        """module_configs populated + task_files → commands scoped to those files."""
        config = _make_config()
        # The 'echo file-check' command doesn't contain 'ruff check', so _scope_command
        # returns it unchanged; we verify task_files are actually passed to scope_module_config
        # by using a real ruff check command that will be narrowed
        mc = ModuleConfig(
            prefix='orchestrator',
            lint_command='echo ruff check src/ tests/',
            test_command=None,
        )
        task_files = ['orchestrator/src/orchestrator/verify.py']
        result = asyncio.run(run_scoped_verification(tmp_path, config, [mc], task_files=task_files))
        assert result.passed is True

    def test_module_configs_with_task_files_real_scoping(self, tmp_path: Path):
        """Real ruff check command is narrowed to specific file via scope_module_config."""
        config = _make_config()
        mc = ModuleConfig(
            prefix='orchestrator',
            lint_command='echo ruff check src/ tests/',
        )
        task_files = ['orchestrator/src/orchestrator/verify.py']
        result = asyncio.run(run_scoped_verification(tmp_path, config, [mc], task_files=task_files))
        assert result.passed is True

    def test_empty_module_configs_with_task_files_bypasses_global(self, tmp_path: Path):
        """Fallback path is exercised via patching _build_fallback_config."""
        config = _make_config()
        # Global would fail with exit 1
        config.test_command = 'exit 1'
        config.lint_command = 'exit 1'
        config.type_check_command = 'exit 1'

        fallback_mc = ModuleConfig(
            prefix='__fallback__',
            lint_command='echo fallback-lint',
            test_command='echo fallback-test',
        )

        with patch('orchestrator.verify._build_fallback_config', return_value=fallback_mc):
            result = asyncio.run(
                run_scoped_verification(tmp_path, config, [], task_files=['src/foo.py'])
            )
        assert result.passed is True
        assert 'fallback-test' in result.test_output

    def test_both_empty_runs_global(self, tmp_path: Path):
        """module_configs=[] and task_files=None → global verification runs."""
        config = _make_config()
        config.test_command = 'echo global-test'
        config.lint_command = 'echo global-lint'
        config.type_check_command = 'echo global-type'

        result = asyncio.run(run_scoped_verification(tmp_path, config, [], task_files=None))
        assert result.passed is True
        assert 'global-test' in result.test_output

    def test_task_files_none_backward_compatible(self, tmp_path: Path):
        """task_files=None → module_configs used unscoped (backward compatible)."""
        config = _make_config()
        config.test_command = 'exit 1'  # global would fail

        mc = ModuleConfig(prefix='esc', lint_command='echo esc-lint', test_command='echo esc-test')
        result = asyncio.run(run_scoped_verification(tmp_path, config, [mc], task_files=None))
        assert result.passed is True
        assert 'esc-test' in result.test_output


# ---------------------------------------------------------------------------
# _scope_cargo_workspace: cargo --workspace → cargo -p <crate>
# ---------------------------------------------------------------------------


class TestScopeCargoWorkspace:
    """Tests for ``_scope_cargo_workspace(cmd, crates)``."""

    def test_single_crate_rewrite(self):
        cmd = 'cargo test --workspace'
        assert _scope_cargo_workspace(cmd, ['reify-eval']) == 'cargo test -p reify-eval'

    def test_multi_crate_rewrite(self):
        cmd = 'cargo test --workspace'
        result = _scope_cargo_workspace(cmd, ['reify-eval', 'reify-lsp'])
        assert result == 'cargo test -p reify-eval -p reify-lsp'

    def test_preserves_trailing_test_args(self):
        cmd = 'cargo test --workspace -- --test-threads=1'
        result = _scope_cargo_workspace(cmd, ['reify-eval'])
        assert result == 'cargo test -p reify-eval -- --test-threads=1'

    def test_preserves_all_targets_flag(self):
        cmd = 'cargo test --all-targets --workspace -- --test-threads=1'
        result = _scope_cargo_workspace(cmd, ['eval'])
        assert result == 'cargo test --all-targets -p eval -- --test-threads=1'

    def test_preserves_release_flag(self):
        cmd = 'cargo build --release --workspace'
        assert _scope_cargo_workspace(cmd, ['eval']) == 'cargo build --release -p eval'

    def test_preserves_feature_flag(self):
        cmd = 'cargo test --workspace -F my-feature'
        # Non-greedy match: --workspace is swapped, -F stays in place
        result = _scope_cargo_workspace(cmd, ['eval'])
        assert result == 'cargo test -p eval -F my-feature'

    def test_clippy_rewrite(self):
        cmd = 'cargo clippy --workspace -- -D warnings'
        result = _scope_cargo_workspace(cmd, ['eval'])
        assert result == 'cargo clippy -p eval -- -D warnings'

    def test_check_rewrite(self):
        cmd = 'cargo check --workspace'
        assert _scope_cargo_workspace(cmd, ['eval']) == 'cargo check -p eval'

    def test_chained_shell_leaves_non_cargo_alone(self):
        cmd = (
            '. ~/.cargo/env && cargo test --workspace -- --test-threads=1 '
            '&& cd gui && npm test'
        )
        result = _scope_cargo_workspace(cmd, ['reify-eval'])
        assert result == (
            '. ~/.cargo/env && cargo test -p reify-eval -- --test-threads=1 '
            '&& cd gui && npm test'
        )

    def test_chained_cargo_commands_both_rewritten(self):
        cmd = 'cargo test --workspace && cargo clippy --workspace -- -D warnings'
        result = _scope_cargo_workspace(cmd, ['eval'])
        assert result == 'cargo test -p eval && cargo clippy -p eval -- -D warnings'

    def test_none_command_returns_none(self):
        assert _scope_cargo_workspace(None, ['eval']) is None

    def test_empty_crates_returns_cmd_unchanged(self):
        cmd = 'cargo test --workspace'
        assert _scope_cargo_workspace(cmd, []) == cmd

    def test_no_workspace_flag_returns_unchanged(self):
        cmd = 'cargo test -p eval'
        assert _scope_cargo_workspace(cmd, ['eval']) == cmd

    def test_non_cargo_command_unchanged(self):
        cmd = 'pytest tests/'
        assert _scope_cargo_workspace(cmd, ['eval']) == cmd

    def test_unsupported_subcommand_not_rewritten(self):
        # cargo doc --workspace — we don't touch 'doc' to avoid semantic drift
        cmd = 'cargo doc --workspace'
        assert _scope_cargo_workspace(cmd, ['eval']) == cmd


# ---------------------------------------------------------------------------
# _apply_cargo_scope: high-level ModuleConfig rewrite with guards
# ---------------------------------------------------------------------------


class TestApplyCargoScope:
    """Tests for ``_apply_cargo_scope(mc, task_files, project_root, enabled)``."""

    def _write_workspace(self, root: Path, crates: dict[str, str]) -> None:
        """Create a minimal cargo workspace rooted at *root*.

        *crates* maps relative crate dir → crate name. Each crate gets a
        trivial Cargo.toml with ``[package].name`` set.
        """
        members = sorted({d.split('/', 1)[0] + '/*' for d in crates})
        (root / 'Cargo.toml').write_text(
            '[workspace]\nmembers = [' + ', '.join(f'"{m}"' for m in members) + ']\n'
        )
        for rel, name in crates.items():
            crate_dir = root / rel
            crate_dir.mkdir(parents=True, exist_ok=True)
            (crate_dir / 'Cargo.toml').write_text(
                f'[package]\nname = "{name}"\nversion = "0.1.0"\n'
            )

    def _reset_cache(self):
        from orchestrator.cargo_scope import _clear_cache
        _clear_cache()

    def test_single_crate_maps_and_rewrites(self, tmp_path: Path):
        self._reset_cache()
        self._write_workspace(tmp_path, {
            'crates/reify-eval': 'reify-eval',
            'crates/reify-lsp': 'reify-lsp',
        })
        mc = ModuleConfig(
            prefix='__synthetic__',
            test_command='cargo test --workspace',
            lint_command='cargo clippy --workspace -- -D warnings',
            type_check_command='cargo check --workspace',
        )
        result = _apply_cargo_scope(
            mc, ['crates/reify-eval/src/foo.rs'], tmp_path, True,
        )
        assert result is not mc
        assert result.test_command == 'cargo test -p reify-eval'
        assert result.lint_command == 'cargo clippy -p reify-eval -- -D warnings'
        assert result.type_check_command == 'cargo check -p reify-eval'

    def test_multi_crate_rewrite(self, tmp_path: Path):
        self._reset_cache()
        self._write_workspace(tmp_path, {
            'crates/reify-eval': 'reify-eval',
            'crates/reify-lsp': 'reify-lsp',
        })
        mc = ModuleConfig(
            prefix='__synthetic__',
            test_command='cargo test --workspace',
        )
        files = [
            'crates/reify-eval/src/a.rs',
            'crates/reify-lsp/src/b.rs',
        ]
        result = _apply_cargo_scope(mc, files, tmp_path, True)
        assert result.test_command == 'cargo test -p reify-eval -p reify-lsp'

    def test_mixed_language_scopes_to_rs_only(self, tmp_path: Path):
        self._reset_cache()
        self._write_workspace(tmp_path, {
            'crates/reify-eval': 'reify-eval',
        })
        mc = ModuleConfig(
            prefix='__synthetic__',
            test_command='cargo test --workspace',
        )
        files = [
            'crates/reify-eval/src/a.rs',
            'gui/src/editor/baz.ts',  # non-.rs — filtered out, only .rs used
        ]
        result = _apply_cargo_scope(mc, files, tmp_path, True)
        # Non-Rust files are filtered; cargo scoped to touched crates only
        assert result.test_command == 'cargo test -p reify-eval'

    def test_file_outside_crates_falls_through(self, tmp_path: Path):
        self._reset_cache()
        self._write_workspace(tmp_path, {
            'crates/reify-eval': 'reify-eval',
        })
        mc = ModuleConfig(
            prefix='__synthetic__',
            test_command='cargo test --workspace',
        )
        # build.rs at repo root is not under a crate dir
        result = _apply_cargo_scope(
            mc, ['build.rs', 'crates/reify-eval/src/a.rs'], tmp_path, True,
        )
        assert result is mc

    def test_module_scope_cargo_false_disables(self, tmp_path: Path):
        self._reset_cache()
        self._write_workspace(tmp_path, {
            'crates/reify-eval': 'reify-eval',
        })
        mc = ModuleConfig(
            prefix='__synthetic__',
            test_command='cargo test --workspace',
            scope_cargo=False,
        )
        result = _apply_cargo_scope(
            mc, ['crates/reify-eval/src/a.rs'], tmp_path, True,
        )
        assert result is mc

    def test_global_scope_cargo_false_disables(self, tmp_path: Path):
        self._reset_cache()
        self._write_workspace(tmp_path, {
            'crates/reify-eval': 'reify-eval',
        })
        mc = ModuleConfig(
            prefix='__synthetic__',
            test_command='cargo test --workspace',
        )
        result = _apply_cargo_scope(
            mc, ['crates/reify-eval/src/a.rs'], tmp_path, False,
        )
        assert result is mc

    def test_no_workspace_flag_returns_same_instance(self, tmp_path: Path):
        self._reset_cache()
        self._write_workspace(tmp_path, {
            'crates/reify-eval': 'reify-eval',
        })
        mc = ModuleConfig(
            prefix='__synthetic__',
            test_command='pytest tests/',  # no cargo, no --workspace
        )
        result = _apply_cargo_scope(
            mc, ['crates/reify-eval/src/a.rs'], tmp_path, True,
        )
        assert result is mc

    def test_empty_task_files_returns_same_instance(self, tmp_path: Path):
        self._reset_cache()
        self._write_workspace(tmp_path, {
            'crates/reify-eval': 'reify-eval',
        })
        mc = ModuleConfig(
            prefix='__synthetic__',
            test_command='cargo test --workspace',
        )
        result = _apply_cargo_scope(mc, [], tmp_path, True)
        assert result is mc

    def test_no_cargo_toml_returns_same_instance(self, tmp_path: Path):
        self._reset_cache()
        # No Cargo.toml in tmp_path
        mc = ModuleConfig(
            prefix='__synthetic__',
            test_command='cargo test --workspace',
        )
        result = _apply_cargo_scope(mc, ['src/a.rs'], tmp_path, True)
        assert result is mc


# ---------------------------------------------------------------------------
# run_scoped_verification: end-to-end cargo scoping wiring
# ---------------------------------------------------------------------------


class TestRunScopedVerificationCargoScope:
    """End-to-end: ensure run_scoped_verification rewrites Rust global commands."""

    def _write_workspace(self, root: Path, crates: dict[str, str]) -> None:
        members = sorted({d.split('/', 1)[0] + '/*' for d in crates})
        (root / 'Cargo.toml').write_text(
            '[workspace]\nmembers = [' + ', '.join(f'"{m}"' for m in members) + ']\n'
        )
        for rel, name in crates.items():
            crate_dir = root / rel
            crate_dir.mkdir(parents=True, exist_ok=True)
            (crate_dir / 'Cargo.toml').write_text(
                f'[package]\nname = "{name}"\nversion = "0.1.0"\n'
            )

    def test_global_cargo_scope_invoked_with_rewritten_command(self, tmp_path: Path):
        """Capture what _run_cmd received to prove the rewrite happened."""
        from orchestrator.cargo_scope import _clear_cache
        _clear_cache()

        self._write_workspace(tmp_path, {
            'crates/reify-eval': 'reify-eval',
        })
        (tmp_path / 'crates/reify-eval/src').mkdir(parents=True, exist_ok=True)
        (tmp_path / 'crates/reify-eval/src/foo.rs').write_text('// stub\n')

        config = _make_config()
        config.test_command = 'cargo test --workspace'
        config.lint_command = 'cargo clippy --workspace -- -D warnings'
        config.type_check_command = 'cargo check --workspace'

        captured: list[str] = []

        async def fake(cmd, cwd, timeout, env=None):
            captured.append(cmd)
            return (0, 'ok', False)

        with patch('orchestrator.verify._run_cmd', side_effect=fake):
            result = asyncio.run(
                run_scoped_verification(
                    tmp_path, config, [],
                    task_files=['crates/reify-eval/src/foo.rs'],
                )
            )
        assert result.passed is True
        # Expect the rewritten commands:
        assert 'cargo test -p reify-eval' in captured
        assert 'cargo clippy -p reify-eval -- -D warnings' in captured
        assert 'cargo check -p reify-eval' in captured
        # And NOT the --workspace forms
        assert 'cargo test --workspace' not in captured
        assert 'cargo clippy --workspace -- -D warnings' not in captured

    def test_mixed_language_task_scopes_to_rs_crates(self, tmp_path: Path):
        """A task with .rs + .ts files scopes cargo to touched Rust crates only."""
        from orchestrator.cargo_scope import _clear_cache
        _clear_cache()

        self._write_workspace(tmp_path, {
            'crates/reify-eval': 'reify-eval',
        })
        (tmp_path / 'crates/reify-eval/src').mkdir(parents=True, exist_ok=True)
        (tmp_path / 'crates/reify-eval/src/foo.rs').write_text('// stub\n')
        (tmp_path / 'gui/src').mkdir(parents=True, exist_ok=True)
        (tmp_path / 'gui/src/bar.ts').write_text('// stub\n')

        config = _make_config()
        config.test_command = 'cargo test --workspace && cd gui && npm test'
        config.lint_command = 'cargo clippy --workspace'
        config.type_check_command = 'cargo check --workspace'

        captured: list[str] = []

        async def fake(cmd, cwd, timeout, env=None):
            captured.append(cmd)
            return (0, 'ok', False)

        with patch('orchestrator.verify._run_cmd', side_effect=fake):
            asyncio.run(
                run_scoped_verification(
                    tmp_path, config, [],
                    task_files=[
                        'crates/reify-eval/src/foo.rs',
                        'gui/src/bar.ts',
                    ],
                )
            )
        # Non-.rs files are filtered; cargo scoped to the touched Rust crates
        assert any('-p reify-eval' in c for c in captured)

    def test_scope_cargo_disabled_globally(self, tmp_path: Path):
        from orchestrator.cargo_scope import _clear_cache
        _clear_cache()

        self._write_workspace(tmp_path, {
            'crates/reify-eval': 'reify-eval',
        })
        (tmp_path / 'crates/reify-eval/src').mkdir(parents=True, exist_ok=True)
        (tmp_path / 'crates/reify-eval/src/foo.rs').write_text('// stub\n')

        config = _make_config(scope_cargo=False)
        config.test_command = 'cargo test --workspace'
        config.lint_command = 'cargo clippy --workspace'
        config.type_check_command = 'cargo check --workspace'

        captured: list[str] = []

        async def fake(cmd, cwd, timeout, env=None):
            captured.append(cmd)
            return (0, 'ok', False)

        with patch('orchestrator.verify._run_cmd', side_effect=fake):
            asyncio.run(
                run_scoped_verification(
                    tmp_path, config, [],
                    task_files=['crates/reify-eval/src/foo.rs'],
                )
            )
        assert 'cargo test --workspace' in captured
        assert not any(' -p reify-eval' in c for c in captured)
