"""Tests for verify.py — _run_cmd, run_verification, and run_scoped_verification."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from orchestrator.config import ModuleConfig
from orchestrator.verify import _run_cmd, run_scoped_verification, run_verification

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
            rc, output = asyncio.run(_run_cmd('echo hello', Path('/tmp')))

        mock_shell.assert_called_once()
        call_kwargs = mock_shell.call_args
        assert call_kwargs.kwargs.get('executable') == '/bin/bash', (
            '_run_cmd must pass executable="/bin/bash" to create_subprocess_shell'
        )


class TestRunCmdBashBuiltins:
    """Integration test: bash-specific builtins work through _run_cmd."""

    def test_source_dev_null_succeeds(self):
        """'source /dev/null' is bash-only and should succeed via _run_cmd."""
        rc, output = asyncio.run(_run_cmd('source /dev/null && echo ok', Path('/tmp')))
        assert rc == 0, f'source should work under bash, got rc={rc}: {output}'
        assert 'ok' in output


# ---------------------------------------------------------------------------
# _run_cmd: timeout handling with bash executable
# ---------------------------------------------------------------------------


class TestRunCmdTimeout:
    """Verify timeout handling still works correctly with bash executable."""

    def test_timeout_returns_error(self):
        """A command that exceeds the timeout should return rc=1 and a message."""
        rc, output = asyncio.run(_run_cmd('sleep 10', Path('/tmp'), timeout=0.1))
        assert rc == 1
        assert 'timed out' in output.lower()

    def test_fast_command_within_timeout(self):
        """A fast command should complete successfully within the timeout."""
        rc, output = asyncio.run(_run_cmd('echo fast', Path('/tmp'), timeout=5))
        assert rc == 0
        assert 'fast' in output


# ---------------------------------------------------------------------------
# run_verification: passes results through correctly
# ---------------------------------------------------------------------------


class TestRunVerification:
    """Verify run_verification aggregates _run_cmd results correctly."""

    def test_all_pass(self, tmp_path: Path):
        """When all commands succeed, result.passed is True."""
        config = MagicMock()
        config.test_command = 'echo tests-ok'
        config.lint_command = 'echo lint-ok'
        config.type_check_command = 'echo types-ok'

        result = asyncio.run(run_verification(tmp_path, config))
        assert result.passed is True
        assert result.summary == 'All checks passed'
        assert 'tests-ok' in result.test_output

    def test_test_failure(self, tmp_path: Path):
        """When tests fail, result.passed is False and summary mentions tests."""
        config = MagicMock()
        config.test_command = 'exit 1'
        config.lint_command = 'echo lint-ok'
        config.type_check_command = 'echo types-ok'

        result = asyncio.run(run_verification(tmp_path, config))
        assert result.passed is False
        assert 'tests failed' in result.summary

    def test_lint_failure(self, tmp_path: Path):
        """When lint fails, result.passed is False and summary mentions lint."""
        config = MagicMock()
        config.test_command = 'echo tests-ok'
        config.lint_command = 'exit 1'
        config.type_check_command = 'echo types-ok'

        result = asyncio.run(run_verification(tmp_path, config))
        assert result.passed is False
        assert 'lint issues' in result.summary

    def test_multiple_failures(self, tmp_path: Path):
        """When multiple checks fail, summary lists all failures."""
        config = MagicMock()
        config.test_command = 'exit 1'
        config.lint_command = 'exit 1'
        config.type_check_command = 'exit 1'

        result = asyncio.run(run_verification(tmp_path, config))
        assert result.passed is False
        assert 'tests failed' in result.summary
        assert 'lint issues' in result.summary
        assert 'type errors' in result.summary


# ---------------------------------------------------------------------------
# run_verification: ModuleConfig overrides
# ---------------------------------------------------------------------------


class TestRunVerificationModuleConfig:
    """Verify that ModuleConfig overrides the global config commands."""

    def test_module_config_overrides_commands(self, tmp_path: Path):
        """ModuleConfig commands take precedence over global config."""
        config = MagicMock()
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
        config = MagicMock()
        config.test_command = 'exit 1'
        config.lint_command = 'echo lint-global'
        config.type_check_command = 'echo type-global'

        mc = ModuleConfig(prefix='dashboard', test_command='echo test-override')
        result = asyncio.run(run_verification(tmp_path, config, module_config=mc))
        assert result.passed is True
        assert 'test-override' in result.test_output

    def test_none_module_config_unchanged(self, tmp_path: Path):
        """Passing None for module_config uses global config (backward compat)."""
        config = MagicMock()
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
        config = MagicMock()
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
        config = MagicMock()
        config.test_command = 'exit 1'
        config.lint_command = 'exit 1'
        config.type_check_command = 'exit 1'

        mc = ModuleConfig(prefix='empty')
        result = asyncio.run(run_verification(tmp_path, config, module_config=mc))
        assert result.passed is True
        assert result.summary == 'All checks passed'

    def test_lint_only_module(self, tmp_path: Path):
        """Simulates shared/ — only lint defined, test and type are None."""
        config = MagicMock()
        config.test_command = 'exit 1'
        config.lint_command = 'exit 1'
        config.type_check_command = 'exit 1'

        mc = ModuleConfig(prefix='shared', lint_command='echo lint-ok')
        result = asyncio.run(run_verification(tmp_path, config, module_config=mc))
        assert result.passed is True
        assert result.test_output == ''

    def test_lint_failure_still_fails(self, tmp_path: Path):
        """Even with tests skipped, a lint failure still fails the result."""
        config = MagicMock()
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
        config = MagicMock()
        config.test_command = 'echo global-test'
        config.lint_command = 'echo global-lint'
        config.type_check_command = 'echo global-type'

        result = asyncio.run(run_scoped_verification(tmp_path, config, []))
        assert result.passed is True
        assert 'global-test' in result.test_output

    def test_single_module_config(self, tmp_path: Path):
        """Single ModuleConfig delegates to run_verification with that config."""
        config = MagicMock()
        config.test_command = 'exit 1'

        mc = ModuleConfig(prefix='esc', test_command='echo esc-test', lint_command='echo esc-lint')
        result = asyncio.run(run_scoped_verification(tmp_path, config, [mc]))
        assert result.passed is True
        assert 'esc-test' in result.test_output

    def test_multiple_module_configs_all_pass(self, tmp_path: Path):
        """Multiple ModuleConfigs all pass → aggregated result passes."""
        config = MagicMock()

        mc1 = ModuleConfig(prefix='a', lint_command='echo a-lint')
        mc2 = ModuleConfig(prefix='b', lint_command='echo b-lint', test_command='echo b-test')
        result = asyncio.run(run_scoped_verification(tmp_path, config, [mc1, mc2]))
        assert result.passed is True

    def test_one_failure_fails_aggregate(self, tmp_path: Path):
        """One subproject fails → aggregated result fails."""
        config = MagicMock()

        mc_good = ModuleConfig(prefix='a', lint_command='echo ok')
        mc_bad = ModuleConfig(prefix='b', lint_command='exit 1')
        result = asyncio.run(run_scoped_verification(tmp_path, config, [mc_good, mc_bad]))
        assert result.passed is False
        assert 'lint issues' in result.summary

    def test_aggregate_combines_output(self, tmp_path: Path):
        """Aggregation merges test_output from multiple subprojects."""
        config = MagicMock()

        mc1 = ModuleConfig(prefix='a', test_command='echo test-a')
        mc2 = ModuleConfig(prefix='b', test_command='echo test-b')
        result = asyncio.run(run_scoped_verification(tmp_path, config, [mc1, mc2]))
        assert 'test-a' in result.test_output
        assert 'test-b' in result.test_output
