"""Tests for verify.py — _run_cmd and run_verification."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from orchestrator.config import ModuleConfig
from orchestrator.verify import _run_cmd, run_verification


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

    def test_partial_override_falls_back(self, tmp_path: Path):
        """ModuleConfig with only test_command; lint/type fall back to global."""
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
