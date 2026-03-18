"""Unit tests for orchestrator.verify — _run_cmd and run_verification."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.verify import VerifyResult, _run_cmd, run_verification


# ---------------------------------------------------------------------------
# _run_cmd — executable parameter
# ---------------------------------------------------------------------------


class TestRunCmdBashExecutable:
    """Verify _run_cmd passes executable='/bin/bash' to create_subprocess_shell."""

    @pytest.mark.asyncio
    async def test_passes_bash_executable(self, tmp_path: Path):
        """Mock create_subprocess_shell and assert executable='/bin/bash' is passed."""
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b'ok\n', b'')
        mock_proc.returncode = 0

        with patch('orchestrator.verify.asyncio.create_subprocess_shell', return_value=mock_proc) as mock_shell:
            rc, out = await _run_cmd('echo hello', tmp_path)

            mock_shell.assert_called_once()
            call_kwargs = mock_shell.call_args
            assert call_kwargs.kwargs.get('executable') == '/bin/bash', (
                '_run_cmd must pass executable="/bin/bash" to create_subprocess_shell'
            )
            assert rc == 0
            assert out == 'ok\n'

    @pytest.mark.asyncio
    async def test_bash_builtins_work(self, tmp_path: Path):
        """Integration test: bash-specific 'source' builtin succeeds via _run_cmd.

        Under dash, 'source' is not available (only '.' is). This test confirms
        _run_cmd uses bash where 'source /dev/null' succeeds.
        """
        rc, out = await _run_cmd('source /dev/null && echo bash_ok', tmp_path)
        assert rc == 0, f'bash builtin "source" failed (rc={rc}): {out}'
        assert 'bash_ok' in out


# ---------------------------------------------------------------------------
# _run_cmd — timeout handling
# ---------------------------------------------------------------------------


class TestRunCmdTimeout:
    """Verify _run_cmd timeout behaviour with bash executable."""

    @pytest.mark.asyncio
    async def test_timeout_returns_error(self, tmp_path: Path):
        """A command exceeding timeout returns rc=1 and a timeout message."""
        rc, out = await _run_cmd('sleep 10', tmp_path, timeout=0.1)
        assert rc == 1
        assert 'timed out' in out.lower()

    @pytest.mark.asyncio
    async def test_timeout_message_includes_command(self, tmp_path: Path):
        """Timeout error message includes the original command string."""
        rc, out = await _run_cmd('sleep 60', tmp_path, timeout=0.1)
        assert 'sleep 60' in out

    @pytest.mark.asyncio
    async def test_fast_command_within_timeout(self, tmp_path: Path):
        """A fast command completes before timeout and succeeds."""
        rc, out = await _run_cmd('echo fast', tmp_path, timeout=10)
        assert rc == 0
        assert 'fast' in out


# ---------------------------------------------------------------------------
# _run_cmd — error handling
# ---------------------------------------------------------------------------


class TestRunCmdErrors:
    """Verify _run_cmd handles subprocess errors gracefully."""

    @pytest.mark.asyncio
    async def test_nonexistent_command_returns_error(self, tmp_path: Path):
        """Running a command that doesn't exist returns non-zero."""
        rc, out = await _run_cmd('nonexistent_command_xyz_12345', tmp_path)
        assert rc != 0

    @pytest.mark.asyncio
    async def test_cwd_passed_to_subprocess(self, tmp_path: Path):
        """The cwd argument is forwarded to the subprocess."""
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b'', b'')
        mock_proc.returncode = 0

        with patch('orchestrator.verify.asyncio.create_subprocess_shell', return_value=mock_proc) as mock_shell:
            await _run_cmd('echo test', tmp_path)

            call_kwargs = mock_shell.call_args
            assert call_kwargs.kwargs.get('cwd') == str(tmp_path)


# ---------------------------------------------------------------------------
# run_verification — end-to-end with mocked _run_cmd
# ---------------------------------------------------------------------------


class TestRunVerification:
    """Verify run_verification assembles results from all three commands."""

    @pytest.fixture
    def config(self, tmp_path: Path) -> OrchestratorConfig:
        return OrchestratorConfig(
            project_root=tmp_path,
            test_command='pytest',
            lint_command='ruff check',
            type_check_command='pyright',
            git=GitConfig(),
        )

    @pytest.mark.asyncio
    async def test_all_pass(self, tmp_path: Path, config: OrchestratorConfig):
        """When all commands succeed, result.passed is True."""
        async def mock_run_cmd(cmd, cwd, timeout=300):
            return 0, 'ok'

        with patch('orchestrator.verify._run_cmd', side_effect=mock_run_cmd):
            result = await run_verification(tmp_path, config)

        assert result.passed is True
        assert result.summary == 'All checks passed'
        assert result.lint_output == ''
        assert result.type_output == ''

    @pytest.mark.asyncio
    async def test_test_failure(self, tmp_path: Path, config: OrchestratorConfig):
        """When tests fail, result.passed is False and summary mentions tests."""
        async def mock_run_cmd(cmd, cwd, timeout=300):
            if 'pytest' in cmd:
                return 1, 'FAILED test_foo.py::test_bar'
            return 0, 'ok'

        with patch('orchestrator.verify._run_cmd', side_effect=mock_run_cmd):
            result = await run_verification(tmp_path, config)

        assert result.passed is False
        assert 'tests failed' in result.summary
        assert 'FAILED' in result.test_output

    @pytest.mark.asyncio
    async def test_lint_failure(self, tmp_path: Path, config: OrchestratorConfig):
        """When linter fails, result.passed is False and lint_output is populated."""
        async def mock_run_cmd(cmd, cwd, timeout=300):
            if 'ruff' in cmd:
                return 1, 'E501 line too long'
            return 0, 'ok'

        with patch('orchestrator.verify._run_cmd', side_effect=mock_run_cmd):
            result = await run_verification(tmp_path, config)

        assert result.passed is False
        assert 'lint issues' in result.summary
        assert 'E501' in result.lint_output

    @pytest.mark.asyncio
    async def test_type_check_failure(self, tmp_path: Path, config: OrchestratorConfig):
        """When type checker fails, result.passed is False and type_output is populated."""
        async def mock_run_cmd(cmd, cwd, timeout=300):
            if 'pyright' in cmd:
                return 1, 'error: type mismatch'
            return 0, 'ok'

        with patch('orchestrator.verify._run_cmd', side_effect=mock_run_cmd):
            result = await run_verification(tmp_path, config)

        assert result.passed is False
        assert 'type errors' in result.summary
        assert 'error' in result.type_output

    @pytest.mark.asyncio
    async def test_multiple_failures(self, tmp_path: Path, config: OrchestratorConfig):
        """When all commands fail, summary mentions all failure types."""
        async def mock_run_cmd(cmd, cwd, timeout=300):
            return 1, 'failed'

        with patch('orchestrator.verify._run_cmd', side_effect=mock_run_cmd):
            result = await run_verification(tmp_path, config)

        assert result.passed is False
        assert 'tests failed' in result.summary
        assert 'lint issues' in result.summary
        assert 'type errors' in result.summary
