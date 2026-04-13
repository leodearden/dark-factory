"""Tests for verify.py - verification runner for test/lint/typecheck stages."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from orchestrator.verify import _run_cmd, run_verification, VerifyResult
from orchestrator.config import OrchestratorConfig


class TestRunCmd:
    """Tests for the _run_cmd function."""

    @pytest.mark.asyncio
    async def test_uses_bash_executable(self, tmp_path: Path):
        """Verify that bash is explicitly passed as the executable to subprocess."""
        # We need to mock the create_subprocess_shell to capture the executable parameter
        captured_executable = []

        async def mock_create(cmd, *args, **kwargs):
            captured_executable.append(kwargs.get('executable'))
            # Return a mock process
            mock_proc = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.communicate = AsyncMock(return_value=(b'output', b''))
            return mock_proc

        with patch('orchestrator.verify.asyncio.create_subprocess_shell', side_effect=mock_create):
            await _run_cmd('echo test', tmp_path, timeout=300)

        # Assert that executable was explicitly set to /bin/bash
        assert len(captured_executable) > 0, "Mock was not called!"
        assert captured_executable[0] == '/bin/bash', (
            f"Expected executable='/bin/bash', got {captured_executable[0]!r}"
        )

    @pytest.mark.asyncio
    async def test_bash_builtin_source_works(self, tmp_path: Path):
        """Verify that bash-specific builtins like 'source' work in commands."""
        # Create a test script that uses 'source' (bash-specific)
        script_path = tmp_path / 'test_env.sh'
        script_path.write_text('export TEST_VAR=hello\n')

        # This command uses 'source' which fails in dash but works in bash
        cmd = f'source {script_path} && echo $TEST_VAR'

        # Mock the subprocess to actually run with bash
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b'hello', b''))

        with patch('orchestrator.verify.asyncio.create_subprocess_shell', return_value=mock_proc):
            returncode, output = await _run_cmd(cmd, tmp_path)

        assert returncode == 0
        assert 'hello' in output


class TestRunVerification:
    """Tests for run_verification function."""

    @pytest.mark.asyncio
    async def test_verification_passes_when_all_commands_succeed(self, tmp_path: Path):
        """Verify that verification passes when all checks succeed."""
        config = OrchestratorConfig(
            test_command='echo "test ok"',
            lint_command='echo "lint ok"',
            type_check_command='echo "type ok"',
        )

        # Mock _run_cmd to return success
        async def mock_run_cmd(cmd, cwd, timeout=300):
            return 0, 'ok'

        with patch('orchestrator.verify._run_cmd', side_effect=mock_run_cmd):
            result = await run_verification(tmp_path, config)

        assert result.passed is True
        assert 'All checks passed' in result.summary

    @pytest.mark.asyncio
    async def test_verification_fails_when_tests_fail(self, tmp_path: Path):
        """Verify that verification fails when tests fail."""
        config = OrchestratorConfig(
            test_command='pytest',
            lint_command='echo "lint ok"',
            type_check_command='echo "type ok"',
        )

        async def mock_run_cmd(cmd, cwd, timeout=300):
            if cmd == 'pytest':
                return 1, 'FAILED: 5 tests failed'
            return 0, 'ok'

        with patch('orchestrator.verify._run_cmd', side_effect=mock_run_cmd):
            result = await run_verification(tmp_path, config)

        assert result.passed is False
        assert 'tests failed' in result.summary
        assert 'FAILED' in result.test_output


class TestVerifyResult:
    """Tests for VerifyResult dataclass."""

    def test_failure_report_includes_test_failures(self):
        """Verify that test failures are included in failure report."""
        result = VerifyResult(
            passed=False,
            test_output='FAILED: test_example.py::test_one FAILED',
            lint_output='',
            type_output='',
            summary='Failures: tests failed',
        )
        report = result.failure_report()
        assert 'Test Failures' in report
        assert 'FAILED' in report

    def test_failure_report_includes_lint_issues(self):
        """Verify that lint issues are included in failure report."""
        result = VerifyResult(
            passed=False,
            test_output='',
            lint_output='E501 line too long',
            type_output='',
            summary='Failures: lint issues',
        )
        report = result.failure_report()
        assert 'Lint Issues' in report

    def test_failure_report_includes_type_errors(self):
        """Verify that type errors are included in failure report."""
        result = VerifyResult(
            passed=False,
            test_output='',
            lint_output='',
            type_output='error: Cannot find module',
            summary='Failures: type errors',
        )
        report = result.failure_report()
        assert 'Type Errors' in report

    def test_failure_report_empty_when_all_pass(self):
        """Verify that failure report is empty when all checks pass."""
        result = VerifyResult(
            passed=True,
            test_output='',
            lint_output='',
            type_output='',
            summary='All checks passed',
        )
        report = result.failure_report()
        # When passed, returns summary
        assert report == 'All checks passed'