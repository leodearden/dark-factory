"""Tests for orchestrator/verify.py, specifically _run_cmd bash executable handling."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from orchestrator.verify import _run_cmd


class TestRunCmdBashExecutable:
    """Tests for _run_cmd ensuring /bin/bash is used as the shell executable."""

    @pytest.mark.asyncio
    async def test_bash_executable_is_passed_to_subprocess(self, tmp_path: Path):
        """_run_cmd should pass executable='/bin/bash' to create_subprocess_shell.

        This test runs a command that only works in bash (not dash), verifying
        that bash is actually being used. The `source` builtin is bash-specific
        and fails with 'source: not found' in dash.
        """
        # Create a temporary file to source
        env_file = tmp_path / "test_env.sh"
        env_file.write_text('export TEST_VAR="success"\n')

        # Test command that uses bash-specific `source` builtin
        cmd = f'source {env_file} && echo "$TEST_VAR"'
        rc, stdout, timed_out = await _run_cmd(cmd, tmp_path, timeout=5.0)

        # With bash, source works and TEST_VAR should be "success"
        # With dash, we'd get 'source: not found' and TEST_VAR would be unset
        assert rc == 0, f"Command should succeed with bash; got: {stdout}"
        assert "success" in stdout, (
            f"Expected 'success' from sourced env var; got: {stdout}"
        )

    @pytest.mark.asyncio
    async def test_source_builtin_works_in_run_cmd(self, tmp_path: Path):
        """The `source` bash builtin should work in _run_cmd commands.

        This is the core bug that was fixed: on dash systems, `source` fails
        because dash only supports the POSIX `.` command.
        """
        # Create a script to source
        script = tmp_path / "helper.sh"
        script.write_text('my_func() { echo "Hello from sourced function"; }\n')

        # Use source (bash builtin) - this should NOT fail
        cmd = f'source {script} && my_func'
        rc, stdout, timed_out = await _run_cmd(cmd, tmp_path, timeout=5.0)

        assert rc == 0, f"Source builtin failed; got: {stdout}"
        assert "Hello from sourced function" in stdout

    @pytest.mark.asyncio
    async def test_bash_arrays_work_in_run_cmd(self, tmp_path: Path):
        """Bash-specific features like arrays should work in _run_cmd.

        Arrays are a bash-specific feature not available in dash.
        """
        # Test bash array syntax
        cmd = 'arr=(a b c) && echo "${arr[1]}"'
        rc, stdout, timed_out = await _run_cmd(cmd, tmp_path, timeout=5.0)

        assert rc == 0, f"Bash array syntax failed; got: {stdout}"
        assert "b" in stdout, f"Expected 'b' from array index 1; got: {stdout}"

    @pytest.mark.asyncio
    async def test_bash_parameter_expansion_works(self, tmp_path: Path):
        """Bash-specific parameter expansion ${var:offset:length} should work."""
        cmd = 'str="hello world" && echo "${str:0:5}"'
        rc, stdout, timed_out = await _run_cmd(cmd, tmp_path, timeout=5.0)

        assert rc == 0, f"Parameter expansion failed; got: {stdout}"
        assert "hello" in stdout

    @pytest.mark.asyncio
    async def test_simple_command_still_works(self, tmp_path: Path):
        """Basic commands without bash-specific features should still work."""
        cmd = 'echo "hello world"'
        rc, stdout, timed_out = await _run_cmd(cmd, tmp_path, timeout=5.0)

        assert rc == 0
        assert "hello world" in stdout

    @pytest.mark.asyncio
    async def test_command_timeout_is_enforced(self, tmp_path: Path):
        """Long-running commands should be terminated after timeout."""
        cmd = 'sleep 10'
        rc, stdout, timed_out = await _run_cmd(cmd, tmp_path, timeout=0.5)

        assert timed_out is True
        assert "Command timed out" in stdout

    @pytest.mark.asyncio
    async def test_failed_command_returns_nonzero_rc(self, tmp_path: Path):
        """Commands that fail should return non-zero return code."""
        cmd = 'exit 42'
        rc, stdout, timed_out = await _run_cmd(cmd, tmp_path, timeout=5.0)

        assert rc == 42
        assert not timed_out

    @pytest.mark.asyncio
    async def test_env_variables_can_be_injected(self, tmp_path: Path):
        """Environment variables should be merged with subprocess env."""
        env = {"CUSTOM_VAR": "custom_value"}
        cmd = 'echo "$CUSTOM_VAR"'
        rc, stdout, timed_out = await _run_cmd(cmd, tmp_path, timeout=5.0, env=env)

        assert rc == 0
        assert "custom_value" in stdout


class TestRunCmdProcessGroup:
    """Tests for _run_cmd process-group safety (SIGTERM propagation fix)."""

    @pytest.mark.asyncio
    async def test_run_cmd_passes_start_new_session_true(self, tmp_path: Path):
        """_run_cmd must spawn subprocess with start_new_session=True.

        Failing test — production _run_cmd doesn't pass that kwarg yet.
        """
        captured_kwargs: dict = {}

        async def fake_shell(cmd, **kwargs):
            captured_kwargs.update(kwargs)
            proc = MagicMock()
            proc.communicate = AsyncMock(return_value=(b'hi\n', None))
            proc.returncode = 0
            return proc

        with patch('orchestrator.verify.asyncio.create_subprocess_shell', side_effect=fake_shell):
            rc, stdout, timed_out = await _run_cmd('echo hi', tmp_path, timeout=5.0)

        assert captured_kwargs.get('start_new_session') is True, (
            f'start_new_session not in subprocess kwargs: {captured_kwargs}'
        )
        assert rc == 0
        assert not timed_out

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_run_cmd_killpg_sequence_on_timeout(self, tmp_path: Path):
        """On timeout, _run_cmd must call os.killpg(pgid, SIGTERM) then SIGKILL.

        We mock asyncio.wait_for so that:
          - call 1 (communicate timeout in _run_cmd) raises TimeoutError
          - call 2 (SIGTERM grace in terminate_process_group) raises TimeoutError
          - call 3 (SIGKILL grace) returns normally
        This avoids real 5-second grace waits in the unit test.
        """
        import signal as signal_module

        wait_for_call_n = 0

        async def counting_wait_for(coro, timeout, **_kwargs):
            nonlocal wait_for_call_n
            wait_for_call_n += 1
            n = wait_for_call_n
            # Eagerly consume / close the coroutine to avoid ResourceWarnings.
            try:
                if n <= 2:
                    coro.close()
                    raise TimeoutError()
                else:
                    # Let it run (AsyncMock returns immediately).
                    result = await coro
                    return result
            except (AttributeError, RuntimeError):
                raise TimeoutError() from None

        async def fake_shell(cmd, **kwargs):
            proc = MagicMock()
            proc.pid = 12345
            proc.returncode = None
            async def _hang():
                await asyncio.sleep(3600)
            proc.communicate = _hang
            proc.wait = AsyncMock(return_value=None)
            return proc

        mock_getpgid = MagicMock(return_value=12345)
        mock_killpg = MagicMock()

        with (
            patch('orchestrator.verify.asyncio.create_subprocess_shell', side_effect=fake_shell),
            patch('shared.proc_group.os.getpgid', mock_getpgid),
            patch('shared.proc_group.os.killpg', mock_killpg),
            patch('asyncio.wait_for', side_effect=counting_wait_for),
        ):
            rc, stdout, timed_out = await _run_cmd('sleep 100', tmp_path, timeout=0.1)

        assert timed_out is True, 'Expected timed_out=True'
        assert mock_killpg.call_count == 2, (
            f'Expected 2 killpg calls, got {mock_killpg.call_count}: {mock_killpg.call_args_list}'
        )
        assert mock_killpg.call_args_list[0] == call(12345, signal_module.SIGTERM)
        assert mock_killpg.call_args_list[1] == call(12345, signal_module.SIGKILL)

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_run_cmd_killpg_on_cancellation(self, tmp_path: Path):
        """When the calling task is cancelled, _run_cmd must killpg the subprocess.

        Creates a task running _run_cmd, cancels it after one event-loop tick,
        then asserts that os.killpg(pgid, SIGTERM) was called and that
        CancelledError is re-raised.
        """
        import signal as signal_module

        async def fake_shell(cmd, **kwargs):
            proc = MagicMock()
            proc.pid = 12345
            proc.returncode = None
            async def _hang():
                await asyncio.sleep(3600)
            proc.communicate = _hang
            proc.wait = AsyncMock(return_value=None)
            return proc

        mock_getpgid = MagicMock(return_value=12345)
        mock_killpg = MagicMock()

        async def run_it():
            with (
                patch('orchestrator.verify.asyncio.create_subprocess_shell', side_effect=fake_shell),
                patch('shared.proc_group.os.getpgid', mock_getpgid),
                patch('shared.proc_group.os.killpg', mock_killpg),
            ):
                await _run_cmd('sleep 100', tmp_path, timeout=60.0)

        task = asyncio.create_task(run_it())
        # Yield one tick to let the subprocess be spawned and communicate() be awaited.
        await asyncio.sleep(0)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert any(
            c == call(12345, signal_module.SIGTERM)
            for c in mock_killpg.call_args_list
        ), f'SIGTERM killpg not called; calls: {mock_killpg.call_args_list}'
