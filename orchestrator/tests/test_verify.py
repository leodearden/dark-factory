"""Tests for orchestrator/verify.py, specifically _run_cmd bash executable handling."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from orchestrator.config import ModuleConfig, OrchestratorConfig
from orchestrator.verify import (
    _apply_cargo_scope,
    _run_cmd,
    run_scoped_verification,
    run_verification,
    scope_module_config,
)


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


class TestVerifyWarmMarker:
    """Tests for the _is_verify_cold / _mark_verify_warm filesystem helpers."""

    def test_is_cold_when_task_dir_exists_without_marker(self, tmp_path: Path):
        """`_is_verify_cold` returns True when .task/ exists but verify_warmed is absent."""
        from orchestrator.verify import _is_verify_cold
        (tmp_path / '.task').mkdir()
        assert _is_verify_cold(tmp_path) is True

    def test_is_warm_when_marker_present(self, tmp_path: Path):
        """`_is_verify_cold` returns False when .task/verify_warmed exists."""
        from orchestrator.verify import _is_verify_cold
        task_dir = tmp_path / '.task'
        task_dir.mkdir()
        (task_dir / 'verify_warmed').touch()
        assert _is_verify_cold(tmp_path) is False

    def test_is_warm_when_task_dir_absent(self, tmp_path: Path):
        """`_is_verify_cold` returns False when .task/ itself is missing (e.g. project_root)."""
        from orchestrator.verify import _is_verify_cold
        # No .task/ directory created — project_root / review-checkpoint case.
        assert _is_verify_cold(tmp_path) is False

    def test_mark_verify_warm_creates_marker(self, tmp_path: Path):
        """`_mark_verify_warm` creates .task/verify_warmed when .task/ already exists."""
        from orchestrator.verify import _mark_verify_warm
        task_dir = tmp_path / '.task'
        task_dir.mkdir()
        _mark_verify_warm(tmp_path)
        assert (task_dir / 'verify_warmed').exists()

    def test_mark_verify_warm_is_idempotent(self, tmp_path: Path):
        """`_mark_verify_warm` called twice does not raise."""
        from orchestrator.verify import _mark_verify_warm
        task_dir = tmp_path / '.task'
        task_dir.mkdir()
        _mark_verify_warm(tmp_path)
        _mark_verify_warm(tmp_path)  # must not raise
        assert (task_dir / 'verify_warmed').exists()

    def test_mark_verify_warm_noop_without_task_dir(self, tmp_path: Path):
        """`_mark_verify_warm` is a no-op and does not create .task/ when it's absent."""
        from orchestrator.verify import _mark_verify_warm
        _mark_verify_warm(tmp_path)
        assert not (tmp_path / '.task').exists()

    def test_is_cold_per_prefix_independent_of_shared_marker(self, tmp_path: Path):
        """Per-prefix marker is independent: touching verify_warmed doesn't warm a prefixed check."""
        from orchestrator.verify import _is_verify_cold, _mark_verify_warm
        task_dir = tmp_path / '.task'
        task_dir.mkdir()
        # Mark the shared (no-prefix) marker
        _mark_verify_warm(tmp_path, None)
        # The 'orchestrator' subproject should still be cold
        assert _is_verify_cold(tmp_path, 'orchestrator') is True

    def test_is_cold_per_prefix_becomes_warm_after_prefix_marker(self, tmp_path: Path):
        """Touching the per-prefix marker makes that prefix warm but leaves others cold."""
        from orchestrator.verify import _is_verify_cold, _mark_verify_warm
        task_dir = tmp_path / '.task'
        task_dir.mkdir()
        _mark_verify_warm(tmp_path, 'orchestrator')
        assert _is_verify_cold(tmp_path, 'orchestrator') is False
        # 'fused-memory' subproject should still be cold
        assert _is_verify_cold(tmp_path, 'fused-memory') is True

    def test_warm_marker_name_sanitizes_path_separators(self, tmp_path: Path):
        """Path separators in prefix are replaced with underscores in marker filename."""
        from orchestrator.verify import _mark_verify_warm, _warm_marker_name
        task_dir = tmp_path / '.task'
        task_dir.mkdir()
        _mark_verify_warm(tmp_path, 'a/b/c')
        expected_name = _warm_marker_name('a/b/c')
        assert '/' not in expected_name
        assert (task_dir / expected_name).exists()


class TestResolveVerifyTimeout:
    """Tests for the updated _resolve_verify_timeout(config, module_config, *, is_cold) helper."""

    def _make_config(self, warm=1800.0, cold=None):
        """Build a minimal OrchestratorConfig with the given timeout values."""
        from orchestrator.config import OrchestratorConfig
        return OrchestratorConfig(
            verify_command_timeout_secs=warm,
            verify_cold_command_timeout_secs=cold,
        )

    def _make_mc(self, warm=None, cold=None):
        """Build a minimal ModuleConfig with the given timeout values."""
        from orchestrator.config import ModuleConfig
        return ModuleConfig(
            prefix='test',
            verify_command_timeout_secs=warm,
            verify_cold_command_timeout_secs=cold,
        )

    def test_warm_no_module_override_returns_config_warm(self):
        """is_cold=False + no module override → config.verify_command_timeout_secs."""
        from orchestrator.verify import _resolve_verify_timeout
        config = self._make_config(warm=1800.0, cold=5400.0)
        assert _resolve_verify_timeout(config, None, is_cold=False) == 1800.0

    def test_cold_config_cold_set_returns_cold(self):
        """is_cold=True + config.verify_cold_command_timeout_secs=5400 → 5400."""
        from orchestrator.verify import _resolve_verify_timeout
        config = self._make_config(warm=1800.0, cold=5400.0)
        assert _resolve_verify_timeout(config, None, is_cold=True) == 5400.0

    def test_cold_all_none_falls_back_to_warm(self):
        """is_cold=True + cold values all None at every level → warm timeout."""
        from orchestrator.verify import _resolve_verify_timeout
        config = self._make_config(warm=1800.0, cold=None)
        mc = self._make_mc(warm=None, cold=None)
        assert _resolve_verify_timeout(config, mc, is_cold=True) == 1800.0

    def test_module_cold_wins_over_config_cold(self):
        """module_config.verify_cold_command_timeout_secs wins over config-level when is_cold=True."""
        from orchestrator.verify import _resolve_verify_timeout
        config = self._make_config(warm=1800.0, cold=5400.0)
        mc = self._make_mc(cold=7200.0)
        assert _resolve_verify_timeout(config, mc, is_cold=True) == 7200.0

    def test_module_warm_wins_when_cold_false(self):
        """module_config.verify_command_timeout_secs wins for the warm track when is_cold=False."""
        from orchestrator.verify import _resolve_verify_timeout
        config = self._make_config(warm=1800.0, cold=5400.0)
        mc = self._make_mc(warm=2000.0)
        assert _resolve_verify_timeout(config, mc, is_cold=False) == 2000.0

    def test_module_warm_only_cold_fallthrough(self):
        """module has warm override but no cold; is_cold=True falls through to config.cold, then module.warm.

        Fallthrough order: module.cold (None) → top.cold (None) → module.warm.
        """
        from orchestrator.verify import _resolve_verify_timeout
        config = self._make_config(warm=1800.0, cold=None)
        mc = self._make_mc(warm=2000.0, cold=None)
        # cold cascade: mc.cold=None → config.cold=None → mc.warm=2000
        assert _resolve_verify_timeout(config, mc, is_cold=True) == 2000.0


class TestRunVerificationColdFirstUse:
    """Integration tests for cold-first-use timeout selection in run_verification.

    Patches orchestrator.verify._run_cmd to capture the timeout argument
    without spawning real subprocesses.
    """

    def _make_config(self, warm=1800.0, cold: float | None = 5400.0, retries=0):
        from orchestrator.config import OrchestratorConfig
        return OrchestratorConfig(
            verify_command_timeout_secs=warm,
            verify_cold_command_timeout_secs=cold,
            verify_timeout_retries=retries,
            test_command='echo test',
            lint_command='echo lint',
            type_check_command='echo type',
        )

    def _make_success_mock(self):
        """AsyncMock that records timeout kwargs and returns (0, '', False) — success."""
        captured_timeouts: list[float] = []

        async def fake_run_cmd(cmd, cwd, timeout, env=None):
            captured_timeouts.append(timeout)
            return 0, '', False

        return fake_run_cmd, captured_timeouts

    def _make_timeout_mock(self):
        """AsyncMock that always returns a pure-timeout result (1, 'Command timed out…', True)."""
        async def fake_run_cmd(cmd, cwd, timeout, env=None):
            return 1, f'Command timed out after {timeout}s: {cmd}', True

        return fake_run_cmd

    @pytest.mark.asyncio
    async def test_first_run_uses_cold_timeout(self, tmp_path: Path):
        """First run with .task/ present (no marker) uses cold timeout (5400)."""
        from orchestrator.verify import run_verification
        (tmp_path / '.task').mkdir()

        config = self._make_config(warm=1800.0, cold=5400.0)
        fake_cmd, captured = self._make_success_mock()

        with patch('orchestrator.verify._run_cmd', side_effect=fake_cmd):
            await run_verification(tmp_path, config)

        assert 5400.0 in captured, f'Expected cold timeout 5400 in {captured}'

    @pytest.mark.asyncio
    async def test_first_run_creates_warm_marker(self, tmp_path: Path):
        """After a successful first run, .task/verify_warmed marker must exist."""
        from orchestrator.verify import run_verification
        (tmp_path / '.task').mkdir()

        config = self._make_config(warm=1800.0, cold=5400.0)
        fake_cmd, _ = self._make_success_mock()

        with patch('orchestrator.verify._run_cmd', side_effect=fake_cmd):
            await run_verification(tmp_path, config)

        assert (tmp_path / '.task' / 'verify_warmed').exists()

    @pytest.mark.asyncio
    async def test_second_run_uses_warm_timeout(self, tmp_path: Path):
        """Second run (marker already present) uses warm timeout (1800)."""
        from orchestrator.verify import run_verification
        task_dir = tmp_path / '.task'
        task_dir.mkdir()
        (task_dir / 'verify_warmed').touch()

        config = self._make_config(warm=1800.0, cold=5400.0)
        fake_cmd, captured = self._make_success_mock()

        with patch('orchestrator.verify._run_cmd', side_effect=fake_cmd):
            await run_verification(tmp_path, config)

        assert 1800.0 in captured, f'Expected warm timeout 1800 in {captured}'
        assert 5400.0 not in captured, f'Unexpected cold timeout in second run: {captured}'

    @pytest.mark.asyncio
    async def test_pure_timeout_does_not_create_marker(self, tmp_path: Path):
        """When all retries time out, the warm marker must NOT be created."""
        from orchestrator.verify import run_verification
        (tmp_path / '.task').mkdir()

        config = self._make_config(warm=1800.0, cold=5400.0, retries=0)
        fake_cmd = self._make_timeout_mock()

        with patch('orchestrator.verify._run_cmd', side_effect=fake_cmd):
            result = await run_verification(tmp_path, config)

        assert result.timed_out is True
        assert not (tmp_path / '.task' / 'verify_warmed').exists()

    @pytest.mark.asyncio
    async def test_non_timeout_failure_creates_marker(self, tmp_path: Path):
        """A real test failure (non-timeout) marks the worktree warm (build completed)."""
        from orchestrator.verify import run_verification
        (tmp_path / '.task').mkdir()

        config = self._make_config(warm=1800.0, cold=5400.0, retries=0)

        async def fake_real_failure(cmd, cwd, timeout, env=None):
            return 1, 'FAILED some_test', False

        with patch('orchestrator.verify._run_cmd', side_effect=fake_real_failure):
            result = await run_verification(tmp_path, config)

        assert result.timed_out is False
        assert (tmp_path / '.task' / 'verify_warmed').exists()

    @pytest.mark.asyncio
    async def test_cold_log_message_emitted_when_timeout_differs(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        """'Cold-cache verify' log is emitted when cold timeout differs from warm."""
        import logging

        from orchestrator.verify import run_verification
        (tmp_path / '.task').mkdir()

        config = self._make_config(warm=1800.0, cold=5400.0)
        fake_cmd, _ = self._make_success_mock()

        with caplog.at_level(logging.INFO, logger='orchestrator.verify'), patch('orchestrator.verify._run_cmd', side_effect=fake_cmd):
            await run_verification(tmp_path, config)

        assert any('Cold-cache verify' in r.message for r in caplog.records), (
            f'Expected "Cold-cache verify" log; got: {[r.message for r in caplog.records]}'
        )

    @pytest.mark.asyncio
    async def test_cold_log_message_suppressed_when_cold_same_as_warm(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        """'Cold-cache verify' log is suppressed when cold timeout is not set (fallback to warm)."""
        import logging

        from orchestrator.verify import run_verification
        (tmp_path / '.task').mkdir()

        # cold=None means cold falls back to warm — no log should fire
        config = self._make_config(warm=1800.0, cold=None)
        fake_cmd, _ = self._make_success_mock()

        with caplog.at_level(logging.INFO, logger='orchestrator.verify'), patch('orchestrator.verify._run_cmd', side_effect=fake_cmd):
            await run_verification(tmp_path, config)

        assert not any('Cold-cache verify' in r.message for r in caplog.records), (
            f'Unexpected "Cold-cache verify" log when cold==warm: {[r.message for r in caplog.records]}'
        )

    @pytest.mark.asyncio
    async def test_allow_cold_cache_false_uses_warm_timeout(self, tmp_path: Path):
        """allow_cold_cache=False forces warm timeout even when .task/ is present."""
        from orchestrator.verify import run_verification
        (tmp_path / '.task').mkdir()  # cold filesystem state, but caller opts out

        config = self._make_config(warm=1800.0, cold=5400.0)
        fake_cmd, captured = self._make_success_mock()

        with patch('orchestrator.verify._run_cmd', side_effect=fake_cmd):
            await run_verification(tmp_path, config, allow_cold_cache=False)

        assert 1800.0 in captured, f'Expected warm timeout 1800; got: {captured}'
        assert 5400.0 not in captured, f'Unexpected cold timeout when allow_cold_cache=False: {captured}'

    @pytest.mark.asyncio
    async def test_is_merge_verify_forces_cold_timeout(self, tmp_path: Path):
        """is_merge_verify=True forces cold timeout even without .task/ present.

        Merge worktrees have .taskmaster/ but not .task/, so
        _is_verify_cold's filesystem heuristic returns False (treats them
        as warm).  But merge worktrees are always freshly created = cold
        cargo build.  Asserting the flag bypasses the heuristic is Fix #1.
        """
        from orchestrator.verify import run_verification
        # No .task/ dir — filesystem heuristic would say "warm".
        assert not (tmp_path / '.task').exists()

        config = self._make_config(warm=1800.0, cold=5400.0)
        fake_cmd, captured = self._make_success_mock()

        with patch('orchestrator.verify._run_cmd', side_effect=fake_cmd):
            await run_verification(tmp_path, config, is_merge_verify=True)

        assert 5400.0 in captured, (
            f'is_merge_verify=True must force cold timeout; got: {captured}'
        )
        assert 1800.0 not in captured, (
            f'Unexpected warm timeout when is_merge_verify=True: {captured}'
        )

    @pytest.mark.asyncio
    async def test_is_merge_verify_does_not_write_warm_marker(
        self, tmp_path: Path,
    ):
        """is_merge_verify success must NOT write the warm marker.

        Merge worktrees are ephemeral (cleaned up right after), so writing
        a marker is at best a no-op and at worst leaks into a path that
        happens to have .task/ (e.g. when a test fixture shares a dir).
        Skipping the marker keeps the intent explicit.
        """
        from orchestrator.verify import run_verification
        # Create .task/ so _mark_verify_warm would otherwise succeed.
        (tmp_path / '.task').mkdir()

        config = self._make_config(warm=1800.0, cold=5400.0)
        fake_cmd, _ = self._make_success_mock()

        with patch('orchestrator.verify._run_cmd', side_effect=fake_cmd):
            result = await run_verification(
                tmp_path, config, is_merge_verify=True,
            )

        assert result.passed
        assert not (tmp_path / '.task' / 'verify_warmed').exists(), (
            'is_merge_verify=True must not write the warm marker'
        )

    @pytest.mark.asyncio
    async def test_is_merge_verify_default_preserves_existing_behaviour(
        self, tmp_path: Path,
    ):
        """Default is_merge_verify=False keeps the existing cold-detection path.

        Ensures non-merge callers (workflow-level task verify, review
        checkpoints) continue to use _is_verify_cold as before.
        """
        from orchestrator.verify import run_verification
        # No .task/ dir → heuristic says "warm" (project_root case).
        assert not (tmp_path / '.task').exists()

        config = self._make_config(warm=1800.0, cold=5400.0)
        fake_cmd, captured = self._make_success_mock()

        with patch('orchestrator.verify._run_cmd', side_effect=fake_cmd):
            await run_verification(tmp_path, config)  # default flags

        # With no .task/ and default flags, _is_verify_cold returns False →
        # warm timeout applies.
        assert 1800.0 in captured, (
            f'Default call without .task/ must use warm timeout; got: {captured}'
        )
        assert 5400.0 not in captured


def test_apply_cargo_scope_preserves_verify_cold_command_timeout_secs(tmp_path: Path):
    """_apply_cargo_scope propagates verify_cold_command_timeout_secs to the rebuilt ModuleConfig.

    Constructs a ModuleConfig with both warm and cold timeout values, forces the
    cargo-scope rewrite path via mocked workspace discovery, and asserts the
    returned ModuleConfig carries both timeout fields unchanged.
    """
    from unittest.mock import patch

    from orchestrator.config import ModuleConfig
    from orchestrator.verify import _apply_cargo_scope

    mc = ModuleConfig(
        prefix='crates',
        test_command='cargo test --workspace',
        lint_command='cargo clippy --workspace',
        type_check_command=None,
        verify_command_timeout_secs=2000.0,
        verify_cold_command_timeout_secs=6000.0,
    )

    # Force the rewrite path: workspace has one crate, and the .rs file maps to it.
    with (
        patch('orchestrator.verify.discover_workspace_crates', return_value={'crates/foo': 'foo'}),
        patch('orchestrator.verify.files_to_crates', return_value=['foo']),
    ):
        result = _apply_cargo_scope(
            mc,
            task_files=['crates/foo/src/lib.rs'],
            project_root=tmp_path,
            scope_cargo_enabled=True,
        )

    assert result.verify_command_timeout_secs == 2000.0
    assert result.verify_cold_command_timeout_secs == 6000.0


class TestScopeModuleConfigReturnsNone:
    """`scope_module_config` returns None when no task_files match the prefix.

    Callers must treat None as "skip this subproject" rather than falling
    back to the full unscoped suite — running a subproject with zero
    matching files was the root cause of the 2026-04-20 merge-queue stall
    (a fused-memory/** merge ran the whole shared/ suite and hung).
    """

    def test_returns_none_when_no_files_under_prefix(self):
        mc = ModuleConfig(
            prefix='shared',
            test_command='uv run pytest tests/',
            lint_command='ruff check src/',
            type_check_command='pyright',
        )
        result = scope_module_config(mc, ['fused-memory/src/foo.py'])
        assert result is None

    def test_returns_none_when_task_files_empty(self):
        mc = ModuleConfig(prefix='shared', test_command='pytest')
        result = scope_module_config(mc, [])
        assert result is None

    def test_returns_scoped_config_when_files_match(self):
        mc = ModuleConfig(
            prefix='shared',
            test_command='uv run --directory shared pytest tests/',
            lint_command='uv run --directory shared ruff check src/',
        )
        result = scope_module_config(mc, ['shared/tests/test_x.py', 'shared/src/y.py'])
        assert result is not None
        # The specific files should appear in the scoped commands.
        assert result.test_command is not None
        assert 'shared/tests/test_x.py' in result.test_command
        assert result.lint_command is not None
        assert 'shared/src/y.py' in result.lint_command


class TestRunScopedVerificationSkipsUntouched:
    """End-to-end: run_scoped_verification skips subprojects with no matching files."""

    @pytest.mark.asyncio
    async def test_skips_subprojects_with_no_matching_files(
        self, tmp_path: Path,
    ):
        """A task that only touches module A should not run module B's commands."""
        # Pretend the task touched a fused-memory test file.
        (tmp_path / 'fused-memory' / 'tests').mkdir(parents=True)
        touched = tmp_path / 'fused-memory' / 'tests' / 'test_changed.py'
        touched.write_text('def test_x(): pass\n')

        config = OrchestratorConfig(project_root=tmp_path)
        module_configs = [
            ModuleConfig(
                prefix='shared',
                test_command='uv run --directory shared pytest tests/',
            ),
            ModuleConfig(
                prefix='fused-memory',
                test_command='uv run --directory fused-memory pytest tests/',
            ),
        ]

        # Spy on _run_cmd; pretend every command passes instantly.
        calls: list[str] = []

        async def fake_run_cmd(cmd, cwd, timeout, env=None):
            calls.append(cmd)
            return 0, '', False

        with patch('orchestrator.verify._run_cmd', side_effect=fake_run_cmd):
            result = await run_scoped_verification(
                tmp_path, config, module_configs,
                task_files=['fused-memory/tests/test_changed.py'],
            )

        assert result.passed
        joined = ' | '.join(calls)
        assert calls, 'expected at least one command to run'
        # shared's command (identified by its --directory shared flag) must
        # NOT appear — that's the whole point of the fix.
        assert '--directory shared' not in joined, (
            f'shared subproject was run despite no matching files: {calls}'
        )
        # And something fused-memory-related did run.
        assert 'fused-memory' in joined or 'test_changed.py' in joined, (
            f'fused-memory command should have run; calls: {calls}'
        )

    @pytest.mark.asyncio
    async def test_max_retries_zero_disables_timeout_retry(self, tmp_path: Path):
        """max_retries=0 (merge-queue path) short-circuits the retry loop.

        Without the plumbing, a deterministic hang triples the stall.
        """
        config = OrchestratorConfig(
            project_root=tmp_path,
            test_command='__test_cmd__',
            lint_command='__lint_cmd__',
            type_check_command='__type_cmd__',
            verify_command_timeout_secs=0.2,
            verify_timeout_retries=5,
        )

        test_calls = 0

        async def fake_run_cmd(cmd, cwd, timeout, env=None):
            nonlocal test_calls
            if cmd == '__test_cmd__':
                test_calls += 1
                return 1, f'Command timed out after {timeout}s: {cmd}', True
            return 0, '', False

        with patch('orchestrator.verify._run_cmd', side_effect=fake_run_cmd):
            result = await run_verification(tmp_path, config, max_retries=0)

        assert not result.passed
        assert result.timed_out
        assert test_calls == 1, (
            f'max_retries=0 should run test cmd once; ran {test_calls} times'
        )

    @pytest.mark.asyncio
    async def test_max_retries_default_uses_config(self, tmp_path: Path):
        """When max_retries is None, the config value is used (regression guard)."""
        config = OrchestratorConfig(
            project_root=tmp_path,
            test_command='__test_cmd__',
            lint_command='__lint_cmd__',
            type_check_command='__type_cmd__',
            verify_command_timeout_secs=0.1,
            verify_timeout_retries=2,
        )

        test_calls = 0

        async def fake_run_cmd(cmd, cwd, timeout, env=None):
            nonlocal test_calls
            if cmd == '__test_cmd__':
                test_calls += 1
                return 1, f'Command timed out after {timeout}s: {cmd}', True
            return 0, '', False

        with patch('orchestrator.verify._run_cmd', side_effect=fake_run_cmd):
            result = await run_verification(tmp_path, config)

        assert not result.passed
        # 1 initial + 2 retries = 3 invocations.
        assert test_calls == 3, (
            f'default should retry per config; got {test_calls} invocations'
        )


class TestApplyCargoScopePolyglotGuard:
    """_apply_cargo_scope polyglot-diff guard: whitelist safe non-.rs extensions.

    A diff mixing .rs files with executable source (.py, .ts, .js, .go) or
    unknown extensions must bail to --workspace so chained non-Rust commands
    (e.g. ``cargo test --workspace && uv run pytest``) are not under-protected.
    A diff mixing .rs with safe config/data files (.toml, .yaml, .yml, .json,
    .md) should still scope to the affected crates.
    """

    # -----------------------------------------------------------------------
    # Shared helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _make_mc() -> ModuleConfig:
        return ModuleConfig(
            prefix='crates',
            test_command='cargo test --workspace',
            lint_command='cargo clippy --workspace',
        )

    @staticmethod
    def _call_scoped(task_files: list[str]) -> ModuleConfig:
        """Call _apply_cargo_scope with mocked workspace, primed to scope if guard passes."""
        mc = TestApplyCargoScopePolyglotGuard._make_mc()
        with (
            patch(
                'orchestrator.verify.discover_workspace_crates',
                return_value={'crates/foo': 'foo'},
            ),
            patch('orchestrator.verify.files_to_crates', return_value=['foo']),
        ):
            return _apply_cargo_scope(
                mc,
                task_files=task_files,
                project_root=Path('/fake/root'),
                scope_cargo_enabled=True,
            )

    # -----------------------------------------------------------------------
    # TDD driver — must FAIL on pre-guard code, PASS after step-2 impl
    # -----------------------------------------------------------------------

    def test_rs_plus_py_bails_to_workspace(self):
        """.rs + .py diff must return mc unchanged (polyglot executable guard)."""
        result = self._call_scoped(
            ['crates/foo/src/lib.rs', 'orchestrator/src/foo.py']
        )
        assert result.test_command == 'cargo test --workspace', (
            f'expected unchanged --workspace, got {result.test_command!r}'
        )
        assert result.lint_command == 'cargo clippy --workspace', (
            f'expected unchanged --workspace, got {result.lint_command!r}'
        )

    # -----------------------------------------------------------------------
    # Task-mandated regression net — .rs + .toml must still scope to crate
    # -----------------------------------------------------------------------

    def test_rs_plus_toml_scopes_to_crate(self):
        """.rs + .toml diff must scope to the matched crate (safe whitelist ext)."""
        result = self._call_scoped(['crates/foo/src/lib.rs', 'Cargo.toml'])
        assert '-p foo' in (result.test_command or ''), (
            f'expected -p foo in test_command, got {result.test_command!r}'
        )
        assert '--workspace' not in (result.test_command or ''), (
            f'expected --workspace absent in test_command, got {result.test_command!r}'
        )
        assert '-p foo' in (result.lint_command or ''), (
            f'expected -p foo in lint_command, got {result.lint_command!r}'
        )
        assert '--workspace' not in (result.lint_command or ''), (
            f'expected --workspace absent in lint_command, got {result.lint_command!r}'
        )

    # -----------------------------------------------------------------------
    # Extension-whitelist boundary coverage (step 4)
    # -----------------------------------------------------------------------

    # -- should SCOPE (extension in whitelist) --

    def test_rs_plus_yaml_scopes(self):
        """.rs + .yaml scopes to crate (safe config ext)."""
        result = self._call_scoped(['crates/foo/src/lib.rs', 'orchestrator.yaml'])
        assert '-p foo' in (result.test_command or '')
        assert '--workspace' not in (result.test_command or '')

    def test_rs_plus_yml_scopes(self):
        """.rs + .yml scopes to crate (safe config ext)."""
        result = self._call_scoped(['crates/foo/src/lib.rs', 'config.yml'])
        assert '-p foo' in (result.test_command or '')
        assert '--workspace' not in (result.test_command or '')

    def test_rs_plus_json_scopes(self):
        """.rs + .json scopes to crate (safe data ext)."""
        result = self._call_scoped(['crates/foo/src/lib.rs', 'package.json'])
        assert '-p foo' in (result.test_command or '')
        assert '--workspace' not in (result.test_command or '')

    def test_rs_plus_md_scopes(self):
        """.rs + .md scopes to crate (safe documentation ext)."""
        result = self._call_scoped(['crates/foo/src/lib.rs', 'README.md'])
        assert '-p foo' in (result.test_command or '')
        assert '--workspace' not in (result.test_command or '')

    def test_rs_plus_uppercase_ext_scopes(self):
        """.rs + uppercase ext (e.g. README.MD) scopes — covers Path.suffix.lower() path."""
        result = self._call_scoped(['crates/foo/src/lib.rs', 'README.MD'])
        assert '-p foo' in (result.test_command or '')
        assert '--workspace' not in (result.test_command or '')

    def test_rs_plus_cargo_lock_scopes(self):
        """.rs + Cargo.lock must scope to crate (Cargo.lock is Rust-only, not a polyglot indicator)."""
        result = self._call_scoped(['crates/foo/src/lib.rs', 'Cargo.lock'])
        assert '-p foo' in (result.test_command or ''), (
            f'expected -p foo in test_command, got {result.test_command!r}'
        )
        assert '--workspace' not in (result.test_command or ''), (
            f'expected --workspace absent in test_command, got {result.test_command!r}'
        )
        assert '-p foo' in (result.lint_command or ''), (
            f'expected -p foo in lint_command, got {result.lint_command!r}'
        )
        assert '--workspace' not in (result.lint_command or ''), (
            f'expected --workspace absent in lint_command, got {result.lint_command!r}'
        )

    def test_rs_plus_rust_toolchain_scopes(self):
        """.rs + rust-toolchain must scope to crate (rustup pin file, no extension)."""
        result = self._call_scoped(['crates/foo/src/lib.rs', 'rust-toolchain'])
        assert '-p foo' in (result.test_command or ''), (
            f'expected -p foo in test_command, got {result.test_command!r}'
        )
        assert '--workspace' not in (result.test_command or ''), (
            f'expected --workspace absent in test_command, got {result.test_command!r}'
        )
        assert '-p foo' in (result.lint_command or ''), (
            f'expected -p foo in lint_command, got {result.lint_command!r}'
        )
        assert '--workspace' not in (result.lint_command or ''), (
            f'expected --workspace absent in lint_command, got {result.lint_command!r}'
        )

    # -- should BAIL (extension outside whitelist) --

    def test_rs_plus_ts_bails(self):
        """.rs + .ts bails to --workspace (executable source)."""
        result = self._call_scoped(['crates/foo/src/lib.rs', 'gui/app.ts'])
        assert result.test_command == 'cargo test --workspace'
        assert result.lint_command == 'cargo clippy --workspace'

    def test_rs_plus_js_bails(self):
        """.rs + .js bails to --workspace (executable source)."""
        result = self._call_scoped(['crates/foo/src/lib.rs', 'scripts/build.js'])
        assert result.test_command == 'cargo test --workspace'
        assert result.lint_command == 'cargo clippy --workspace'

    def test_rs_plus_no_extension_bails(self):
        """.rs + Dockerfile (no suffix) bails to --workspace (conservative default)."""
        result = self._call_scoped(['crates/foo/src/lib.rs', 'Dockerfile'])
        assert result.test_command == 'cargo test --workspace'
        assert result.lint_command == 'cargo clippy --workspace'

    def test_rs_plus_unknown_ext_bails(self):
        """.rs + .rst bails to --workspace (unknown extension, conservative)."""
        result = self._call_scoped(['crates/foo/src/lib.rs', 'notes.rst'])
        assert result.test_command == 'cargo test --workspace'
        assert result.lint_command == 'cargo clippy --workspace'

    # -- baseline regression --

    def test_only_rs_still_scopes(self):
        """Pure .rs diff still scopes to crate (guard is transparent)."""
        result = self._call_scoped(['crates/foo/src/lib.rs'])
        assert '-p foo' in (result.test_command or '')
        assert '--workspace' not in (result.test_command or '')

    def test_no_rs_still_bails(self):
        """No .rs files at all bails via the pre-existing early-exit guard."""
        result = self._call_scoped(['Cargo.toml'])
        assert result.test_command == 'cargo test --workspace'
        assert result.lint_command == 'cargo clippy --workspace'
