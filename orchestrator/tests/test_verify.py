"""Tests for orchestrator/verify.py, specifically _run_cmd bash executable handling."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from orchestrator.config import ModuleConfig, OrchestratorConfig
from orchestrator.verify import (
    VerifyResult,
    _aggregate_results,
    _apply_cargo_scope,
    _build_fallback_config,
    _extract_cause_hint,
    _is_test_file,
    _run_cmd,
    _scope_cargo_workspace,
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


class TestIsTestFile:
    """`_is_test_file` returns True only for concrete test files, never for conftest.py.

    The docstring contract: conftest.py is excluded at any depth so callers can
    pass the result straight to pytest without a follow-up filter.
    """

    def test_returns_false_for_conftest_under_tests_dir(self):
        """conftest.py under a tests/ directory must not be treated as a test file.

        Tracing the old code: '/tests/' in 'orchestrator/tests/conftest.py' is
        True, so the function returned True — violating the docstring invariant.
        """
        assert _is_test_file('orchestrator/tests/conftest.py') is False

    def test_returns_false_for_conftest_under_root_tests_prefix(self):
        """conftest.py at the root tests/ prefix must not be treated as a test file.

        'tests/conftest.py'.startswith('tests/') is True in the old code, so
        this also returned True erroneously.
        """
        assert _is_test_file('tests/conftest.py') is False

    def test_returns_false_for_tests_data_conftest(self):
        """conftest.py inside a 'tests_data/' directory must return False.

        Guards the startswith('tests/') prefix boundary: 'tests_data/' starts
        with 'tests' but not 'tests/', so it must not be mistaken for a tests/
        directory.  Without the trailing-slash boundary the path would falsely
        match and expose a conftest to pytest.
        """
        assert _is_test_file('tests_data/conftest.py') is False

    def test_returns_true_for_test_file_under_tests_dir(self):
        """A real test file under tests/ must still return True after the fix."""
        assert _is_test_file('orchestrator/tests/test_foo.py') is True

    def test_returns_true_for_test_prefixed_file(self):
        """A test_*.py file matched by name prefix must return True."""
        assert _is_test_file('src/test_x.py') is True

    def test_returns_true_for_test_suffixed_file(self):
        """A *_test.py file matched by name suffix must return True."""
        assert _is_test_file('src/foo_test.py') is True

    def test_returns_false_for_regular_source_file(self):
        """A regular source file must return False (negative baseline)."""
        assert _is_test_file('orchestrator/src/orchestrator/verify.py') is False


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

    def test_conftest_only_uses_full_test_suite(self):
        """conftest.py alone should return the full unscoped test_command.

        conftest.py defines fixtures/hooks that affect every test in the
        directory subtree, so the correct scope is the full unscoped suite
        already expressed by mc.test_command.
        """
        mc = ModuleConfig(
            prefix='orchestrator',
            test_command='uv run --project orchestrator --directory orchestrator pytest tests/ --tb=short -q',
            lint_command='uv run --directory orchestrator ruff check src/',
        )
        result = scope_module_config(mc, ['orchestrator/tests/conftest.py'])
        assert result is not None
        assert result.test_command == mc.test_command

    def test_conftest_with_test_files_uses_full_suite(self):
        """conftest.py mixed with test files should still use the full suite.

        Even when a task touches both conftest.py and concrete test files,
        the full unscoped suite is the correct scope — conftest may shadow
        tests that aren't in the diff.
        """
        mc = ModuleConfig(
            prefix='orchestrator',
            test_command='uv run --project orchestrator --directory orchestrator pytest tests/',
            lint_command='uv run --directory orchestrator ruff check src/',
        )
        result = scope_module_config(
            mc,
            ['orchestrator/tests/conftest.py', 'orchestrator/tests/test_foo.py'],
        )
        assert result is not None
        assert result.test_command == mc.test_command


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

    def test_rs_plus_uv_lock_bails(self):
        """.rs + uv.lock bails to --workspace.

        The ``.lock`` extension is NOT globally safe — it is shared by non-Rust
        ecosystem lockfiles (``yarn.lock``, ``poetry.lock``, ``uv.lock``).  Only
        ``Cargo.lock`` by exact filename is allowed; a bare ``.lock`` suffix must
        keep triggering the polyglot guard so that chained commands such as
        ``cargo test --workspace && uv run pytest`` are not silently under-protected.
        """
        result = self._call_scoped(['crates/foo/src/lib.rs', 'uv.lock'])
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


class TestScopeCargoWorkspaceRewrite:
    """Regression guards for `_scope_cargo_workspace` — the --workspace rewriter.

    The fix that strips ``--exclude`` flags after rewriting ``--workspace`` →
    ``-p <crate>`` landed in commit fd4758fcff.  These tests keep it from
    regressing.  They invoke `_scope_cargo_workspace` directly (pure function,
    no mocks needed).
    """

    # A1: single --exclude stripped
    def test_single_exclude_stripped(self):
        """A single ``--exclude foo`` is stripped after ``--workspace`` rewrite."""
        cmd = 'cargo test --workspace --exclude foo -- --test-threads=1'
        result = _scope_cargo_workspace(cmd, ['bar'])
        assert result is not None
        assert '-p bar' in result
        assert '--workspace' not in result
        assert '--exclude' not in result
        assert 'foo' not in result  # the excluded crate name is gone too
        assert '-- --test-threads=1' in result

    # A2: three chained --exclude flags all stripped via the while-loop
    def test_multiple_excludes_all_stripped(self):
        """Three consecutive ``--exclude`` flags are all removed by the while-loop."""
        cmd = (
            'cargo test --workspace '
            '--exclude alpha --exclude beta --exclude gamma '
            '-- --test-threads=1'
        )
        result = _scope_cargo_workspace(cmd, ['delta'])
        assert result is not None
        assert '-p delta' in result
        assert '--workspace' not in result
        assert '--exclude' not in result
        assert 'alpha' not in result
        assert 'beta' not in result
        assert 'gamma' not in result

    # A3: --exclude=foo equals-form stripped
    def test_exclude_equals_form_stripped(self):
        """``--exclude=foo`` (equals-sign form) is also stripped."""
        cmd = 'cargo test --workspace --exclude=foo -- --test-threads=1'
        result = _scope_cargo_workspace(cmd, ['bar'])
        assert result is not None
        assert '-p bar' in result
        assert '--workspace' not in result
        assert '--exclude' not in result

    # A4: full reify orchestrator.yaml 4-segment command
    def test_reify_chained_command_rewrites_ungated_segments_only(self):
        """Four-segment command: gated wrappers untouched, ungated segments rewritten.

        This mirrors the real-world ``orchestrator.yaml`` test_command from the
        reify project: two gated wrapper segments (no ``--workspace``) followed
        by two ungated segments (with ``--workspace --exclude ...``).
        Rewriting for ``crates=['reify-compiler']`` must:
          - leave the gated segments byte-identical
          - rewrite both ungated segments to ``-p reify-compiler``
          - strip ALL ``--exclude`` flags from the ungated segments
        """
        gated_1 = (
            './scripts/cargo-test-occt-gated.sh cargo test '
            '-p reify-kernel-occt -p reify-eval -p reify-cli'
        )
        gated_2 = (
            './scripts/cargo-test-occt-gated.sh cargo test '
            '-p reify-kernel-occt-extra'
        )
        ungated_excludes = (
            '--exclude reify-kernel-occt --exclude reify-eval '
            '--exclude reify-cli --exclude reify-kernel-occt-extra'
        )
        ungated_1 = f'cargo test --workspace {ungated_excludes} -- --test-threads=1'
        ungated_2 = f'cargo test --workspace {ungated_excludes} -- --test-threads=1'
        cmd = f'{gated_1} && {gated_2} && {ungated_1} && {ungated_2}'

        result = _scope_cargo_workspace(cmd, ['reify-compiler'])

        assert result is not None
        # Gated segments must be present verbatim
        assert gated_1 in result, f'gated_1 missing from result: {result!r}'
        assert gated_2 in result, f'gated_2 missing from result: {result!r}'
        # Ungated segments must have been rewritten
        assert '--workspace' not in result, f'--workspace still present: {result!r}'
        assert '-p reify-compiler' in result
        # Trailing args preserved in the rewritten segments
        assert '-- --test-threads=1' in result

    # A5: non-cargo --exclude in a chained command is NOT stripped
    def test_non_cargo_exclude_not_stripped(self):
        """A trailing ``npm test --exclude foo`` must not have its ``--exclude`` removed.

        ``_CARGO_EXCLUDE_RE`` is anchored to ``cargo <subcmd>`` so it must not
        bleed into adjacent non-cargo shell segments.
        """
        cmd = (
            'cargo test --workspace --exclude some-crate -- --test-threads=1'
            ' && npm test --exclude foo'
        )
        result = _scope_cargo_workspace(cmd, ['my-crate'])
        assert result is not None
        # The npm segment must be intact
        assert 'npm test --exclude foo' in result, (
            f'npm --exclude was incorrectly removed: {result!r}'
        )
        # But the cargo --exclude must be gone
        # (the cargo segment is the only one _CARGO_EXCLUDE_RE touches)
        # Split on '&&' to isolate the cargo part
        cargo_part = result.split('&&')[0]
        assert '--exclude' not in cargo_part, (
            f'cargo --exclude not stripped from: {cargo_part!r}'
        )

    # A6: idempotency — already -p-scoped command returns byte-identical
    def test_idempotent_on_already_scoped_command(self):
        """Rewriting a command that is already ``-p``-scoped (no ``--workspace``) is a no-op."""
        cmd = 'cargo test -p my-crate -- --test-threads=1'
        result = _scope_cargo_workspace(cmd, ['my-crate'])
        assert result == cmd, f'Expected unchanged cmd, got {result!r}'

    # A7: no --exclude substring in the rewritten A4 cargo segments
    def test_no_exclude_token_in_rewritten_ungated_segments(self):
        """After rewrite, neither ``--exclude`` nor any excluded crate name appears in the ungated cargo segments."""
        gated = (
            './scripts/cargo-test-occt-gated.sh cargo test '
            '-p reify-kernel-occt -p reify-eval -p reify-cli'
        )
        ungated = (
            'cargo test --workspace '
            '--exclude reify-kernel-occt --exclude reify-eval --exclude reify-cli '
            '-- --test-threads=1'
        )
        cmd = f'{gated} && {ungated}'
        result = _scope_cargo_workspace(cmd, ['reify-compiler'])

        assert result is not None
        # Split on && to isolate the rewritten cargo segment
        segments = [s.strip() for s in result.split('&&')]
        # The last segment is the rewritten ungated cargo test
        rewritten = segments[-1]
        assert '--exclude' not in rewritten, (
            f'--exclude token found in rewritten segment: {rewritten!r}'
        )


class TestExtractCauseHint:
    """Tests for the ``_extract_cause_hint(output: str) -> str`` helper.

    These tests will *fail* until step 3 implements the helper — the import
    itself triggers an ImportError on the current codebase.
    """

    # (a) cargo/clippy surface error → first ``error: …`` line returned
    def test_cargo_error_line_returned(self):
        """A cargo/clippy ``error: …`` line is detected and returned."""
        output = (
            'Compiling my-crate v0.1.0\n'
            'error: --exclude can only be used together with --workspace\n'
            'error[E0308]: mismatched types\n'
            'Other noise\n'
        )
        hint = _extract_cause_hint(output)
        assert hint == 'error: --exclude can only be used together with --workspace', (
            f'Unexpected hint: {hint!r}'
        )

    # (b) Rust test runner FAILED line
    def test_rust_test_failed_line_returned(self):
        """A ``test my::mod::it FAILED`` line is detected and returned."""
        output = (
            'running 3 tests\n'
            'test my::mod::passes ... ok\n'
            'test my::mod::it FAILED\n'
            'test my::mod::another ... ok\n'
        )
        hint = _extract_cause_hint(output)
        assert hint == 'test my::mod::it FAILED', f'Unexpected hint: {hint!r}'

    # (c) our own timeout message
    def test_timeout_message_returned(self):
        """``Command timed out after Ns: …`` is returned directly."""
        output = 'Command timed out after 1800s: cargo test --workspace'
        hint = _extract_cause_hint(output)
        assert hint == 'Command timed out after 1800s: cargo test --workspace', (
            f'Unexpected hint: {hint!r}'
        )

    # (d) flock/wrapper ERROR: line
    def test_flock_error_line_returned(self):
        """A flock/wrapper ``ERROR: …`` line is detected and returned."""
        output = (
            'Starting cargo test...\n'
            'ERROR: cargo-test-occt-gated.sh: timed out\n'
            'Cleanup done.\n'
        )
        hint = _extract_cause_hint(output)
        assert hint == 'ERROR: cargo-test-occt-gated.sh: timed out', (
            f'Unexpected hint: {hint!r}'
        )

    # (e) npm ERR! / npm error
    def test_npm_err_line_returned(self):
        """An ``npm ERR!`` line is detected and returned."""
        output = (
            'running build...\n'
            'npm ERR! code ELIFECYCLE\n'
            'npm ERR! errno 1\n'
        )
        hint = _extract_cause_hint(output)
        assert 'npm ERR!' in hint, f'Unexpected hint: {hint!r}'

    def test_npm_error_lowercase_returned(self):
        """An ``npm error …`` line (lowercase) is also detected."""
        output = 'npm error peer dep missing: react@^18\n'
        hint = _extract_cause_hint(output)
        assert 'npm error' in hint, f'Unexpected hint: {hint!r}'

    # (f) fallback — last non-blank line when no pattern matches
    def test_fallback_last_nonblank_line(self):
        """When no pattern matches, the last non-blank line is returned."""
        output = 'Line one\nLine two\nLine three\n\n'
        hint = _extract_cause_hint(output)
        assert hint == 'Line three', f'Unexpected hint: {hint!r}'

    # (g) empty / whitespace-only input → returns ''
    def test_empty_input_returns_empty_string(self):
        """Empty input returns ''."""
        assert _extract_cause_hint('') == ''

    def test_whitespace_only_returns_empty_string(self):
        """Whitespace-only input returns ''."""
        assert _extract_cause_hint('   \n\n\t  \n') == ''

    # (h) a 500-char line is truncated to 200 chars
    def test_long_line_truncated_to_200_chars(self):
        """A line longer than 200 chars is capped at 200 characters."""
        long_line = 'error: ' + 'x' * 500
        output = long_line + '\n'
        hint = _extract_cause_hint(output)
        assert len(hint) == 200, f'Expected 200 chars, got {len(hint)}: {hint!r}'
        assert hint == long_line[:200]

    # (i) when multiple ``error:`` lines exist, the FIRST one is returned
    def test_first_error_line_returned_when_multiple(self):
        """When multiple ``error: …`` lines exist, the first one wins."""
        output = (
            'error: first error here\n'
            'error: second error here\n'
            'error: third error here\n'
        )
        hint = _extract_cause_hint(output)
        assert hint == 'error: first error here', f'Unexpected hint: {hint!r}'


class TestVerifyResultCauseHint:
    """Tests for the ``cause_hint`` field on ``VerifyResult`` and its population.

    Tests (a) will fail until step 5 adds ``cause_hint`` to ``VerifyResult``.
    Tests (b)–(f) will also fail until step 5 wires hint population into
    ``run_verification`` and ``_aggregate_results``.
    """

    # (a) VerifyResult accepts cause_hint kwarg with default ''
    def test_verify_result_accepts_cause_hint_field(self):
        """VerifyResult can be constructed with cause_hint kwarg; default is ''."""
        vr_default = VerifyResult(
            passed=True,
            test_output='',
            lint_output='',
            type_output='',
            summary='All checks passed',
        )
        assert vr_default.cause_hint == '', (
            f'Default cause_hint should be empty, got {vr_default.cause_hint!r}'
        )

        vr_explicit = VerifyResult(
            passed=False,
            test_output='error: bad',
            lint_output='',
            type_output='',
            summary='Failures: tests failed',
            cause_hint='error: bad',
        )
        assert vr_explicit.cause_hint == 'error: bad', (
            f'Explicit cause_hint mismatch: {vr_explicit.cause_hint!r}'
        )

    # (b) run_verification populates cause_hint from test output when test_rc != 0
    @pytest.mark.asyncio
    async def test_run_verification_populates_cause_hint_from_test_failure(
        self, tmp_path: Path,
    ):
        """run_verification sets cause_hint from _extract_cause_hint(test_output) when test fails."""
        config = OrchestratorConfig(
            project_root=tmp_path,
            test_command='cargo test --workspace',
            lint_command='echo ok',
            type_check_command='echo ok',
        )

        test_output = 'error: --exclude can only be used together with --workspace\nOther noise'

        async def fake_run_cmd(cmd, cwd, timeout, env=None):
            if 'cargo test' in cmd:
                return 1, test_output, False
            return 0, '', False

        with patch('orchestrator.verify._run_cmd', side_effect=fake_run_cmd):
            result = await run_verification(tmp_path, config, max_retries=0)

        assert not result.passed
        assert result.cause_hint.startswith('error: --exclude'), (
            f'Expected cause_hint to start with error: --exclude, got {result.cause_hint!r}'
        )

    # (c) both test and lint fail → cause_hint has both hints joined by ' | '
    @pytest.mark.asyncio
    async def test_run_verification_joins_hints_from_multiple_failures(
        self, tmp_path: Path,
    ):
        """When test and lint both fail, cause_hint joins their hints with ' | '."""
        config = OrchestratorConfig(
            project_root=tmp_path,
            test_command='cargo test --workspace',
            lint_command='ruff check src/',
            type_check_command='echo ok',
        )

        async def fake_run_cmd(cmd, cwd, timeout, env=None):
            if 'cargo test' in cmd:
                return 1, 'error: cargo-bad', False
            if 'ruff' in cmd:
                return 1, 'error: ruff-bad', False
            return 0, '', False

        with patch('orchestrator.verify._run_cmd', side_effect=fake_run_cmd):
            result = await run_verification(tmp_path, config, max_retries=0)

        assert not result.passed
        assert ' | ' in result.cause_hint, (
            f'Expected " | " separator in cause_hint, got {result.cause_hint!r}'
        )
        assert 'cargo-bad' in result.cause_hint
        assert 'ruff-bad' in result.cause_hint

    # (d) _aggregate_results surfaces first non-empty cause_hint from failing children
    def test_aggregate_results_surfaces_cause_hint_from_children(self):
        """_aggregate_results joins cause_hint from failing child results."""
        child1 = VerifyResult(
            passed=False,
            test_output='error: bad in child1',
            lint_output='',
            type_output='',
            summary='Failures: tests failed',
            cause_hint='error: bad in child1',
        )
        child2 = VerifyResult(
            passed=True,
            test_output='',
            lint_output='',
            type_output='',
            summary='All checks passed',
            cause_hint='',
        )
        agg = _aggregate_results([child1, child2])
        assert agg.cause_hint, (
            f'Expected non-empty cause_hint in aggregated result, got {agg.cause_hint!r}'
        )
        assert 'bad in child1' in agg.cause_hint

    # (e) passed=True → cause_hint is ''
    @pytest.mark.asyncio
    async def test_run_verification_cause_hint_empty_on_pass(self, tmp_path: Path):
        """When verification passes, cause_hint must be ''."""
        config = OrchestratorConfig(
            project_root=tmp_path,
            test_command='echo ok',
            lint_command='echo ok',
            type_check_command='echo ok',
        )

        async def fake_run_cmd(cmd, cwd, timeout, env=None):
            return 0, '', False

        with patch('orchestrator.verify._run_cmd', side_effect=fake_run_cmd):
            result = await run_verification(tmp_path, config, max_retries=0)

        assert result.passed
        assert result.cause_hint == '', (
            f'Passed verification should have empty cause_hint, got {result.cause_hint!r}'
        )

    # (f) pure timeout failure → cause_hint contains 'Command timed out after'
    @pytest.mark.asyncio
    async def test_run_verification_cause_hint_from_timeout(self, tmp_path: Path):
        """When failure is a pure timeout, cause_hint contains 'Command timed out after'."""
        config = OrchestratorConfig(
            project_root=tmp_path,
            test_command='cargo test --workspace',
            lint_command='echo ok',
            type_check_command='echo ok',
            verify_command_timeout_secs=0.1,
        )

        async def fake_run_cmd(cmd, cwd, timeout, env=None):
            if 'cargo test' in cmd:
                return 1, f'Command timed out after {timeout}s: {cmd}', True
            return 0, '', False

        with patch('orchestrator.verify._run_cmd', side_effect=fake_run_cmd):
            result = await run_verification(tmp_path, config, max_retries=0)

        assert not result.passed
        assert result.timed_out
        assert 'Command timed out after' in result.cause_hint, (
            f'Expected timeout hint in cause_hint, got {result.cause_hint!r}'
        )


class TestVerificationFailedLogCauseHint:
    """The 'Verification failed' log line should carry the cause hint.

    Tests will fail until step 7 appends `` — {cause_hint}`` to the log format.
    """

    @pytest.mark.asyncio
    async def test_verification_failed_log_line_contains_cause_hint(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ):
        """When tests fail, the log record contains both the bucket label and the hint."""
        import logging

        config = OrchestratorConfig(
            project_root=tmp_path,
            test_command='cargo test --workspace',
            lint_command='echo ok',
            type_check_command='echo ok',
        )

        test_output = 'error: --exclude can only be used together with --workspace\nOther noise'

        async def fake_run_cmd(cmd, cwd, timeout, env=None):
            if 'cargo test' in cmd:
                return 1, test_output, False
            return 0, '', False

        with (
            caplog.at_level(logging.INFO, logger='orchestrator.verify'),
            patch('orchestrator.verify._run_cmd', side_effect=fake_run_cmd),
        ):
            await run_verification(tmp_path, config, max_retries=0)

        # At least one log record must carry both the bucket label and the hint.
        matching = [
            r for r in caplog.records
            if 'tests failed' in r.getMessage() and 'error: --exclude' in r.getMessage()
        ]
        assert matching, (
            f'Expected a log record containing "tests failed" AND "error: --exclude"; '
            f'got records: {[r.getMessage() for r in caplog.records]}'
        )

    @pytest.mark.asyncio
    async def test_verification_passed_log_unchanged_when_hint_empty(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ):
        """When verification passes, the log line does not contain ' — ' or double spaces."""
        import logging

        config = OrchestratorConfig(
            project_root=tmp_path,
            test_command='echo ok',
            lint_command='echo ok',
            type_check_command='echo ok',
        )

        async def fake_run_cmd(cmd, cwd, timeout, env=None):
            return 0, '', False

        with (
            caplog.at_level(logging.INFO, logger='orchestrator.verify'),
            patch('orchestrator.verify._run_cmd', side_effect=fake_run_cmd),
        ):
            result = await run_verification(tmp_path, config, max_retries=0)

        assert result.passed
        # The 'Verification passed' log line must not contain a trailing '—' dash
        for r in caplog.records:
            msg = r.getMessage()
            if 'Verification passed' in msg:
                assert ' — ' not in msg, (
                    f'Passed log line should not have " — " hint: {msg!r}'
                )
                assert '  ' not in msg, (
                    f'Passed log line should not have double spaces: {msg!r}'
                )


@pytest.mark.asyncio
class TestVerificationFailedLogIncludesPath:
    """When run_verification persists logs, the log record format includes the path.

    Tests will fail until step 16 updates the logger call to emit
    ``Verification failed: <category> — <cause_hint> (full log: <path>)``
    when worktree_log_paths is non-empty.
    """

    async def test_failed_log_includes_category_hint_and_path(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ):
        """When persistence is active, log line is '<category> — <hint> (full log: <path>)'."""
        import logging

        task_dir = tmp_path / '.task'
        task_dir.mkdir()
        archive_root = tmp_path / 'data' / 'verify-logs'

        config = OrchestratorConfig(
            project_root=tmp_path,
            test_command='cargo test --workspace',
            lint_command='echo ok',
            type_check_command='echo ok',
        )

        test_output = 'error: --exclude can only be used together with --workspace\nOther noise'

        async def fake_run_cmd(cmd, cwd, timeout, env=None):
            if 'cargo test' in cmd:
                return 1, test_output, False
            return 0, 'ok', False

        with (
            caplog.at_level(logging.INFO, logger='orchestrator.verify'),
            patch('orchestrator.verify._run_cmd', side_effect=fake_run_cmd),
        ):
            result = await run_verification(
                tmp_path, config,
                max_retries=0,
                attempt_id=1,
                task_id='42',
                archive_root=archive_root,
            )

        # Result must have a worktree log path
        assert result.worktree_log_paths, 'Persistence should have produced worktree log paths'
        expected_path = result.worktree_log_paths[0]

        # Find the 'Verification failed' log record
        failing_records = [
            r for r in caplog.records if 'Verification failed' in r.getMessage()
        ]
        assert failing_records, f'Expected a "Verification failed" log record; got {[r.getMessage() for r in caplog.records]}'
        msg = failing_records[0].getMessage()

        # Must contain category
        assert 'cargo_cli_error' in msg, (
            f'Expected "cargo_cli_error" in log line; got: {msg!r}'
        )
        # Must contain cause hint
        assert 'error: --exclude' in msg, (
            f'Expected cause_hint "error: --exclude" in log line; got: {msg!r}'
        )
        # Must contain path reference
        assert expected_path in msg, (
            f'Expected worktree log path {expected_path!r} in log line; got: {msg!r}'
        )
        # Must NOT contain a 1000+ char raw blob
        assert len(msg) < 1000, (
            f'Log line is suspiciously long ({len(msg)} chars) — raw blob may have leaked: {msg[:200]!r}'
        )

    async def test_failed_log_fallback_when_no_persistence(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ):
        """When no attempt_id/path, log falls back to '<summary> — <hint>' format."""
        import logging

        config = OrchestratorConfig(
            project_root=tmp_path,
            test_command='cargo test --workspace',
            lint_command='echo ok',
            type_check_command='echo ok',
        )

        test_output = 'error: --exclude can only be used together with --workspace\nOther noise'

        async def fake_run_cmd(cmd, cwd, timeout, env=None):
            if 'cargo test' in cmd:
                return 1, test_output, False
            return 0, '', False

        with (
            caplog.at_level(logging.INFO, logger='orchestrator.verify'),
            patch('orchestrator.verify._run_cmd', side_effect=fake_run_cmd),
        ):
            result = await run_verification(tmp_path, config, max_retries=0)

        # No persistence kwargs → no log paths
        assert result.worktree_log_paths == [], 'No persistence kwargs → empty paths'

        # Log must still contain the summary and cause hint (original format)
        failing_records = [
            r for r in caplog.records if 'Verification failed' in r.getMessage()
        ]
        assert failing_records, 'Expected a "Verification failed" log record'
        msg = failing_records[0].getMessage()
        assert 'tests failed' in msg, f'Expected summary in fallback log: {msg!r}'
        assert 'error: --exclude' in msg, f'Expected hint in fallback log: {msg!r}'
        # Must NOT contain "(full log:" string (path-format only when paths exist)
        assert '(full log:' not in msg, (
            f'Fallback log should not contain path reference: {msg!r}'
        )


class TestFailureReportCauseHint:
    """Tests for ``VerifyResult.failure_report()`` interaction with ``cause_hint``.

    Tests (a), (b), (d) will fail until step 9 inserts the ``## Failure Cause``
    section.  Test (c) acts as a regression guard on the existing section order.
    """

    # (a) non-empty cause_hint → report starts with ## Failure Cause heading
    def test_failure_report_starts_with_cause_hint_section(self):
        """When cause_hint is non-empty, report starts with ## Failure Cause."""
        vr = VerifyResult(
            passed=False,
            test_output='test my::mod::it FAILED\n',
            lint_output='',
            type_output='',
            summary='Failures: tests failed',
            cause_hint='test my::mod::it FAILED',
        )
        report = vr.failure_report()
        assert report.startswith('## Failure Cause'), (
            f'Expected report to start with "## Failure Cause"; got:\n{report[:200]!r}'
        )
        assert 'test my::mod::it FAILED' in report

    # (b) empty cause_hint → ## Failure Cause must NOT appear
    def test_failure_report_no_cause_section_when_hint_empty(self):
        """When cause_hint is '', the report contains no ## Failure Cause section."""
        vr = VerifyResult(
            passed=False,
            test_output='test my::mod::it FAILED\n',
            lint_output='',
            type_output='',
            summary='Failures: tests failed',
            cause_hint='',
        )
        report = vr.failure_report()
        assert '## Failure Cause' not in report, (
            f'"## Failure Cause" present despite empty cause_hint:\n{report[:300]!r}'
        )

    # (c) existing sections still appear after the cause hint (regression guard)
    def test_failure_report_existing_sections_appear_in_order(self):
        """## Test Failures, ## Lint Issues, ## Type Errors appear after ## Failure Cause."""
        vr = VerifyResult(
            passed=False,
            test_output='test my::mod::it FAILED\nline2',
            lint_output='lint error here',
            type_output='error: type mismatch',
            summary='Failures: tests failed, lint issues, type errors',
            cause_hint='test my::mod::it FAILED',
        )
        report = vr.failure_report()
        # All three sections must be present
        assert '## Test Failures' in report, f'Missing ## Test Failures in:\n{report!r}'
        assert '## Lint Issues' in report, f'Missing ## Lint Issues in:\n{report!r}'
        assert '## Type Errors' in report, f'Missing ## Type Errors in:\n{report!r}'
        # And they must come AFTER ## Failure Cause
        cause_pos = report.find('## Failure Cause')
        test_pos = report.find('## Test Failures')
        lint_pos = report.find('## Lint Issues')
        type_pos = report.find('## Type Errors')
        assert cause_pos < test_pos, 'Expected ## Failure Cause before ## Test Failures'
        assert test_pos < lint_pos, 'Expected ## Test Failures before ## Lint Issues'
        assert lint_pos < type_pos, 'Expected ## Lint Issues before ## Type Errors'

    # (d) timed_out=True: ## Verify Timed Out first, then ## Failure Cause
    def test_failure_report_timeout_preamble_comes_before_cause_hint(self):
        """When timed_out=True, ## Verify Timed Out appears first, ## Failure Cause after."""
        vr = VerifyResult(
            passed=False,
            test_output='Command timed out after 1800s: cargo test --workspace',
            lint_output='',
            type_output='',
            summary='Verification timed out',
            timed_out=True,
            cause_hint='Command timed out after 1800s: cargo test --workspace',
        )
        report = vr.failure_report()
        assert '## Verify Timed Out' in report, (
            f'Missing ## Verify Timed Out:\n{report!r}'
        )
        assert '## Failure Cause' in report, (
            f'Missing ## Failure Cause:\n{report!r}'
        )
        timeout_pos = report.find('## Verify Timed Out')
        cause_pos = report.find('## Failure Cause')
        assert timeout_pos < cause_pos, (
            f'Expected ## Verify Timed Out (pos {timeout_pos}) before '
            f'## Failure Cause (pos {cause_pos})'
        )


class TestFailureReportLogPaths:
    """Tests for ``## Verify Logs`` section in ``VerifyResult.failure_report()``.

    Tests will fail until step 18 inserts the ``## Verify Logs`` section into
    ``failure_report()`` between ``## Failure Cause`` and ``## Test Failures``.
    """

    # (a) non-empty worktree_log_paths → ## Verify Logs section appears
    def test_verify_logs_section_appears_when_paths_set(self):
        """## Verify Logs section is present when worktree_log_paths is non-empty."""
        vr = VerifyResult(
            passed=False,
            test_output='test my::mod::it FAILED\n',
            lint_output='',
            type_output='',
            summary='Failures: tests failed',
            cause_hint='test my::mod::it FAILED',
            worktree_log_paths=['/wt/.task/verify/attempt-1.test.log'],
            archive_log_paths=['/data/verify-logs/42/attempt-1-20260426T120000Z.log'],
        )
        report = vr.failure_report()
        assert '## Verify Logs' in report, (
            f'"## Verify Logs" missing from failure_report():\n{report!r}'
        )

    # (b) ## Verify Logs appears after ## Failure Cause and before ## Test Failures
    def test_verify_logs_section_ordering(self):
        """## Verify Logs appears after ## Failure Cause and before ## Test Failures."""
        vr = VerifyResult(
            passed=False,
            test_output='test my::mod::it FAILED\nline2',
            lint_output='',
            type_output='',
            summary='Failures: tests failed',
            cause_hint='test my::mod::it FAILED',
            worktree_log_paths=['/wt/.task/verify/attempt-1.test.log'],
            archive_log_paths=[],
        )
        report = vr.failure_report()
        cause_pos = report.find('## Failure Cause')
        logs_pos = report.find('## Verify Logs')
        test_pos = report.find('## Test Failures')
        assert cause_pos >= 0, '## Failure Cause missing'
        assert logs_pos >= 0, '## Verify Logs missing'
        assert test_pos >= 0, '## Test Failures missing'
        assert cause_pos < logs_pos, (
            f'Expected ## Failure Cause before ## Verify Logs: '
            f'cause={cause_pos}, logs={logs_pos}'
        )
        assert logs_pos < test_pos, (
            f'Expected ## Verify Logs before ## Test Failures: '
            f'logs={logs_pos}, test={test_pos}'
        )

    # (c) when both lists are empty, ## Verify Logs must NOT appear
    def test_no_verify_logs_section_when_paths_empty(self):
        """## Verify Logs is absent when both log path lists are empty."""
        vr = VerifyResult(
            passed=False,
            test_output='test my::mod::it FAILED\n',
            lint_output='',
            type_output='',
            summary='Failures: tests failed',
            cause_hint='test my::mod::it FAILED',
        )
        report = vr.failure_report()
        assert '## Verify Logs' not in report, (
            f'"## Verify Logs" present despite empty paths:\n{report!r}'
        )

    # (d) section contains Worktree: with paths, Archive: with archive paths
    def test_verify_logs_section_contains_paths(self):
        """## Verify Logs section contains Worktree: and Archive: subsections."""
        wt_path = '/wt/.task/verify/attempt-1.test.log'
        arch_path = '/data/verify-logs/42/attempt-1-20260426T120000Z.log'
        vr = VerifyResult(
            passed=False,
            test_output='',
            lint_output='lint error here',
            type_output='',
            summary='Failures: lint issues',
            cause_hint='lint error here',
            worktree_log_paths=[wt_path],
            archive_log_paths=[arch_path],
        )
        report = vr.failure_report()
        assert 'Worktree:' in report, f'"Worktree:" missing from ## Verify Logs:\n{report!r}'
        assert wt_path in report, f'Worktree path {wt_path!r} missing from report:\n{report!r}'
        assert 'Archive' in report, f'"Archive" missing from ## Verify Logs:\n{report!r}'
        assert arch_path in report, f'Archive path {arch_path!r} missing from report:\n{report!r}'

    # (d-extra) when archive_log_paths is empty, Archive section is omitted
    def test_no_archive_subsection_when_archive_empty(self):
        """When archive_log_paths is empty, no Archive subsection appears."""
        vr = VerifyResult(
            passed=False,
            test_output='',
            lint_output='lint error here',
            type_output='',
            summary='Failures: lint issues',
            cause_hint='lint error here',
            worktree_log_paths=['/wt/.task/verify/attempt-1.lint.log'],
            archive_log_paths=[],
        )
        report = vr.failure_report()
        assert '## Verify Logs' in report, '## Verify Logs should appear (worktree paths set)'
        # No archive paths → no "Archive" subsection
        assert 'Archive' not in report, (
            f'"Archive" present despite empty archive_log_paths:\n{report!r}'
        )

    # (e) Existing tests: existing sections still appear, regression guard
    def test_existing_sections_still_present_with_paths(self):
        """All existing sections still appear when ## Verify Logs is added."""
        vr = VerifyResult(
            passed=False,
            test_output='test my::mod::it FAILED\nline2',
            lint_output='lint error',
            type_output='error: type mismatch',
            summary='Failures: tests failed, lint issues, type errors',
            cause_hint='test my::mod::it FAILED',
            worktree_log_paths=['/wt/.task/verify/attempt-1.test.log'],
            archive_log_paths=[],
        )
        report = vr.failure_report()
        assert '## Failure Cause' in report
        assert '## Verify Logs' in report
        assert '## Test Failures' in report
        assert '## Lint Issues' in report
        assert '## Type Errors' in report


class TestClassifyFailure:
    """Tests for ``_classify_failure(output, rc, timed_out) -> str``.

    Each test imports ``_classify_failure`` locally so that existing tests in
    this file are not disrupted by the missing function during step 1.  Tests
    will fail with ImportError until step 2 implements the function.
    """

    def _classify(self, output: str, rc: int, timed_out: bool) -> str:
        from orchestrator.verify import _classify_failure  # noqa: PLC0415
        return _classify_failure(output, rc, timed_out)

    # (a) rc == 0 → 'passed' regardless of output content
    def test_passed_when_rc_zero(self):
        """rc=0 always yields 'passed'."""
        assert self._classify('some output FAILED', rc=0, timed_out=False) == 'passed'

    def test_passed_when_rc_zero_and_timed_out_false(self):
        """rc=0 with timed_out=False is 'passed'."""
        assert self._classify('', rc=0, timed_out=False) == 'passed'

    # (b) timed_out=True wins over any pattern match
    def test_infra_timeout_wins_over_rc(self):
        """timed_out=True yields 'infra_timeout' even when output matches other patterns."""
        assert self._classify('error: some error', rc=1, timed_out=True) == 'infra_timeout'

    def test_infra_timeout_when_timed_out_true_no_output(self):
        """timed_out=True with empty output still yields 'infra_timeout'."""
        assert self._classify('', rc=1, timed_out=True) == 'infra_timeout'

    # (c) cargo_cli_error: --exclude used outside --workspace
    def test_cargo_cli_error_exclude_pattern(self):
        """'error: --exclude can only be used together with --workspace' → cargo_cli_error."""
        output = (
            'Compiling my-crate v0.1.0\n'
            'error: --exclude can only be used together with --workspace\n'
        )
        assert self._classify(output, rc=1, timed_out=False) == 'cargo_cli_error'

    def test_cargo_cli_error_generic_cargo_error(self):
        """A generic 'error: <cargo cli message>' → cargo_cli_error."""
        output = 'error: no such subcommand: `tset`\n'
        assert self._classify(output, rc=1, timed_out=False) == 'cargo_cli_error'

    def test_cargo_cli_error_failed_to_compile(self):
        """'error: failed to compile …' (cargo CLI wrapper) → cargo_cli_error.

        Documents that the 'failed to (parse|compile|read|find)' allowlist token
        is intentional: cargo emits this form as a CLI-level wrapper (distinct from
        rustc's 'error: could not compile `…`').  If this token is ever removed
        from the allowlist, this test will catch the regression.
        """
        output = 'error: failed to compile `my-crate` (lib)\n'
        assert self._classify(output, rc=1, timed_out=False) == 'cargo_cli_error'

    def test_rustc_top_level_diagnostics_not_cargo_cli_error(self):
        """rustc top-level diagnostics without error[Exxxx] code must NOT → cargo_cli_error.

        When a rustc compile failure escapes without a coded error[E\\d+]: line (e.g. parse
        errors that promote to a top-level diagnostic), the run must fall through to
        'unknown_test_failure', not be mis-bucketed as 'cargo_cli_error'.
        """
        output = (
            'Compiling my-crate v0.1.0\n'
            'error: aborting due to previous errors\n'
            'error: could not compile `my-crate` (lib) due to previous error\n'
        )
        result = self._classify(output, rc=1, timed_out=False)
        assert result != 'cargo_cli_error', (
            f"rustc top-level diagnostics must not be mis-bucketed as cargo_cli_error, got {result!r}"
        )
        assert result == 'unknown_test_failure', (
            f"rustc top-level diagnostics should fall through to unknown_test_failure, got {result!r}"
        )

    # (d) compile_error: rustc diagnostic error codes
    def test_compile_error_rustc_code(self):
        """'error[E0308]: mismatched types' → compile_error."""
        output = (
            'Compiling my-crate v0.1.0\n'
            'error[E0308]: mismatched types\n'
            '  --> src/lib.rs:10:5\n'
        )
        assert self._classify(output, rc=1, timed_out=False) == 'compile_error'

    def test_compile_error_compile_error_string(self):
        """'compile error' string → compile_error."""
        output = 'compile error in foo.py line 5\n'
        assert self._classify(output, rc=1, timed_out=False) == 'compile_error'

    # (e) test_failure: '… FAILED' rust/pytest pattern
    def test_test_failure_rust_test_runner(self):
        """'test my::mod::it FAILED' → test_failure."""
        output = (
            'running 3 tests\n'
            'test my::mod::it FAILED\n'
            'test my::mod::another ... ok\n'
        )
        assert self._classify(output, rc=1, timed_out=False) == 'test_failure'

    def test_test_failure_pytest_failed(self):
        """'FAILED tests/test_foo.py::test_bar' → test_failure."""
        output = 'FAILED tests/test_foo.py::test_bar - AssertionError\n'
        assert self._classify(output, rc=1, timed_out=False) == 'test_failure'

    # (f) unknown_test_failure: rc!=0 with no matching pattern
    def test_unknown_test_failure_fallback(self):
        """rc!=0 with output that matches no specific pattern → unknown_test_failure."""
        output = 'Something went wrong but no recognizable pattern\n'
        assert self._classify(output, rc=1, timed_out=False) == 'unknown_test_failure'

    def test_unknown_test_failure_empty_output(self):
        """rc!=0 with empty output → unknown_test_failure."""
        assert self._classify('', rc=1, timed_out=False) == 'unknown_test_failure'

    # (g) npm_error
    def test_npm_err_exclamation(self):
        """'npm ERR! ...' → npm_error."""
        output = 'npm ERR! code ELIFECYCLE\nnpm ERR! errno 1\n'
        assert self._classify(output, rc=1, timed_out=False) == 'npm_error'

    def test_npm_error_lowercase(self):
        """'npm error ...' → npm_error."""
        output = 'npm error peer dep missing: react@^18\n'
        assert self._classify(output, rc=1, timed_out=False) == 'npm_error'

    # (h) flock_error
    def test_flock_error_pattern(self):
        """'flock: failed to acquire lock' → flock_error."""
        output = 'flock: failed to acquire lock on /var/lock/mylock\n'
        assert self._classify(output, rc=1, timed_out=False) == 'flock_error'

    # (i) tree_sitter_generate_error
    def test_tree_sitter_generate_error(self):
        """tree-sitter generate failure → tree_sitter_generate_error."""
        output = (
            'Running tree-sitter generate\n'
            'tree-sitter generate failed: unexpected token\n'
        )
        assert self._classify(output, rc=1, timed_out=False) == 'tree_sitter_generate_error'


class TestVerifyResultCategoryAndPaths:
    """Tests for the new ``category``, ``worktree_log_paths``, and
    ``archive_log_paths`` fields on ``VerifyResult``, and their population
    through ``run_verification`` / ``_aggregate_results``.

    Tests fail until step 6 adds the fields to the dataclass and wires
    category classification into ``run_verification`` / ``_aggregate_results``.
    """

    # (a) VerifyResult defaults: category='', worktree_log_paths=[], archive_log_paths=[]
    def test_verify_result_category_defaults(self):
        """VerifyResult defaults: category='', log path lists are empty."""
        vr = VerifyResult(
            passed=True,
            test_output='',
            lint_output='',
            type_output='',
            summary='All checks passed',
        )
        assert vr.category == '', f'Expected empty category default, got {vr.category!r}'
        assert vr.worktree_log_paths == [], f'Expected empty list, got {vr.worktree_log_paths!r}'
        assert vr.archive_log_paths == [], f'Expected empty list, got {vr.archive_log_paths!r}'

    # (b) explicit values round-trip
    def test_verify_result_explicit_category_and_paths(self):
        """Explicit category/paths round-trip via dataclass field access."""
        vr = VerifyResult(
            passed=False,
            test_output='error: bad',
            lint_output='',
            type_output='',
            summary='Failures: tests failed',
            category='cargo_cli_error',
            worktree_log_paths=['/wt/.task/verify/attempt-1.test.log'],
            archive_log_paths=['/data/verify-logs/42/attempt-1-20260426T120000Z.log'],
        )
        assert vr.category == 'cargo_cli_error'
        assert vr.worktree_log_paths == ['/wt/.task/verify/attempt-1.test.log']
        assert vr.archive_log_paths == ['/data/verify-logs/42/attempt-1-20260426T120000Z.log']

    # (c) run_verification populates result.category from _classify_failure
    @pytest.mark.asyncio
    async def test_run_verification_populates_category_from_test_failure(
        self, tmp_path: Path,
    ):
        """run_verification sets result.category from _classify_failure on the failing check."""
        config = OrchestratorConfig(
            project_root=tmp_path,
            test_command='cargo test --workspace',
            lint_command='echo ok',
            type_check_command='echo ok',
        )
        test_output = 'error: --exclude can only be used together with --workspace\nOther noise'

        async def fake_run_cmd(cmd, cwd, timeout, env=None):
            if 'cargo test' in cmd:
                return 1, test_output, False
            return 0, '', False

        with patch('orchestrator.verify._run_cmd', side_effect=fake_run_cmd):
            result = await run_verification(tmp_path, config, max_retries=0)

        assert not result.passed
        assert result.category == 'cargo_cli_error', (
            f'Expected cargo_cli_error, got {result.category!r}'
        )

    # (d) _aggregate_results concatenates worktree_log_paths and archive_log_paths
    def test_aggregate_results_concatenates_log_paths(self):
        """_aggregate_results flattens child worktree_log_paths and archive_log_paths."""
        r1 = VerifyResult(
            passed=False, test_output='', lint_output='lint error', type_output='',
            summary='Failures: lint issues',
            worktree_log_paths=['/wt/.task/verify/attempt-1.lint.log'],
            archive_log_paths=['/data/verify-logs/42/attempt-1-lint.log'],
        )
        r2 = VerifyResult(
            passed=False, test_output='error: bad', lint_output='', type_output='',
            summary='Failures: tests failed',
            worktree_log_paths=['/wt/.task/verify/attempt-1.test.log'],
            archive_log_paths=[],
        )
        agg = _aggregate_results([r1, r2])
        assert '/wt/.task/verify/attempt-1.lint.log' in agg.worktree_log_paths
        assert '/wt/.task/verify/attempt-1.test.log' in agg.worktree_log_paths
        assert '/data/verify-logs/42/attempt-1-lint.log' in agg.archive_log_paths

    # (e) _aggregate_results picks worst child category by priority
    def test_aggregate_results_picks_worst_category(self):
        """_aggregate_results selects the highest-priority non-passed category."""
        r_test = VerifyResult(
            passed=False, test_output='', lint_output='', type_output='',
            summary='Failures: tests failed',
            category='test_failure',
        )
        r_cargo = VerifyResult(
            passed=False, test_output='', lint_output='', type_output='',
            summary='Failures: tests failed',
            category='cargo_cli_error',
        )
        agg = _aggregate_results([r_test, r_cargo])
        # cargo_cli_error has higher priority than test_failure
        assert agg.category == 'cargo_cli_error', (
            f'Expected cargo_cli_error (higher priority), got {agg.category!r}'
        )


class TestPersistAttemptLogs:
    """Tests for ``_persist_attempt_logs(worktree, attempt_id, runs, category, cause_hint)``.

    Tests fail until step 8 implements the helper.
    """

    def _persist(self, worktree, attempt_id, runs, category='cargo_cli_error', cause_hint='error: bad'):
        import asyncio  # noqa: PLC0415

        from orchestrator.verify import _persist_attempt_logs  # noqa: PLC0415
        return asyncio.get_event_loop().run_until_complete(
            _persist_attempt_logs(worktree, attempt_id, runs, category, cause_hint)
        ) if asyncio.iscoroutinefunction(_persist_attempt_logs) else _persist_attempt_logs(
            worktree, attempt_id, runs, category, cause_hint
        )

    def _make_runs(self):
        """Return a list of fake per-command run dicts for test+lint+type."""
        return [
            {
                'label': 'test',
                'cmd': 'cargo test --workspace',
                'rc': 1,
                'output': 'error: --exclude can only be used together with --workspace\nLine2\n',
                'timed_out': False,
                'started_at': '2026-04-26T12:00:00+00:00',
                'duration_secs': 5.1,
            },
            {
                'label': 'lint',
                'cmd': 'ruff check src/',
                'rc': 0,
                'output': '',
                'timed_out': False,
                'started_at': '2026-04-26T12:00:05+00:00',
                'duration_secs': 1.2,
            },
            {
                'label': 'type',
                'cmd': 'mypy src/',
                'rc': 0,
                'output': '',
                'timed_out': False,
                'started_at': '2026-04-26T12:00:06+00:00',
                'duration_secs': 0.8,
            },
        ]

    # (a-c) basic write: three log files + summary.json created
    def test_writes_log_files_and_summary(self, tmp_path: Path):
        """Log files and summary.json are created under .task/verify/."""
        (tmp_path / '.task').mkdir()
        runs = self._make_runs()
        self._persist(tmp_path, attempt_id=1, runs=runs)

        verify_dir = tmp_path / '.task' / 'verify'
        test_log = verify_dir / 'attempt-1.test.log'
        lint_log = verify_dir / 'attempt-1.lint.log'
        type_log = verify_dir / 'attempt-1.type.log'
        summary = verify_dir / 'attempt-1.summary.json'

        assert test_log.exists(), f'test log missing: {test_log}'
        assert test_log.read_text() == runs[0]['output']
        # lint and type have empty output but cmd is set → files created
        assert lint_log.exists(), 'lint log missing'
        assert type_log.exists(), 'type log missing'
        assert summary.exists(), 'summary.json missing'

    # (d) summary.json contains required keys
    def test_summary_json_has_required_keys(self, tmp_path: Path):
        """summary.json has category, cause_hint, rc, timed_out, cmd, started_at, duration_secs, commands."""
        import json
        (tmp_path / '.task').mkdir()
        runs = self._make_runs()
        self._persist(tmp_path, attempt_id=1, runs=runs, category='cargo_cli_error', cause_hint='error: bad')
        summary_path = tmp_path / '.task' / 'verify' / 'attempt-1.summary.json'
        data = json.loads(summary_path.read_text())
        for key in ('category', 'cause_hint', 'rc', 'timed_out', 'cmd', 'started_at', 'duration_secs', 'commands'):
            assert key in data, f'Missing key {key!r} in summary: {data}'
        assert data['category'] == 'cargo_cli_error'
        assert data['cause_hint'] == 'error: bad'
        assert isinstance(data['commands'], list)
        assert len(data['commands']) >= 1

    # (e) returned list contains the written log paths
    def test_returns_written_log_paths(self, tmp_path: Path):
        """Return value lists exactly the written log paths in test/lint/type order."""
        (tmp_path / '.task').mkdir()
        runs = self._make_runs()
        paths = self._persist(tmp_path, attempt_id=1, runs=runs)
        verify_dir = tmp_path / '.task' / 'verify'
        # All three cmd-based logs should be in the returned list
        path_strs = [str(p) for p in paths]
        assert str(verify_dir / 'attempt-1.test.log') in path_strs
        assert str(verify_dir / 'attempt-1.lint.log') in path_strs
        assert str(verify_dir / 'attempt-1.type.log') in path_strs

    # non-clobber: attempt-1 and attempt-2 files coexist
    def test_non_clobbering_multiple_attempts(self, tmp_path: Path):
        """Different attempt_ids produce separate files that coexist."""
        (tmp_path / '.task').mkdir()
        runs = self._make_runs()
        self._persist(tmp_path, attempt_id=1, runs=runs)
        self._persist(tmp_path, attempt_id=2, runs=runs)
        verify_dir = tmp_path / '.task' / 'verify'
        assert (verify_dir / 'attempt-1.test.log').exists()
        assert (verify_dir / 'attempt-2.test.log').exists()

    # no-op: when .task/ is absent, returns empty list and no files created
    def test_noop_when_task_dir_absent(self, tmp_path: Path):
        """When .task/ does not exist, returns [] and creates no files."""
        runs = self._make_runs()
        paths = self._persist(tmp_path, attempt_id=1, runs=runs)
        assert paths == [] or paths == [], f'Expected empty list, got {paths!r}'
        assert not (tmp_path / '.task').exists(), '.task/ should not have been created'

    # skipped cmd (None) does not produce a log file
    def test_none_cmd_run_skipped(self, tmp_path: Path):
        """A run with cmd=None produces no log file."""
        (tmp_path / '.task').mkdir()
        runs = [
            {
                'label': 'test',
                'cmd': None,
                'rc': 0,
                'output': '',
                'timed_out': False,
                'started_at': '2026-04-26T12:00:00+00:00',
                'duration_secs': 0.0,
            }
        ]
        self._persist(tmp_path, attempt_id=1, runs=runs)
        verify_dir = tmp_path / '.task' / 'verify'
        # No .log files (only summary.json may exist)
        log_files = list(verify_dir.glob('*.log'))
        assert log_files == [], f'Expected no log files, got {log_files}'


class TestArchiveAttemptLog:
    """Tests for ``_archive_attempt_log(worktree_log_paths, archive_root, task_id, attempt_id, category)``.

    Tests fail until step 10 implements the helper.
    """

    def _archive(self, worktree_log_paths, archive_root, task_id, attempt_id, category):
        from orchestrator.verify import _archive_attempt_log  # noqa: PLC0415
        return _archive_attempt_log(worktree_log_paths, archive_root, task_id, attempt_id, category)

    def _make_source_logs(self, tmp_path: Path) -> list[Path]:
        """Create two fake worktree log files and return their paths."""
        log_dir = tmp_path / '.task' / 'verify'
        log_dir.mkdir(parents=True)
        test_log = log_dir / 'attempt-1.test.log'
        lint_log = log_dir / 'attempt-1.lint.log'
        test_log.write_text('error: --exclude ...\n')
        lint_log.write_text('')
        return [test_log, lint_log]

    # (a) cargo_cli_error → archives created under <archive_root>/<task_id>/
    def test_archives_for_cargo_cli_error(self, tmp_path: Path):
        """cargo_cli_error is in the archive bucket; files are copied."""
        sources = self._make_source_logs(tmp_path)
        archive_root = tmp_path / 'data' / 'verify-logs'
        dest_paths = self._archive(sources, archive_root, '42', 1, 'cargo_cli_error')
        assert len(dest_paths) == len(sources), f'Expected {len(sources)} archived, got {dest_paths}'
        for dest in dest_paths:
            assert dest.exists(), f'Archive file missing: {dest}'

    # (b) unknown_test_failure → archives created
    def test_archives_for_unknown_test_failure(self, tmp_path: Path):
        """unknown_test_failure is archived."""
        sources = self._make_source_logs(tmp_path)
        archive_root = tmp_path / 'data' / 'verify-logs'
        dest_paths = self._archive(sources, archive_root, '42', 1, 'unknown_test_failure')
        assert len(dest_paths) == len(sources)
        for dest in dest_paths:
            assert dest.exists()

    # (c) test_failure → NO archive files created
    def test_no_archive_for_test_failure(self, tmp_path: Path):
        """test_failure is not archived; returns empty list."""
        sources = self._make_source_logs(tmp_path)
        archive_root = tmp_path / 'data' / 'verify-logs'
        dest_paths = self._archive(sources, archive_root, '42', 1, 'test_failure')
        assert dest_paths == [], f'Expected [], got {dest_paths}'

    # (d) compile_error → NO archive files
    def test_no_archive_for_compile_error(self, tmp_path: Path):
        """compile_error is not archived."""
        sources = self._make_source_logs(tmp_path)
        archive_root = tmp_path / 'data' / 'verify-logs'
        dest_paths = self._archive(sources, archive_root, '42', 1, 'compile_error')
        assert dest_paths == []

    # (e) infra_timeout → NO archive files
    def test_no_archive_for_infra_timeout(self, tmp_path: Path):
        """infra_timeout is not archived."""
        sources = self._make_source_logs(tmp_path)
        archive_root = tmp_path / 'data' / 'verify-logs'
        dest_paths = self._archive(sources, archive_root, '42', 1, 'infra_timeout')
        assert dest_paths == []

    # (f) archive_root=None → returns [] with no error
    def test_noop_when_archive_root_none(self, tmp_path: Path):
        """archive_root=None returns empty list, no exception."""
        sources = self._make_source_logs(tmp_path)
        dest_paths = self._archive(sources, None, '42', 1, 'cargo_cli_error')
        assert dest_paths == []

    # (g) archive_root parent does not exist → function creates it
    def test_creates_archive_root_when_missing(self, tmp_path: Path):
        """archive_root hierarchy is created by mkdir(parents=True)."""
        sources = self._make_source_logs(tmp_path)
        archive_root = tmp_path / 'deep' / 'nested' / 'verify-logs'
        assert not archive_root.exists()
        dest_paths = self._archive(sources, archive_root, '42', 1, 'cargo_cli_error')
        assert len(dest_paths) > 0
        assert archive_root.exists()

    # content preserved
    def test_archived_content_matches_source(self, tmp_path: Path):
        """The content of archived files matches the source."""
        sources = self._make_source_logs(tmp_path)
        archive_root = tmp_path / 'data' / 'verify-logs'
        dest_paths = self._archive(sources, archive_root, '42', 1, 'cargo_cli_error')
        for src, dest in zip(sources, dest_paths, strict=True):
            assert dest.read_text() == src.read_text(), f'Content mismatch: {src} vs {dest}'


class TestPruneArchive:
    """Tests for ``_prune_archive(archive_root, max_age_days, max_total_bytes)``.

    Tests import the function locally; they are expected to pass once step 12
    (already folded into step 10's commit) provides the implementation.
    """

    def _prune(self, archive_root, max_age_days=30, max_total_bytes=1_000_000_000):
        from orchestrator.verify import _prune_archive  # noqa: PLC0415
        return _prune_archive(archive_root, max_age_days=max_age_days, max_total_bytes=max_total_bytes)

    # (a) old files deleted, fresh files kept
    def test_old_files_deleted_fresh_kept(self, tmp_path: Path):
        """Files older than max_age_days are deleted; fresh files survive."""
        import os
        import time
        archive_root = tmp_path / 'archive'
        archive_root.mkdir()
        old = archive_root / 'old.log'
        fresh = archive_root / 'fresh.log'
        old.write_text('old')
        fresh.write_text('fresh')
        # Set old file's mtime to 31 days ago
        old_mtime = time.time() - 31 * 86_400
        os.utime(old, (old_mtime, old_mtime))
        self._prune(archive_root, max_age_days=30)
        assert not old.exists(), 'Old file should have been deleted'
        assert fresh.exists(), 'Fresh file should remain'

    # (b) size cap: oldest files deleted until under cap
    def test_size_cap_deletes_oldest_first(self, tmp_path: Path):
        """When total size > max_total_bytes, oldest files deleted until under cap."""
        import os
        import time
        archive_root = tmp_path / 'archive'
        archive_root.mkdir()
        # Create 3 files of 50 bytes each; cap at 100 bytes → one must be deleted
        t = time.time() - 60  # base mtime
        files = []
        for i in range(3):
            f = archive_root / f'f{i}.log'
            f.write_text('x' * 50)  # 50 bytes
            os.utime(f, (t + i, t + i))
            files.append(f)
        # Total = 150 bytes, cap = 100 → must delete at least 1 (oldest = f0)
        self._prune(archive_root, max_age_days=365, max_total_bytes=100)
        assert not files[0].exists(), 'Oldest file should be deleted'
        # At least one of the newer files must remain
        assert files[1].exists() or files[2].exists()

    # (c) non-existent archive_root → no error
    def test_noop_on_missing_archive_root(self, tmp_path: Path):
        """Returns silently when archive_root does not exist."""
        missing = tmp_path / 'nonexistent'
        self._prune(missing)  # must not raise

    # (d) directory structure preserved (only files deleted, not dirs)
    def test_directories_preserved(self, tmp_path: Path):
        """Only log files are deleted; empty directories are left intact."""
        import os
        import time
        archive_root = tmp_path / 'archive'
        sub = archive_root / 'sub'
        sub.mkdir(parents=True)
        old = sub / 'old.log'
        old.write_text('content')
        old_mtime = time.time() - 31 * 86_400
        os.utime(old, (old_mtime, old_mtime))
        self._prune(archive_root, max_age_days=30)
        assert not old.exists(), 'Old log file should be deleted'
        assert sub.exists(), 'Sub-directory should remain'


@pytest.mark.asyncio
class TestRunVerificationPersistence:
    """Tests for wire-through of persistence kwargs in ``run_verification``.

    Tests will fail until step 14 adds ``attempt_id``, ``task_id``, and
    ``archive_root`` keyword args to ``run_verification`` and
    ``run_scoped_verification`` and calls the persistence helpers.
    """

    async def test_run_verification_with_persistence_kwargs_sets_paths(
        self, tmp_path: Path,
    ):
        """When attempt_id/task_id/archive_root are passed, result carries log paths."""
        # Pre-create .task/ so persistence is active
        task_dir = tmp_path / '.task'
        task_dir.mkdir()
        archive_root = tmp_path / 'data' / 'verify-logs'

        config = OrchestratorConfig(
            project_root=tmp_path,
            test_command='cargo test --workspace',
            lint_command='echo ok',
            type_check_command='echo ok',
        )

        test_output = 'error: --exclude can only be used together with --workspace\nOther noise'

        async def fake_run_cmd(cmd, cwd, timeout, env=None):
            if 'cargo test' in cmd:
                return 1, test_output, False
            return 0, 'ok', False

        with patch('orchestrator.verify._run_cmd', side_effect=fake_run_cmd):
            result = await run_verification(
                tmp_path, config,
                max_retries=0,
                attempt_id=1,
                task_id='42',
                archive_root=archive_root,
            )

        # (a) category is set correctly
        assert result.category == 'cargo_cli_error', (
            f'Expected cargo_cli_error, got {result.category!r}'
        )

        # (b) worktree_log_paths lists the test log file
        assert result.worktree_log_paths, 'Expected non-empty worktree_log_paths'
        test_log_path = tmp_path / '.task' / 'verify' / 'attempt-1.test.log'
        assert str(test_log_path) in result.worktree_log_paths, (
            f'Expected {test_log_path} in {result.worktree_log_paths}'
        )

        # (c) archive_log_paths lists at least one archived file
        assert result.archive_log_paths, 'Expected non-empty archive_log_paths (cargo_cli_error triggers archival)'
        archived = result.archive_log_paths[0]
        assert '42' in archived, f'Archive path should contain task_id "42": {archived}'
        assert 'attempt-1' in archived, f'Archive path should contain attempt_id: {archived}'

        # (d) worktree log content matches patched output
        assert test_log_path.exists(), 'Worktree test log file should exist'
        content = test_log_path.read_text()
        assert content == test_output, (
            f'Worktree log content mismatch: {content!r} != {test_output!r}'
        )

    async def test_run_verification_without_persistence_kwargs_no_paths(
        self, tmp_path: Path,
    ):
        """(e) When attempt_id/task_id/archive_root are omitted, paths are empty (backward-compat)."""
        task_dir = tmp_path / '.task'
        task_dir.mkdir()

        config = OrchestratorConfig(
            project_root=tmp_path,
            test_command='cargo test --workspace',
            lint_command='echo ok',
            type_check_command='echo ok',
        )

        async def fake_run_cmd(cmd, cwd, timeout, env=None):
            if 'cargo test' in cmd:
                return 1, 'error: --exclude bad', False
            return 0, '', False

        with patch('orchestrator.verify._run_cmd', side_effect=fake_run_cmd):
            result = await run_verification(tmp_path, config, max_retries=0)

        assert result.worktree_log_paths == [], (
            f'Expected empty worktree_log_paths when no attempt_id; got {result.worktree_log_paths!r}'
        )
        assert result.archive_log_paths == [], (
            f'Expected empty archive_log_paths when no attempt_id; got {result.archive_log_paths!r}'
        )

    async def test_run_verification_no_task_dir_no_paths(
        self, tmp_path: Path,
    ):
        """When .task/ is absent, persistence is a no-op and paths are empty."""
        # NOTE: do NOT create .task/ — simulate merge-queue / review-checkpoint paths
        archive_root = tmp_path / 'data' / 'verify-logs'

        config = OrchestratorConfig(
            project_root=tmp_path,
            test_command='cargo test --workspace',
            lint_command='echo ok',
            type_check_command='echo ok',
        )

        async def fake_run_cmd(cmd, cwd, timeout, env=None):
            if 'cargo test' in cmd:
                return 1, 'error: --exclude bad', False
            return 0, '', False

        with patch('orchestrator.verify._run_cmd', side_effect=fake_run_cmd):
            result = await run_verification(
                tmp_path, config,
                max_retries=0,
                attempt_id=1,
                task_id='42',
                archive_root=archive_root,
            )

        assert result.worktree_log_paths == [], (
            f'Expected empty paths when .task/ absent; got {result.worktree_log_paths!r}'
        )
        assert result.archive_log_paths == [], (
            f'Expected empty archive_paths when .task/ absent; got {result.archive_log_paths!r}'
        )


class TestPersistAttemptLogsModulePrefix:
    """Tests for the ``module_prefix`` parameter added to ``_persist_attempt_logs``.

    Tests fail until step 22 adds ``module_prefix`` to ``_persist_attempt_logs``
    and wires it through ``run_verification``.
    """

    def _persist(
        self,
        worktree,
        attempt_id,
        runs,
        category='cargo_cli_error',
        cause_hint='error: bad',
        module_prefix=None,
    ):
        from orchestrator.verify import _persist_attempt_logs  # noqa: PLC0415

        kwargs = {}
        if module_prefix is not None:
            kwargs['module_prefix'] = module_prefix
        return _persist_attempt_logs(worktree, attempt_id, runs, category, cause_hint, **kwargs)

    def _make_runs(self):
        return [
            {
                'label': 'test',
                'cmd': 'cargo test -p cratea',
                'rc': 1,
                'output': 'cratea test failure\n',
                'timed_out': False,
                'started_at': '2026-04-26T12:00:00+00:00',
                'duration_secs': 3.0,
            },
            {
                'label': 'lint',
                'cmd': 'ruff check cratea/src/',
                'rc': 0,
                'output': '',
                'timed_out': False,
                'started_at': '2026-04-26T12:00:03+00:00',
                'duration_secs': 0.5,
            },
            {
                'label': 'type',
                'cmd': 'mypy cratea/src/',
                'rc': 0,
                'output': '',
                'timed_out': False,
                'started_at': '2026-04-26T12:00:04+00:00',
                'duration_secs': 0.4,
            },
        ]

    def test_with_module_prefix_uses_infix_filenames(self, tmp_path: Path):
        """module_prefix='cratea' → attempt-1.cratea.test.log, etc."""
        (tmp_path / '.task').mkdir()
        runs = self._make_runs()
        self._persist(tmp_path, attempt_id=1, runs=runs, module_prefix='cratea')

        verify_dir = tmp_path / '.task' / 'verify'
        assert (verify_dir / 'attempt-1.cratea.test.log').exists(), 'Expected cratea test log'
        assert (verify_dir / 'attempt-1.cratea.lint.log').exists(), 'Expected cratea lint log'
        assert (verify_dir / 'attempt-1.cratea.type.log').exists(), 'Expected cratea type log'
        assert (verify_dir / 'attempt-1.cratea.summary.json').exists(), 'Expected cratea summary.json'
        # No plain (prefix-less) filenames should exist
        assert not (verify_dir / 'attempt-1.test.log').exists(), 'Unexpected prefix-less test log'

    def test_without_module_prefix_uses_plain_filenames(self, tmp_path: Path):
        """When module_prefix is omitted, filenames are attempt-1.{test,lint,type}.log (unchanged)."""
        (tmp_path / '.task').mkdir()
        runs = self._make_runs()
        self._persist(tmp_path, attempt_id=1, runs=runs)

        verify_dir = tmp_path / '.task' / 'verify'
        assert (verify_dir / 'attempt-1.test.log').exists(), 'Expected plain test log'
        assert (verify_dir / 'attempt-1.lint.log').exists(), 'Expected plain lint log'
        assert (verify_dir / 'attempt-1.type.log').exists(), 'Expected plain type log'
        assert (verify_dir / 'attempt-1.summary.json').exists(), 'Expected plain summary.json'

    def test_module_prefix_sanitization_slashes(self, tmp_path: Path):
        """module_prefix='path/to/crate' → slashes replaced with underscores."""
        (tmp_path / '.task').mkdir()
        runs = self._make_runs()
        self._persist(tmp_path, attempt_id=1, runs=runs, module_prefix='path/to/crate')

        verify_dir = tmp_path / '.task' / 'verify'
        assert (verify_dir / 'attempt-1.path_to_crate.test.log').exists(), (
            'Expected slash-sanitized test log'
        )
        assert (verify_dir / 'attempt-1.path_to_crate.summary.json').exists(), (
            'Expected slash-sanitized summary.json'
        )

    def test_module_prefix_sanitization_spaces(self, tmp_path: Path):
        """module_prefix containing spaces → spaces replaced with underscores."""
        (tmp_path / '.task').mkdir()
        runs = self._make_runs()
        self._persist(tmp_path, attempt_id=1, runs=runs, module_prefix='my crate')

        verify_dir = tmp_path / '.task' / 'verify'
        assert (verify_dir / 'attempt-1.my_crate.test.log').exists(), (
            'Expected space-sanitized test log'
        )

    def test_two_prefixes_same_attempt_id_no_clobber(self, tmp_path: Path):
        """Two calls with same attempt_id but different module_prefix produce distinct files."""
        (tmp_path / '.task').mkdir()
        runs_a = self._make_runs()
        runs_b = [dict(r, output=r['output'].replace('cratea', 'crateb')) for r in self._make_runs()]

        self._persist(tmp_path, attempt_id=1, runs=runs_a, module_prefix='cratea')
        self._persist(tmp_path, attempt_id=1, runs=runs_b, module_prefix='crateb')

        verify_dir = tmp_path / '.task' / 'verify'
        assert (verify_dir / 'attempt-1.cratea.test.log').exists(), 'cratea test log missing'
        assert (verify_dir / 'attempt-1.crateb.test.log').exists(), 'crateb test log missing'
        # Content must differ (no last-writer-wins clobber)
        cratea_content = (verify_dir / 'attempt-1.cratea.test.log').read_text()
        crateb_content = (verify_dir / 'attempt-1.crateb.test.log').read_text()
        assert 'cratea' in cratea_content
        assert 'crateb' in crateb_content


@pytest.mark.asyncio
class TestRunScopedVerificationConcurrentNoClobber:
    """Integration tests ensuring ``run_scoped_verification`` with multiple ModuleConfigs
    writes per-module log files without last-writer-wins clobber.

    Tests fail until step 22 passes ``module_prefix`` to ``_persist_attempt_logs``
    inside ``run_verification``.
    """

    def _make_config(self, tmp_path: Path) -> 'OrchestratorConfig':
        return OrchestratorConfig(
            project_root=tmp_path,
            test_command='cargo test --workspace',
            lint_command='echo ok',
            type_check_command='echo ok',
        )

    def _make_module_configs(self) -> 'list[ModuleConfig]':
        mc_a = ModuleConfig(
            prefix='cratea',
            test_command='cargo test -p cratea',
            lint_command=None,
            type_check_command=None,
        )
        mc_b = ModuleConfig(
            prefix='crateb',
            test_command='cargo test -p crateb',
            lint_command=None,
            type_check_command=None,
        )
        return [mc_a, mc_b]

    async def test_concurrent_no_clobber(self, tmp_path: Path):
        """Two ModuleConfigs write distinct per-module log files without clobber."""
        (tmp_path / '.task').mkdir()
        archive_root = tmp_path / 'data' / 'verify-logs'
        config = self._make_config(tmp_path)
        module_configs = self._make_module_configs()

        async def fake_run_cmd(cmd, cwd, timeout, env=None):
            if 'cratea' in cmd:
                return 1, 'cratea ERR\nfoo\nbar\n', False
            if 'crateb' in cmd:
                return 1, 'crateb ERR\nbaz\nqux\n', False
            return 0, 'ok', False

        with patch('orchestrator.verify._run_cmd', side_effect=fake_run_cmd):
            result = await run_scoped_verification(
                tmp_path, config, module_configs,
                attempt_id=1,
                task_id='42',
                archive_root=archive_root,
            )

        verify_dir = tmp_path / '.task' / 'verify'

        # (e) both files exist
        cratea_log = verify_dir / 'attempt-1.cratea.test.log'
        crateb_log = verify_dir / 'attempt-1.crateb.test.log'
        assert cratea_log.exists(), f'cratea log missing: {list(verify_dir.iterdir())}'
        assert crateb_log.exists(), f'crateb log missing: {list(verify_dir.iterdir())}'

        # (f) content is module-specific (no clobber)
        assert 'cratea ERR' in cratea_log.read_text(), 'cratea log has wrong content'
        assert 'crateb ERR' in crateb_log.read_text(), 'crateb log has wrong content'

        # (g) both summary.json files exist with the correct category
        cratea_summary = verify_dir / 'attempt-1.cratea.summary.json'
        crateb_summary = verify_dir / 'attempt-1.crateb.summary.json'
        assert cratea_summary.exists(), 'cratea summary.json missing'
        assert crateb_summary.exists(), 'crateb summary.json missing'

        # (h) result.worktree_log_paths contains both modules' paths
        all_paths = result.worktree_log_paths
        assert any('cratea' in p for p in all_paths), (
            f'cratea not in worktree_log_paths: {all_paths}'
        )
        assert any('crateb' in p for p in all_paths), (
            f'crateb not in worktree_log_paths: {all_paths}'
        )


class TestShouldArchiveCategory:
    """Tests for ``_should_archive_category(category: str) -> bool``.

    Tests will fail with ImportError until step 4 implements the helper.
    """

    def _should_archive(self, category: str) -> bool:
        from orchestrator.verify import _should_archive_category  # noqa: PLC0415
        return _should_archive_category(category)

    # Categories that MUST be archived (human triage required)
    def test_unknown_test_failure_is_archived(self):
        """'unknown_test_failure' → True (human must look)."""
        assert self._should_archive('unknown_test_failure') is True

    def test_cargo_cli_error_is_archived(self):
        """'cargo_cli_error' ends with '_error' → True."""
        assert self._should_archive('cargo_cli_error') is True

    def test_npm_error_is_archived(self):
        """'npm_error' ends with '_error' → True."""
        assert self._should_archive('npm_error') is True

    def test_flock_error_is_archived(self):
        """'flock_error' ends with '_error' → True."""
        assert self._should_archive('flock_error') is True

    def test_tree_sitter_generate_error_is_archived(self):
        """'tree_sitter_generate_error' ends with '_error' → True."""
        assert self._should_archive('tree_sitter_generate_error') is True

    # Categories that must NOT be archived (debugger can handle without human)
    def test_test_failure_not_archived(self):
        """'test_failure' does not end with '_error' → False."""
        assert self._should_archive('test_failure') is False

    def test_compile_error_not_archived(self):
        """'compile_error' is explicitly excluded despite '_error' suffix → False."""
        assert self._should_archive('compile_error') is False

    def test_infra_timeout_not_archived(self):
        """'infra_timeout' → False."""
        assert self._should_archive('infra_timeout') is False

    def test_passed_not_archived(self):
        """'passed' → False."""
        assert self._should_archive('passed') is False

    def test_empty_string_not_archived(self):
        """'' (empty) → False."""
        assert self._should_archive('') is False


class TestBuildFallbackConfigConftest:
    """`_build_fallback_config` handles conftest.py correctly.

    The fallback path has no parent ModuleConfig, so it cannot reuse
    mc.test_command as the full suite.  Instead it uses the parent directory
    of each conftest.py as the pytest target — that directory contains every
    test the conftest can affect and pytest discovers them correctly.
    """

    def test_conftest_only_uses_directory_not_conftest_file(self):
        """conftest.py alone → pytest targets its parent directory."""
        result = _build_fallback_config(['orchestrator/tests/conftest.py'])
        assert result is not None
        assert result.test_command is not None
        assert 'conftest.py' not in result.test_command
        assert result.test_command == 'pytest orchestrator/tests'

    def test_conftest_with_test_files_uses_directory(self):
        """conftest.py mixed with test files → directory scope, no conftest.py token.

        Even when concrete test files are present, the conftest directory scope
        takes priority — conftest may affect tests outside the diff, and pytest
        discovers them correctly from the directory.
        """
        result = _build_fallback_config(
            ['orchestrator/tests/conftest.py', 'orchestrator/tests/test_foo.py'],
        )
        assert result is not None
        assert result.test_command is not None
        assert result.test_command == 'pytest orchestrator/tests'
        assert 'conftest.py' not in result.test_command

    def test_multiple_conftest_dirs_sorted_deterministically(self):
        """Multiple conftest.py files in distinct dirs → sorted unique parent dirs."""
        result = _build_fallback_config([
            'shared/tests/conftest.py',
            'orchestrator/tests/conftest.py',
        ])
        assert result is not None
        assert result.test_command is not None
        # sorted: orchestrator/tests before shared/tests
        assert result.test_command == 'pytest orchestrator/tests shared/tests'
        assert 'conftest.py' not in result.test_command

    def test_conftest_with_test_file_in_different_directory(self):
        """Test files outside the conftest directory are included alongside it.

        e.g. ['a/conftest.py', 'b/test_x.py'] → 'pytest a b/test_x.py', so
        tests in b/ are not silently skipped.
        """
        result = _build_fallback_config([
            'a/tests/conftest.py',
            'b/tests/test_x.py',
        ])
        assert result is not None
        assert result.test_command is not None
        assert 'a/tests' in result.test_command
        assert 'b/tests/test_x.py' in result.test_command
        assert 'conftest.py' not in result.test_command

    def test_root_conftest_maps_to_dot(self):
        """A conftest.py at the worktree root uses '.' as target, not the file itself."""
        result = _build_fallback_config(['conftest.py'])
        assert result is not None
        assert result.test_command is not None
        assert result.test_command == 'pytest .'
        assert 'conftest.py' not in result.test_command

    def test_nested_test_under_conftest_dir_uses_directory_only(self):
        """A test file inside the conftest's directory subtree is NOT added redundantly.

        ['a/conftest.py', 'a/sub/test_x.py'] → 'pytest a' because 'a/sub/test_x.py'
        starts with 'a/' and is therefore covered by the conftest directory target.
        Regression guard for the `t.startswith(d + '/')` boundary check.
        """
        result = _build_fallback_config(['a/conftest.py', 'a/sub/test_x.py'])
        assert result is not None
        assert result.test_command == 'pytest a'
        assert 'a/sub/test_x.py' not in result.test_command
        assert 'conftest.py' not in result.test_command

    def test_sibling_prefix_test_file_included_alongside_conftest_dir(self):
        """Directory 'a' does NOT swallow a sibling 'ab/test_x.py'.

        ['a/conftest.py', 'ab/test_x.py'] → 'pytest a ab/test_x.py' because
        'ab/test_x.py'.startswith('a/') is False — the `d + '/'` boundary check
        correctly distinguishes directory 'a' from sibling prefix 'ab'.
        Regression guard for the boundary-check contract.
        """
        result = _build_fallback_config(['a/conftest.py', 'ab/test_x.py'])
        assert result is not None
        assert result.test_command == 'pytest a ab/test_x.py'
        assert 'conftest.py' not in result.test_command


class TestPruneArchiveDedupedAtAggregateSite:
    """Tests ensuring ``_prune_archive`` is called exactly once per
    ``run_scoped_verification``, not once-per-module inside
    ``_archive_attempt_log``.

    Tests (a) and (b) fail until step 24 removes the inner ``_prune_archive``
    call from ``_archive_attempt_log`` and adds a single call at the aggregate
    site in ``run_scoped_verification``.
    """

    @pytest.fixture(autouse=True)
    def _reset_prune_throttle(self, monkeypatch):
        from orchestrator import verify  # noqa: PLC0415
        monkeypatch.setattr(verify, '_LAST_PRUNE_AT', None)

    def _make_config(self, tmp_path: Path) -> OrchestratorConfig:
        return OrchestratorConfig(
            project_root=tmp_path,
            test_command='cargo test --workspace',
            lint_command='echo ok',
            type_check_command='echo ok',
        )

    def _make_module_configs(self) -> list[ModuleConfig]:
        return [
            ModuleConfig(
                prefix='cratea',
                test_command='cargo test -p cratea',
                lint_command=None,
                type_check_command=None,
            ),
            ModuleConfig(
                prefix='crateb',
                test_command='cargo test -p crateb',
                lint_command=None,
                type_check_command=None,
            ),
        ]

    def test_archive_attempt_log_does_not_call_prune(self, tmp_path: Path):
        """(a) After step 24, _archive_attempt_log must NOT call _prune_archive.

        ``_prune_archive`` belongs at the aggregate site in
        ``run_scoped_verification``, not inside the per-file copy loop.
        """
        from orchestrator import verify  # noqa: PLC0415
        from orchestrator.verify import _archive_attempt_log  # noqa: PLC0415

        log_file = tmp_path / 'attempt-1.test.log'
        log_file.write_text('cratea ERR\n')
        archive_root = tmp_path / 'data' / 'verify-logs'

        with patch.object(verify, '_prune_archive') as spy:
            _archive_attempt_log(
                [log_file],
                archive_root,
                task_id='42',
                attempt_id=1,
                category='cargo_cli_error',
            )
            assert spy.call_count == 0, (
                f'_archive_attempt_log must not call _prune_archive; '
                f'got {spy.call_count} call(s)'
            )

    @pytest.mark.asyncio
    async def test_gather_calls_prune_exactly_once(self, tmp_path: Path):
        """(b) Two concurrent modules each archiving → _prune_archive called once total.

        Before step 24, _archive_attempt_log calls _prune_archive once per
        module, yielding 2 calls.  After step 24, the aggregate site fires once.
        """
        (tmp_path / '.task').mkdir()
        archive_root = tmp_path / 'data' / 'verify-logs'
        config = self._make_config(tmp_path)
        module_configs = self._make_module_configs()

        async def fake_run_cmd(cmd, cwd, timeout, env=None):
            if 'cratea' in cmd:
                return 1, 'error: --exclude cratea\ncratea ERR\n', False
            if 'crateb' in cmd:
                return 1, 'error: --exclude crateb\ncrateb ERR\n', False
            return 0, 'ok', False

        from orchestrator import verify  # noqa: PLC0415

        with patch('orchestrator.verify._run_cmd', side_effect=fake_run_cmd), \
                patch.object(verify, '_prune_archive') as spy:
            await run_scoped_verification(
                tmp_path, config, module_configs,
                attempt_id=1,
                task_id='42',
                archive_root=archive_root,
            )
        assert spy.call_count == 1, (
            f'_prune_archive should be called exactly once across gather; '
            f'got {spy.call_count} call(s)'
        )
        assert spy.call_args[0][0] == archive_root, (
            f'First positional arg to _prune_archive should be archive_root; '
            f'got {spy.call_args[0][0]!r}'
        )

    @pytest.mark.asyncio
    async def test_no_prune_when_archive_root_is_none(self, tmp_path: Path):
        """(c) archive_root=None → _prune_archive is never called."""
        (tmp_path / '.task').mkdir()
        config = self._make_config(tmp_path)
        module_configs = self._make_module_configs()

        async def fake_run_cmd(cmd, cwd, timeout, env=None):
            return 1, 'error: --exclude\n', False

        from orchestrator import verify  # noqa: PLC0415

        with patch('orchestrator.verify._run_cmd', side_effect=fake_run_cmd), \
                patch.object(verify, '_prune_archive') as spy:
            await run_scoped_verification(
                tmp_path, config, module_configs,
                attempt_id=1,
                task_id='42',
                archive_root=None,
            )
        assert spy.call_count == 0, (
            f'_prune_archive must not be called when archive_root is None; '
            f'got {spy.call_count} call(s)'
        )

    @pytest.mark.asyncio
    async def test_single_module_global_path_calls_prune_once(self, tmp_path: Path):
        """(d) Global path (module_configs=[], task_files=None) still calls prune once.

        After step 24, the single aggregate-site call in ``run_scoped_verification``
        fires even on the non-gather (global) fallback paths.
        """
        (tmp_path / '.task').mkdir()
        archive_root = tmp_path / 'data' / 'verify-logs'
        config = self._make_config(tmp_path)

        async def fake_run_cmd(cmd, cwd, timeout, env=None):
            return 1, 'error: --exclude\nfoo\n', False

        from orchestrator import verify  # noqa: PLC0415

        with patch('orchestrator.verify._run_cmd', side_effect=fake_run_cmd), \
                patch.object(verify, '_prune_archive') as spy:
            await run_scoped_verification(
                tmp_path, config, [],
                attempt_id=1,
                task_id='42',
                archive_root=archive_root,
            )
        assert spy.call_count == 1, (
            f'Global path should call _prune_archive exactly once; '
            f'got {spy.call_count} call(s)'
        )


class TestPruneArchiveThrottle:
    """Tests for the ``_maybe_prune_archive`` throttle wrapper.

    Each test resets ``_LAST_PRUNE_AT`` to ``None`` via an autouse fixture so
    module-level state doesn't leak between cases.
    """

    @pytest.fixture(autouse=True)
    def _reset_prune_throttle(self, monkeypatch):
        from orchestrator import verify  # noqa: PLC0415
        monkeypatch.setattr(verify, '_LAST_PRUNE_AT', None)

    def test_first_call_invokes_prune(self, tmp_path: Path):
        """First call to ``_maybe_prune_archive`` always fires ``_prune_archive``."""
        from orchestrator import verify  # noqa: PLC0415
        from orchestrator.verify import _maybe_prune_archive  # noqa: PLC0415

        archive_root = tmp_path / 'data' / 'verify-logs'

        with patch.object(verify, '_prune_archive') as spy:
            _maybe_prune_archive(archive_root)

        assert spy.call_count == 1, (
            f'First call should invoke _prune_archive once; got {spy.call_count}'
        )
        assert spy.call_args[0][0] == archive_root, (
            f'First positional arg should be archive_root; got {spy.call_args[0][0]!r}'
        )

    def test_second_call_within_window_skips_prune(self, tmp_path: Path):
        """Second immediate call is throttled — ``_prune_archive`` called only once."""
        from orchestrator import verify  # noqa: PLC0415
        from orchestrator.verify import _maybe_prune_archive  # noqa: PLC0415

        archive_root = tmp_path / 'data' / 'verify-logs'

        with patch.object(verify, '_prune_archive') as spy:
            _maybe_prune_archive(archive_root)
            _maybe_prune_archive(archive_root)

        assert spy.call_count == 1, (
            f'Second immediate call must be throttled; expected 1, got {spy.call_count}'
        )

    def test_call_after_throttle_elapsed_fires_again(self, monkeypatch, tmp_path: Path):
        """After the throttle window elapses, the next call fires ``_prune_archive``."""
        import time as time_mod  # noqa: PLC0415

        from orchestrator import verify  # noqa: PLC0415
        from orchestrator.verify import _maybe_prune_archive, _PRUNE_THROTTLE_SECS  # noqa: PLC0415

        archive_root = tmp_path / 'data' / 'verify-logs'
        base_time = 0.0

        with patch.object(verify, '_prune_archive') as spy:
            monkeypatch.setattr(time_mod, 'monotonic', lambda: base_time)
            _maybe_prune_archive(archive_root)  # first call — fires

            # Advance time past the throttle window
            elapsed = _PRUNE_THROTTLE_SECS + 1
            monkeypatch.setattr(time_mod, 'monotonic', lambda: base_time + elapsed)
            _maybe_prune_archive(archive_root)  # second call — window elapsed, fires again

        assert spy.call_count == 2, (
            f'Call after throttle elapsed should fire again; expected 2, got {spy.call_count}'
        )

    def test_third_call_within_new_window_skips(self, monkeypatch, tmp_path: Path):
        """After second fire, the window slides — an immediate third call is throttled."""
        import time as time_mod  # noqa: PLC0415

        from orchestrator import verify  # noqa: PLC0415
        from orchestrator.verify import _maybe_prune_archive, _PRUNE_THROTTLE_SECS  # noqa: PLC0415

        archive_root = tmp_path / 'data' / 'verify-logs'
        base_time = 0.0
        elapsed = _PRUNE_THROTTLE_SECS + 1

        with patch.object(verify, '_prune_archive') as spy:
            monkeypatch.setattr(time_mod, 'monotonic', lambda: base_time)
            _maybe_prune_archive(archive_root)  # first fire

            monkeypatch.setattr(time_mod, 'monotonic', lambda: base_time + elapsed)
            _maybe_prune_archive(archive_root)  # second fire (window elapsed)

            # Third call immediately after second — still at the same "elapsed" time
            _maybe_prune_archive(archive_root)  # should be throttled

        assert spy.call_count == 2, (
            f'Third call within new window must be throttled; expected 2, got {spy.call_count}'
        )

    def test_archive_root_none_no_op_does_not_update_throttle(self, tmp_path: Path):
        """None call doesn't burn the throttle — subsequent real call still fires."""
        from orchestrator import verify  # noqa: PLC0415
        from orchestrator.verify import _maybe_prune_archive  # noqa: PLC0415

        archive_root = tmp_path / 'data' / 'verify-logs'

        with patch.object(verify, '_prune_archive') as spy:
            result_none = _maybe_prune_archive(None)
            assert spy.call_count == 0, 'None call must not invoke _prune_archive'
            assert result_none is False

            result_real = _maybe_prune_archive(archive_root)
            assert spy.call_count == 1, (
                f'Real call after None must fire _prune_archive; got {spy.call_count}'
            )
            assert result_real is True

    @pytest.mark.asyncio
    async def test_run_scoped_verification_finally_uses_wrapper(self, monkeypatch, tmp_path: Path):
        """run_scoped_verification goes through the wrapper on every call; the
        wrapper throttles so _prune_archive fires only once across two calls."""
        (tmp_path / '.task').mkdir()
        archive_root = tmp_path / 'data' / 'verify-logs'
        from orchestrator import verify  # noqa: PLC0415
        from orchestrator.config import OrchestratorConfig  # noqa: PLC0415

        config = OrchestratorConfig(
            project_root=tmp_path,
            test_command='echo ok',
            lint_command='echo ok',
            type_check_command='echo ok',
        )

        async def fake_run_cmd(cmd, cwd, timeout, env=None):
            return 1, 'error: --exclude\nfoo\n', False

        wrapper_calls: list[object] = []
        original_maybe = verify._maybe_prune_archive

        def counting_wrapper(ar):
            wrapper_calls.append(ar)
            return original_maybe(ar)

        with patch('orchestrator.verify._run_cmd', side_effect=fake_run_cmd), \
                patch.object(verify, '_maybe_prune_archive', side_effect=counting_wrapper), \
                patch.object(verify, '_prune_archive') as prune_spy:
            # First call — wrapper fires _prune_archive
            await run_scoped_verification(
                tmp_path, config, [],
                attempt_id=1, task_id='42', archive_root=archive_root,
            )
            # Second call — wrapper is invoked again but _prune_archive is throttled
            await run_scoped_verification(
                tmp_path, config, [],
                attempt_id=2, task_id='42', archive_root=archive_root,
            )

        assert len(wrapper_calls) == 2, (
            f'_maybe_prune_archive should be called twice (once per run); '
            f'got {len(wrapper_calls)}'
        )
        assert prune_spy.call_count == 1, (
            f'_prune_archive should be called only once (throttled on second); '
            f'got {prune_spy.call_count}'
        )
