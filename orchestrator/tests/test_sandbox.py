"""Tests for bwrap sandbox probe and fallback."""

import logging
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from orchestrator.agents.sandbox import _reset_probe, build_bwrap_command, is_bwrap_available


@pytest.fixture(autouse=True)
def _reset():
    """Reset cached probe before and after each test."""
    _reset_probe()
    yield
    _reset_probe()


def _bind_pairs(cmd: list[str]) -> list[tuple[str, str]]:
    return [(cmd[i + 1], cmd[i + 2]) for i, arg in enumerate(cmd) if arg == '--bind']


class TestBuildBwrapCommandExtras:
    """argv shape only — no bwrap binary is needed."""

    def test_existing_file_extra_is_bound(self, tmp_path: Path):
        granted = tmp_path / 'granted.md'
        granted.write_text('x')

        cmd = build_bwrap_command(['true'], tmp_path / 'wt', [], writable_extras=[str(granted)])

        assert (str(granted), str(granted)) in _bind_pairs(cmd)

    def test_existing_dir_extra_is_bound(self, tmp_path: Path):
        extra = tmp_path / 'extra'
        extra.mkdir()

        cmd = build_bwrap_command(['true'], tmp_path / 'wt', [], writable_extras=[str(extra)])

        assert (str(extra), str(extra)) in _bind_pairs(cmd)

    def test_missing_extra_is_skipped_with_warning(self, tmp_path: Path, caplog):
        missing = tmp_path / 'absent.md'

        with caplog.at_level(logging.WARNING, logger='orchestrator.agents.sandbox'):
            cmd = build_bwrap_command(['true'], tmp_path / 'wt', [], writable_extras=[str(missing)])

        assert str(missing) not in cmd
        assert any(str(missing) in r.getMessage() for r in caplog.records)

    def test_extras_follow_task_bind_in_given_order_before_separator(self, tmp_path: Path):
        wt = tmp_path / 'wt'
        a_file = tmp_path / 'a.md'
        a_file.write_text('a')
        b_dir = tmp_path / 'b'
        b_dir.mkdir()
        c_file = tmp_path / 'c.md'
        c_file.write_text('c')
        extras = [str(c_file), str(b_dir), str(a_file)]

        cmd = build_bwrap_command(['true'], wt, [], writable_extras=extras)

        task_dir = str((wt / '.task').resolve())
        assert [src for src, _ in _bind_pairs(cmd)] == [task_dir, *extras]
        assert cmd.index('--') > max(cmd.index(e) for e in extras)


class TestIsBwrapAvailable:
    def test_returns_false_when_not_in_path(self):
        with patch('orchestrator.agents.sandbox.shutil.which', return_value=None):
            assert is_bwrap_available() is False

    def test_returns_false_on_nonzero_exit(self):
        with patch('orchestrator.agents.sandbox.shutil.which', return_value='/usr/bin/bwrap'), \
             patch('orchestrator.agents.sandbox.subprocess.run') as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=1, stderr=b'bwrap: setting up uid map: Permission denied',
            )
            assert is_bwrap_available() is False

    def test_returns_true_on_success(self):
        with patch('orchestrator.agents.sandbox.shutil.which', return_value='/usr/bin/bwrap'), \
             patch('orchestrator.agents.sandbox.subprocess.run') as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stderr=b'',
            )
            assert is_bwrap_available() is True

    def test_returns_false_on_timeout(self):
        with patch('orchestrator.agents.sandbox.shutil.which', return_value='/usr/bin/bwrap'), \
             patch('orchestrator.agents.sandbox.subprocess.run',
                   side_effect=subprocess.TimeoutExpired(cmd='bwrap', timeout=5)):
            assert is_bwrap_available() is False

    def test_returns_false_on_oserror(self):
        with patch('orchestrator.agents.sandbox.shutil.which', return_value='/usr/bin/bwrap'), \
             patch('orchestrator.agents.sandbox.subprocess.run',
                   side_effect=OSError('exec format error')):
            assert is_bwrap_available() is False

    def test_caches_result(self):
        with patch('orchestrator.agents.sandbox.shutil.which', return_value='/usr/bin/bwrap'), \
             patch('orchestrator.agents.sandbox.subprocess.run') as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stderr=b'',
            )
            assert is_bwrap_available() is True
            assert is_bwrap_available() is True
            # subprocess.run called only once despite two calls
            mock_run.assert_called_once()


class TestInvokeAgentSandboxFallback:
    @pytest.mark.asyncio
    async def test_runs_unsandboxed_when_no_backend_available(self, tmp_path, monkeypatch):
        """When every sandbox backend is unavailable, invoke_agent falls
        through to the shared (unsandboxed) invocation path."""
        claude = tmp_path / 'claude'
        claude.write_text('#!/bin/sh\nexit 0\n')
        claude.chmod(0o755)
        monkeypatch.setenv('CLAUDE_BINARY', str(claude))
        # Root conftest's _restore_sandbox_backend autouse fixture restores
        # the prior backend after this test, so no try/finally needed here.
        from orchestrator.agents import sandbox_dispatch
        sandbox_dispatch.set_backend('auto')
        with patch('orchestrator.agents.sandbox.is_bwrap_available', return_value=False), \
             patch('orchestrator.agents.landlock.is_landlock_available', return_value=False), \
             patch('asyncio.create_subprocess_exec', new_callable=AsyncMock) as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate.return_value = (b'{"result": "ok"}', b'')
            mock_proc.returncode = 0
            mock_exec.return_value = mock_proc

            from orchestrator.agents.invoke import invoke_agent
            await invoke_agent(
                prompt='test',
                system_prompt='sys',
                cwd=tmp_path,
                sandbox_modules=['mod_a'],
            )
            call_args = mock_exec.call_args[0]
            assert call_args[0] == str(claude)
