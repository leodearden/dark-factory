"""Tests for bwrap sandbox probe and fallback."""

import subprocess
from unittest.mock import AsyncMock, patch

import pytest

from orchestrator.agents.sandbox import _reset_probe, is_bwrap_available


@pytest.fixture(autouse=True)
def _reset():
    """Reset cached probe before and after each test."""
    _reset_probe()
    yield
    _reset_probe()


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
    async def test_runs_unsandboxed_when_no_backend_available(self, tmp_path):
        """When every sandbox backend is unavailable, invoke_agent falls
        through to the shared (unsandboxed) invocation path."""
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
            assert call_args[0] == 'claude'
