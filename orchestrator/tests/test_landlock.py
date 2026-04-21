"""Tests for landlock sandbox backend and backend dispatcher."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from orchestrator.agents import landlock as landlock_mod
from orchestrator.agents.landlock import (
    _reset_probe as _landlock_reset_probe,
    build_landlock_command,
    is_landlock_available,
)


@pytest.fixture(autouse=True)
def _reset_landlock_probe():
    """Reset cached probe before and after each test."""
    _landlock_reset_probe()
    yield
    _landlock_reset_probe()


# ---------------------------------------------------------------------------
# Test 1: is_landlock_available probe semantics
# ---------------------------------------------------------------------------

class TestIsLandlockAvailable:
    def test_returns_true_when_abi_positive(self):
        with patch.object(landlock_mod, '_syscall_probe_abi', return_value=7):
            assert is_landlock_available() is True

    def test_returns_false_when_abi_negative(self):
        with patch.object(landlock_mod, '_syscall_probe_abi', return_value=-1):
            assert is_landlock_available() is False

    def test_returns_false_when_abi_zero(self):
        with patch.object(landlock_mod, '_syscall_probe_abi', return_value=0):
            assert is_landlock_available() is False

    def test_caches_result(self):
        with patch.object(landlock_mod, '_syscall_probe_abi', return_value=7) as mock_probe:
            assert is_landlock_available() is True
            assert is_landlock_available() is True
            mock_probe.assert_called_once()


# ---------------------------------------------------------------------------
# Test 2: build_landlock_command argv shape
# ---------------------------------------------------------------------------

class TestBuildLandlockCommand:
    def test_wrapper_argv_shape(self, tmp_path: Path):
        worktree = tmp_path
        (worktree / 'mod_a').mkdir()
        (worktree / 'mod_b').mkdir()

        inner = ['claude', '--print', '--model', 'haiku']
        cmd = build_landlock_command(
            inner, worktree, ['mod_a', 'mod_b'], writable_extras=['/var/tmp/extra'],
        )

        assert cmd[0] == sys.executable
        # Wrapper script path
        assert cmd[1].endswith('landlock_exec.py')
        # Contains the per-module writable flags
        assert '--writable' in cmd
        w_idx = [i for i, x in enumerate(cmd) if x == '--writable']
        w_values = {cmd[i + 1] for i in w_idx}
        assert str((worktree / 'mod_a').resolve()) in w_values
        assert str((worktree / 'mod_b').resolve()) in w_values
        assert '/var/tmp/extra' in w_values
        # .task is always writable
        assert str((worktree / '.task').resolve()) in w_values
        # Terminates with `--` then the inner command
        assert '--' in cmd
        dash_idx = cmd.index('--')
        assert cmd[dash_idx + 1:] == inner

    def test_creates_module_and_task_dirs(self, tmp_path: Path):
        worktree = tmp_path
        assert not (worktree / 'mod_a').exists()
        assert not (worktree / '.task').exists()

        build_landlock_command(['claude'], worktree, ['mod_a'])

        assert (worktree / 'mod_a').is_dir()
        assert (worktree / '.task').is_dir()


# ---------------------------------------------------------------------------
# Test 3: integration — wrapper actually enforces the sandbox
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not is_landlock_available(),
    reason='landlock not supported on this kernel',
)
class TestLandlockEnforcement:
    def test_allowed_and_denied_writes(self, tmp_path: Path):
        worktree = tmp_path / 'wt'
        worktree.mkdir()
        (worktree / 'mod_a').mkdir()
        (worktree / 'mod_b').mkdir()  # sibling — must be denied
        denied_parent = tmp_path / 'outside'
        denied_parent.mkdir()

        inner = [
            '/bin/sh', '-c',
            (
                f'touch {worktree}/mod_a/ok_allowed && '
                f'(touch {worktree}/mod_b/bad_sibling 2>/dev/null || echo sibling_denied) && '
                f'(touch {denied_parent}/bad_outside 2>/dev/null || echo outside_denied) && '
                'echo end'
            ),
        ]
        cmd = build_landlock_command(inner, worktree, ['mod_a'])
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)

        assert result.returncode == 0, f'stderr={result.stderr}'
        assert (worktree / 'mod_a' / 'ok_allowed').exists()
        assert not (worktree / 'mod_b' / 'bad_sibling').exists()
        assert not (denied_parent / 'bad_outside').exists()
        assert 'sibling_denied' in result.stdout
        assert 'outside_denied' in result.stdout
        assert 'end' in result.stdout


# ---------------------------------------------------------------------------
# Test 4: invoke.py dispatches to the correct backend
# ---------------------------------------------------------------------------

class TestInvokeBackendDispatch:
    @pytest.mark.asyncio
    @pytest.mark.parametrize('backend,expect_bwrap,expect_landlock', [
        ('bwrap', True, False),
        ('landlock', False, True),
        ('none', False, False),
    ])
    async def test_dispatches_by_backend(
        self, tmp_path: Path, backend: str, expect_bwrap: bool, expect_landlock: bool,
    ):
        from orchestrator.agents import sandbox_dispatch

        # Set the backend preference (test fixture would normally reset this)
        sandbox_dispatch.set_backend(backend)
        try:
            with patch(
                'orchestrator.agents.sandbox.is_bwrap_available', return_value=True,
            ), patch(
                'orchestrator.agents.sandbox.build_bwrap_command',
                return_value=['bwrap-wrapped', 'claude'],
            ) as mock_bwrap, patch(
                'orchestrator.agents.landlock.is_landlock_available', return_value=True,
            ), patch(
                'orchestrator.agents.landlock.build_landlock_command',
                return_value=['landlock-wrapped', 'claude'],
            ) as mock_landlock, patch(
                'asyncio.create_subprocess_exec', new_callable=AsyncMock,
            ) as mock_exec:
                mock_proc = AsyncMock()
                mock_proc.communicate.return_value = (b'{"result": "ok"}', b'')
                mock_proc.returncode = 0
                mock_exec.return_value = mock_proc

                from orchestrator.agents.invoke import invoke_agent
                await invoke_agent(
                    prompt='test', system_prompt='sys', cwd=tmp_path,
                    sandbox_modules=['mod_a'],
                )

                assert mock_bwrap.called == expect_bwrap
                assert mock_landlock.called == expect_landlock
        finally:
            sandbox_dispatch.set_backend('auto')


# ---------------------------------------------------------------------------
# Test 5: SandboxConfig has a backend field
# ---------------------------------------------------------------------------

class TestSandboxConfigBackendField:
    def test_default_backend_is_auto(self):
        from orchestrator.config import SandboxConfig
        cfg = SandboxConfig()
        assert cfg.backend == 'auto'

    def test_accepts_landlock(self):
        from orchestrator.config import SandboxConfig
        cfg = SandboxConfig(backend='landlock')
        assert cfg.backend == 'landlock'

    def test_accepts_bwrap(self):
        from orchestrator.config import SandboxConfig
        cfg = SandboxConfig(backend='bwrap')
        assert cfg.backend == 'bwrap'

    def test_accepts_none(self):
        from orchestrator.config import SandboxConfig
        cfg = SandboxConfig(backend='none')
        assert cfg.backend == 'none'

    def test_rejects_garbage(self):
        from orchestrator.config import SandboxConfig
        with pytest.raises(Exception):  # pydantic ValidationError
            SandboxConfig(backend='seccomp-magic')  # type: ignore[arg-type]
