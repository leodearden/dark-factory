"""Unit tests for orchestrator.verify — _run_cmd and run_verification."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
