"""Tests for the sandbox_wrap hook in shared.cli_invoke.

The hook allows callers to wrap the claude argv (e.g. with Landlock or bwrap)
immediately before the subprocess is spawned, without shared needing to know
about any specific sandbox backend.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from shared.cli_invoke import (
    _SubprocessResult,
    invoke_claude_agent,
)


def _make_subprocess_result(**overrides: Any) -> _SubprocessResult:
    """Build a minimal _SubprocessResult that _parse_claude_output can parse."""
    stdout = json.dumps({
        'result': 'ok',
        'is_error': False,
        'subtype': 'success',
        'cost_usd': 0.01,
        'duration_ms': 100,
        'num_turns': 1,
        'session_id': 'test-session',
    })
    defaults: dict[str, Any] = {
        'stdout': stdout,
        'stderr': '',
        'returncode': 0,
        'duration_ms': 100,
        'timed_out': False,
    }
    defaults.update(overrides)
    return _SubprocessResult(**defaults)


@pytest.mark.asyncio
async def test_sandbox_wrap_applied_to_argv(tmp_path: Path) -> None:
    """When sandbox_wrap is set, _invoke_claude applies it to cmd before spawning.

    The wrapped argv should start with the wrap token ('WRAP') but still
    contain 'claude' as the wrapped command.  Confirms the hook fires between
    temp-file creation and _run_subprocess, as designed.
    """
    captured: list[list[str]] = []

    async def fake_run_subprocess(cmd: list[str], *args: Any, **kwargs: Any) -> _SubprocessResult:
        captured.append(list(cmd))
        return _make_subprocess_result()

    with patch('shared.cli_invoke._run_subprocess', side_effect=fake_run_subprocess):
        await invoke_claude_agent(
            prompt='hello',
            system_prompt='sys',
            cwd=tmp_path,
            sandbox_wrap=lambda cmd: ['WRAP'] + cmd,
        )

    assert len(captured) == 1, 'Expected exactly one subprocess invocation'
    argv = captured[0]
    assert argv[0] == 'WRAP', f'Expected wrapped argv to start with WRAP; got {argv[:3]}'
    assert 'claude' in argv, f'Expected wrapped argv to contain claude; got {argv[:5]}'


@pytest.mark.asyncio
async def test_sandbox_wrap_none_leaves_argv_unchanged(tmp_path: Path) -> None:
    """When sandbox_wrap is None (default), cmd is passed to _run_subprocess unchanged.

    The first token in the captured argv must be 'claude', confirming no
    accidental transformation.
    """
    captured: list[list[str]] = []

    async def fake_run_subprocess(cmd: list[str], *args: Any, **kwargs: Any) -> _SubprocessResult:
        captured.append(list(cmd))
        return _make_subprocess_result()

    with patch('shared.cli_invoke._run_subprocess', side_effect=fake_run_subprocess):
        await invoke_claude_agent(
            prompt='hello',
            system_prompt='sys',
            cwd=tmp_path,
            # sandbox_wrap intentionally omitted (defaults to None)
        )

    assert len(captured) == 1
    argv = captured[0]
    assert argv[0] == 'claude', f'Expected unchanged argv to start with claude; got {argv[:3]}'
