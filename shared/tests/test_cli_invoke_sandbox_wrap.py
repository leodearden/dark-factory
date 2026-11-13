"""Tests for the sandbox_wrap hook in shared.cli_invoke.

The hook allows callers to wrap the claude argv (e.g. with Landlock or bwrap)
immediately before the subprocess is spawned, without shared needing to know
about any specific sandbox backend.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from shared.cli_invoke import (
    AgentResult,
    SubprocessResult,
    invoke_claude_agent,
    invoke_with_cap_retry,
)


@pytest.fixture
def pinned_claude(monkeypatch, tmp_path: Path) -> str:
    """Pin CLAUDE_BINARY at a real executable and yield its absolute path.

    build_claude_argv now resolves argv[0] to an absolute path rather than
    emitting the bare name 'claude' (task 4448), so these argv assertions need
    a deterministic value that does not depend on whether the CLI happens to
    be on the CI PATH.
    """
    exec_file = tmp_path / 'claude'
    exec_file.write_text('#!/bin/sh\nexit 0\n')
    exec_file.chmod(0o755)
    monkeypatch.setenv('CLAUDE_BINARY', str(exec_file))
    return str(exec_file)


def _make_subprocess_result(**overrides: Any) -> SubprocessResult:
    """Build a minimal SubprocessResult that parse_claude_output can parse."""
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
    return SubprocessResult(**defaults)


@pytest.mark.asyncio
async def test_sandbox_wrap_reaches_the_runner_as_wrap_argv(
    tmp_path: Path, pinned_claude: str,
) -> None:
    """When sandbox_wrap is set, the runner gets it as wrap_argv with the unwrapped argv.

    The runner applies the wrap once, before spawn (pinned in
    test_cli_runner.py), so _invoke_claude must hand it over untouched rather
    than pre-wrapping the claude argv itself.
    """
    captured: list[tuple[list[str], dict[str, Any]]] = []

    async def fake_runner(cmd: list[str], cwd: Path, env: dict, **kwargs: Any) -> SubprocessResult:
        captured.append((list(cmd), kwargs))
        return _make_subprocess_result()

    def wrap(cmd: list[str]) -> list[str]:
        return ['WRAP', *cmd]

    await invoke_claude_agent(
        prompt='hello',
        system_prompt='sys',
        cwd=tmp_path,
        sandbox_wrap=wrap,
        runner=fake_runner,
    )

    assert len(captured) == 1, 'Expected exactly one runner invocation'
    argv, kwargs = captured[0]
    assert argv[0] == pinned_claude, f'Expected the unwrapped claude argv; got {argv[:3]}'
    assert kwargs['wrap_argv'] is wrap


@pytest.mark.asyncio
async def test_sandbox_wrap_none_reaches_the_runner_as_none(
    tmp_path: Path, pinned_claude: str,
) -> None:
    """When sandbox_wrap is None (default), the runner gets wrap_argv=None.

    The first token in the captured argv must be the resolved claude binary,
    confirming no accidental transformation — in particular that NO sandbox
    wrapper was prepended.
    """
    captured: list[tuple[list[str], dict[str, Any]]] = []

    async def fake_runner(cmd: list[str], cwd: Path, env: dict, **kwargs: Any) -> SubprocessResult:
        captured.append((list(cmd), kwargs))
        return _make_subprocess_result()

    await invoke_claude_agent(
        prompt='hello',
        system_prompt='sys',
        cwd=tmp_path,
        runner=fake_runner,
    )

    assert len(captured) == 1
    argv, kwargs = captured[0]
    assert argv[0] == pinned_claude, (
        f'Expected unchanged argv to start with the claude binary; got {argv[:3]}'
    )
    assert kwargs['wrap_argv'] is None


# ── invoke_with_cap_retry forwarding tests ────────────────────────────────────
# These tests confirm that sandbox_wrap survives passage through the
# **invoke_kwargs mechanism in invoke_with_cap_retry.  A regression here
# (e.g. invoke_kwargs filtering that strips unknown kwargs) would silently run
# agents unconfined with no failing test in the mocked wiring tests above.


def _make_agent_result() -> AgentResult:
    """Minimal successful AgentResult for stub invoke_fn."""
    return AgentResult(
        success=True,
        output='ok',
        cost_usd=0.0,
        stderr='',
        structured_output=None,
    )


@pytest.mark.asyncio
async def test_invoke_with_cap_retry_forwards_sandbox_wrap_no_gate(tmp_path: Path) -> None:
    """sandbox_wrap kwarg reaches the invoke_fn in the no-gate (fast-path) branch.

    When usage_gate=None, invoke_with_cap_retry runs a single direct invocation
    without entering the cap-retry loop.  This test verifies sandbox_wrap still
    arrives at the underlying callable.
    """
    received_kwargs: dict = {}

    async def stub_invoke(**kwargs) -> AgentResult:
        received_kwargs.update(kwargs)
        return _make_agent_result()

    sentinel_wrap = lambda cmd: ['SENTINEL'] + cmd  # noqa: E731

    await invoke_with_cap_retry(
        usage_gate=None,
        label='test-no-gate',
        invoke_fn=stub_invoke,
        prompt='hello',
        system_prompt='sys',
        cwd=tmp_path,
        sandbox_wrap=sentinel_wrap,
    )

    assert 'sandbox_wrap' in received_kwargs, (
        f'sandbox_wrap missing from invoke_fn kwargs (no-gate path); got {list(received_kwargs)}'
    )
    assert received_kwargs['sandbox_wrap'] is sentinel_wrap, (
        f'Expected sentinel_wrap; got {received_kwargs["sandbox_wrap"]!r}'
    )


@pytest.mark.asyncio
async def test_invoke_with_cap_retry_forwards_sandbox_wrap_with_gate(tmp_path: Path) -> None:
    """sandbox_wrap kwarg reaches the invoke_fn in the usage-gate (retry-loop) branch.

    When a UsageGate is supplied, invoke_with_cap_retry enters the cap-retry
    loop.  This test verifies sandbox_wrap is still forwarded to the invoke_fn
    in that path so that the Landlock wrap is not silently dropped on gated calls.
    """
    import contextlib
    from unittest.mock import MagicMock

    received_kwargs: dict = {}

    async def stub_invoke(**kwargs) -> AgentResult:
        received_kwargs.update(kwargs)
        return _make_agent_result()

    sentinel_wrap = lambda cmd: ['WRAPPED'] + cmd  # noqa: E731

    # Build a minimal UsageGate stub: usage_gate.invoke_slot() must be an async
    # context manager that yields a slot with .account_name and .token attributes.
    slot = MagicMock()
    slot.account_name = 'test-account'
    slot.token = None
    # detect_cap_hit must return False so the retry loop exits after one iteration.
    slot.detect_cap_hit = MagicMock(return_value=False)

    class _FakeGate:
        account_count = 1
        soonest_resets_at = None

        @contextlib.asynccontextmanager
        async def invoke_slot(self, scope=None, *, park=True, max_wait_secs=None):
            yield slot

    await invoke_with_cap_retry(
        usage_gate=_FakeGate(),  # type: ignore[arg-type]
        label='test-with-gate',
        invoke_fn=stub_invoke,
        prompt='hello',
        system_prompt='sys',
        cwd=tmp_path,
        sandbox_wrap=sentinel_wrap,
    )

    assert 'sandbox_wrap' in received_kwargs, (
        f'sandbox_wrap missing from invoke_fn kwargs (with-gate path); got {list(received_kwargs)}'
    )
    assert received_kwargs['sandbox_wrap'] is sentinel_wrap, (
        f'Expected sentinel_wrap; got {received_kwargs["sandbox_wrap"]!r}'
    )
