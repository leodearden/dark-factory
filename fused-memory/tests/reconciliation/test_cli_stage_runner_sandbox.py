"""Tests for sandbox confinement wiring in cli_stage_runner.run_stage_via_cli (task 1935).

Verifies:
  (a) confinement on → invoke_with_cap_retry receives sandbox_wrap= the sentinel callable
  (b) fail-closed: RemediationSandboxUnavailable → StageResult.success=False, error set,
      and invoke_with_cap_retry NOT called (agent not launched unconfined)
  (c) confinement off (sandbox_recon_agents=False) → sandbox_wrap=None passed (or absent)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from shared.cli_invoke import AgentResult

from fused_memory.config.schema import ReconciliationConfig
from fused_memory.reconciliation.cli_stage_runner import run_stage_via_cli
from fused_memory.reconciliation.sandbox_guard import RemediationSandboxUnavailable


def _make_agent_result(**overrides) -> AgentResult:
    """Build a successful AgentResult for mocking invoke_with_cap_retry."""
    defaults = {
        'success': True,
        'output': '{"summary": "ok"}',
        'cost_usd': 0.01,
        'stderr': '',
        'structured_output': {'summary': 'ok', 'flagged_items': []},
    }
    defaults.update(overrides)
    return AgentResult(**defaults)


def _make_config(**overrides) -> ReconciliationConfig:
    """Build a minimal ReconciliationConfig for testing."""
    defaults: dict = {
        'sandbox_recon_agents': True,
        'sandbox_recon_writable_extras': [],
    }
    defaults.update(overrides)
    return ReconciliationConfig(**defaults)


def _make_mcp_config() -> dict:
    return {'mcpServers': {}}


@pytest.mark.asyncio
async def test_confinement_on_passes_wrap_to_invoke(tmp_path):
    """When sandbox_recon_agents=True, invoke_with_cap_retry receives sandbox_wrap= sentinel.

    Confirms the wrap callable produced by resolve_recon_sandbox_wrap is
    forwarded as sandbox_wrap= kwarg into invoke_with_cap_retry (which then
    passes it through its **invoke_kwargs to invoke_claude_agent).
    """
    sentinel_wrap = lambda cmd: ['SENTINEL'] + cmd  # noqa: E731

    mock_invoke = AsyncMock(return_value=_make_agent_result())

    with patch(
        'fused_memory.reconciliation.cli_stage_runner.invoke_with_cap_retry',
        mock_invoke,
    ), patch(
        'fused_memory.reconciliation.cli_stage_runner.resolve_recon_sandbox_wrap',
        return_value=sentinel_wrap,
    ) as mock_resolve:
        config = _make_config(sandbox_recon_agents=True)
        await run_stage_via_cli(
            system_prompt='sys',
            payload='pay',
            disallowed_tools=[],
            config=config,
            mcp_config=_make_mcp_config(),
            cwd=tmp_path,
        )

    # resolve_recon_sandbox_wrap should have been called
    assert mock_resolve.called, 'Expected resolve_recon_sandbox_wrap to be called'

    # invoke_with_cap_retry should have been called with sandbox_wrap= sentinel
    assert mock_invoke.called, 'Expected invoke_with_cap_retry to be called'
    _, call_kwargs = mock_invoke.call_args
    assert 'sandbox_wrap' in call_kwargs, (
        f'Expected sandbox_wrap in invoke kwargs; got keys={list(call_kwargs.keys())}'
    )
    assert call_kwargs['sandbox_wrap'] is sentinel_wrap, (
        f'Expected sentinel_wrap to be forwarded; got {call_kwargs["sandbox_wrap"]}'
    )


@pytest.mark.asyncio
async def test_fail_closed_not_launched(tmp_path):
    """When resolve_recon_sandbox_wrap raises RemediationSandboxUnavailable,
    the StageResult carries an error AND invoke_with_cap_retry is NOT called.

    This is the explicit fail-closed leaf: the agent must not be launched
    unconfined when confinement is enabled but no backend is available.
    """
    mock_invoke = AsyncMock(return_value=_make_agent_result())

    with patch(
        'fused_memory.reconciliation.cli_stage_runner.invoke_with_cap_retry',
        mock_invoke,
    ), patch(
        'fused_memory.reconciliation.cli_stage_runner.resolve_recon_sandbox_wrap',
        side_effect=RemediationSandboxUnavailable('no sandbox'),
    ):
        config = _make_config(sandbox_recon_agents=True)
        result = await run_stage_via_cli(
            system_prompt='sys',
            payload='pay',
            disallowed_tools=[],
            config=config,
            mcp_config=_make_mcp_config(),
            cwd=tmp_path,
        )

    # Agent must NOT have been launched
    assert not mock_invoke.called, (
        'invoke_with_cap_retry must NOT be called when confinement fails (fail-closed)'
    )
    # StageResult must carry an error
    assert result.success is False, 'StageResult.success must be False on fail-closed'
    assert result.error, f'StageResult.error must be non-empty; got {result.error!r}'


@pytest.mark.asyncio
async def test_confinement_off_no_wrap(tmp_path):
    """When sandbox_recon_agents=False, invoke_with_cap_retry is called with sandbox_wrap=None.

    Operators who explicitly opt out of confinement should get unconfined
    invocations (sandbox_wrap=None, which cli_invoke treats as no-wrap).
    """
    mock_invoke = AsyncMock(return_value=_make_agent_result())

    with patch(
        'fused_memory.reconciliation.cli_stage_runner.invoke_with_cap_retry',
        mock_invoke,
    ), patch(
        'fused_memory.reconciliation.cli_stage_runner.resolve_recon_sandbox_wrap',
    ) as mock_resolve:
        config = _make_config(sandbox_recon_agents=False)
        await run_stage_via_cli(
            system_prompt='sys',
            payload='pay',
            disallowed_tools=[],
            config=config,
            mcp_config=_make_mcp_config(),
            cwd=tmp_path,
        )

    # resolve_recon_sandbox_wrap should NOT be called when confinement is off
    assert not mock_resolve.called, (
        'resolve_recon_sandbox_wrap must not be called when sandbox_recon_agents=False'
    )

    # invoke_with_cap_retry should have been called
    assert mock_invoke.called, 'Expected invoke_with_cap_retry to be called'
    _, call_kwargs = mock_invoke.call_args
    # sandbox_wrap should be absent or None
    sandbox_wrap_val = call_kwargs.get('sandbox_wrap')
    assert sandbox_wrap_val is None, (
        f'Expected sandbox_wrap=None when confinement is off; got {sandbox_wrap_val!r}'
    )
