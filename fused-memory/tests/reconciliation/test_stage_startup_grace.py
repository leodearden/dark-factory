"""Recon stage agents get their own startup-wedge grace (task 5982).

Before this, ``run_stage_via_cli`` passed no ``startup_grace_secs``, so every
recon stage rode shared's generic 120s default and Stage 2, whose first turn
tails to >=111s, was routinely killed before it produced a turn. The basis for
the 450s default is plans/recon-stage-startup-wedge-rca-2026-09-27.md.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from fused_memory.config.schema import ReconciliationConfig
from fused_memory.reconciliation.cli_stage_runner import run_stage_via_cli

_SHARED_DEFAULT_STARTUP_GRACE_SECS = 120.0


def _make_capture_invoke(captured_kwargs: dict):
    async def _capture_invoke(**kwargs):
        captured_kwargs.update(kwargs)
        result = MagicMock()
        result.structured_output = None
        result.output = None
        result.turns = 0
        result.cost_usd = 0.0
        result.success = True
        return result

    return _capture_invoke


async def _forwarded_kwargs(config: ReconciliationConfig) -> dict:
    captured_kwargs: dict = {}
    with patch(
        'fused_memory.reconciliation.cli_stage_runner.invoke_with_cap_retry',
        new=_make_capture_invoke(captured_kwargs),
    ):
        await run_stage_via_cli(
            system_prompt='test',
            payload='test payload',
            disallowed_tools=[],
            config=config,
            mcp_config={'mcpServers': {}},
        )
    return captured_kwargs


class TestStageStartupGraceConfig:
    def test_default_matches_the_hosts_orchestrator_ruling(self):
        assert ReconciliationConfig().stage_startup_grace_secs == 450.0

    def test_grace_past_the_stage_ceiling_is_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            ReconciliationConfig(stage_startup_grace_secs=4000, stage_timeout_seconds=3600)
        message = str(exc_info.value)
        assert 'stage_startup_grace_secs' in message
        assert 'stage_timeout_seconds' in message

    def test_zero_grace_is_rejected(self):
        with pytest.raises(ValidationError):
            ReconciliationConfig(stage_startup_grace_secs=0)


class TestRunStageViaCliForwardsStartupGrace:
    @pytest.mark.asyncio
    async def test_configured_grace_reaches_the_cli_invocation(self):
        config = ReconciliationConfig(
            stage_startup_grace_secs=333.0, sandbox_recon_agents=False,
        )

        kwargs = await _forwarded_kwargs(config)

        assert kwargs.get('startup_grace_secs') == 333.0

    @pytest.mark.asyncio
    async def test_default_grace_no_longer_rides_the_shared_default(self):
        config = ReconciliationConfig(sandbox_recon_agents=False)

        kwargs = await _forwarded_kwargs(config)

        assert kwargs.get('startup_grace_secs') == config.stage_startup_grace_secs
        assert kwargs['startup_grace_secs'] > _SHARED_DEFAULT_STARTUP_GRACE_SECS
