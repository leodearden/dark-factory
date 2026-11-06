"""Recon stage agents get their own startup-wedge grace (task 5982).

Before this, ``run_stage_via_cli`` passed no ``startup_grace_secs``, so every
recon stage rode shared's generic 120s default and Stage 2, whose first turn
tails to >=111s, was routinely killed before it produced a turn. The basis for
the configured default is plans/recon-stage-startup-wedge-rca-2026-09-27.md.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from shared.cli_invoke import AgentResult

from fused_memory.config.schema import ReconciliationConfig
from reconciliation.stubbed_stage_cli import run_stage_answered_by

_SHARED_DEFAULT_STARTUP_GRACE_SECS = 120.0


async def _forwarded_kwargs(config: ReconciliationConfig, cwd: Path) -> Mapping[str, Any]:
    _, invoke = await run_stage_answered_by(
        AgentResult(success=True, output=''), cwd, config,
    )
    return invoke.call_args.kwargs


class TestStageStartupGraceConfig:
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
    async def test_configured_grace_reaches_the_cli_invocation(self, tmp_path):
        config = ReconciliationConfig(
            stage_startup_grace_secs=333.0, sandbox_recon_agents=False,
        )

        kwargs = await _forwarded_kwargs(config, tmp_path)

        assert kwargs.get('startup_grace_secs') == 333.0

    @pytest.mark.asyncio
    async def test_default_grace_no_longer_rides_the_shared_default(self, tmp_path):
        config = ReconciliationConfig(sandbox_recon_agents=False)

        kwargs = await _forwarded_kwargs(config, tmp_path)

        assert kwargs.get('startup_grace_secs') == config.stage_startup_grace_secs
        assert kwargs['startup_grace_secs'] > _SHARED_DEFAULT_STARTUP_GRACE_SECS
