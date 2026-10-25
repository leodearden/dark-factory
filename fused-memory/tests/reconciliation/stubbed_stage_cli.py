"""Drive ``run_stage_via_cli`` with its CLI invocation stubbed (task 5982).

Shared by the startup-grace forwarding guard and the agent-outcome
classification guard: each needs only what ``run_stage_via_cli`` hands to
``invoke_with_cap_retry``, or what it makes of the ``AgentResult`` it receives.
A plain non-test module imported as ``from reconciliation.stubbed_stage_cli
import …``, after the ``payload_section_parity.py`` precedent.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

from shared.cli_invoke import AgentResult

from fused_memory.config.schema import ReconciliationConfig
from fused_memory.reconciliation.cli_stage_runner import StageResult, run_stage_via_cli

INVOKE_WITH_CAP_RETRY = 'fused_memory.reconciliation.cli_stage_runner.invoke_with_cap_retry'


async def run_stage(config: ReconciliationConfig, cwd: Path) -> StageResult:
    return await run_stage_via_cli(
        system_prompt='sys',
        payload='pay',
        disallowed_tools=[],
        config=config,
        mcp_config={'mcpServers': {}},
        cwd=cwd,
    )


async def run_stage_answered_by(
    agent_result: AgentResult,
    cwd: Path,
    config: ReconciliationConfig | None = None,
) -> tuple[StageResult, AsyncMock]:
    invoke = AsyncMock(return_value=agent_result)
    effective_config = (
        config if config is not None else ReconciliationConfig(sandbox_recon_agents=False)
    )
    with patch(INVOKE_WITH_CAP_RETRY, invoke):
        result = await run_stage(effective_config, cwd)
    return result, invoke
