"""A recon stage's one CLI invocation lands one invocations-ledger row (task 10414)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from _invocation_ledger import invocation_rows
from shared.cli_invoke import AgentResult
from shared.cost_store import CostStore

from fused_memory.config.schema import ReconciliationConfig
from fused_memory.models.invocation_ledger import InvocationLedger, LedgerRole
from fused_memory.models.reconciliation import StageAgentOutcome
from fused_memory.reconciliation.cli_stage_runner import StageResult, run_stage_via_cli
from reconciliation.stubbed_stage_cli import run_stage_answered_by

pytestmark = pytest.mark.asyncio

_CONFIG = ReconciliationConfig(sandbox_recon_agents=False)
_TOKENS = {
    'input_tokens': 1200,
    'output_tokens': 300,
    'cache_read_tokens': 50000,
    'cache_create_tokens': 4000,
}


async def _run_with_ledger(
    tmp_path: Path, served: AgentResult,
) -> tuple[StageResult, list[dict]]:
    db_path = tmp_path / 'curator_events.db'
    async with CostStore(db_path) as store:
        ledger = InvocationLedger(
            store, LedgerRole.RECON_STAGE_TASK_KNOWLEDGE_SYNC, 'run-s', 'proj-s',
        )
        with patch('shared.cli_invoke.invoke_claude_agent', new=AsyncMock(return_value=served)):
            result = await run_stage_via_cli(
                system_prompt='sys',
                payload='pay',
                disallowed_tools=[],
                config=_CONFIG,
                mcp_config={'mcpServers': {}},
                cwd=tmp_path,
                ledger=ledger,
            )
    return result, await invocation_rows(db_path)


async def test_successful_stage_lands_one_row_and_carries_its_tokens(tmp_path: Path):
    served = AgentResult(
        success=True,
        output='',
        structured_output={'summary': 'ok'},
        cost_usd=0.42,
        duration_ms=1234,
        **_TOKENS,
    )

    result, rows = await _run_with_ledger(tmp_path, served)

    assert len(rows) == 1
    row = rows[0]
    assert row['role'] == 'recon_stage_task_knowledge_sync'
    assert row['run_id'] == 'run-s'
    assert row['project_id'] == 'proj-s'
    assert row['model'] == _CONFIG.agent_llm_model
    assert {key: row[key] for key in _TOKENS} == _TOKENS
    assert row['cost_usd'] == 0.42
    assert row['capped'] == 0
    assert result.tokens_used == 1500
    assert result.cache_read_tokens == 50000
    assert result.cache_create_tokens == 4000


async def test_returned_failure_still_lands_a_row_and_carries_its_tokens(tmp_path: Path):
    hung = AgentResult(success=False, output='', timed_out=True, duration_ms=60_000, **_TOKENS)

    result, rows = await _run_with_ledger(tmp_path, hung)

    assert len(rows) == 1
    assert result.success is False
    assert result.outcome is not StageAgentOutcome.completed
    assert result.tokens_used == 1500
    assert result.cache_read_tokens == 50000
    assert result.cache_create_tokens == 4000


async def test_unreported_tokens_are_zero(tmp_path: Path):
    served = AgentResult(success=True, output='', structured_output={'summary': 'ok'})

    result, _rows = await _run_with_ledger(tmp_path, served)

    assert result.tokens_used == 0
    assert result.cache_read_tokens == 0
    assert result.cache_create_tokens == 0


async def test_no_ledger_passes_no_cost_store(tmp_path: Path):
    _result, invoke = await run_stage_answered_by(
        AgentResult(success=True, output='', structured_output={'summary': 'ok'}), tmp_path,
    )

    assert invoke.await_args.kwargs.get('cost_store') is None
