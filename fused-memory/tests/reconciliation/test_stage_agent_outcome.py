"""Each recon stage agent's fate is recorded as a typed outcome (task 5982).

Before this, a stage whose CLI agent was killed by the startup-wedge watchdog
wrote a cycle summary indistinguishable from a stage that ran and chose to do
nothing. See plans/recon-stage-startup-wedge-rca-2026-09-27.md.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest
from shared.cli_invoke import AgentResult

from fused_memory.config.schema import ReconciliationConfig
from fused_memory.models.reconciliation import StageAgentOutcome, StageId, StageReport
from fused_memory.reconciliation.cli_stage_runner import StageResult, run_stage_via_cli
from fused_memory.reconciliation.sandbox_guard import RemediationSandboxUnavailable

_INVOKE = 'fused_memory.reconciliation.cli_stage_runner.invoke_with_cap_retry'
_RESOLVE_SANDBOX = 'fused_memory.reconciliation.cli_stage_runner.resolve_recon_sandbox_wrap'


async def _run_stage(config: ReconciliationConfig, tmp_path) -> StageResult:
    return await run_stage_via_cli(
        system_prompt='sys',
        payload='pay',
        disallowed_tools=[],
        config=config,
        mcp_config={'mcpServers': {}},
        cwd=tmp_path,
    )


async def _outcome_of(agent_result: AgentResult, tmp_path) -> StageResult:
    with patch(_INVOKE, AsyncMock(return_value=agent_result)):
        return await _run_stage(ReconciliationConfig(sandbox_recon_agents=False), tmp_path)


class TestRunStageViaCliClassifiesOutcome:
    @pytest.mark.asyncio
    async def test_successful_agent_is_completed(self, tmp_path):
        result = await _outcome_of(AgentResult(success=True, output='', turns=3), tmp_path)

        assert result.success is True
        assert result.outcome is StageAgentOutcome.completed

    @pytest.mark.asyncio
    async def test_startup_wedge_kill_is_zero_output_timeout(self, tmp_path):
        wedge_kill = AgentResult(
            success=False,
            output='Agent produced no output',
            timed_out=True,
            transcript_turns=0,
        )

        result = await _outcome_of(wedge_kill, tmp_path)

        assert result.success is False
        assert result.outcome is StageAgentOutcome.zero_output_timeout

    @pytest.mark.asyncio
    async def test_kill_after_work_is_timed_out(self, tmp_path):
        killed_mid_work = AgentResult(
            success=False, output='', timed_out=True, transcript_turns=7,
        )

        result = await _outcome_of(killed_mid_work, tmp_path)

        assert result.outcome is StageAgentOutcome.timed_out

    @pytest.mark.asyncio
    async def test_unsuccessful_exit_is_failed(self, tmp_path):
        result = await _outcome_of(
            AgentResult(success=False, output='boom', timed_out=False), tmp_path,
        )

        assert result.outcome is StageAgentOutcome.failed

    @pytest.mark.asyncio
    async def test_raising_invocation_is_failed(self, tmp_path):
        with patch(_INVOKE, AsyncMock(side_effect=RuntimeError('spawn exploded'))):
            result = await _run_stage(
                ReconciliationConfig(sandbox_recon_agents=False), tmp_path,
            )

        assert result.success is False
        assert result.outcome is StageAgentOutcome.failed

    @pytest.mark.asyncio
    async def test_sandbox_fail_closed_is_not_launched(self, tmp_path):
        mock_invoke = AsyncMock(return_value=AgentResult(success=True, output=''))

        with patch(_INVOKE, mock_invoke), patch(
            _RESOLVE_SANDBOX, side_effect=RemediationSandboxUnavailable('no sandbox'),
        ):
            result = await _run_stage(
                ReconciliationConfig(sandbox_recon_agents=True), tmp_path,
            )

        assert not mock_invoke.called
        assert result.success is False
        assert result.outcome is StageAgentOutcome.not_launched


class TestStageResultOutcomeInvariant:
    def test_success_without_outcome_derives_completed(self):
        assert StageResult(success=True).outcome is StageAgentOutcome.completed

    def test_failure_without_outcome_derives_failed(self):
        assert StageResult(success=False).outcome is StageAgentOutcome.failed

    def test_success_with_a_failure_outcome_is_rejected(self):
        with pytest.raises(ValueError):
            StageResult(success=True, outcome=StageAgentOutcome.zero_output_timeout)

    def test_failure_with_the_completed_outcome_is_rejected(self):
        with pytest.raises(ValueError):
            StageResult(success=False, outcome=StageAgentOutcome.completed)


def _stage_report(**overrides) -> StageReport:
    fields: dict = {
        'stage': StageId.task_knowledge_sync,
        'started_at': datetime(2026, 9, 27, 0, 31, 35, tzinfo=UTC),
        'completed_at': datetime(2026, 9, 27, 0, 33, 37, tzinfo=UTC),
    }
    fields.update(overrides)
    return StageReport(**fields)


class TestStageReportAgentOutcomeField:
    def test_omitted_outcome_is_unrecorded_not_completed(self):
        assert _stage_report().agent_outcome is None

    def test_outcome_survives_the_journal_round_trip(self):
        report = _stage_report(agent_outcome=StageAgentOutcome.zero_output_timeout)

        rebuilt = StageReport(**report.model_dump(mode='json'))

        assert rebuilt == report
        assert rebuilt.agent_outcome is StageAgentOutcome.zero_output_timeout

    def test_legacy_journal_row_without_the_key_rebuilds_unrecorded(self):
        legacy = _stage_report().model_dump(mode='json')
        legacy.pop('agent_outcome', None)

        assert StageReport(**legacy).agent_outcome is None
