"""The invocations-ledger keys a recon or link-adjudicator LLM call is written under."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from shared.cli_invoke import AgentResult

from fused_memory.models.invocation_ledger import (
    InvocationLedger,
    LedgerRole,
    input_output_tokens,
    invocations_ledger_path,
    ledger_invoke_kwargs,
)
from fused_memory.models.reconciliation import StageId


class TestLedgerRole:
    def test_role_values_are_the_audit_contract(self):
        assert {role.value for role in LedgerRole} == {
            'recon_stage_memory_consolidator',
            'recon_stage_task_knowledge_sync',
            'recon_stage_integrity_check',
            'recon_judge',
            'recon_verify',
            'link_adjudicator',
        }

    @pytest.mark.parametrize(
        ('stage', 'role'),
        [
            (StageId.memory_consolidator, 'recon_stage_memory_consolidator'),
            (StageId.task_knowledge_sync, 'recon_stage_task_knowledge_sync'),
            (StageId.integrity_check, 'recon_stage_integrity_check'),
        ],
    )
    def test_for_stage_names_each_stage(self, stage: StageId, role: str):
        assert LedgerRole.for_stage(stage) == role

    def test_every_stage_has_its_own_role(self):
        assert len({LedgerRole.for_stage(stage) for stage in StageId}) == len(StageId)


class TestLedgerInvokeKwargs:
    def test_spreads_every_key(self):
        store = MagicMock(name='cost_store')
        ledger = InvocationLedger(store, LedgerRole.RECON_JUDGE, run_id='r1', project_id='p1')

        assert ledger_invoke_kwargs(ledger) == {
            'cost_store': store,
            'role': 'recon_judge',
            'run_id': 'r1',
            'project_id': 'p1',
        }

    def test_no_ledger_gives_no_kwargs(self):
        assert ledger_invoke_kwargs(None) == {}


class TestInputOutputTokens:
    def test_sums_input_and_output_only(self):
        result = AgentResult(
            success=True,
            output='',
            input_tokens=1200,
            output_tokens=300,
            cache_read_tokens=50000,
            cache_create_tokens=4000,
        )

        assert input_output_tokens(result) == 1500

    def test_unreported_tokens_count_as_zero(self):
        assert input_output_tokens(AgentResult(success=True, output='')) == 0


def test_ledger_path_is_the_curator_events_file(tmp_path: Path):
    assert invocations_ledger_path(tmp_path) == tmp_path / 'curator_events.db'
