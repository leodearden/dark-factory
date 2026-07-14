"""Tests for τ2 (task 2437): Stage 3 consults the ReconLedgerStore
authoritatively for cycle-summary presence, via τ1's (task 2436)
``get_cycle_summary_presence`` tool, falling back to the existing best-effort
Mem0 two-path check only when the ledger read is inconclusive.

Two cohesive pieces of the same G2 signal live here (PRD
plans/stage3-ledger-presence-prd.md §12):

- ``TestStage3PromptLedgerAuthoritative`` — a prompt-content contract test.
  The Stage-3 agent's in-loop reasoning is not unit-testable; the LLM prompt
  string IS the observable behaviour surface (the agent cannot call a tool or
  apply a rule it is not told about), so this asserts the new tool is named
  and the ledger-authoritative rule markers are present, plus a closure gate
  that the retired "Known gap" comment is actually gone from the module
  source. Mirrors test_stages.py::TestStage3PromptAlignment.
- ``TestWriteThenReadLedgerSeam`` (added separately) — a write→read
  boundary/seam test proving the mechanical path the rewritten prompt now
  trusts is real, not faked.
"""

from __future__ import annotations

import inspect
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

import fused_memory.reconciliation.prompts.stage3 as stage3_module
from fused_memory.models.reconciliation import StageId, StageReport
from fused_memory.reconciliation.prompts.stage3 import STAGE3_SYSTEM_PROMPT
from fused_memory.reconciliation.recon_ledger import ReconLedgerStore
from fused_memory.reconciliation.summary_pool import write_cycle_summary
from fused_memory.services.memory_service import MemoryService

_PROJECT_ID = 'dark_factory'
_STAGE = 'task_knowledge_sync'


class TestStage3PromptLedgerAuthoritative:
    """STAGE3_SYSTEM_PROMPT names the new ledger-presence tool and carries
    the ledger-authoritative rule; the retired "Known gap" comment is gone
    from the module source."""

    def test_prompt_names_get_cycle_summary_presence_tool(self):
        assert 'get_cycle_summary_presence' in STAGE3_SYSTEM_PROMPT, (
            "STAGE3_SYSTEM_PROMPT must name the get_cycle_summary_presence tool"
        )

    def test_prompt_carries_ledger_authoritative_rule_markers(self):
        assert 'ledger_available' in STAGE3_SYSTEM_PROMPT, (
            "STAGE3_SYSTEM_PROMPT must reference the 'ledger_available' return field"
        )
        assert 'authoritative' in STAGE3_SYSTEM_PROMPT, (
            "STAGE3_SYSTEM_PROMPT must describe the ledger read as authoritative"
        )

    def test_known_gap_comment_removed_from_module_source(self):
        assert 'Known gap' not in inspect.getsource(stage3_module), (
            'The retired "Known gap" comment must be deleted from stage3.py, '
            'not merely absent from the prompt string'
        )


class TestWriteThenReadLedgerSeam:
    """Write→read boundary/seam test (G2 integration signal + regression
    guard): a real MemoryService(mock_config) wired to a real
    ReconLedgerStore proves write_cycle_summary (τ1's write half) and
    get_cycle_summary_presence (τ1's read half) transact against the same
    ledger row. Passes on first write — τ1 (task 2436, merged) already
    delivered both ends; this is a characterization test for the seam the
    rewritten Stage-3 prompt now relies on, not a RED test with an impl
    counterpart in this task."""

    def _report(self, **overrides) -> StageReport:
        defaults: dict[str, Any] = dict(
            stage=StageId.task_knowledge_sync,
            started_at=datetime(2026, 7, 10, 11, 0, 0, tzinfo=UTC),
            completed_at=datetime(2026, 7, 10, 11, 5, 0, tzinfo=UTC),
            items_flagged=[{'description': 'a'}],
            stats={},
            llm_calls=1,
            tokens_used=10,
        )
        defaults.update(overrides)
        return StageReport(**defaults)

    @pytest.mark.asyncio
    async def test_written_summary_is_present_unwritten_run_id_is_absent(self, mock_config, tmp_path):
        service = MemoryService(mock_config)
        store = ReconLedgerStore(tmp_path / 'reconciliation.db')
        await store.initialize()
        service.set_recon_ledger(store)
        # Stub only the best-effort Mem0 paths — the ledger write+read runs fully real.
        service.add_system_record = AsyncMock(return_value=SimpleNamespace(memory_ids=['m1']))
        service.get_memories_by_metadata = AsyncMock(return_value=[])

        try:
            run_id = 'run-seam-present'
            await write_cycle_summary(
                service,
                _PROJECT_ID,
                self._report(),
                run_id,
                stage=_STAGE,
                recon_pool='stage2_cycle_summary',
                trim_source='stage2_cycle_summary_trim',
                cap=2,
            )

            present_result = await service.get_cycle_summary_presence(
                project_id=_PROJECT_ID, run_id=run_id, stage=_STAGE,
            )
            absent_result = await service.get_cycle_summary_presence(
                project_id=_PROJECT_ID, run_id='run-seam-never-written', stage=_STAGE,
            )

            # Anti-inversion: assert BOTH directions in the same test so the
            # seam cannot be wired backwards.
            assert present_result.get('present') is True, (
                f'Expected present=True for a written run_id, got: {present_result!r}'
            )
            assert present_result.get('ledger_available') is True
            assert absent_result.get('present') is False, (
                f'Expected present=False for an unwritten run_id, got: {absent_result!r}'
            )
            assert absent_result.get('ledger_available') is True
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_no_ledger_wired_is_inconclusive(self, mock_config):
        service = MemoryService(mock_config)
        assert service.recon_ledger is None

        result = await service.get_cycle_summary_presence(
            project_id=_PROJECT_ID, run_id='any-run', stage=_STAGE,
        )

        assert result.get('present') is False
        assert result.get('ledger_available') is False, (
            'ledger_available=False (inconclusive) must drive Stage 3 to the Mem0 fallback'
        )
