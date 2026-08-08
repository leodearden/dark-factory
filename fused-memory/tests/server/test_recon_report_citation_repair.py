"""``ReconReportState.repair_memory_citation`` and its MCP tool (task 3065).

The state method is a thin wrapper: it keeps the existing ``run_id`` contract
(the CALLER's own live run, resolved through the unchanged ``_resolve_entry``,
which also supplies the ``repaired_by`` attribution for free) and adds a
SEPARATE ``target_run_id`` for the run that owns the finding. The incident that
produced this task had two failed attempts that were both defensible readings of
one overloaded ``run_id``; splitting the two is what makes the call site
unambiguous.

The repair semantics themselves are tested in
``tests/reconciliation/test_citation_repair.py`` — these tests assert the
wrapper's delegation, its degradation posture, and the tool surface.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from _fm_helpers import FakeMemoryLookup, build_journal_with_closed_run

from fused_memory.server.recon_report import (
    RECON_REPORT_INSTRUCTIONS,
    ReconReportState,
    create_recon_report_server,
    get_recon_report_tool_signatures,
)

CALLER_RUN = 'caller-1'
TARGET_RUN = '06a4466d-cdc0-49ac-8e99-e6723be39392'
DANGLING = 'beacf7fc-b76a-4c0b-876d-f4cf6d906d42'
SUCCESSOR = '746b4ab9-ca3c-418b-982a-32b85bfcf94b'

SUCCESSOR_RECORD: dict[str, Any] = {
    'id': SUCCESSOR,
    'content': 'the surviving consolidated entry',
    'metadata': {
        'category': 'procedural_knowledge',
        'agent_id': 'recon-stage-memory_consolidator',
        'created_at': '2026-07-26T04:34:05Z',
    },
}


def _finding(finding_id: str, memory_id: str) -> dict[str, Any]:
    return {
        'finding_id': finding_id,
        'description': f'finding {finding_id}',
        'cited_memories': [{'memory_id': memory_id, 'store': 'mem0'}],
    }


async def _seeded_journal(tmp_path, *, status: str = 'completed', run_id: str = TARGET_RUN):
    return await build_journal_with_closed_run(
        tmp_path,
        run_id=run_id,
        status=status,
        findings=[_finding('f-1', DANGLING)],
    )


def _state(**kwargs) -> ReconReportState:
    return ReconReportState(ttl_seconds=300, clock=lambda: 0.0, **kwargs)


class TestStateRepairMemoryCitation:
    """The wrapper's contract: caller-run resolution, injection, degradation."""

    @pytest.mark.asyncio
    async def test_delegates_and_stamps_caller_identity(self, tmp_path):
        journal = await _seeded_journal(tmp_path)
        try:
            memory = FakeMemoryLookup({DANGLING: None, SUCCESSOR: SUCCESSOR_RECORD})
            state = _state(memory_service=memory, journal=journal)
            state.start_report(
                run_id=CALLER_RUN, stage='memory_consolidator', project_id='reify'
            )

            outcome = await state.repair_memory_citation(
                run_id=CALLER_RUN,
                target_run_id=TARGET_RUN,
                finding_id='f-1',
                memory_id=DANGLING,
                store='mem0',
                replacement_memory_id=SUCCESSOR,
            )

            assert outcome['status'] == 'repaired'
            run = await journal.get_run(TARGET_RUN)
            finding = run.stage_reports['memory_consolidator'].items_flagged[0]
            # repaired_by comes from the caller's own live run — no new
            # LLM-typed identifier to get wrong.
            assert finding['citation_repairs'][0]['repaired_by'] == f'run:{CALLER_RUN}'
        finally:
            await journal.close()

    @pytest.mark.asyncio
    async def test_unknown_caller_run_is_unchanged_contract(self, tmp_path):
        """No active entry for ``run_id`` -> the existing refusal, no journal call."""
        journal = await _seeded_journal(tmp_path)
        try:
            journal.get_run = AsyncMock(wraps=journal.get_run)
            memory = FakeMemoryLookup({DANGLING: None, SUCCESSOR: SUCCESSOR_RECORD})
            state = _state(memory_service=memory, journal=journal)

            outcome = await state.repair_memory_citation(
                run_id='never-started',
                target_run_id=TARGET_RUN,
                finding_id='f-1',
                memory_id=DANGLING,
                store='mem0',
                replacement_memory_id=SUCCESSOR,
            )

            assert outcome['error'] == 'run_id_unknown'
            assert outcome['error_type'] == 'ReconReportRunUnknown'
            assert journal.get_run.await_count == 0
        finally:
            await journal.close()

    @pytest.mark.asyncio
    async def test_no_journal_degrades_loudly(self):
        """Reconciliation disabled -> a structured refusal, never a half-repair."""
        memory = FakeMemoryLookup({DANGLING: None, SUCCESSOR: SUCCESSOR_RECORD})
        state = _state(memory_service=memory)
        state.start_report(
            run_id=CALLER_RUN, stage='memory_consolidator', project_id='reify'
        )

        outcome = await state.repair_memory_citation(
            run_id=CALLER_RUN,
            target_run_id=TARGET_RUN,
            finding_id='f-1',
            memory_id=DANGLING,
            store='mem0',
            replacement_memory_id=SUCCESSOR,
        )

        assert outcome['error'] == 'journal_unavailable'
        assert outcome['error_type'] == 'ReconReportJournalUnavailable'

    @pytest.mark.asyncio
    async def test_no_memory_service_degrades_loudly(self, tmp_path):
        journal = await _seeded_journal(tmp_path)
        try:
            state = _state(journal=journal)
            state.start_report(
                run_id=CALLER_RUN, stage='memory_consolidator', project_id='reify'
            )

            outcome = await state.repair_memory_citation(
                run_id=CALLER_RUN,
                target_run_id=TARGET_RUN,
                finding_id='f-1',
                memory_id=DANGLING,
                store='mem0',
                replacement_memory_id=SUCCESSOR,
            )

            assert outcome['error'] == 'service_not_configured'
            assert outcome['error_type'] == 'ReconReportServiceUnavailable'
        finally:
            await journal.close()

    @pytest.mark.asyncio
    async def test_repairing_own_live_run_is_refused(self, tmp_path):
        """``target_run_id == run_id`` falls out of the live_run_ids guard.

        Splitting the two parameters gives this special case for free: the
        caller's own run is by definition in ``_state``.
        """
        journal = await _seeded_journal(tmp_path, run_id=CALLER_RUN)
        try:
            memory = FakeMemoryLookup({DANGLING: None, SUCCESSOR: SUCCESSOR_RECORD})
            state = _state(memory_service=memory, journal=journal)
            state.start_report(
                run_id=CALLER_RUN, stage='memory_consolidator', project_id='reify'
            )

            outcome = await state.repair_memory_citation(
                run_id=CALLER_RUN,
                target_run_id=CALLER_RUN,
                finding_id='f-1',
                memory_id=DANGLING,
                store='mem0',
                replacement_memory_id=SUCCESSOR,
            )

            assert outcome['error'] == 'run_still_live'
            assert 'cite_memory' in outcome['hint']
            assert 'delete_finding' in outcome['hint']
        finally:
            await journal.close()


class TestRepairToolViaFastMCP:
    """The MCP surface: registration, signature, end-to-end call, docs."""

    @pytest.mark.asyncio
    async def test_tool_is_registered(self):
        mcp = create_recon_report_server(_state())
        names = {tool.name for tool in await mcp.list_tools()}
        assert 'repair_memory_citation' in names

    def test_signature_shape(self):
        sig = get_recon_report_tool_signatures()['repair_memory_citation']
        assert list(sig.parameters) == [
            'run_id',
            'target_run_id',
            'finding_id',
            'memory_id',
            'store',
            'replacement_memory_id',
        ]
        assert sig.parameters['replacement_memory_id'].default is None
        # Mirrors cite_memory's declared shape so a bad store is rejected at the
        # schema boundary too, not only by the unsupported_store gate.
        # recon_report.py carries `from __future__ import annotations`, so the
        # signature holds the SOURCE TEXT of the annotation, not the object.
        assert sig.parameters['store'].annotation == "Literal['graphiti', 'mem0']"

    @pytest.mark.asyncio
    async def test_end_to_end_tool_call(self, tmp_path):
        journal = await _seeded_journal(tmp_path)
        try:
            memory = FakeMemoryLookup({DANGLING: None, SUCCESSOR: SUCCESSOR_RECORD})
            state = _state(memory_service=memory, journal=journal)
            state.start_report(
                run_id=CALLER_RUN, stage='memory_consolidator', project_id='reify'
            )
            mcp = create_recon_report_server(state)

            result = await mcp._tool_manager.call_tool(
                'repair_memory_citation',
                {
                    'run_id': CALLER_RUN,
                    'target_run_id': TARGET_RUN,
                    'finding_id': 'f-1',
                    'memory_id': DANGLING,
                    'store': 'mem0',
                    'replacement_memory_id': SUCCESSOR,
                },
            )

            assert result['status'] == 'repaired'
            assert result['replacement_memory_id'] == SUCCESSOR
        finally:
            await journal.close()

    def test_instructions_document_the_two_run_ids(self):
        assert 'repair_memory_citation' in RECON_REPORT_INSTRUCTIONS
        # The one thing an agent must not get wrong, given the incident.
        assert 'target_run_id' in RECON_REPORT_INSTRUCTIONS

    def test_generated_stage_guidance_excludes_the_repair_tool(self):
        """The rendered stage-prompt guidance must NOT gain this tool.

        ``render_recon_report_tool_guidance`` renders only the routine
        agent-called report tools. Repair is exceptional and evidence-gated, and
        for Stage 3 it is denied outright — advertising it in every stage's
        system prompt would add standing prompt weight and, there, name a tool
        the stage cannot call.
        """
        from fused_memory.reconciliation.prompts import (
            render_recon_report_tool_guidance,
        )

        rendered = render_recon_report_tool_guidance()
        assert rendered.strip()
        assert 'repair_memory_citation' not in rendered
