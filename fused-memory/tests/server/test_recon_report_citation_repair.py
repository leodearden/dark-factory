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

from fused_memory.models.reconciliation import StageReport
from fused_memory.server.recon_report import (
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
            assert run is not None
            # Still a parsed StageReport, not a raw dict: the repair rewrites
            # the blob through the journal's own model round-trip.
            report = run.stage_reports['memory_consolidator']
            assert isinstance(report, StageReport)
            finding = report.items_flagged[0]
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

    @pytest.mark.asyncio
    async def test_cross_project_target_run_is_refused_before_any_lookup(self, tmp_path):
        """A caller in project A may not repair a run owned by project B.

        ``reconciliation.data_dir`` is ONE journal per process holding runs for
        every project, so ``get_run(target_run_id)`` resolves another project's
        run perfectly well. Without the caller-project gate a Stage-1/Stage-2
        agent reconciling A would rewrite B's durable audit record — and issue
        the corroboration point reads in B's mem0 scope to do it. Asserting on
        ``memory.calls`` is the load-bearing half: the refusal must land BEFORE
        any cross-project read, not merely before the write.
        """
        journal = await build_journal_with_closed_run(
            tmp_path,
            run_id=TARGET_RUN,
            project_id='other_project',
            status='completed',
            findings=[_finding('f-1', DANGLING)],
        )
        try:
            memory = FakeMemoryLookup({DANGLING: None, SUCCESSOR: SUCCESSOR_RECORD})
            state = _state(memory_service=memory, journal=journal)
            state.start_report(
                run_id=CALLER_RUN, stage='memory_consolidator', project_id='dark_factory'
            )

            outcome = await state.repair_memory_citation(
                run_id=CALLER_RUN,
                target_run_id=TARGET_RUN,
                finding_id='f-1',
                memory_id=DANGLING,
                store='mem0',
                replacement_memory_id=SUCCESSOR,
            )

            assert outcome['error'] == 'project_mismatch'
            assert outcome['caller_project_id'] == 'dark_factory'
            assert outcome['target_project_id'] == 'other_project'
            # No mem0 point read was issued in the other project's scope.
            assert memory.calls == []
            # And the other project's audit record is byte-for-byte untouched.
            run = await journal.get_run(TARGET_RUN)
            assert run is not None
            report = run.stage_reports['memory_consolidator']
            assert isinstance(report, StageReport)
            finding = report.items_flagged[0]
            assert finding['cited_memories'] == [
                {'memory_id': DANGLING, 'store': 'mem0'}
            ]
            assert 'citation_repairs' not in finding
        finally:
            await journal.close()

    @pytest.mark.asyncio
    async def test_same_project_target_run_still_repairs(self, tmp_path):
        """The gate is a project COMPARISON, not a blanket cross-run refusal.

        Guards the obvious over-correction: the whole point of task 3065 is
        repairing ANOTHER run's finding, so the isolation gate must stay silent
        when that run is owned by the caller's own project.
        """
        journal = await build_journal_with_closed_run(
            tmp_path,
            run_id=TARGET_RUN,
            project_id='reify',
            status='completed',
            findings=[_finding('f-1', DANGLING)],
        )
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
            assert outcome['project_id'] == 'reify'
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


_QUALIFIED_TOOL = 'mcp__recon-report__repair_memory_citation'


class TestRepairToolDisallowList:
    """Per-stage gating: Stage 3 must not repair what Stage 3 detects.

    Mirrors ``test_recon_report_standing_decision.py::TestStandingDecisionDisallowList``
    — this is the SECOND recon-report tool that writes past in-process state, so
    it rides its own additive sublist. The journal is not the ledger, so folding
    it into ``DISALLOW_RECON_REPORT_LEDGER_WRITES`` would make that name lie and
    would couple two grants that need to move independently.

    The per-stage split is the substance:

    * Stage 3 (``integrity_check``) is read-only by contract AND is the stage
      that DETECTS a dangling cross-run citation. Letting it also repair would
      break the contract and collapse detect-and-repair into one unaccountable
      actor.
    * Stage 1 keeps it — Stage 1 is where a supersession deletes the predecessor
      memory, i.e. where cross-run citations become dangling in the first place,
      and it is already ``citation_verifier``'s own caller.
    * Stage 2 keeps it — Stage 2 is the remediation stage whose blocked attempt
      originated this task.
    """

    def test_tool_in_dedicated_sublist(self):
        from fused_memory.reconciliation.cli_stage_runner import (
            DISALLOW_RECON_REPORT_JOURNAL_WRITES,
        )

        assert _QUALIFIED_TOOL in DISALLOW_RECON_REPORT_JOURNAL_WRITES

    def test_tool_in_stage3_disallowed(self):
        from fused_memory.reconciliation.cli_stage_runner import STAGE3_DISALLOWED

        assert _QUALIFIED_TOOL in STAGE3_DISALLOWED

    def test_tool_not_in_stage1_disallowed(self):
        from fused_memory.reconciliation.cli_stage_runner import STAGE1_DISALLOWED

        assert _QUALIFIED_TOOL not in STAGE1_DISALLOWED

    def test_tool_not_in_stage2_disallowed(self):
        from fused_memory.reconciliation.cli_stage_runner import STAGE2_DISALLOWED

        assert _QUALIFIED_TOOL not in STAGE2_DISALLOWED

    @pytest.mark.asyncio
    async def test_denial_names_tools_that_actually_exist(self):
        """A denial that names a non-existent tool is decoration, not gating.

        The same anti-decoration check ``test_stages.py`` applies to
        ``DISALLOW_ESCALATION_READS``: enumerate the LIVE tool surface and
        require every denied name to be on it. A typo'd or renamed entry denies
        nothing and reads, to the next person, as though it does.
        """
        from fused_memory.reconciliation.cli_stage_runner import (
            DISALLOW_RECON_REPORT_JOURNAL_WRITES,
        )

        mcp = create_recon_report_server(_state())
        registered = {tool.name for tool in await mcp.list_tools()}

        prefix = 'mcp__recon-report__'
        denied = set()
        for qualified in DISALLOW_RECON_REPORT_JOURNAL_WRITES:
            assert qualified.startswith(prefix), qualified
            denied.add(qualified[len(prefix) :])

        stale = denied - registered
        assert not stale, (
            'DISALLOW_RECON_REPORT_JOURNAL_WRITES names recon-report tools that '
            f'are not registered by create_recon_report_server: {sorted(stale)}. '
            'A denial that names nothing denies nothing.'
        )


class TestReconReportComponentsJournalWiring:
    """``_build_recon_report_components`` threads the journal, or degrades loudly.

    Mirrors ``test_recon_report_citations.py::TestReconReportComponentsWiring``.
    The journal rides the same optional-service injection pattern as
    ``memory_service`` / ``task_interceptor``: a defaulted keyword arg, so every
    existing caller — two ``main.py`` sites and three test modules — keeps
    working untouched.

    The reconciliation-DISABLED boot path has no journal to pass, and the tool
    must then answer ``journal_unavailable`` rather than half-work against a
    store that does not exist.
    """

    def _make_config(self):
        from fused_memory.config.schema import (
            FusedMemoryConfig,
            ReconciliationConfig,
            ServerConfig,
        )

        return FusedMemoryConfig(
            server=ServerConfig(recon_report_port=8003, host='127.0.0.1'),
            reconciliation=ReconciliationConfig(recon_report_state_ttl_seconds=300),
        )

    def test_journal_is_injected_when_supplied(self):
        from fused_memory.server.main import _build_recon_report_components

        sentinel = object()
        state, _, _ = _build_recon_report_components(
            self._make_config(), recon_journal=sentinel
        )
        assert state._journal is sentinel

    def test_existing_call_shape_still_works_without_a_journal(self):
        """Backward compatibility: the arg is optional, and absent means None."""
        from fused_memory.server.main import _build_recon_report_components
        from fused_memory.server.recon_report import ReconReportState

        state, _, _ = _build_recon_report_components(self._make_config())
        assert isinstance(state, ReconReportState)
        assert state._journal is None

    @pytest.mark.asyncio
    async def test_state_without_journal_degrades_rather_than_raising(self):
        from fused_memory.server.main import _build_recon_report_components

        state, _, _ = _build_recon_report_components(self._make_config())
        state.start_report(
            run_id=CALLER_RUN, stage='memory_consolidator', project_id='reify'
        )

        result = await state.repair_memory_citation(
            run_id=CALLER_RUN,
            target_run_id=TARGET_RUN,
            finding_id='f-1',
            memory_id=DANGLING,
            store='mem0',
            replacement_memory_id=SUCCESSOR,
        )

        assert result['error'] == 'journal_unavailable'
        assert result['error_type'] == 'ReconReportJournalUnavailable'
