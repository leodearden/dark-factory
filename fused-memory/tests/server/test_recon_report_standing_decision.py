"""Behaviour tests for the recon-report write_entity_standing_decision MCP tool
(task 2895 β) and its stage disallow-list wiring.

The tool is the single un-bypassable Stage-2 entry to the entity-standing-decision
write path: it delegates to ``write_entity_standing_decision`` WITHOUT
authorized_by, so Stage 2 is always gated (operator/backfill η bypass lives on
the helper, not the tool). Drives the tool via
``mcp._tool_manager.call_tool`` (mirrors test_recon_report_citations.py).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from fused_memory.models.memory import AddMemoryResponse
from fused_memory.reconciliation.recon_ledger import ReconLedgerStore
from fused_memory.reconciliation.standing_decision_constants import (
    GROUNDS_STRUCTURAL_SIZE_CONFLATION,
    MEM0_KIND_INVESTIGATION_OUTCOME,
    STATE_ACTIVE,
)
from fused_memory.server.recon_report import (
    ReconReportState,
    create_recon_report_server,
    get_recon_report_tool_signatures,
)

_PROJECT_ID = 'dark_factory'
_ENTITY_UUID = 'aaaaaaaa-1111-1111-1111-111111111111'
_TOOL = 'write_entity_standing_decision'


def _outcome(run_id):
    """An investigation_outcome mem0 record dict (arm-2 substrate)."""
    return {
        'id': f'oc-{run_id}',
        'created_at': '2026-07-01T00:00:00+00:00',
        'metadata': {
            'kind': MEM0_KIND_INVESTIGATION_OUTCOME,
            'actionable': False,
            'run_id': run_id,
        },
    }


@pytest_asyncio.fixture
async def env(tmp_path):
    """ReconReportState + FastMCP server wired to an AsyncMock memory_service
    with a REAL tmp_path ReconLedgerStore on ``.recon_ledger``.

    Defaults gate-reject (empty arm-2 pool, no resolvable evidence); a test that
    wants a successful write overrides ``service.get_memories_by_metadata``.
    """
    ledger = ReconLedgerStore(tmp_path / 'reconciliation.db')
    await ledger.initialize()
    service = AsyncMock()
    service.recon_ledger = ledger
    service.get_memory_by_id = AsyncMock(return_value=None)
    service.get_memories_by_metadata = AsyncMock(return_value=[])
    service.add_memory = AsyncMock(return_value=AddMemoryResponse(memory_ids=['mirror-id']))
    service.graphiti.get_valid_edges_for_node = AsyncMock(
        return_value=[{'uuid': 'e1'}, {'uuid': 'e2'}]
    )
    state = ReconReportState(ttl_seconds=300, clock=lambda: 0.0, memory_service=service)
    mcp = create_recon_report_server(state)
    try:
        yield SimpleNamespace(service=service, state=state, mcp=mcp, ledger=ledger)
    finally:
        await ledger.close()


class TestWriteEntityStandingDecisionTool:
    @pytest.mark.asyncio
    async def test_insufficient_evidence_returns_rejection_no_row(self, env):
        """Neither arm satisfied ⇒ the tool returns the structured
        insufficient_evidence rejection (unmet_arms + hint) and writes no row."""
        result = await env.mcp._tool_manager.call_tool(
            _TOOL,
            {
                'project_id': _PROJECT_ID,
                'entity_uuid': _ENTITY_UUID,
                'grounds': GROUNDS_STRUCTURAL_SIZE_CONFLATION,
                'evidence': [],
            },
        )
        assert result['error'] == 'insufficient_evidence'
        assert 'unmet_arms' in result
        assert 'hint' in result
        rows = await env.ledger.list_entity_standing_decisions(_PROJECT_ID)
        assert rows == []

    @pytest.mark.asyncio
    async def test_arm2_satisfied_writes_active_row(self, env):
        """Arm 2 satisfied (3 distinct investigation_outcome run_ids) ⇒ success
        dict with entity_uuid/grounds/edge_count_at_decision/expires_at and one
        listable ACTIVE row."""
        env.service.get_memories_by_metadata = AsyncMock(
            return_value=[_outcome('run-a'), _outcome('run-b'), _outcome('run-c')]
        )
        result = await env.mcp._tool_manager.call_tool(
            _TOOL,
            {
                'project_id': _PROJECT_ID,
                'entity_uuid': _ENTITY_UUID,
                'grounds': GROUNDS_STRUCTURAL_SIZE_CONFLATION,
                'evidence': [],
            },
        )
        assert result['entity_uuid'] == _ENTITY_UUID
        assert result['grounds'] == GROUNDS_STRUCTURAL_SIZE_CONFLATION
        assert result['edge_count_at_decision'] == 2
        assert 'expires_at' in result
        rows = await env.ledger.list_entity_standing_decisions(
            _PROJECT_ID, state=STATE_ACTIVE
        )
        assert len(rows) == 1
        assert rows[0].entity_uuid == _ENTITY_UUID

    @pytest.mark.asyncio
    async def test_invalid_grounds_returns_structured_error_no_row(self, env):
        """A grounds value outside GROUNDS_ENUM ⇒ a structured invalid_grounds
        error (from the ledger's ValueError) and no row — even when the gate
        would otherwise pass (arm 2 satisfied)."""
        env.service.get_memories_by_metadata = AsyncMock(
            return_value=[_outcome('run-a'), _outcome('run-b'), _outcome('run-c')]
        )
        result = await env.mcp._tool_manager.call_tool(
            _TOOL,
            {
                'project_id': _PROJECT_ID,
                'entity_uuid': _ENTITY_UUID,
                'grounds': 'not_a_real_grounds_value',
                'evidence': [],
            },
        )
        assert result['error'] == 'invalid_grounds'
        assert 'error_type' in result
        rows = await env.ledger.list_entity_standing_decisions(_PROJECT_ID)
        assert rows == []

    @pytest.mark.asyncio
    async def test_service_unavailable_when_memory_service_none(self):
        """memory_service=None ⇒ a structured service-unavailable error."""
        state = ReconReportState(ttl_seconds=300, clock=lambda: 0.0, memory_service=None)
        mcp = create_recon_report_server(state)
        result = await mcp._tool_manager.call_tool(
            _TOOL,
            {
                'project_id': _PROJECT_ID,
                'entity_uuid': _ENTITY_UUID,
                'grounds': GROUNDS_STRUCTURAL_SIZE_CONFLATION,
                'evidence': [],
            },
        )
        assert result['error'] == 'service_not_configured'

    def test_tool_signature_has_no_authorized_by(self):
        """The Stage-2 tool exposes project_id/entity_uuid/grounds/evidence and
        NO authorized_by — Stage 2 cannot bypass the evidence gate."""
        sigs = get_recon_report_tool_signatures()
        assert _TOOL in sigs
        params = set(sigs[_TOOL].parameters)
        assert 'authorized_by' not in params
        assert {'project_id', 'entity_uuid', 'grounds', 'evidence'} <= params
