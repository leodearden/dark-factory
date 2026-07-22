"""Tests for fused_memory.reconciliation.standing_decision_writer (task 2895 β).

Covers the two-armed evidence gate (``evaluate_evidence_gate``), the
evidence-ref resolver, and the ``write_entity_standing_decision`` helper that
writes an entity standing decision to α's ledger.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from fused_memory.models.memory import AddMemoryResponse
from fused_memory.reconciliation.recon_ledger import ReconLedgerStore
from fused_memory.reconciliation.standing_decision_constants import (
    GROUNDS_STRUCTURAL_SIZE_CONFLATION,
    MEM0_KIND_INVESTIGATION_OUTCOME,
    RECORD_KIND_ENTITY_STANDING_DECISION,
    STANDING_DECISION_TTL_DAYS,
    STATE_ACTIVE,
)
from fused_memory.reconciliation.standing_decision_writer import (
    EvidenceGateRejected,
    evaluate_evidence_gate,
    write_entity_standing_decision,
)

_PROJECT_ID = 'dark_factory'
_ENTITY_UUID = 'aaaaaaaa-1111-1111-1111-111111111111'
_MEM0_REF_ID = 'bbbbbbbb-2222-2222-2222-222222222222'


def _outcome(run_id):
    """An investigation_outcome mem0 record dict as returned by the metadata
    scroll; *run_id* of None models a record whose metadata omits run_id."""
    metadata = {'kind': MEM0_KIND_INVESTIGATION_OUTCOME, 'actionable': False}
    if run_id is not None:
        metadata['run_id'] = run_id
    return {'id': f'oc-{run_id}', 'created_at': '2026-07-01T00:00:00+00:00', 'metadata': metadata}


def _memory_service(*, memory_record=None, metadata_records=None):
    """AsyncMock memory_service exposing the two evidence-read entrypoints.

    ``get_memory_by_id`` resolves a cited mem0 evidence ref (arm 1); it returns
    the single *memory_record* (or ``None`` to model an unresolvable id).
    ``get_memories_by_metadata`` backs the arm-2 investigation_outcome scroll.
    """
    service = AsyncMock()
    service.get_memory_by_id = AsyncMock(return_value=memory_record)
    service.get_memories_by_metadata = AsyncMock(
        return_value=list(metadata_records or [])
    )
    return service


class TestEvidenceGateArm1:
    """Arm 1 — ≥1 locally-resolvable, human-authored mem0 evidence ref."""

    @pytest.mark.asyncio
    async def test_human_authored_mem0_ref_satisfies_arm1(self):
        """A resolvable mem0 ref authored by claude-interactive-* satisfies arm 1
        (and, with arm 2 empty, the whole gate); its resolved entry is stamped
        locally_resolved=true."""
        service = _memory_service(
            memory_record={
                'id': _MEM0_REF_ID,
                'content': 'operator investigation note',
                'metadata': {'agent_id': 'claude-interactive-leo'},
            },
            metadata_records=[],
        )
        result = await evaluate_evidence_gate(
            service,
            project_id=_PROJECT_ID,
            entity_uuid=_ENTITY_UUID,
            evidence=[{'type': 'mem0', 'id': _MEM0_REF_ID}],
        )
        assert result.arm1_satisfied is True
        assert result.satisfied is True
        stamped = result.resolved_evidence[0]
        assert stamped['locally_resolved'] is True
        assert stamped['agent_id'] == 'claude-interactive-leo'

    @pytest.mark.asyncio
    async def test_agent_authored_mem0_ref_does_not_satisfy_arm1(self):
        """A resolvable mem0 ref authored by a stage/agent id (not human) is
        locally_resolved but does NOT satisfy arm 1 (under-suppression bias)."""
        service = _memory_service(
            memory_record={
                'id': _MEM0_REF_ID,
                'content': 'stage-authored note',
                'metadata': {'agent_id': 'reconciliation-stage-2'},
            },
            metadata_records=[],
        )
        result = await evaluate_evidence_gate(
            service,
            project_id=_PROJECT_ID,
            entity_uuid=_ENTITY_UUID,
            evidence=[{'type': 'mem0', 'id': _MEM0_REF_ID}],
        )
        assert result.arm1_satisfied is False
        assert result.satisfied is False
        stamped = result.resolved_evidence[0]
        assert stamped['locally_resolved'] is True
        assert stamped['agent_id'] == 'reconciliation-stage-2'

    @pytest.mark.asyncio
    async def test_foreign_escalation_ref_never_counts(self):
        """A foreign (escalation) ref is stamped locally_resolved=false verbatim,
        is never counted toward arm 1, and triggers no mem0 lookup."""
        service = _memory_service(memory_record=None, metadata_records=[])
        result = await evaluate_evidence_gate(
            service,
            project_id=_PROJECT_ID,
            entity_uuid=_ENTITY_UUID,
            evidence=[{'type': 'escalation', 'id': 'esc-123'}],
        )
        assert result.arm1_satisfied is False
        assert result.satisfied is False
        stamped = result.resolved_evidence[0]
        assert stamped['locally_resolved'] is False
        service.get_memory_by_id.assert_not_called()

    @pytest.mark.asyncio
    async def test_unresolvable_mem0_ref_not_counted(self):
        """A mem0 ref whose id does not resolve (get_memory_by_id→None) is stamped
        locally_resolved=false and is not counted toward arm 1."""
        service = _memory_service(memory_record=None, metadata_records=[])
        result = await evaluate_evidence_gate(
            service,
            project_id=_PROJECT_ID,
            entity_uuid=_ENTITY_UUID,
            evidence=[{'type': 'mem0', 'id': _MEM0_REF_ID}],
        )
        assert result.arm1_satisfied is False
        assert result.satisfied is False
        stamped = result.resolved_evidence[0]
        assert stamped['locally_resolved'] is False


class TestEvidenceGateArm2:
    """Arm 2 — ≥3 DISTINCT investigation_outcome run_ids for this entity."""

    @pytest.mark.asyncio
    async def test_filters_the_scroll_on_kind_entity_and_actionable_false(self):
        """The arm-2 scroll queries get_memories_by_metadata with exactly
        {kind: investigation_outcome, entity_uuid, actionable: False}."""
        service = _memory_service(metadata_records=[])
        await evaluate_evidence_gate(
            service,
            project_id=_PROJECT_ID,
            entity_uuid=_ENTITY_UUID,
            evidence=[],
        )
        service.get_memories_by_metadata.assert_called_once_with(
            _PROJECT_ID,
            {
                'kind': MEM0_KIND_INVESTIGATION_OUTCOME,
                'entity_uuid': _ENTITY_UUID,
                'actionable': False,
            },
        )

    @pytest.mark.asyncio
    async def test_three_distinct_run_ids_satisfy_arm2_without_evidence(self):
        """3 investigation_outcome records with 3 distinct run_ids satisfy arm 2
        (count==3) and thus the whole gate, even with empty cited evidence."""
        service = _memory_service(
            metadata_records=[_outcome('run-a'), _outcome('run-b'), _outcome('run-c')]
        )
        result = await evaluate_evidence_gate(
            service,
            project_id=_PROJECT_ID,
            entity_uuid=_ENTITY_UUID,
            evidence=[],
        )
        assert result.arm2_distinct_run_count == 3
        assert result.arm2_satisfied is True
        assert result.satisfied is True

    @pytest.mark.asyncio
    async def test_three_records_two_distinct_run_ids_does_not_satisfy_arm2(self):
        """3 records collapsing to only 2 distinct run_ids do NOT satisfy arm 2."""
        service = _memory_service(
            metadata_records=[_outcome('run-a'), _outcome('run-a'), _outcome('run-b')]
        )
        result = await evaluate_evidence_gate(
            service,
            project_id=_PROJECT_ID,
            entity_uuid=_ENTITY_UUID,
            evidence=[],
        )
        assert result.arm2_distinct_run_count == 2
        assert result.arm2_satisfied is False
        assert result.satisfied is False

    @pytest.mark.asyncio
    async def test_records_missing_run_id_are_not_counted(self):
        """Records whose metadata omits (or empties) run_id cannot establish
        independence and are excluded from the distinct count."""
        service = _memory_service(
            metadata_records=[
                _outcome('run-a'),
                _outcome('run-b'),
                _outcome(None),
                _outcome(''),
            ]
        )
        result = await evaluate_evidence_gate(
            service,
            project_id=_PROJECT_ID,
            entity_uuid=_ENTITY_UUID,
            evidence=[],
        )
        assert result.arm2_distinct_run_count == 2
        assert result.arm2_satisfied is False

    @pytest.mark.asyncio
    async def test_no_investigation_outcomes_yields_zero(self):
        """An empty investigation_outcome pool (day-one) yields count 0, arm 2
        unsatisfied."""
        service = _memory_service(metadata_records=[])
        result = await evaluate_evidence_gate(
            service,
            project_id=_PROJECT_ID,
            entity_uuid=_ENTITY_UUID,
            evidence=[],
        )
        assert result.arm2_distinct_run_count == 0
        assert result.arm2_satisfied is False
        assert result.satisfied is False


class TestEvidenceGateRejection:
    """The structured rejection payload when the gate is not satisfied."""

    @pytest.mark.asyncio
    async def test_neither_arm_builds_structured_rejection(self):
        """Neither arm: satisfied False and rejection is a structured dict naming
        BOTH unmet arms + a hint that names each arm's remedy and embeds the
        observed distinct-run count."""
        service = _memory_service(memory_record=None, metadata_records=[])
        result = await evaluate_evidence_gate(
            service,
            project_id=_PROJECT_ID,
            entity_uuid=_ENTITY_UUID,
            evidence=[],
        )
        assert result.satisfied is False
        rej = result.rejection
        assert isinstance(rej, dict)
        assert rej['error'] == 'insufficient_evidence'
        assert rej['error_type'] == 'StandingDecisionEvidenceInsufficient'
        # Both arms are unmet.
        assert isinstance(rej['unmet_arms'], list)
        assert len(rej['unmet_arms']) == 2
        hint = rej['hint']
        assert isinstance(hint, str)
        assert 'arm 1' in hint.lower()
        assert 'arm 2' in hint.lower()
        assert 'human-authored' in hint.lower()
        assert 'investigation_outcome' in hint
        # The observed distinct-run count (0 here) is embedded in the hint.
        assert str(result.arm2_distinct_run_count) in hint

    @pytest.mark.asyncio
    async def test_arm2_satisfied_has_no_rejection(self):
        """Exactly one arm (arm 2) satisfied ⇒ satisfied True, rejection None."""
        service = _memory_service(
            metadata_records=[_outcome('run-a'), _outcome('run-b'), _outcome('run-c')]
        )
        result = await evaluate_evidence_gate(
            service,
            project_id=_PROJECT_ID,
            entity_uuid=_ENTITY_UUID,
            evidence=[],
        )
        assert result.satisfied is True
        assert result.rejection is None

    @pytest.mark.asyncio
    async def test_arm1_satisfied_has_no_rejection(self):
        """Exactly one arm (arm 1) satisfied ⇒ satisfied True, rejection None."""
        service = _memory_service(
            memory_record={
                'id': _MEM0_REF_ID,
                'content': 'operator note',
                'metadata': {'agent_id': 'claude-interactive-leo'},
            },
            metadata_records=[],
        )
        result = await evaluate_evidence_gate(
            service,
            project_id=_PROJECT_ID,
            entity_uuid=_ENTITY_UUID,
            evidence=[{'type': 'mem0', 'id': _MEM0_REF_ID}],
        )
        assert result.satisfied is True
        assert result.rejection is None


# ---------------------------------------------------------------------------
# write_entity_standing_decision — end-to-end against a REAL ledger
# (mirrors test_flag_dedup.py's ledger_memory_service fixture)
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def writer_memory_service(tmp_path):
    """AsyncMock memory_service with a REAL initialized ReconLedgerStore on
    ``.recon_ledger`` and mocked graphiti/evidence-read/add_memory backends.

    Defaults: graphiti samples 2 edges (edge_count N=2); arm-2 pool empty;
    no resolvable cited mem0 evidence; add_memory returns a mirror id.
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
    try:
        yield service
    finally:
        await ledger.close()


class TestWriteEntityStandingDecision:
    """The writer helper: bypass, gated, fingerprint, mirror, loud-fail."""

    @pytest.mark.asyncio
    async def test_authorized_bypass_writes_active_row(self, writer_memory_service):
        """authorized_by bypasses the gate (empty evidence + empty arm-2 pool) and
        writes exactly one ACTIVE row: correct entity_uuid/grounds, the sampled
        edge_count fingerprint (N=2), a 90-day decided→expires delta, resolved
        evidence + an operator provenance entry in the payload, and a best-effort
        mem0 mirror carrying record kind + entity_uuid + grounds."""
        service = writer_memory_service
        result = await write_entity_standing_decision(
            service,
            project_id=_PROJECT_ID,
            entity_uuid=_ENTITY_UUID,
            grounds=GROUNDS_STRUCTURAL_SIZE_CONFLATION,
            evidence=[{'type': 'mem0', 'id': _MEM0_REF_ID}],
            authorized_by='operator-x',
        )
        assert result['status'] == 'written'
        assert result['edge_count_at_decision'] == 2

        # Graphiti sampled with the exact (uuid, group_id) shape.
        service.graphiti.get_valid_edges_for_node.assert_called_once_with(
            _ENTITY_UUID, group_id=_PROJECT_ID
        )

        rows = await service.recon_ledger.list_entity_standing_decisions(_PROJECT_ID)
        assert len(rows) == 1
        row = rows[0]
        assert row.state == STATE_ACTIVE
        assert row.entity_uuid == _ENTITY_UUID

        payload = json.loads(row.payload_json)
        assert payload['grounds'] == GROUNDS_STRUCTURAL_SIZE_CONFLATION
        assert payload['edge_count_at_decision'] == 2

        decided_at = datetime.fromisoformat(payload['decided_at'])
        expires_at = datetime.fromisoformat(row.expires_at)
        assert expires_at - decided_at == timedelta(days=STANDING_DECISION_TTL_DAYS)

        # The cited mem0 ref is resolved+stamped, and an operator provenance
        # entry names the bypassing operator.
        ev = payload['evidence']
        assert any(
            e.get('type') == 'mem0' and e.get('locally_resolved') is False for e in ev
        )
        assert any(e.get('authorized_by') == 'operator-x' for e in ev)

        # Best-effort mem0 mirror carries the record kind + entity_uuid + grounds.
        service.add_memory.assert_called_once()
        md = service.add_memory.call_args.kwargs['metadata']
        assert md['kind'] == RECORD_KIND_ENTITY_STANDING_DECISION
        assert md['entity_uuid'] == _ENTITY_UUID
        assert md['grounds'] == GROUNDS_STRUCTURAL_SIZE_CONFLATION

    @pytest.mark.asyncio
    async def test_gated_write_when_arm2_satisfied(self, writer_memory_service):
        """No authorized_by, but arm 2 satisfied (3 distinct investigation_outcome
        run_ids) ⇒ the gate passes and the row is written."""
        service = writer_memory_service
        service.get_memories_by_metadata = AsyncMock(
            return_value=[_outcome('run-a'), _outcome('run-b'), _outcome('run-c')]
        )
        result = await write_entity_standing_decision(
            service,
            project_id=_PROJECT_ID,
            entity_uuid=_ENTITY_UUID,
            grounds=GROUNDS_STRUCTURAL_SIZE_CONFLATION,
            evidence=[],
        )
        assert result['status'] == 'written'
        rows = await service.recon_ledger.list_entity_standing_decisions(
            _PROJECT_ID, state=STATE_ACTIVE
        )
        assert len(rows) == 1

    @pytest.mark.asyncio
    async def test_gate_rejection_raises_and_writes_nothing(self, writer_memory_service):
        """No authorized_by + neither arm ⇒ EvidenceGateRejected carrying the
        rejection dict; no ledger row, no graphiti sample, no mem0 mirror."""
        service = writer_memory_service
        with pytest.raises(EvidenceGateRejected) as excinfo:
            await write_entity_standing_decision(
                service,
                project_id=_PROJECT_ID,
                entity_uuid=_ENTITY_UUID,
                grounds=GROUNDS_STRUCTURAL_SIZE_CONFLATION,
                evidence=[],
            )
        assert excinfo.value.rejection['error'] == 'insufficient_evidence'

        rows = await service.recon_ledger.list_entity_standing_decisions(_PROJECT_ID)
        assert rows == []
        service.graphiti.get_valid_edges_for_node.assert_not_called()
        service.add_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_graphiti_sampling_failure_propagates_no_row(self, writer_memory_service):
        """A graphiti edge-count sampling failure RAISES (loud-over-silent) — no
        poisoned row, no mem0 mirror."""
        service = writer_memory_service
        service.graphiti.get_valid_edges_for_node = AsyncMock(
            side_effect=RuntimeError('graphiti unavailable')
        )
        with pytest.raises(RuntimeError):
            await write_entity_standing_decision(
                service,
                project_id=_PROJECT_ID,
                entity_uuid=_ENTITY_UUID,
                grounds=GROUNDS_STRUCTURAL_SIZE_CONFLATION,
                evidence=[],
                authorized_by='operator-x',
            )
        rows = await service.recon_ledger.list_entity_standing_decisions(_PROJECT_ID)
        assert rows == []
        service.add_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_mem0_mirror_failure_is_swallowed(self, writer_memory_service):
        """A raising best-effort mem0 mirror never fails the authoritative ledger
        write — the ACTIVE row is still present."""
        service = writer_memory_service
        service.add_memory = AsyncMock(side_effect=RuntimeError('mem0 down'))
        result = await write_entity_standing_decision(
            service,
            project_id=_PROJECT_ID,
            entity_uuid=_ENTITY_UUID,
            grounds=GROUNDS_STRUCTURAL_SIZE_CONFLATION,
            evidence=[],
            authorized_by='operator-x',
        )
        assert result['status'] == 'written'
        rows = await service.recon_ledger.list_entity_standing_decisions(_PROJECT_ID)
        assert len(rows) == 1
