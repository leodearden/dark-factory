"""Tests for fused_memory.reconciliation.standing_decision_writer (task 2895 β).

Covers the two-armed evidence gate (``evaluate_evidence_gate``), the
evidence-ref resolver, and the ``write_entity_standing_decision`` helper that
writes an entity standing decision to α's ledger.
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from fused_memory.reconciliation.standing_decision_writer import (
    evaluate_evidence_gate,
)

_PROJECT_ID = 'dark_factory'
_ENTITY_UUID = 'aaaaaaaa-1111-1111-1111-111111111111'
_MEM0_REF_ID = 'bbbbbbbb-2222-2222-2222-222222222222'


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
