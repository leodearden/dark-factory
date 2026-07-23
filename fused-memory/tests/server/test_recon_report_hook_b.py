"""Behaviour tests for Hook B (task 2897 δ): cite_entity annotates a finding
with an ACTIVE entity standing decision, best-effort and NEVER-drops.

Hook B is annotate-only (PRD ``plans/stage1-entity-standing-decision-prd.md``
§Hook B semantics, decision 3): a cited entity that carries an active standing
decision is *annotated* on the returned citation response (under a
``standing_decision`` key) and on the persisted ``_Finding.standing_decision_id``
field, but NO finding is ever created or dropped and the stored citation is left
untouched.  The lookup is fail-open — an absent ``recon_ledger`` or any lookup
error skips annotation and never breaks entity citation.

Fixture model: the β Hook-B env (test_recon_report_standing_decision.py:50-73) —
a REAL ``tmp_path`` :class:`ReconLedgerStore` mounted on an ``AsyncMock``
memory_service's ``.recon_ledger``, with ``get_entity`` faked to resolve the
decided node.  Active rows are seeded directly via the α writer
``upsert_entity_standing_decision`` (no need to go through β's gated tool).
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from fused_memory.reconciliation.recon_ledger import ReconLedgerStore
from fused_memory.reconciliation.standing_decision_constants import (
    GROUNDS_STRUCTURAL_SIZE_CONFLATION,
    STATE_ACTIVE,
)
from fused_memory.server.recon_report import ReconReportState

_PROJECT_ID = 'dark_factory'
_ENTITY_UUID = 'aaaaaaaa-1111-1111-1111-111111111111'
_OTHER_UUID = 'bbbbbbbb-2222-2222-2222-222222222222'
_DECIDED_AT = '2026-07-01T00:00:00+00:00'
_EXPIRES_AT = '2026-09-29T00:00:00+00:00'


def _expected_id(uuid: str = _ENTITY_UUID) -> str:
    return f'{uuid}:{GROUNDS_STRUCTURAL_SIZE_CONFLATION}'


@pytest_asyncio.fixture
async def ledger(tmp_path):
    """A REAL, function-scoped :class:`ReconLedgerStore` on a fresh tmp db."""
    store = ReconLedgerStore(tmp_path / 'reconciliation.db')
    await store.initialize()
    try:
        yield store
    finally:
        await store.close()


async def _seed_active(store: ReconLedgerStore, uuid: str = _ENTITY_UUID) -> None:
    """Seed one ACTIVE structural_size_conflation standing decision for *uuid*."""
    await store.upsert_entity_standing_decision(
        project_id=_PROJECT_ID,
        entity_uuid=uuid,
        grounds=GROUNDS_STRUCTURAL_SIZE_CONFLATION,
        decided_at=_DECIDED_AT,
        expires_at=_EXPIRES_AT,
        edge_count_at_decision=2,
        evidence=[],
        state=STATE_ACTIVE,
    )


def _service(ledger_obj, *, uuid: str = _ENTITY_UUID, name: str = 'X'):
    """AsyncMock memory service resolving get_entity to one node, with
    *ledger_obj* mounted on ``.recon_ledger`` (may be ``None``)."""
    service = AsyncMock()
    service.recon_ledger = ledger_obj
    service.get_entity = AsyncMock(
        return_value={'nodes': [{'uuid': uuid, 'name': name}], 'edges': []}
    )
    return service


def _state_with_finding(service):
    """(state, run_id, finding_id) — one started report + one actionable finding."""
    state = ReconReportState(ttl_seconds=300, clock=lambda: 0.0, memory_service=service)
    run_id = 'run-1'
    state.start_report(run_id=run_id, stage='reconciler', project_id=_PROJECT_ID)
    result = state.add_finding(
        run_id=run_id,
        severity='moderate',
        category='systemic_pattern',
        description='d',
        suggested_action='a',
        actionable=True,
        task_id='42',
        flag_type='orphaned_knowledge',
    )
    return state, run_id, result['finding_id']


class TestHookBCiteEntityAnnotation:
    @pytest.mark.asyncio
    async def test_active_decision_annotates_response_and_finding(self, ledger):
        """(a)+(b): an ACTIVE decision on the cited entity annotates BOTH the
        returned response (under ``standing_decision``) AND the finding."""
        await _seed_active(ledger)
        service = _service(ledger)
        state, run_id, finding_id = _state_with_finding(service)

        result = await state.cite_entity(run_id, finding_id, 'X')

        # (a) the returned dict keeps the {entity_uuid, canonical_name} contract
        # AND nests the annotation with the four required keys.
        assert result['entity_uuid'] == _ENTITY_UUID
        assert result['canonical_name'] == 'X'
        sd = result['standing_decision']
        assert set(sd) >= {'standing_decision_id', 'grounds', 'decided_at', 'summary'}
        assert sd['standing_decision_id'] == _expected_id()
        assert sd['grounds'] == GROUNDS_STRUCTURAL_SIZE_CONFLATION
        assert sd['decided_at'] == _DECIDED_AT
        assert isinstance(sd['summary'], str) and sd['summary']  # non-empty, no wording pin

        # (b) the persisted finding carries the same standing_decision_id.
        resolved = state._resolve_finding(run_id, finding_id)
        assert resolved is not None
        _, finding = resolved
        assert finding.standing_decision_id == _expected_id()

    @pytest.mark.asyncio
    async def test_no_active_decision_returns_plain_citation(self, ledger):
        """(c): a cited entity with NO active decision → plain citation, no
        ``standing_decision`` key, finding.standing_decision_id stays None."""
        # ledger is empty for this entity.
        service = _service(ledger)
        state, run_id, finding_id = _state_with_finding(service)

        result = await state.cite_entity(run_id, finding_id, 'X')

        assert 'standing_decision' not in result
        assert result['entity_uuid'] == _ENTITY_UUID
        resolved = state._resolve_finding(run_id, finding_id)
        assert resolved is not None
        _, finding = resolved
        assert finding.standing_decision_id is None

    @pytest.mark.asyncio
    async def test_absent_recon_ledger_skips_annotation(self):
        """(c): ``recon_ledger`` is None (fail-open) → plain citation, no
        annotation, finding.standing_decision_id stays None."""
        service = _service(None)
        state, run_id, finding_id = _state_with_finding(service)

        result = await state.cite_entity(run_id, finding_id, 'X')

        assert 'standing_decision' not in result
        resolved = state._resolve_finding(run_id, finding_id)
        assert resolved is not None
        _, finding = resolved
        assert finding.standing_decision_id is None


class TestHookBFindingProjections:
    """standing_decision_id flows through BOTH finding-dict projections
    (get_findings_for_run — the Stage-2 recon_report channel — and
    get_assembled_report) and NEVER changes finding counts."""

    @pytest.mark.asyncio
    async def test_projections_expose_standing_decision_id(self, ledger):
        await _seed_active(ledger)
        service = _service(ledger)
        state, run_id, finding_id = _state_with_finding(service)
        await state.cite_entity(run_id, finding_id, 'X')

        # (a) get_findings_for_run (the task-1966 channel Stage 2 polls).
        findings = state.get_findings_for_run(run_id)
        assert len(findings) == 1
        assert findings[0]['standing_decision_id'] == _expected_id()

        # (b) get_assembled_report flagged_items (sibling projection).
        report = state.get_assembled_report(run_id, 'reconciler')
        assert report is not None
        flagged = report['flagged_items']
        assert len(flagged) == 1
        assert flagged[0]['standing_decision_id'] == _expected_id()

    @pytest.mark.asyncio
    async def test_never_drops_finding_count_unchanged(self, ledger):
        """NEVER-DROPS: two equivalent runs — one whose cited entity has an
        active decision, one whose entity has none — yield the SAME finding
        count, and the no-decision finding carries standing_decision_id None."""
        # Run A: cited entity HAS an active decision.
        await _seed_active(ledger)
        service_a = _service(ledger, uuid=_ENTITY_UUID, name='X')
        state_a, run_a, finding_a = _state_with_finding(service_a)
        await state_a.cite_entity(run_a, finding_a, 'X')
        findings_a = state_a.get_findings_for_run(run_a)

        # Run B: cited entity has NO active decision (fresh, unseeded uuid).
        service_b = _service(ledger, uuid=_OTHER_UUID, name='Y')
        state_b, run_b, finding_b = _state_with_finding(service_b)
        await state_b.cite_entity(run_b, finding_b, 'Y')
        findings_b = state_b.get_findings_for_run(run_b)

        # Finding count is identical with/without a decision (annotation-only).
        assert len(findings_a) == len(findings_b) == 1
        assert findings_a[0]['standing_decision_id'] == _expected_id()
        assert findings_b[0]['standing_decision_id'] is None
