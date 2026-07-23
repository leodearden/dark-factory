"""Tests for the merge_entities → entity_standing_decision invalidation hook (task 2899 ζ).

Covers ``MemoryService._expire_standing_decisions_for_merge`` (the merge-invalidation
helper) and its best-effort wiring into ``MemoryService.merge_entities``:

* a successful merge expires ACTIVE standing decisions on EITHER merged uuid with
  ``expiry_reason='merge'`` (the post-merge entity is a new subject, so a prior
  dismissal no longer applies — research fact 9's dangling-uuid fix), and
* a hook failure NEVER breaks the authoritative merge (the invalidation is a
  secondary consequence — a failed hook simply leaves the row ACTIVE, caught
  later by TTL or the growth sweep).

Scoped to ζ's own capabilities only: the row-state flips (``expired/merge``) and
the fail-safe wiring. NOT Hook A re-read (γ/η) nor the TTL gc flip (α).
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from fused_memory.reconciliation.recon_ledger import (
    ReconLedgerRecord,
    ReconLedgerStore,
)
from fused_memory.reconciliation.standing_decision_constants import (
    EXPIRY_REASON_MERGE,
    GROUNDS_STRUCTURAL_SIZE_CONFLATION,
    RECORD_KIND_ENTITY_STANDING_DECISION,
    STATE_ACTIVE,
    STATE_EXPIRED,
)

_PROJECT_ID = 'dark_factory'
_DEPRECATED_UUID = 'dddddddd-1111-1111-1111-111111111111'
_SURVIVING_UUID = 'ssssssss-2222-2222-2222-222222222222'
_UNRELATED_UUID = 'uuuuuuuu-3333-3333-3333-333333333333'
_DECIDED_AT = '2026-07-01T00:00:00+00:00'
_EXPIRES_AT = '2026-09-29T00:00:00+00:00'


async def _seed_active_decision(ledger, *, entity_uuid, edge_count=5):
    """Seed one ACTIVE entity_standing_decision row on *entity_uuid*."""
    await ledger.upsert_entity_standing_decision(
        project_id=_PROJECT_ID,
        entity_uuid=entity_uuid,
        grounds=GROUNDS_STRUCTURAL_SIZE_CONFLATION,
        decided_at=_DECIDED_AT,
        expires_at=_EXPIRES_AT,
        edge_count_at_decision=edge_count,
        evidence=[{'type': 'mem0', 'id': 'ev-1'}],
        state=STATE_ACTIVE,
    )


async def _active_uuids(ledger):
    """The set of entity_uuids that still carry an ACTIVE standing decision."""
    rows = await ledger.list_entity_standing_decisions(_PROJECT_ID, state=STATE_ACTIVE)
    return {r.entity_uuid for r in rows}


async def _expired_row(ledger, entity_uuid):
    """The single EXPIRED standing-decision row for *entity_uuid*, or None."""
    rows = await ledger.list_entity_standing_decisions(_PROJECT_ID, state=STATE_EXPIRED)
    return next((r for r in rows if r.entity_uuid == entity_uuid), None)


@pytest_asyncio.fixture
async def merge_memory_service(mock_config, tmp_path):
    """MemoryService with a REAL initialized ReconLedgerStore wired via
    ``set_recon_ledger`` and a mocked graphiti backend whose ``merge_entities``
    succeeds (mirrors test_merge_entities.py's service fixture + the
    writer test's real-ledger-on-tmp_path fixture)."""
    from fused_memory.services.memory_service import MemoryService

    ledger = ReconLedgerStore(tmp_path / 'reconciliation.db')
    await ledger.initialize()

    svc = MemoryService(mock_config)
    svc.set_recon_ledger(ledger)
    svc.graphiti = MagicMock()
    svc.graphiti.merge_entities = AsyncMock(return_value={
        'surviving_uuid': _SURVIVING_UUID,
        'surviving_name': 'SurvivingName',
        'deprecated_uuid': _DEPRECATED_UUID,
        'deprecated_name': 'DeprecatedName',
        'edges_redirected': {
            'outgoing_redirected': 2, 'incoming_redirected': 1, 'inter_node_deleted': 0,
        },
        'surviving_summary': {'before': 'old', 'after': 'new', 'edge_count': 3},
    })
    svc.mem0 = MagicMock()
    svc.durable_queue = MagicMock()
    svc.durable_queue.enqueue = AsyncMock(return_value=1)
    try:
        yield svc
    finally:
        await ledger.close()


class TestExpireStandingDecisionsForMerge:
    """MemoryService._expire_standing_decisions_for_merge(project_id, dep, sur)
    expires ACTIVE decisions on EITHER merged uuid with reason='merge', leaving
    unrelated rows untouched, and returns the count expired."""

    @pytest.mark.asyncio
    async def test_expires_row_on_deprecated_uuid(self, merge_memory_service):
        """(a) A row on the DEPRECATED uuid flips to expired with
        payload expiry_reason=='merge'."""
        svc = merge_memory_service
        ledger = svc.recon_ledger
        await _seed_active_decision(ledger, entity_uuid=_DEPRECATED_UUID)

        count = await svc._expire_standing_decisions_for_merge(
            _PROJECT_ID, _DEPRECATED_UUID, _SURVIVING_UUID
        )

        assert count == 1
        assert _DEPRECATED_UUID not in await _active_uuids(ledger)
        flipped = await _expired_row(ledger, _DEPRECATED_UUID)
        assert flipped is not None
        assert flipped.state == STATE_EXPIRED
        assert json.loads(flipped.payload_json)['expiry_reason'] == EXPIRY_REASON_MERGE

    @pytest.mark.asyncio
    async def test_expires_row_on_surviving_uuid(self, merge_memory_service):
        """(b) A row on the SURVIVING uuid also flips to expired/merge."""
        svc = merge_memory_service
        ledger = svc.recon_ledger
        await _seed_active_decision(ledger, entity_uuid=_SURVIVING_UUID)

        count = await svc._expire_standing_decisions_for_merge(
            _PROJECT_ID, _DEPRECATED_UUID, _SURVIVING_UUID
        )

        assert count == 1
        assert _SURVIVING_UUID not in await _active_uuids(ledger)
        flipped = await _expired_row(ledger, _SURVIVING_UUID)
        assert flipped is not None
        assert flipped.state == STATE_EXPIRED
        assert json.loads(flipped.payload_json)['expiry_reason'] == EXPIRY_REASON_MERGE

    @pytest.mark.asyncio
    async def test_unrelated_uuid_stays_active(self, merge_memory_service):
        """(c) A row on an unrelated uuid is left ACTIVE (count 0)."""
        svc = merge_memory_service
        ledger = svc.recon_ledger
        await _seed_active_decision(ledger, entity_uuid=_UNRELATED_UUID)

        count = await svc._expire_standing_decisions_for_merge(
            _PROJECT_ID, _DEPRECATED_UUID, _SURVIVING_UUID
        )

        assert count == 0
        assert _UNRELATED_UUID in await _active_uuids(ledger)
        assert await _expired_row(ledger, _UNRELATED_UUID) is None

    @pytest.mark.asyncio
    async def test_ledger_none_returns_zero(self, merge_memory_service):
        """(d) recon_ledger unwired ⇒ returns 0, no raise (did-not-run)."""
        svc = merge_memory_service
        svc.recon_ledger = None
        count = await svc._expire_standing_decisions_for_merge(
            _PROJECT_ID, _DEPRECATED_UUID, _SURVIVING_UUID
        )
        assert count == 0

    @pytest.mark.asyncio
    async def test_return_value_counts_both_expired(self, merge_memory_service):
        """Return value == number of rows expired: both merged uuids flip while
        the unrelated row stays ACTIVE (robust to multiple grounds/rows)."""
        svc = merge_memory_service
        ledger = svc.recon_ledger
        await _seed_active_decision(ledger, entity_uuid=_DEPRECATED_UUID)
        await _seed_active_decision(ledger, entity_uuid=_SURVIVING_UUID)
        await _seed_active_decision(ledger, entity_uuid=_UNRELATED_UUID)

        count = await svc._expire_standing_decisions_for_merge(
            _PROJECT_ID, _DEPRECATED_UUID, _SURVIVING_UUID
        )

        assert count == 2
        assert await _active_uuids(ledger) == {_UNRELATED_UUID}

    @pytest.mark.asyncio
    async def test_malformed_row_does_not_block_sibling_flip(self, merge_memory_service):
        """Per-row fail-safe: a malformed ACTIVE row on ONE merged uuid (payload
        missing ``edge_count_at_decision`` → ``KeyError`` inside the flip helper)
        must NOT abort the loop before the well-formed sibling row on the OTHER
        merged uuid is flipped. The bad row is left ACTIVE (re-caught later by
        TTL or the growth sweep); the good row still flips to expired/merge; the
        returned count reflects only the successful flip — mirroring the per-row
        guard in ``_sweep_entity_standing_decision_growth``."""
        svc = merge_memory_service
        ledger = svc.recon_ledger
        # Well-formed ACTIVE row on the deprecated uuid.
        await _seed_active_decision(ledger, entity_uuid=_DEPRECATED_UUID)
        # Malformed ACTIVE row on the surviving uuid seeded via the low-level
        # upsert: its payload OMITS edge_count_at_decision, which
        # expire_entity_standing_decision reads directly (→ KeyError). The normal
        # upsert_entity_standing_decision path always writes that key, so the raw
        # record is the only faithful way to reproduce the malformed-payload case.
        await ledger.upsert(
            ReconLedgerRecord(
                project_id=_PROJECT_ID,
                record_kind=RECORD_KIND_ENTITY_STANDING_DECISION,
                flag_type=GROUNDS_STRUCTURAL_SIZE_CONFLATION,
                run_id=_SURVIVING_UUID,
                entity_uuid=_SURVIVING_UUID,
                payload_json=json.dumps(
                    {'grounds': GROUNDS_STRUCTURAL_SIZE_CONFLATION}
                ),
                state=STATE_ACTIVE,
                created_at=_DECIDED_AT,
                expires_at=_EXPIRES_AT,
            )
        )

        count = await svc._expire_standing_decisions_for_merge(
            _PROJECT_ID, _DEPRECATED_UUID, _SURVIVING_UUID
        )

        # Only the well-formed row flipped; the malformed row's KeyError was
        # swallowed per-row and did NOT abort the sibling's flip.
        assert count == 1
        good = await _expired_row(ledger, _DEPRECATED_UUID)
        assert good is not None
        assert good.state == STATE_EXPIRED
        assert json.loads(good.payload_json)['expiry_reason'] == EXPIRY_REASON_MERGE
        # The malformed row is left ACTIVE for TTL / the growth sweep to re-catch.
        assert _SURVIVING_UUID in await _active_uuids(ledger)


class TestMergeEntitiesInvalidationWiring:
    """merge_entities best-effort-invalidates standing decisions on a SUCCESSFUL
    merge, and a hook failure never breaks the authoritative merge."""

    @pytest.mark.asyncio
    async def test_merge_expires_active_decision_on_either_uuid(self, merge_memory_service):
        """A seeded ACTIVE row on either merged uuid ends expired/merge after a
        successful merge_entities, and the graphiti merge result is returned
        unchanged."""
        svc = merge_memory_service
        ledger = svc.recon_ledger
        await _seed_active_decision(ledger, entity_uuid=_DEPRECATED_UUID)
        await _seed_active_decision(ledger, entity_uuid=_SURVIVING_UUID)

        result = await svc.merge_entities(
            deprecated_uuid=_DEPRECATED_UUID,
            surviving_uuid=_SURVIVING_UUID,
            project_id=_PROJECT_ID,
        )

        # The authoritative merge result is unchanged.
        assert result['surviving_uuid'] == _SURVIVING_UUID
        svc.graphiti.merge_entities.assert_awaited_once_with(
            _DEPRECATED_UUID, _SURVIVING_UUID, group_id=_PROJECT_ID
        )

        # Both rows flipped to expired/merge; none remain ACTIVE.
        assert await _active_uuids(ledger) == set()
        for entity_uuid in (_DEPRECATED_UUID, _SURVIVING_UUID):
            flipped = await _expired_row(ledger, entity_uuid)
            assert flipped is not None
            assert flipped.state == STATE_EXPIRED
            assert (
                json.loads(flipped.payload_json)['expiry_reason'] == EXPIRY_REASON_MERGE
            )

    @pytest.mark.asyncio
    async def test_hook_failure_does_not_break_merge(self, merge_memory_service):
        """A ledger failure inside the invalidation hook is swallowed: the hook is
        attempted (list called) but merge_entities still returns the graphiti
        result and does not propagate the error."""
        svc = merge_memory_service
        # Make the ledger enumeration raise so the hook fails mid-merge.
        svc.recon_ledger.list_entity_standing_decisions = AsyncMock(
            side_effect=RuntimeError('ledger down')
        )

        result = await svc.merge_entities(
            deprecated_uuid=_DEPRECATED_UUID,
            surviving_uuid=_SURVIVING_UUID,
            project_id=_PROJECT_ID,
        )

        # Merge is authoritative — its result survives the hook failure.
        assert result['surviving_uuid'] == _SURVIVING_UUID
        svc.graphiti.merge_entities.assert_awaited_once()
        # The hook WAS attempted (its failure was swallowed, not skipped).
        svc.recon_ledger.list_entity_standing_decisions.assert_awaited()
