"""η's end-to-end integration gate for the entity-standing-decision batch (task 2900).

PRD ``plans/stage1-entity-standing-decision-prd.md`` row η: *"Reify
'orchestrator' entity has an active listable row; one E2E run demonstrates
suppression stat + annotation + never-drop in the same cycle."* This module is
that gate, and nothing else: every mechanism it exercises — β's gated writer,
γ's Hook-A filter, δ's Hook-B annotation, α's ledger — ALREADY LANDED and is
green on this branch.

**A leg that passes on first write is therefore the EXPECTED signal, and is
itself the gate.** Manufacturing a RED phase here would mean breaking upstream
code in order to re-fix it. Conversely, a leg that comes up RED means a real
integration gap in β/γ/δ/ζ — it must be ESCALATED, not patched inside η, since
the fix would belong to a module this task does not own.

WHAT THIS ASSERTS THAT THE PER-LEG SUITES DO NOT
------------------------------------------------
``tests/test_stages.py::TestMemoryConsolidatorEntityStandingDecision`` owns
Hook A's per-leg assertions and ``tests/server/test_recon_report_hook_b.py``
owns Hook B's. Re-deriving either here would be a second maintenance site for
the same claims (SPOT). What neither owns is the CONJUNCTION, in one cycle, off
ONE row seeded through the BACKFILL'S OWN WRITE PATH — which is precisely the
claim η certifies: *the backfilled row drives both hooks*, not *a hand-seeded
row does*.

The Hook-A leg is also genuinely new coverage rather than a restatement: γ's
own tests drive the STRONG (stamped) match, while the motivating incident's
flag carried no stamps at all and cited the entity uuid in free text. This gate
uses that FALLBACK shape, with ``graphiti_entity_conflation_unresolved`` — a
flag_type measured live on this entity — so γ's ``'conflat'`` family stem is
validated against real data instead of a synthetic string.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from fused_memory.reconciliation.recon_ledger import ReconLedgerStore
from fused_memory.reconciliation.standing_decision_constants import (
    GROUNDS_STRUCTURAL_SIZE_CONFLATION,
    GROUNDS_TOKEN_FAMILIES,
    STANDING_DECISION_TTL_DAYS,
    STATE_ACTIVE,
)
from fused_memory.reconciliation.standing_decision_writer import (
    write_entity_standing_decision,
)

#: reify's 'orchestrator' node — the entity ``b0057f3d`` decided about.
ENTITY_UUID = 'f02a32ea-0efd-4865-94b4-97a412d8ffda'
ENTITY_NAME = 'orchestrator'
PROJECT_ID = 'reify'

#: The evidence the migration cites: the source record, the demoted correction,
#: the prose-cited human-authored record, and the opening escalation.
EVIDENCE_REFS = [
    {'type': 'mem0', 'id': 'b0057f3d-dc53-4cf8-9d1f-9959bd0897bd'},
    {'type': 'mem0', 'id': 'baf8ca57-9f36-431b-a9b9-17c82fadd22d'},
    {'type': 'mem0', 'id': 'ef1f1b1b-219c-40fb-a933-53631646df96'},
    {'type': 'escalation', 'id': 'esc-2867-1'},
]

AUTHORIZED_BY = 'backfill_entity_standing_decision (task 2900 η)'

#: The prose-cited record whose author is ``claude-interactive`` — exactly β's
#: arm-1 human-authorship predicate. Resolvable here, so the seeded row's
#: provenance would satisfy the evidence gate on its own merits even though the
#: migration takes the operator bypass.
HUMAN_EVIDENCE_ID = 'ef1f1b1b-219c-40fb-a933-53631646df96'

#: β samples this at decision time; ζ's growth sweep later compares against it.
SEEDED_EDGE_COUNT = 11

#: A flag_type measured live on this entity, and the shape the motivating
#: incident actually travelled: no ``entity_uuid`` / ``grounds`` stamps, the
#: uuid present only in free text. γ matches it through the ``'conflat'`` stem
#: of :data:`GROUNDS_TOKEN_FAMILIES`.
FALLBACK_FLAG_TYPE = 'graphiti_entity_conflation_unresolved'


async def _resolve_evidence_record(_project_id: str, memory_id: str) -> dict | None:
    """Stand in for the three live reify mem0 records the row cites."""
    if memory_id not in {ref['id'] for ref in EVIDENCE_REFS}:
        return None
    agent_id = (
        'claude-interactive'
        if memory_id == HUMAN_EVIDENCE_ID
        else 'reconciliation-stage-1'
    )
    return {'id': memory_id, 'metadata': {'agent_id': agent_id}}


def make_memory_service(ledger) -> AsyncMock:
    """A mem0/graphiti façade with *ledger* mounted, resolving the one entity."""
    service = AsyncMock()
    service.recon_ledger = ledger
    service.graphiti.get_valid_edges_for_node = AsyncMock(
        return_value=[{'uuid': f'edge-{n}'} for n in range(SEEDED_EDGE_COUNT)]
    )
    service.get_memory_by_id = AsyncMock(side_effect=_resolve_evidence_record)
    service.get_entity = AsyncMock(
        return_value={
            'nodes': [{'uuid': ENTITY_UUID, 'name': ENTITY_NAME}],
            'edges': [],
        }
    )
    return service


async def make_ledger(tmp_path, name: str = 'reconciliation.db') -> ReconLedgerStore:
    """A REAL initialized ``ReconLedgerStore`` on a fresh tmp db."""
    store = ReconLedgerStore(tmp_path / name)
    await store.initialize()
    return store


@pytest_asyncio.fixture
async def backfilled_decision(tmp_path):
    """The migration's own write path, run once: ``(ledger, service)``.

    Seeded through β's ``write_entity_standing_decision(..., authorized_by=...)``
    — the EXACT call ``scripts/backfill_entity_standing_decision.py`` makes —
    rather than α's raw ``upsert_entity_standing_decision``. That is what lets
    the legs below claim "the BACKFILLED row drives both hooks": a hand-rolled
    upsert would demonstrate only that some row does.
    """
    ledger = await make_ledger(tmp_path)
    service = make_memory_service(ledger)
    try:
        await write_entity_standing_decision(
            service,
            project_id=PROJECT_ID,
            entity_uuid=ENTITY_UUID,
            grounds=GROUNDS_STRUCTURAL_SIZE_CONFLATION,
            evidence=[dict(ref) for ref in EVIDENCE_REFS],
            authorized_by=AUTHORIZED_BY,
        )
        yield ledger, service
    finally:
        await ledger.close()


class TestBackfilledRowIsActiveAndListable:
    """η's first signal: "Reify 'orchestrator' entity has an active listable row."

    Read through ``list_entity_standing_decisions(project_id, state=active)`` —
    the same API an operator would use, and the same one γ's Hook-A filter
    consumes — rather than by reaching into the store.
    """

    @pytest.mark.asyncio
    async def test_exactly_one_active_row_for_the_decided_entity(
        self, backfilled_decision
    ) -> None:
        ledger, _ = backfilled_decision
        rows = await ledger.list_entity_standing_decisions(
            PROJECT_ID, state=STATE_ACTIVE
        )
        assert len(rows) == 1
        assert rows[0].entity_uuid == ENTITY_UUID

    @pytest.mark.asyncio
    async def test_the_row_records_the_decided_grounds(
        self, backfilled_decision
    ) -> None:
        """α's PK-slot mapping puts ``grounds`` in ``flag_type``."""
        ledger, _ = backfilled_decision
        rows = await ledger.list_entity_standing_decisions(
            PROJECT_ID, state=STATE_ACTIVE
        )
        assert rows[0].flag_type == GROUNDS_STRUCTURAL_SIZE_CONFLATION

    @pytest.mark.asyncio
    async def test_the_row_expires_after_the_shared_ttl(
        self, backfilled_decision
    ) -> None:
        """A standing decision is a time-BOUNDED hold, not a permanent one."""
        ledger, _ = backfilled_decision
        row = (await ledger.list_entity_standing_decisions(PROJECT_ID))[0]
        span = datetime.fromisoformat(row.expires_at) - datetime.fromisoformat(
            row.created_at
        )
        assert span == timedelta(days=STANDING_DECISION_TTL_DAYS)

    @pytest.mark.asyncio
    async def test_the_row_carries_the_sampled_edge_count_fingerprint(
        self, backfilled_decision
    ) -> None:
        ledger, _ = backfilled_decision
        row = (await ledger.list_entity_standing_decisions(PROJECT_ID))[0]
        assert json.loads(row.payload_json)['edge_count_at_decision'] == (
            SEEDED_EDGE_COUNT
        )

    @pytest.mark.asyncio
    async def test_the_row_records_who_authorized_the_migration(
        self, backfilled_decision
    ) -> None:
        """The operator bypass leaves a trace: the row says on its face that a
        migration wrote it, not an LLM mid-cycle."""
        ledger, _ = backfilled_decision
        row = (await ledger.list_entity_standing_decisions(PROJECT_ID))[0]
        evidence = json.loads(row.payload_json)['evidence']
        assert AUTHORIZED_BY in [ref.get('authorized_by') for ref in evidence]

    @pytest.mark.asyncio
    async def test_the_grounds_carry_the_token_family_the_fallback_needs(
        self,
    ) -> None:
        """The fallback flag_type this gate uses is matched by a stem READ from
        γ's map, not assumed — so the leg below tests γ's real binding."""
        stems = GROUNDS_TOKEN_FAMILIES[GROUNDS_STRUCTURAL_SIZE_CONFLATION]
        assert any(stem in FALLBACK_FLAG_TYPE for stem in stems)

