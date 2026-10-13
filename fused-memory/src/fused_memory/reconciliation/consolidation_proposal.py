"""The consolidation-proposal ledger row: one vocabulary for its writer and its reader.

Recon Stage 1 emits a proposal through ``server/tools.py::propose_consolidation``
and the auto-consolidation executor reads it back. Both go through this module,
so the row's identity, payload keys and states have one home: the executor
lists rows with ``ReconLedgerStore.list_by_kind(project_id,
RECORD_KIND_CONSOLIDATION_PROPOSAL, ProposalState.PROPOSED)`` and decodes each
with :func:`from_ledger_record`, rather than growing a second reader.

Clock-injected like :mod:`fused_memory.reconciliation.recon_ledger`: the
builder takes ``now`` rather than reading a clock. ``rationale`` and
``seeded_from`` are reviewer-facing and never reach canonical text (PRD D2);
only ``claim`` does.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

from fused_memory.models.reconciliation import StageId
from fused_memory.reconciliation.recon_ledger import ReconLedgerRecord, ReconLedgerStore

RECORD_KIND_CONSOLIDATION_PROPOSAL = 'consolidation_proposal'

#: The one caller that may propose (PRD D12): recon Stage 1. The stage deny
#: lists enforce the same rule from the other side.
PROPOSER_AGENT_ID = f'recon-stage-{StageId.memory_consolidator.value}'


class ProposalState(StrEnum):
    PROPOSED = 'proposed'


_MEMBER_IDS_KEY = 'member_ids'
_CATEGORY_KEY = 'category'
_CLAIM_KEY = 'claim'
_RATIONALE_KEY = 'rationale'
_SEEDED_FROM_KEY = 'seeded_from'


@dataclass(frozen=True)
class ConsolidationProposal:
    """One proposed near-duplicate cluster, identified by ``(project_id, topic, run_id)``."""

    project_id: str
    run_id: str
    topic: str
    member_ids: tuple[str, ...]
    category: str
    claim: str
    rationale: str
    seeded_from: str | None = None

    def ledger_key(self) -> dict[str, str]:
        """The row's five-part identity, usable as ``ReconLedgerStore.get_by_identity(**key)``."""
        return {
            'project_id': self.project_id,
            'record_kind': RECORD_KIND_CONSOLIDATION_PROPOSAL,
            'task_id': '',
            'flag_type': self.topic,
            'run_id': self.run_id,
        }

    def to_ledger_record(self, *, now: datetime, ttl_hours: int) -> ReconLedgerRecord:
        """This proposal as a ``proposed`` row that expires *ttl_hours* after *now*.

        Never builds a row without ``expires_at``: ``ReconLedgerStore.gc`` reaps
        only rows that carry one, so an unstamped proposal would live forever.
        """
        if now.tzinfo is None:
            raise ValueError(f'now must be timezone-aware, got naive {now!r}')
        if isinstance(ttl_hours, bool) or not isinstance(ttl_hours, int) or ttl_hours <= 0:
            raise ValueError(f'ttl_hours must be a positive int, got {ttl_hours!r}')
        created = now.astimezone(UTC)
        payload = {
            _MEMBER_IDS_KEY: list(self.member_ids),
            _CATEGORY_KEY: self.category,
            _CLAIM_KEY: self.claim,
            _RATIONALE_KEY: self.rationale,
            _SEEDED_FROM_KEY: self.seeded_from,
        }
        return ReconLedgerRecord(
            **self.ledger_key(),
            payload_json=json.dumps(payload),
            state=ProposalState.PROPOSED.value,
            created_at=created.isoformat(),
            expires_at=(created + timedelta(hours=ttl_hours)).isoformat(),
        )


async def record_proposal(
    ledger: ReconLedgerStore,
    proposal: ConsolidationProposal,
    *,
    now: datetime,
    ttl_hours: int,
) -> bool:
    """Write *proposal* as a ``proposed`` row; ``False`` if its identity was already acted on.

    A re-proposal overwrites the earlier row only while that row is still
    ``proposed``. Once code has moved it on, resetting it would hand the
    executor the same cluster twice, so the stored row is left as it is.
    """
    return await ledger.upsert_if_state(
        proposal.to_ledger_record(now=now, ttl_hours=ttl_hours),
        existing_state=ProposalState.PROPOSED.value,
    )


def from_ledger_record(record: ReconLedgerRecord) -> ConsolidationProposal:
    """Decode a proposal row, refusing anything :meth:`ConsolidationProposal.to_ledger_record`
    would not have written.

    Raises ``ValueError`` naming the offending field: a ledger row's JSON is
    never trusted for shape.
    """
    if record.record_kind != RECORD_KIND_CONSOLIDATION_PROPOSAL:
        raise ValueError(
            f'record_kind {record.record_kind!r} is not {RECORD_KIND_CONSOLIDATION_PROPOSAL!r}'
        )
    try:
        payload = json.loads(record.payload_json)
    except ValueError as exc:
        raise ValueError(f'payload_json is not JSON: {record.payload_json[:200]!r}') from exc
    if not isinstance(payload, dict):
        raise ValueError(f'payload_json is not a JSON object: {record.payload_json[:200]!r}')

    member_ids = payload.get(_MEMBER_IDS_KEY)
    if not isinstance(member_ids, list) or not all(isinstance(m, str) for m in member_ids):
        raise ValueError(f'payload {_MEMBER_IDS_KEY!r} is not a list of strings: {member_ids!r}')
    seeded_from = payload.get(_SEEDED_FROM_KEY)
    if seeded_from is not None and not isinstance(seeded_from, str):
        raise ValueError(f'payload {_SEEDED_FROM_KEY!r} is not a string or null: {seeded_from!r}')

    return ConsolidationProposal(
        project_id=record.project_id,
        run_id=record.run_id,
        topic=record.flag_type,
        member_ids=tuple(member_ids),
        category=_payload_str(payload, _CATEGORY_KEY),
        claim=_payload_str(payload, _CLAIM_KEY),
        rationale=_payload_str(payload, _RATIONALE_KEY),
        seeded_from=seeded_from,
    )


def _payload_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f'payload {key!r} is not a string: {value!r}')
    return value
