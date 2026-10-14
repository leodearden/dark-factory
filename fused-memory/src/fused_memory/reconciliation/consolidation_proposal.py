"""The consolidation-proposal ledger row: one vocabulary for its writer and its reader.

Recon Stage 1 emits a proposal through ``server/tools.py::propose_consolidation``
and the auto-consolidation executor reads it back. Both go through this module,
so the row's identity, payload keys, lifecycle states, recorded outcome and the
pure readers over a topic's rows (:func:`refusal_streak`,
:func:`newest_per_topic`) have one home: the executor lists rows with
``ReconLedgerStore.list_by_kind(project_id, RECORD_KIND_CONSOLIDATION_PROPOSAL,
ProposalState.PROPOSED)``, decodes each with :func:`from_ledger_record`, and
moves it on with :func:`record_outcome`, rather than growing a second reader.

Clock-injected like :mod:`fused_memory.reconciliation.recon_ledger`: the
builder takes ``now`` and an outcome carries a caller-supplied ``acted_at``
rather than reading a clock. ``rationale`` and
``seeded_from`` are reviewer-facing and never reach canonical text (PRD D2);
only ``claim`` does.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

from fused_memory.models.reconciliation import StageId
from fused_memory.reconciliation.consolidation_auto import AutoOutcome
from fused_memory.reconciliation.recon_ledger import ReconLedgerRecord, ReconLedgerStore

RECORD_KIND_CONSOLIDATION_PROPOSAL = 'consolidation_proposal'

#: The one caller that may propose (PRD D12): recon Stage 1. The stage deny
#: lists enforce the same rule from the other side.
PROPOSER_AGENT_ID = f'recon-stage-{StageId.memory_consolidator.value}'


class ProposalState(StrEnum):
    PROPOSED = 'proposed'
    ADDRESSED = 'addressed'
    DRY_RUN = 'dry_run'
    REFUSED = 'refused'
    GATED = 'gated'
    SUPERSEDED = 'superseded'
    REVERTED = 'reverted'


_MEMBER_IDS_KEY = 'member_ids'
_CATEGORY_KEY = 'category'
_CLAIM_KEY = 'claim'
_RATIONALE_KEY = 'rationale'
_SEEDED_FROM_KEY = 'seeded_from'
_OUTCOME_KEY = 'outcome'


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
    payload = _payload_object(record)

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


@dataclass(frozen=True)
class ProposalOutcome:
    """What code did with one proposal row, stored on the row itself.

    An outcome that touched the corpus (``executed``) is the authoritative
    record of the run and the source a reversal reads, so its row never expires.
    """

    state: ProposalState
    acted_run_id: str
    acted_at: str
    verdict: AutoOutcome | None = None
    reason_codes: tuple[str, ...] = ()
    predicate_version: str | None = None
    canonical_id: str | None = None
    canonical_minted: bool = False
    tagged_ids: tuple[str, ...] = ()
    tag_failure_ids: tuple[str, ...] = ()
    stripped_ids: tuple[str, ...] = ()
    gate_task_id: str | None = None
    closure_codes: tuple[str, ...] = ()

    @property
    def executed(self) -> bool:
        return (
            self.state in (ProposalState.ADDRESSED, ProposalState.GATED)
            and self.canonical_id is not None
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            'state': self.state.value,
            'acted_run_id': self.acted_run_id,
            'acted_at': self.acted_at,
            'verdict': None if self.verdict is None else self.verdict.value,
            'reason_codes': list(self.reason_codes),
            'predicate_version': self.predicate_version,
            'canonical_id': self.canonical_id,
            'canonical_minted': self.canonical_minted,
            'tagged_ids': list(self.tagged_ids),
            'tag_failure_ids': list(self.tag_failure_ids),
            'stripped_ids': list(self.stripped_ids),
            'gate_task_id': self.gate_task_id,
            'closure_codes': list(self.closure_codes),
        }

    @classmethod
    def from_payload(cls, payload: object) -> ProposalOutcome:
        """Decode :meth:`to_payload`'s shape; ``ValueError`` naming any bad field."""
        if not isinstance(payload, dict):
            raise ValueError(f'outcome is not a JSON object: {payload!r}')
        verdict = _optional_str(payload, 'verdict')
        canonical_minted = payload.get('canonical_minted')
        if not isinstance(canonical_minted, bool):
            raise ValueError(f'outcome canonical_minted is not a bool: {canonical_minted!r}')
        return cls(
            state=ProposalState(_payload_str(payload, 'state')),
            acted_run_id=_payload_str(payload, 'acted_run_id'),
            acted_at=_payload_str(payload, 'acted_at'),
            verdict=None if verdict is None else AutoOutcome(verdict),
            reason_codes=_str_tuple(payload, 'reason_codes'),
            predicate_version=_optional_str(payload, 'predicate_version'),
            canonical_id=_optional_str(payload, 'canonical_id'),
            canonical_minted=canonical_minted,
            tagged_ids=_str_tuple(payload, 'tagged_ids'),
            tag_failure_ids=_str_tuple(payload, 'tag_failure_ids'),
            stripped_ids=_str_tuple(payload, 'stripped_ids'),
            gate_task_id=_optional_str(payload, 'gate_task_id'),
            closure_codes=_str_tuple(payload, 'closure_codes'),
        )


async def record_outcome(
    ledger: ReconLedgerStore,
    record: ReconLedgerRecord,
    outcome: ProposalOutcome,
) -> bool:
    """Move *record* to ``outcome.state`` with *outcome* on its payload.

    A compare-and-set on the state *record* was read in: ``False`` (and no
    write) if the stored row has moved since. The proposal's own payload keys
    are kept, so :func:`from_ledger_record` still decodes the row. An executed
    outcome drops ``expires_at`` so ``ReconLedgerStore.gc`` never reaps it.
    """
    if outcome.state is ProposalState.PROPOSED:
        raise ValueError('an outcome cannot move a row back to proposed')
    payload = _payload_object(record)
    payload[_OUTCOME_KEY] = outcome.to_payload()
    moved = dataclasses.replace(
        record,
        state=outcome.state.value,
        payload_json=json.dumps(payload),
        expires_at=None if outcome.executed else record.expires_at,
    )
    return await ledger.upsert_if_state(moved, existing_state=record.state)


def outcome_from_record(record: ReconLedgerRecord) -> ProposalOutcome | None:
    """The outcome :func:`record_outcome` stored on *record*, or ``None`` if never acted on."""
    payload = _payload_object(record)
    if _OUTCOME_KEY not in payload:
        return None
    return ProposalOutcome.from_payload(payload[_OUTCOME_KEY])


def refusal_streak(rows: Iterable[ReconLedgerRecord]) -> int:
    """Count the trailing ``refused`` rows of one topic, in the given order.

    ``proposed`` rows (not yet acted on) and ``superseded`` rows (never
    evaluated) neither count nor break the streak; any other acted state
    resets it.
    """
    streak = 0
    for row in rows:
        state = ProposalState(row.state)
        if state in (ProposalState.PROPOSED, ProposalState.SUPERSEDED):
            continue
        streak = streak + 1 if state is ProposalState.REFUSED else 0
    return streak


def newest_per_topic(
    records: Sequence[ReconLedgerRecord],
) -> tuple[tuple[ReconLedgerRecord, ...], tuple[ReconLedgerRecord, ...]]:
    """Split *records* into the newest row per topic and the rest.

    Newest is the greatest ``(created_at, run_id)``. Both returned tuples keep
    input order.
    """
    newest: dict[str, ReconLedgerRecord] = {}
    for record in records:
        incumbent = newest.get(record.flag_type)
        if incumbent is None or (record.created_at, record.run_id) > (
            incumbent.created_at,
            incumbent.run_id,
        ):
            newest[record.flag_type] = record
    kept = set(map(id, newest.values()))
    keep = tuple(r for r in records if id(r) in kept)
    superseded = tuple(r for r in records if id(r) not in kept)
    return keep, superseded


def _payload_object(record: ReconLedgerRecord) -> dict[str, Any]:
    try:
        payload = json.loads(record.payload_json)
    except ValueError as exc:
        raise ValueError(f'payload_json is not JSON: {record.payload_json[:200]!r}') from exc
    if not isinstance(payload, dict):
        raise ValueError(f'payload_json is not a JSON object: {record.payload_json[:200]!r}')
    return payload


def _optional_str(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is not None and not isinstance(value, str):
        raise ValueError(f'payload {key!r} is not a string or null: {value!r}')
    return value


def _str_tuple(payload: dict[str, Any], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError(f'payload {key!r} is not a list of strings: {value!r}')
    return tuple(value)
