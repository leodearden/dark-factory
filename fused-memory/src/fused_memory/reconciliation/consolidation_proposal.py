"""The auto-consolidation ledger vocabulary: the proposal row and the two rows beside it.

Recon Stage 1 emits a proposal through ``server/tools.py::propose_consolidation``
and the auto-consolidation executor reads it back. Both go through this module,
so the row's identity, payload keys, lifecycle states, recorded outcome and the
pure readers over a topic's rows (:func:`refusal_streak`,
:func:`newest_per_topic`) have one home: the executor lists rows with
``ReconLedgerStore.list_by_kind(project_id, RECORD_KIND_CONSOLIDATION_PROPOSAL,
ProposalState.PROPOSED)``, decodes each with :func:`from_ledger_record`, and
moves it on with :func:`record_outcome`, rather than growing a second reader.

The outcome also records what the adjudicator flow did to a proposal (PRD
``plans/consolidation-adjudicator-prd.md`` K6). Its decision and reason code
are stored as the strings recorded, so an aged row still decodes after the
adjudicator's vocabulary changes. Beside the proposal row that flow keeps an
attempt row per members key (:func:`adjudication_attempt_key`, K4) and a
calibration row per adjudicator identity (:func:`adjudicator_calibration_key`,
K5); this module defines their identities, and their writers own the rest.

Clock-injected like :mod:`fused_memory.reconciliation.recon_ledger`: the
builder takes ``now`` and an outcome carries a caller-supplied ``acted_at``
rather than reading a clock. ``rationale`` and
``seeded_from`` are reviewer-facing and never reach canonical text (PRD D2);
only ``claim`` does.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from types import MappingProxyType
from typing import Any, TypeVar

from fused_memory.models.reconciliation import StageId
from fused_memory.reconciliation.consolidation_auto import AutoOutcome
from fused_memory.reconciliation.recon_ledger import (
    SERVICE_SCOPE_PROJECT_ID,
    ReconLedgerRecord,
    ReconLedgerStore,
)

RECORD_KIND_CONSOLIDATION_PROPOSAL = 'consolidation_proposal'
#: One row per (project, members key): the adjudicator flow's bounded retry (PRD K4).
RECORD_KIND_ADJUDICATION_ATTEMPT = 'consolidation_adjudication_attempt'
#: One row per adjudicator identity, belonging to no project (PRD K5).
RECORD_KIND_ADJUDICATOR_CALIBRATION = 'consolidation_adjudicator_calibration'

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
    DROPPED = 'dropped'
    UNRESOLVED = 'unresolved'


_MEMBER_IDS_KEY = 'member_ids'
_CATEGORY_KEY = 'category'
_CLAIM_KEY = 'claim'
_RATIONALE_KEY = 'rationale'
_SEEDED_FROM_KEY = 'seeded_from'
_OUTCOME_KEY = 'outcome'
_UNDECODABLE_PAYLOAD_KEY = 'undecodable_payload_json'

_T = TypeVar('_T')


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
class ProposalRef:
    """Another proposal row of the same project, by its topic and run id."""

    topic: str
    run_id: str

    @classmethod
    def from_record(cls, record: ReconLedgerRecord) -> ProposalRef:
        """The ref to *record*, whose topic is its ``flag_type`` (see ``ConsolidationProposal.ledger_key``)."""
        if record.record_kind != RECORD_KIND_CONSOLIDATION_PROPOSAL:
            raise ValueError(
                f'record_kind {record.record_kind!r} is not {RECORD_KIND_CONSOLIDATION_PROPOSAL!r}'
            )
        return cls(topic=record.flag_type, run_id=record.run_id)

    def to_payload(self) -> dict[str, str]:
        return {'topic': self.topic, 'run_id': self.run_id}

    @classmethod
    def from_payload(cls, payload: object) -> ProposalRef:
        obj = _json_object(payload)
        return cls(topic=_payload_str(obj, 'topic'), run_id=_payload_str(obj, 'run_id'))


#: How much of a member's text the adjudication record keeps (PRD K6).
SHOWN_TEXT_CHARS = 1500


@dataclass(frozen=True)
class ShownMember:
    """One member as the adjudicator saw it, its text cut to :data:`SHOWN_TEXT_CHARS`."""

    id: str
    content_sha256: str
    text: str

    def __post_init__(self) -> None:
        if len(self.text) > SHOWN_TEXT_CHARS:
            raise ValueError(
                f'shown text of {self.id} is {len(self.text)} chars, over the '
                f'{SHOWN_TEXT_CHARS}-char cap'
            )

    @classmethod
    def cut(cls, member_id: str, content_sha256: str, seen_text: str) -> ShownMember:
        return cls(id=member_id, content_sha256=content_sha256, text=seen_text[:SHOWN_TEXT_CHARS])

    def to_payload(self) -> dict[str, str]:
        return {'id': self.id, 'content_sha256': self.content_sha256, 'text': self.text}

    @classmethod
    def from_payload(cls, payload: object) -> ShownMember:
        obj = _json_object(payload)
        return cls(
            id=_payload_str(obj, 'id'),
            content_sha256=_payload_str(obj, 'content_sha256'),
            text=_payload_str(obj, 'text'),
        )


@dataclass(frozen=True)
class AdjudicationRecord:
    """What the adjudicator was shown and what it decided, for a reviewer with no store access.

    ``waivers`` are the ``(code, subject member id)`` pairs the executor derived
    from the decision; the model never writes one (PRD K2).
    """

    decision: str
    reason_code: str
    reason: str
    waivers: tuple[tuple[str, str], ...]
    claim_suggestion: str | None
    identity: str
    context_key: str
    resolved_model: str | None
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    latency_s: float | None
    shown: tuple[ShownMember, ...]

    def __post_init__(self) -> None:
        _check_measure('input_tokens', self.input_tokens, integral=True)
        _check_measure('output_tokens', self.output_tokens, integral=True)
        _check_measure('cost_usd', self.cost_usd, integral=False)
        _check_measure('latency_s', self.latency_s, integral=False)

    def to_payload(self) -> dict[str, Any]:
        return {
            'decision': self.decision,
            'reason_code': self.reason_code,
            'reason': self.reason,
            'waivers': [{'code': code, 'subject_id': subject} for code, subject in self.waivers],
            'claim_suggestion': self.claim_suggestion,
            'identity': self.identity,
            'context_key': self.context_key,
            'resolved_model': self.resolved_model,
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'cost_usd': self.cost_usd,
            'latency_s': self.latency_s,
            'shown': [member.to_payload() for member in self.shown],
        }

    @classmethod
    def from_payload(cls, payload: object) -> AdjudicationRecord:
        """Decode :meth:`to_payload`'s shape; every key is required."""
        obj = _json_object(payload)
        missing = [f.name for f in dataclasses.fields(cls) if f.name not in obj]
        if missing:
            raise ValueError(f'adjudication is missing {", ".join(missing)}')
        return cls(
            decision=_payload_str(obj, 'decision'),
            reason_code=_payload_str(obj, 'reason_code'),
            reason=_payload_str(obj, 'reason'),
            waivers=_decode_items(obj, 'waivers', _waiver_from_payload),
            claim_suggestion=_optional_str(obj, 'claim_suggestion'),
            identity=_payload_str(obj, 'identity'),
            context_key=_payload_str(obj, 'context_key'),
            resolved_model=_optional_str(obj, 'resolved_model'),
            input_tokens=obj['input_tokens'],
            output_tokens=obj['output_tokens'],
            cost_usd=obj['cost_usd'],
            latency_s=obj['latency_s'],
            shown=_decode_items(obj, 'shown', ShownMember.from_payload),
        )


def _waiver_from_payload(payload: object) -> tuple[str, str]:
    obj = _json_object(payload)
    return _payload_str(obj, 'code'), _payload_str(obj, 'subject_id')


def _check_measure(key: str, value: object, *, integral: bool) -> None:
    """Refuse a count or measure that is not a finite non-negative number (``None`` allowed)."""
    if value is None:
        return
    kinds: tuple[type, ...] = (int,) if integral else (int, float)
    if (
        isinstance(value, bool)
        or not isinstance(value, kinds)
        or (isinstance(value, float) and not math.isfinite(value))
        or value < 0
    ):
        wanted = 'int' if integral else 'number'
        raise ValueError(f'{key!r} is not a non-negative {wanted} or null: {value!r}')


def _no_repairs() -> Mapping[str, tuple[str, ...]]:
    return MappingProxyType({})


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
    members_key: str | None = None
    kept_ids: tuple[str, ...] = ()
    repaired: Mapping[str, tuple[str, ...]] = field(default_factory=_no_repairs, hash=False)
    effective_claim: str | None = None
    effective_category: str | None = None
    reused_from: ProposalRef | None = None
    adjudication: AdjudicationRecord | None = None

    def __post_init__(self) -> None:
        frozen = MappingProxyType({code: tuple(ids) for code, ids in self.repaired.items()})
        object.__setattr__(self, 'repaired', frozen)

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
            'members_key': self.members_key,
            'kept_ids': list(self.kept_ids),
            'repaired': {code: list(ids) for code, ids in self.repaired.items()},
            'effective_claim': self.effective_claim,
            'effective_category': self.effective_category,
            'reused_from': None if self.reused_from is None else self.reused_from.to_payload(),
            'adjudication': None if self.adjudication is None else self.adjudication.to_payload(),
        }

    @classmethod
    def from_payload(cls, payload: object) -> ProposalOutcome:
        """Decode :meth:`to_payload`'s shape; ``ValueError`` naming any bad field.

        The fields the adjudicator flow added read as their defaults when ABSENT,
        so a row written before them still decodes; present and malformed raises.
        """
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
            members_key=_optional_str(payload, 'members_key'),
            kept_ids=_optional_str_tuple(payload, 'kept_ids'),
            repaired=_str_tuple_mapping(payload, 'repaired'),
            effective_claim=_optional_str(payload, 'effective_claim'),
            effective_category=_optional_str(payload, 'effective_category'),
            reused_from=_optional_object(payload, 'reused_from', ProposalRef.from_payload),
            adjudication=_optional_object(payload, 'adjudication', AdjudicationRecord.from_payload),
        )


async def record_outcome(
    ledger: ReconLedgerStore,
    record: ReconLedgerRecord,
    outcome: ProposalOutcome,
    *,
    retention: timedelta | None = None,
) -> bool:
    """Move *record* to ``outcome.state`` with *outcome* on its payload.

    A compare-and-set on the state *record* was read in: ``False`` (and no
    write) if the stored row has moved since. The proposal's own payload keys
    are kept, so :func:`from_ledger_record` still decodes the row; a payload
    that was never a JSON object is kept verbatim under one key instead. An
    executed outcome drops ``expires_at`` so ``ReconLedgerStore.gc`` never
    reaps it. A non-executed outcome lives *retention* past its ``acted_at``
    when one is passed (PRD consolidation-adjudicator K6), else keeps the
    row's expiry.
    """
    if outcome.state is ProposalState.PROPOSED:
        raise ValueError('an outcome cannot move a row back to proposed')
    expires_at = _outcome_expiry(record, outcome, retention)
    payload: dict[str, Any]
    try:
        payload = _payload_object(record)
    except ValueError:
        payload = {_UNDECODABLE_PAYLOAD_KEY: record.payload_json}
    payload[_OUTCOME_KEY] = outcome.to_payload()
    moved = dataclasses.replace(
        record,
        state=outcome.state.value,
        payload_json=json.dumps(payload),
        expires_at=expires_at,
    )
    return await ledger.upsert_if_state(moved, existing_state=record.state)


def _outcome_expiry(
    record: ReconLedgerRecord, outcome: ProposalOutcome, retention: timedelta | None,
) -> str | None:
    """The ``expires_at`` *record* takes with *outcome*; a bad *retention* raises even when executed."""
    expiry = record.expires_at if retention is None else _retained_until(outcome.acted_at, retention)
    return None if outcome.executed else expiry


def _retained_until(acted_at: str, retention: timedelta) -> str:
    """*retention* after *acted_at*, as canonical UTC text, which ``gc`` compares as a string."""
    if retention <= timedelta(0):
        raise ValueError(f'retention must be positive, got {retention!r}')
    try:
        acted = datetime.fromisoformat(acted_at)
    except ValueError as exc:
        raise ValueError(f'acted_at is not an ISO-8601 timestamp: {acted_at!r}') from exc
    if acted.tzinfo is None:
        raise ValueError(f'acted_at must be timezone-aware, got naive {acted_at!r}')
    return (acted + retention).astimezone(UTC).isoformat()


def outcome_from_record(record: ReconLedgerRecord) -> ProposalOutcome | None:
    """The outcome :func:`record_outcome` stored on *record*, or ``None`` if never acted on."""
    payload = _payload_object(record)
    if _OUTCOME_KEY not in payload:
        return None
    return ProposalOutcome.from_payload(payload[_OUTCOME_KEY])


def refusal_streak(rows: Iterable[ReconLedgerRecord]) -> int:
    """Count the trailing ``refused`` rows of one topic, in the given order.

    ``proposed`` rows (not yet acted on), ``superseded`` rows (never
    evaluated) and rows carrying ``reused_from`` (memo copies, PRD
    consolidation-adjudicator K4) neither count nor break the streak; every
    other acted state, ``dropped`` and ``unresolved`` included, resets it.
    """
    streak = 0
    for row in rows:
        state = ProposalState(row.state)
        if state in (ProposalState.PROPOSED, ProposalState.SUPERSEDED) or _is_memo_copy(row):
            continue
        streak = streak + 1 if state is ProposalState.REFUSED else 0
    return streak


def _is_memo_copy(row: ReconLedgerRecord) -> bool:
    """Does *row*'s outcome reuse an earlier row's? An undecodable outcome counts as an original.

    Fail-safe for an escalation counter: the executor reads the streak on
    every refusal, and a raise here would abort its whole Stage 1 post-step.
    """
    try:
        outcome = outcome_from_record(row)
    except ValueError:
        return False
    return outcome is not None and outcome.reused_from is not None


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


#: :func:`content_sha256` of a record the store does not have. Never a hex digest.
MISSING_CONTENT_SHA256 = 'missing'
#: :func:`content_sha256` of a read that did not answer or answered garbage. Never a hex digest.
UNREADABLE_CONTENT_SHA256 = 'unreadable'


def content_sha256(record: object) -> str:
    """The hash of one member read's content (PRD K4, K6).

    The one content hash behind :func:`members_key`, the adjudicator's context
    key and :attr:`ShownMember.content_sha256`. ``UNREADABLE``, like any read
    that is not a record with text content, hashes as unreadable.
    """
    if record is None:
        return MISSING_CONTENT_SHA256
    content = record.get('content') if isinstance(record, Mapping) else None
    if not isinstance(content, str):
        return UNREADABLE_CONTENT_SHA256
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def members_key(member_ids: Sequence[str], members: Mapping[str, object]) -> str:
    """Which records a proposal names and what they say, in any order (PRD K4).

    No topic and no claim: the proposer words those freshly each cycle. An id
    absent from *members* reads as missing, as the predicate reads it, and a
    repeated id stays in the key.
    """
    pairs = sorted((member_id, content_sha256(members.get(member_id))) for member_id in member_ids)
    return hashlib.sha256(json.dumps(pairs, separators=(',', ':')).encode('utf-8')).hexdigest()


def adjudication_attempt_key(project_id: str, members_key: str) -> dict[str, str]:
    """The attempt row's identity, usable as ``ReconLedgerStore.get_by_identity(**key)``.

    Its payload, states and expiry are its writer's (the executor).
    """
    if not project_id or project_id == SERVICE_SCOPE_PROJECT_ID:
        raise ValueError(f'project_id must name a real project, got {project_id!r}')
    if not members_key:
        raise ValueError('members_key must not be blank')
    return {
        'project_id': project_id,
        'record_kind': RECORD_KIND_ADJUDICATION_ATTEMPT,
        'task_id': '',
        'flag_type': members_key,
        'run_id': '',
    }


def adjudicator_calibration_key(identity: str) -> dict[str, str]:
    """The calibration row's identity, under :data:`SERVICE_SCOPE_PROJECT_ID`.

    Its payload, states and expiry are its writer's (the calibration).
    """
    if not identity:
        raise ValueError('identity must not be blank')
    return {
        'project_id': SERVICE_SCOPE_PROJECT_ID,
        'record_kind': RECORD_KIND_ADJUDICATOR_CALIBRATION,
        'task_id': '',
        'flag_type': identity,
        'run_id': '',
    }


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


def _optional_str_tuple(payload: dict[str, Any], key: str) -> tuple[str, ...]:
    """:func:`_str_tuple`, reading an absent key as empty."""
    return _str_tuple(payload, key) if key in payload else ()


def _str_tuple_mapping(payload: dict[str, Any], key: str) -> dict[str, tuple[str, ...]]:
    """An object of string lists, reading an absent key as empty."""
    value = payload.get(key, {})
    if not isinstance(value, dict) or not all(
        isinstance(ids, list) and all(isinstance(i, str) for i in ids) for ids in value.values()
    ):
        raise ValueError(f'payload {key!r} is not an object of string lists: {value!r}')
    return {code: tuple(ids) for code, ids in value.items()}


def _optional_object(
    payload: dict[str, Any], key: str, decode: Callable[[object], _T],
) -> _T | None:
    """*decode* the object under *key*, reading an absent or null key as ``None``."""
    value = payload.get(key)
    if value is None:
        return None
    try:
        return decode(value)
    except ValueError as exc:
        raise ValueError(f'payload {key!r}: {exc}') from exc


def _decode_items(
    payload: dict[str, Any], key: str, decode: Callable[[object], _T],
) -> tuple[_T, ...]:
    """*decode* each element of the list under *key*, naming the key and index on failure."""
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f'payload {key!r} is not a list: {value!r}')
    items: list[_T] = []
    for index, item in enumerate(value):
        try:
            items.append(decode(item))
        except ValueError as exc:
            raise ValueError(f'payload {key!r}[{index}]: {exc}') from exc
    return tuple(items)


def _json_object(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f'not a JSON object: {value!r}')
    return value
