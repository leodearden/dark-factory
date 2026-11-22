"""The adjudicator's self-calibration over the pinned eval set (plans/consolidation-adjudicator-prd.md K5, E7).

Owns the eval set, the six bounds, the scoring and the calibration row: its
states, its payload and the one-step state machine :func:`run_calibration`
moves it through. The eval set is resolved from the package and pinned by
sha256; the bounds are module constants, changed by code review and never by
reload. It files nothing itself: a result names the finding for its caller to
file. Imports no executor code.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import logging
import math
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

from fused_memory.llm.openai_responses import ResponsesCall
from fused_memory.reconciliation.consolidation_adjudicator import (
    Adjudication,
    AdjudicationRequest,
    AdjudicatorSettings,
    CallMeta,
    ClaimCheck,
    Decision,
    FailureKind,
    Hazard,
    MemberInput,
    Verdict,
    adjudicate,
    adjudicator_identity,
    keep_rule_holds,
)
from fused_memory.reconciliation.consolidation_auto import (
    REASON_CODE_CLASSES,
    AutoReasonCode,
    FlowReasonCode,
    ReasonCodeClass,
)
from fused_memory.reconciliation.consolidation_proposal import (
    RECORD_KIND_ADJUDICATOR_CALIBRATION,
    adjudicator_calibration_key,
)
from fused_memory.reconciliation.recon_ledger import ReconLedgerRecord, ReconLedgerStore

logger = logging.getLogger(__name__)

EVAL_SET_PATH = (
    Path(__file__).resolve().parents[3] / 'calibration' / 'consolidation_adjudicator_eval_set.jsonl'
)
EVAL_SET_SHA256 = 'df2723389d1ee696b3b5bf3e7efa4c1e558263ca08669c7e58fd8c5befc135af'

#: More unanswered cases than this make a run inconclusive.
MAX_UNANSWERED = 5
WALL_TIME_LIMIT_S = 900
CALLS_IN_FLIGHT = 6
#: Attempts per case on a transport failure; a content failure is never retried.
TRANSPORT_ATTEMPTS = 3
#: The pause before a transport retry.
RETRY_DELAY_S = 2.0
#: A ``running`` row older than this is a crashed run.
CRASH_AGE = timedelta(hours=2)
#: Runs ending inconclusive or crashed before the calibration stalls.
STALL_RUN_CAP = 3
STALL_BACKOFF = timedelta(hours=24)


class EvalSetInvalid(ValueError):
    """The eval set is missing, is not the pinned file, or has a shape this code does not read."""


class CaseSplit(StrEnum):
    dev = 'dev'
    test = 'test'


class CaseKind(StrEnum):
    fold = 'fold'
    contaminated = 'contaminated'
    must_not_fold = 'must_not_fold'
    real_proposal = 'real_proposal'


@dataclass(frozen=True)
class EvalMember:
    id: str
    category: str
    text: str


@dataclass(frozen=True)
class EvalHazard:
    code: AutoReasonCode
    #: None for a case-level hazard (``mixed_category``).
    member_id: str | None


@dataclass(frozen=True)
class Expectation:
    decision_set: frozenset[Decision]
    keep_ids: tuple[str, ...]
    #: The member that belongs to another topic, on a contaminated case.
    foreign_id: str | None


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    split: CaseSplit
    kind: CaseKind
    members: tuple[EvalMember, ...]
    claim: str
    hazards: tuple[EvalHazard, ...]
    expected: Expectation
    #: ``provenance.topic`` or ``provenance.seed_topic``; None when the case carries neither.
    topic: str | None


_CASE_KEYS = frozenset({
    'case_id', 'split', 'kind', 'project', 'n_members', 'mixed_category', 'members', 'claim',
    'claim_source', 'hazards', 'expected', 'provenance',
})
_MEMBER_KEYS = frozenset({'id', 'category', 'text'})
_HAZARD_KEYS = frozenset({'code', 'member_id'})
_EXPECTED_KEYS = frozenset({'decision_set', 'keep_ids', 'foreign_id', 'reason_code'})
_EXPECTED_OPTIONAL_KEYS = frozenset({'min_keep_fraction'})


def _invalid(case_id: object, what: str) -> EvalSetInvalid:
    return EvalSetInvalid(f'eval case {case_id!r}: {what}')


def _object(
    raw: object, case_id: object, *, required: frozenset[str], allowed: frozenset[str],
) -> dict[str, Any]:
    if not isinstance(raw, dict) or not required <= raw.keys() <= allowed:
        raise _invalid(case_id, f'unexpected shape {raw!r:.200}')
    return raw


def _str(value: object, case_id: object, what: str) -> str:
    if not isinstance(value, str) or not value:
        raise _invalid(case_id, f'{what} must be a non-empty string')
    return value


def _member(raw: object, case_id: object) -> EvalMember:
    member = _object(raw, case_id, required=_MEMBER_KEYS, allowed=_MEMBER_KEYS)
    return EvalMember(
        id=_str(member['id'], case_id, 'member id'),
        category=_str(member['category'], case_id, 'member category'),
        text=_str(member['text'], case_id, 'member text'),
    )


def _hazard(raw: object, case_id: object, member_ids: frozenset[str]) -> EvalHazard:
    hazard = _object(raw, case_id, required=frozenset({'code'}), allowed=_HAZARD_KEYS)
    try:
        code = AutoReasonCode(hazard['code'])
    except ValueError:
        raise _invalid(case_id, f'unknown hazard code {hazard["code"]!r}') from None
    member_id = hazard.get('member_id')
    if member_id is not None and member_id not in member_ids:
        raise _invalid(case_id, f'hazard {code} names {member_id!r}, not a member')
    return EvalHazard(code=code, member_id=member_id)


def _expectation(raw: object, case_id: object) -> Expectation:
    expected = _object(
        raw, case_id, required=_EXPECTED_KEYS, allowed=_EXPECTED_KEYS | _EXPECTED_OPTIONAL_KEYS,
    )
    try:
        decisions = frozenset(Decision(d) for d in expected['decision_set'])
    except (TypeError, ValueError):
        raise _invalid(case_id, 'bad decision_set') from None
    keep_ids, foreign_id = expected['keep_ids'], expected['foreign_id']
    if not isinstance(keep_ids, list) or not all(isinstance(i, str) for i in keep_ids):
        raise _invalid(case_id, 'keep_ids must be a list of ids')
    if foreign_id is not None and not isinstance(foreign_id, str):
        raise _invalid(case_id, 'foreign_id must be an id or null')
    return Expectation(decision_set=decisions, keep_ids=tuple(keep_ids), foreign_id=foreign_id)


def _topic(raw: object, case_id: object) -> str | None:
    if not isinstance(raw, dict):
        raise _invalid(case_id, 'provenance must be an object')
    topic = raw.get('topic', raw.get('seed_topic'))
    if topic is not None and not isinstance(topic, str):
        raise _invalid(case_id, 'the provenance topic must be a string')
    return topic


def _case(raw: object) -> EvalCase:
    case_id = raw.get('case_id') if isinstance(raw, dict) else None
    row = _object(raw, case_id, required=_CASE_KEYS, allowed=_CASE_KEYS)
    members = tuple(_member(m, case_id) for m in row['members'])
    if len(members) != row['n_members'] or not isinstance(row['mixed_category'], bool):
        raise _invalid(case_id, 'n_members or mixed_category disagrees with the case')
    try:
        split, kind = CaseSplit(row['split']), CaseKind(row['kind'])
    except ValueError:
        raise _invalid(case_id, 'unknown split or kind') from None
    member_ids = frozenset(m.id for m in members)
    return EvalCase(
        case_id=_str(case_id, case_id, 'case_id'),
        split=split,
        kind=kind,
        members=members,
        claim=_str(row['claim'], case_id, 'claim'),
        hazards=tuple(_hazard(h, case_id, member_ids) for h in row['hazards']),
        expected=_expectation(row['expected'], case_id),
        topic=_topic(row['provenance'], case_id),
    )


def load_eval_set(path: Path = EVAL_SET_PATH) -> tuple[EvalCase, ...]:
    """The pinned eval set's cases. The sha is checked before anything is parsed."""
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        raise EvalSetInvalid(f'eval set missing at {path}') from None
    observed = hashlib.sha256(data).hexdigest()
    if observed != EVAL_SET_SHA256:
        raise EvalSetInvalid(
            f'eval set at {path} has sha256 {observed}, not the pinned {EVAL_SET_SHA256}',
        )
    lines = [line for line in data.decode().splitlines() if line.strip()]
    try:
        rows = [json.loads(line) for line in lines]
    except json.JSONDecodeError as exc:
        raise EvalSetInvalid(f'eval set line is not JSON: {exc}') from None
    return tuple(_case(row) for row in rows)


Exclusion = AutoReasonCode | FlowReasonCode


def case_exclusion(case: EvalCase, settings: AdjudicatorSettings) -> Exclusion | None:
    """The code K1 would refuse *case* with before any call, or None when it is adjudicated."""
    for hazard in case.hazards:
        if REASON_CODE_CLASSES[hazard.code] is ReasonCodeClass.REFUSE:
            return hazard.code
    if len(case.members) > settings.member_max:
        return FlowReasonCode.too_many_members_to_adjudicate
    if len(case.members) < settings.member_min:
        return FlowReasonCode.too_few_members_after_repair
    return None


def case_request(case: EvalCase) -> AdjudicationRequest:
    """*case* as the adjudicator sees it: its members' shared category, adjudicator-class hazards."""
    categories = {member.category for member in case.members}
    if len(categories) != 1:
        raise ValueError(f'eval case {case.case_id!r} has no single category: {sorted(categories)}')
    return AdjudicationRequest(
        topic=case.topic,
        claim=case.claim,
        category=categories.pop(),
        members=tuple(MemberInput(id=m.id, category=m.category, text=m.text) for m in case.members),
        hazards=tuple(
            Hazard(code=h.code, subject_id=h.member_id)
            for h in case.hazards
            if h.member_id is not None
            and REASON_CODE_CLASSES[h.code] is ReasonCodeClass.ADJUDICATOR
        ),
    )


class OutcomeKind(StrEnum):
    verdict = 'verdict'
    content_failure = 'content_failure'
    #: Every attempt failed in transport.
    unanswered = 'unanswered'


@dataclass(frozen=True)
class CaseOutcome:
    case: EvalCase
    kind: OutcomeKind
    verdict: Verdict | None

    def __post_init__(self) -> None:
        if (self.kind is OutcomeKind.verdict) != (self.verdict is not None):
            raise ValueError(f'a {self.kind} outcome carries a verdict exactly when it is one')


class BoundName(StrEnum):
    contaminated_false_keep = 'contaminated_false_keep'
    fold_rate = 'fold_rate'
    must_not_merge_test = 'must_not_merge_test'
    must_not_merge_dev = 'must_not_merge_dev'
    real_proposal_folds = 'real_proposal_folds'
    invalid_replies = 'invalid_replies'


class BoundDirection(StrEnum):
    at_most = 'at_most'
    at_least = 'at_least'


class BoundBasis(StrEnum):
    """What a bound rests on (PRD E7's basis column), recorded with every figure."""

    measured = 'measured'
    sanity_floor = 'sanity_floor'
    regression_guard = 'regression_guard'


class CaseCount(StrEnum):
    """Which cases a bound's numerator counts."""

    #: The foreign member is kept by a proceed that passes the keep rule.
    false_keep = 'false_keep'
    #: A proceed that passes the keep rule.
    fold = 'fold'
    #: The same event as ``fold``, counted against a must-not-merge population.
    proceed = 'proceed'
    #: A content failure or an unanswered case.
    invalid = 'invalid'


@dataclass(frozen=True)
class Bound:
    name: BoundName
    #: None: every adjudicated case.
    split: CaseSplit | None
    kinds: frozenset[CaseKind]
    direction: BoundDirection
    numerator: int
    denominator: int
    basis: BoundBasis
    counts: CaseCount


BOUNDS: tuple[Bound, ...] = (
    Bound(BoundName.contaminated_false_keep, CaseSplit.test, frozenset({CaseKind.contaminated}),
          BoundDirection.at_most, 6, 45, BoundBasis.measured, CaseCount.false_keep),
    Bound(BoundName.fold_rate, CaseSplit.test, frozenset({CaseKind.fold}),
          BoundDirection.at_least, 32, 45, BoundBasis.measured, CaseCount.fold),
    Bound(BoundName.must_not_merge_test, CaseSplit.test, frozenset({CaseKind.must_not_fold}),
          BoundDirection.at_most, 15, 30, BoundBasis.sanity_floor, CaseCount.proceed),
    Bound(BoundName.must_not_merge_dev, CaseSplit.dev, frozenset({CaseKind.must_not_fold}),
          BoundDirection.at_most, 5, 13, BoundBasis.regression_guard, CaseCount.proceed),
    Bound(BoundName.real_proposal_folds, CaseSplit.dev, frozenset({CaseKind.real_proposal}),
          BoundDirection.at_least, 5, 6, BoundBasis.regression_guard, CaseCount.fold),
    Bound(BoundName.invalid_replies, None, frozenset(CaseKind),
          BoundDirection.at_most, 1, 20, BoundBasis.measured, CaseCount.invalid),
)
"""PRD E7's six bounds, compared as rationals by integer cross-multiplication."""

_BOUNDS_BY_NAME: Mapping[BoundName, Bound] = {bound.name: bound for bound in BOUNDS}


@dataclass(frozen=True)
class BoundFigure:
    name: BoundName
    numerator: int
    denominator: int
    holds: bool


@dataclass(frozen=True)
class RunFigures:
    """One figure per bound, in :data:`BOUNDS` order."""

    figures: tuple[BoundFigure, ...]

    def to_payload(self) -> list[dict[str, Any]]:
        return [
            {'name': f.name.value, 'numerator': f.numerator, 'denominator': f.denominator,
             'holds': f.holds}
            for f in self.figures
        ]

    @classmethod
    def from_payload(cls, payload: object) -> RunFigures:
        """Decode :meth:`to_payload`'s shape: one figure per bound, in :data:`BOUNDS` order."""
        if not isinstance(payload, list):
            raise ValueError(f'figures are not a list: {payload!r:.200}')
        figures = tuple(_figure_from_payload(item) for item in payload)
        if tuple(f.name for f in figures) != tuple(b.name for b in BOUNDS):
            raise ValueError(f'figures do not name the bounds in order: {payload!r:.200}')
        return cls(figures)


_FIGURE_KEYS = frozenset({'name', 'numerator', 'denominator', 'holds'})


def _figure_from_payload(item: object) -> BoundFigure:
    if not isinstance(item, dict) or item.keys() != _FIGURE_KEYS:
        raise ValueError(f'a figure has an unexpected shape: {item!r:.200}')
    try:
        name = BoundName(item['name'])
    except ValueError:
        raise ValueError(f'unknown bound {item["name"]!r}') from None
    numerator, denominator, holds = item['numerator'], item['denominator'], item['holds']
    if not (_is_count(numerator) and _is_count(denominator) and isinstance(holds, bool)):
        raise ValueError(f'figure {name} has a bad count or verdict: {item!r:.200}')
    return BoundFigure(name=name, numerator=numerator, denominator=denominator, holds=holds)


def _is_count(value: object, minimum: int = 0) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _holds(bound: Bound, numerator: int, denominator: int) -> bool:
    if denominator == 0:
        return False
    observed, limit = numerator * bound.denominator, bound.numerator * denominator
    return observed <= limit if bound.direction is BoundDirection.at_most else observed >= limit


def _kept_after_keep_rule(outcome: CaseOutcome, member_min: int) -> tuple[str, ...]:
    """The kept list of a proceed that passes K1's keep rule; empty for anything else."""
    verdict = outcome.verdict
    if verdict is None or verdict.decision is not Decision.proceed:
        return ()
    kept = verdict.keep_member_ids
    if not keep_rule_holds(len(kept), len(outcome.case.members), member_min=member_min):
        return ()
    return kept


def _in_population(bound: Bound, case: EvalCase) -> bool:
    return (bound.split is None or case.split is bound.split) and case.kind in bound.kinds


def _counted(bound: Bound, outcome: CaseOutcome, member_min: int) -> bool:
    if outcome.verdict is None:
        return bound.direction is BoundDirection.at_most
    if bound.counts is CaseCount.invalid:
        return False
    kept = _kept_after_keep_rule(outcome, member_min)
    if bound.counts is CaseCount.false_keep:
        return outcome.case.expected.foreign_id in kept
    return bool(kept)


def score_run(outcomes: Iterable[CaseOutcome], *, member_min: int) -> RunFigures:
    """Every bound's numerator and denominator over *outcomes*.

    A case without a verdict is wrong on its bound, whichever way the bound points.
    """
    scored = tuple(outcomes)
    figures = []
    for bound in BOUNDS:
        population = [o for o in scored if _in_population(bound, o.case)]
        numerator = sum(_counted(bound, o, member_min) for o in population)
        figures.append(BoundFigure(
            name=bound.name, numerator=numerator, denominator=len(population),
            holds=_holds(bound, numerator, len(population)),
        ))
    return RunFigures(tuple(figures))


def evaluate(figures: RunFigures) -> tuple[BoundName, ...]:
    """The bounds *figures* fail, in :data:`BOUNDS` order."""
    return tuple(figure.name for figure in figures.figures if not figure.holds)


def pool(first: RunFigures, second: RunFigures) -> RunFigures:
    """Two runs as one: same rates, doubled counts."""
    pooled = []
    for a, b in zip(first.figures, second.figures, strict=True):
        if a.name is not b.name:
            raise ValueError(f'cannot pool {a.name} with {b.name}')
        numerator, denominator = a.numerator + b.numerator, a.denominator + b.denominator
        pooled.append(BoundFigure(
            name=a.name, numerator=numerator, denominator=denominator,
            holds=_holds(_BOUNDS_BY_NAME[a.name], numerator, denominator),
        ))
    return RunFigures(tuple(pooled))



class CalibrationState(StrEnum):
    """The calibration row's states (PRD K5); this module is their writer."""

    RUNNING = 'running'
    INCONCLUSIVE = 'inconclusive'
    CRASHED = 'crashed'
    STALLED = 'stalled'
    FAIL_FIRST_RUN = 'fail_first_run'
    PASS = 'pass'
    FAIL = 'fail'
    STALE = 'stale'


#: States no run leaves: the identity is calibrated, one way or the other.
_SETTLED = frozenset({CalibrationState.PASS, CalibrationState.FAIL})

#: Never stored, so a claim against an absent row loses to a concurrent insert.
_NO_ROW = '(no row)'


class CalibrationFinding(StrEnum):
    """What a :class:`CalibrationResult` asks its caller to file (PRD K6's findings)."""

    adjudicator_calibration_failed = 'adjudicator_calibration_failed'
    adjudicator_calibration_stalled = 'adjudicator_calibration_stalled'
    adjudicator_eval_set_invalid = 'adjudicator_eval_set_invalid'


def _is_aware(value: object) -> bool:
    return isinstance(value, datetime) and value.utcoffset() is not None


def _is_measure(value: object) -> bool:
    return value is None or (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def _is_text(value: object) -> bool:
    return isinstance(value, str) and bool(value)


@dataclass(frozen=True)
class CalibrationRecord:
    """One adjudicator identity's calibration row: its state and what its last run measured.

    The identity is the row's key and the state its state column; every other
    field is the payload.
    """

    identity: str
    state: CalibrationState
    #: The run number, from 1; each start counts one.
    attempt: int
    started_at: datetime
    model: str
    endpoint_host: str | None
    #: Consecutive runs that ended inconclusive or crashed.
    stall_count: int = 0
    stalled_at: datetime | None = None
    #: A failed first run, kept for the second run it is pooled with.
    first_run: RunFigures | None = None
    #: The figures the state was decided on: pooled with ``first_run`` on a second run.
    figures: RunFigures | None = None
    unanswered: int = 0
    resolved_models: tuple[str, ...] = ()
    #: The sum over the priced calls; None when no call was priced.
    cost_usd: float | None = None
    wall_time_s: float | None = None
    eval_sha256: str = EVAL_SET_SHA256

    def __post_init__(self) -> None:
        checks = (
            ('identity', _is_text(self.identity), 'a non-empty string'),
            ('state', isinstance(self.state, CalibrationState), 'a CalibrationState'),
            ('attempt', _is_count(self.attempt, 1), 'a positive int'),
            ('started_at', _is_aware(self.started_at), 'a timezone-aware datetime'),
            ('model', _is_text(self.model), 'a non-empty string'),
            ('endpoint_host', self.endpoint_host is None or _is_text(self.endpoint_host),
             'a host or None'),
            ('stall_count', _is_count(self.stall_count), 'a non-negative int'),
            ('stalled_at', self.stalled_at is None or _is_aware(self.stalled_at),
             'a timezone-aware datetime or None'),
            ('first_run', self.first_run is None or isinstance(self.first_run, RunFigures),
             'RunFigures or None'),
            ('figures', self.figures is None or isinstance(self.figures, RunFigures),
             'RunFigures or None'),
            ('unanswered', _is_count(self.unanswered), 'a non-negative int'),
            ('resolved_models', isinstance(self.resolved_models, tuple)
             and all(_is_text(m) for m in self.resolved_models), 'a tuple of model names'),
            ('cost_usd', _is_measure(self.cost_usd), 'a non-negative number or None'),
            ('wall_time_s', _is_measure(self.wall_time_s), 'a non-negative number or None'),
            ('eval_sha256', _is_text(self.eval_sha256), 'a non-empty string'),
        )
        for field, ok, wanted in checks:
            if not ok:
                raise ValueError(
                    f'calibration record {field} must be {wanted}, got {getattr(self, field)!r}',
                )

    @property
    def failed_bounds(self) -> tuple[BoundName, ...]:
        return () if self.figures is None else evaluate(self.figures)

    def to_payload(self) -> dict[str, Any]:
        return {
            'attempt': self.attempt,
            'started_at': self.started_at.astimezone(UTC).isoformat(),
            'model': self.model,
            'endpoint_host': self.endpoint_host,
            'stall_count': self.stall_count,
            'stalled_at': None if self.stalled_at is None else self.stalled_at.astimezone(UTC).isoformat(),
            'first_run': None if self.first_run is None else self.first_run.to_payload(),
            'figures': None if self.figures is None else self.figures.to_payload(),
            'unanswered': self.unanswered,
            'resolved_models': list(self.resolved_models),
            'cost_usd': self.cost_usd,
            'wall_time_s': self.wall_time_s,
            'eval_sha256': self.eval_sha256,
        }

    @classmethod
    def from_payload(
        cls, payload: object, *, identity: str, state: CalibrationState,
    ) -> CalibrationRecord:
        """Decode :meth:`to_payload`'s shape; every key is required. A bad field is named."""
        if not isinstance(payload, dict):
            raise ValueError(f'calibration payload is not an object: {payload!r:.200}')
        missing = [key for key in _PAYLOAD_KEYS if key not in payload]
        if missing:
            raise ValueError(f'calibration payload is missing {", ".join(missing)}')
        resolved_models = payload['resolved_models']
        if not isinstance(resolved_models, list):
            raise ValueError(f'calibration payload resolved_models is not a list: {resolved_models!r}')
        return cls(
            identity=identity,
            state=state,
            attempt=payload['attempt'],
            started_at=_timestamp(payload, 'started_at'),
            model=payload['model'],
            endpoint_host=payload['endpoint_host'],
            stall_count=payload['stall_count'],
            stalled_at=None if payload['stalled_at'] is None else _timestamp(payload, 'stalled_at'),
            first_run=_optional_figures(payload, 'first_run'),
            figures=_optional_figures(payload, 'figures'),
            unanswered=payload['unanswered'],
            resolved_models=tuple(resolved_models),
            cost_usd=payload['cost_usd'],
            wall_time_s=payload['wall_time_s'],
            eval_sha256=payload['eval_sha256'],
        )

    def to_ledger_record(self, *, now: datetime) -> ReconLedgerRecord:
        """This record as its row, written at *now*. No calibration row expires."""
        _require_aware('now', now)
        return ReconLedgerRecord(
            **adjudicator_calibration_key(self.identity),
            payload_json=json.dumps(self.to_payload()),
            state=self.state.value,
            created_at=now.astimezone(UTC).isoformat(),
            expires_at=None,
        )

    @classmethod
    def from_ledger_record(cls, record: ReconLedgerRecord) -> CalibrationRecord:
        if record.record_kind != RECORD_KIND_ADJUDICATOR_CALIBRATION:
            raise ValueError(f'record_kind {record.record_kind!r} is not a calibration row')
        try:
            payload = json.loads(record.payload_json)
            state = CalibrationState(record.state)
        except ValueError as exc:
            raise ValueError(f'calibration row {record.flag_type!r}: {exc}') from None
        return cls.from_payload(payload, identity=record.flag_type, state=state)


_PAYLOAD_KEYS = tuple(
    f.name for f in dataclasses.fields(CalibrationRecord) if f.name not in {'identity', 'state'}
)


def _require_aware(name: str, value: datetime) -> None:
    if not _is_aware(value):
        raise ValueError(f'{name} must be timezone-aware, got {value!r}')


def _timestamp(payload: Mapping[str, Any], key: str) -> datetime:
    value = payload[key]
    if not isinstance(value, str):
        raise ValueError(f'calibration payload {key} is not a timestamp: {value!r}')
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        raise ValueError(f'calibration payload {key} is not a timestamp: {value!r}') from None


def _optional_figures(payload: Mapping[str, Any], key: str) -> RunFigures | None:
    value = payload[key]
    if value is None:
        return None
    try:
        return RunFigures.from_payload(value)
    except ValueError as exc:
        raise ValueError(f'calibration payload {key}: {exc}') from None


@dataclass(frozen=True)
class CalibrationResult:
    """What one :func:`run_calibration` call found, and whether it ran the eval set.

    ``finding`` is set only on the call that produced it, so a caller files it once.
    """

    #: The row as this call left it; None when there is none.
    record: CalibrationRecord | None
    ran: bool
    finding: CalibrationFinding | None = None
    #: Why, for the finding's filing (the eval set's refusal).
    detail: str | None = None

    @property
    def state(self) -> CalibrationState | None:
        return None if self.record is None else self.record.state

    @property
    def failed_bounds(self) -> tuple[BoundName, ...]:
        return () if self.record is None else self.record.failed_bounds

    @property
    def may_adjudicate(self) -> bool:
        return self.state is CalibrationState.PASS


@dataclass(frozen=True)
class _Tally:
    """What one run's calls came back with; ``outcomes`` only for the cases that finished."""

    outcomes: tuple[CaseOutcome, ...]
    metas: tuple[CallMeta, ...]
    timed_out: bool
    wall_time_s: float

    @property
    def unanswered(self) -> int:
        return sum(o.kind is OutcomeKind.unanswered for o in self.outcomes)


async def run_calibration(
    *,
    ledger: ReconLedgerStore,
    settings: AdjudicatorSettings,
    model_call: ResponsesCall,
    claim_check: ClaimCheck,
    now: datetime,
    eval_set_path: Path = EVAL_SET_PATH,
    wall_time_limit_s: float = WALL_TIME_LIMIT_S,
    retry_delay_s: float = RETRY_DELAY_S,
) -> CalibrationResult:
    """Take the identity's calibration row one step: read it, and score a run when one is due.

    A ``pass`` or ``fail`` row is final. Otherwise the eval set is verified
    before any write, the run is claimed by compare-and-set so one caller runs
    it, the cases are scored through :func:`adjudicate`, and the row is
    rewritten with the run's end state.
    """
    _require_aware('now', now)
    identity = adjudicator_identity(settings)
    stored = await _read(ledger, identity)
    if stored is not None and stored.state in _SETTLED:
        return CalibrationResult(record=stored, ran=False)
    try:
        cases = load_eval_set(eval_set_path)
    except EvalSetInvalid as exc:
        return CalibrationResult(
            record=stored, ran=False,
            finding=CalibrationFinding.adjudicator_eval_set_invalid, detail=str(exc),
        )
    running = _running_record(identity, settings, stored, now)
    claimed = await ledger.upsert_if_state(
        running.to_ledger_record(now=now),
        existing_state=_NO_ROW if stored is None else stored.state.value,
    )
    if not claimed:
        return CalibrationResult(record=await _read(ledger, identity), ran=False)
    tally = await _score(
        [case for case in cases if case_exclusion(case, settings) is None],
        settings=settings, model_call=model_call, claim_check=claim_check,
        wall_time_limit_s=wall_time_limit_s, retry_delay_s=retry_delay_s,
    )
    return await _write_end(ledger, _ended(running, tally, member_min=settings.member_min), now)


async def _read(ledger: ReconLedgerStore, identity: str) -> CalibrationRecord | None:
    row = await ledger.get_by_identity(**adjudicator_calibration_key(identity))
    return None if row is None else CalibrationRecord.from_ledger_record(row)


def _running_record(
    identity: str, settings: AdjudicatorSettings, stored: CalibrationRecord | None, now: datetime,
) -> CalibrationRecord:
    """The next run's ``running`` row: one attempt on, a stored first run kept for pooling."""
    return CalibrationRecord(
        identity=identity,
        state=CalibrationState.RUNNING,
        attempt=1 if stored is None else stored.attempt + 1,
        started_at=now,
        model=settings.model,
        endpoint_host=settings.endpoint_host,
        stall_count=0 if stored is None else stored.stall_count,
        first_run=None if stored is None else stored.first_run,
    )


async def _score(
    cases: Sequence[EvalCase],
    *,
    settings: AdjudicatorSettings,
    model_call: ResponsesCall,
    claim_check: ClaimCheck,
    wall_time_limit_s: float,
    retry_delay_s: float,
) -> _Tally:
    """Every case through the adjudicator, :data:`CALLS_IN_FLIGHT` at a time, within the wall time."""
    in_flight = asyncio.Semaphore(CALLS_IN_FLIGHT)
    outcomes: list[CaseOutcome] = []
    metas: list[CallMeta] = []

    async def score(case: EvalCase) -> None:
        outcomes.append(await _score_case(
            case, settings=settings, model_call=model_call, claim_check=claim_check,
            in_flight=in_flight, retry_delay_s=retry_delay_s, metas=metas,
        ))

    started = time.monotonic()
    timed_out = False
    try:
        async with asyncio.timeout(wall_time_limit_s), asyncio.TaskGroup() as group:
            for case in cases:
                group.create_task(score(case))
    except TimeoutError:
        timed_out = True
    return _Tally(tuple(outcomes), tuple(metas), timed_out, time.monotonic() - started)


async def _score_case(
    case: EvalCase,
    *,
    settings: AdjudicatorSettings,
    model_call: ResponsesCall,
    claim_check: ClaimCheck,
    in_flight: asyncio.Semaphore,
    retry_delay_s: float,
    metas: list[CallMeta],
) -> CaseOutcome:
    """One case's outcome: a transport failure is retried, anything else ends the case."""
    request = case_request(case)
    for attempt in range(TRANSPORT_ATTEMPTS):
        if attempt:
            await asyncio.sleep(retry_delay_s)
        async with in_flight:
            result = await adjudicate(
                request, settings=settings, model_call=model_call, claim_check=claim_check,
            )
        if result.meta is not None:
            metas.append(result.meta)
        if isinstance(result, Adjudication):
            return CaseOutcome(case=case, kind=OutcomeKind.verdict, verdict=result.verdict)
        if result.kind is FailureKind.CONTENT:
            return CaseOutcome(case=case, kind=OutcomeKind.content_failure, verdict=None)
    return CaseOutcome(case=case, kind=OutcomeKind.unanswered, verdict=None)


def _ended(running: CalibrationRecord, tally: _Tally, *, member_min: int) -> CalibrationRecord:
    """*running* rewritten with the run's measures and the state K5 assigns them."""
    costs = [meta.cost_usd for meta in tally.metas if meta.cost_usd is not None]
    measured = dataclasses.replace(
        running,
        unanswered=tally.unanswered,
        resolved_models=tuple(sorted({m.resolved_model for m in tally.metas if m.resolved_model})),
        cost_usd=sum(costs) if costs else None,
        wall_time_s=tally.wall_time_s,
    )
    if tally.timed_out or tally.unanswered > MAX_UNANSWERED:
        return dataclasses.replace(measured, state=CalibrationState.INCONCLUSIVE)
    figures = score_run(tally.outcomes, member_min=member_min)
    if running.first_run is None:
        if evaluate(figures):
            return dataclasses.replace(
                measured, state=CalibrationState.FAIL_FIRST_RUN, figures=figures, first_run=figures,
            )
        return dataclasses.replace(measured, state=CalibrationState.PASS, figures=figures)
    pooled = pool(running.first_run, figures)
    state = CalibrationState.FAIL if evaluate(pooled) else CalibrationState.PASS
    return dataclasses.replace(measured, state=state, figures=pooled)


async def _write_end(
    ledger: ReconLedgerStore, ended: CalibrationRecord, now: datetime,
) -> CalibrationResult:
    """Replace this run's ``running`` row with *ended*, unless another caller moved it on."""
    written = await ledger.upsert_if_state(
        ended.to_ledger_record(now=now), existing_state=CalibrationState.RUNNING.value,
    )
    if not written:
        logger.warning(
            'calibration %s: the running row was moved on before the run ended; %s not recorded',
            ended.identity[:12], ended.state,
        )
        return CalibrationResult(record=await _read(ledger, ended.identity), ran=True)
    logger.info(
        'calibration %s: attempt %d ended %s (failed bounds: %s)',
        ended.identity[:12], ended.attempt, ended.state,
        ', '.join(ended.failed_bounds) or 'none',
    )
    finding = (
        CalibrationFinding.adjudicator_calibration_failed
        if ended.state is CalibrationState.FAIL else None
    )
    return CalibrationResult(record=ended, ran=True, finding=finding)
