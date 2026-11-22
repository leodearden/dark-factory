"""The adjudicator's self-calibration over the pinned eval set (plans/consolidation-adjudicator-prd.md K5, E7).

Owns the eval set, the six bounds and the scoring. The eval set is resolved
from the package and pinned by sha256; the bounds are module constants,
changed by code review and never by reload. Imports no executor code.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

from fused_memory.reconciliation.consolidation_adjudicator import (
    AdjudicationRequest,
    AdjudicatorSettings,
    Decision,
    Hazard,
    MemberInput,
    Verdict,
    keep_rule_holds,
)
from fused_memory.reconciliation.consolidation_auto import (
    REASON_CODE_CLASSES,
    AutoReasonCode,
    FlowReasonCode,
    ReasonCodeClass,
)

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

