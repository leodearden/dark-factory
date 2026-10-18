"""The defect-tests vocabulary: enums, the ref grammar, records and the transition table.

This module is the single home of that vocabulary (``plans/defect-tests-prd.md``
decisions 7, 13, 15 and 16; Contract **Types**, invariants **I1** and **I2**).
Consumers import from here and never re-spell an enum value or the ref grammar.

Intentionally NOT re-exported from ``shared/__init__.py``: consumers import
``from shared.defects import …``, the sub-module convention
``shared.governed_exceptions`` follows.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated, Final, Literal, Self

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    PlainValidator,
    TypeAdapter,
    field_validator,
    model_validator,
)

from shared.finding_key import KEY_PATTERN
from shared.governed_exceptions import Debt, MalformedDisposition, owner_from_record, owner_record
from shared.task_metadata import register_metadata_submodel

__all__ = [
    'Closure',
    'DEFECTS_METADATA_KEY',
    'DEFECT_TEST_RETIREMENTS_METADATA_KEY',
    'DEFECT_TRANSITIONS',
    'Declare',
    'DeclareClosure',
    'DeclareEvent',
    'DeclareRetirement',
    'DefectDeclaration',
    'DefectEventKind',
    'DefectMergeRefusal',
    'DefectOrigin',
    'DefectProof',
    'DefectRefStr',
    'DefectStatus',
    'DefectTrigger',
    'Exemption',
    'ExemptionReason',
    'FIX_FINDING_OUTCOMES',
    'IllegalDefectTransition',
    'IssueKind',
    'IssueKindFlag',
    'IssueKindReading',
    'KeepCheck',
    'KeepOutcome',
    'MalformedDefectRef',
    'MissingSymbol',
    'NO_VERDICT_OUTCOMES',
    'NodeId',
    'OutcomeEvent',
    'ParsedRef',
    'PlanCheck',
    'PlanCheckOutcome',
    'Proof',
    'ProofOutcome',
    'RETIREMENT_TRANSITIONS',
    'Retirement',
    'RetirementCode',
    'RetirementReason',
    'RetirementState',
    'Rule',
    'RuledBy',
    'Ruling',
    'RulingSubject',
    'SideOutcome',
    'SideResult',
    'Sighting',
    'UnavailableReason',
    'Withdrawal',
    'WithdrawalCode',
    'WithdrawalReason',
    'apply_event',
    'apply_retirement_event',
    'issue_kind_of',
    'mint_explore_ref',
    'mint_finding_ref',
    'mint_followup_ref',
    'mint_human_ref',
    'mint_portal_ref',
    'mint_redmain_ref',
    'mint_review_ref',
    'parse_declare_event',
    'parse_outcome_event',
    'parse_ref',
]


class DefectOrigin(StrEnum):
    REVIEW = 'review'
    REVIEW_INSTRUMENT = 'review_instrument'
    RED_MAIN = 'red_main'
    AGENT_FOLLOWUP = 'agent_followup'
    HUMAN = 'human'
    CLIENT = 'client'
    EXPLORATORY = 'exploratory'


class IssueKind(StrEnum):
    DEFECT = 'defect'
    TEST_QUALITY = 'test_quality'
    DOCS = 'docs'
    PROCESS = 'process'


class DefectStatus(StrEnum):
    OPEN = 'open'
    PENDING = 'pending'
    PROVEN = 'proven'
    EXEMPTED = 'exempted'
    WITHDRAWN = 'withdrawn'


class ExemptionReason(StrEnum):
    NONDETERMINISTIC_ENVIRONMENT = 'nondeterministic_environment'
    NEEDS_EXTERNAL_SYSTEM = 'needs_external_system'
    NO_RUNNABLE_HARNESS = 'no_runnable_harness'


class WithdrawalCode(StrEnum):
    NOT_A_DEFECT = 'not_a_defect'
    TEST_WAS_WRONG = 'test_was_wrong'
    DUPLICATE_OF = 'duplicate_of'


class RetirementCode(StrEnum):
    FEATURE_REMOVED = 'feature_removed'
    SUPERSEDED_BY = 'superseded_by'
    TEST_RELOCATED = 'test_relocated'


class ProofOutcome(StrEnum):
    PROVEN = 'PROVEN'
    PASSES_AT_BASE = 'PASSES_AT_BASE'
    FAILS_AT_TIP = 'FAILS_AT_TIP'
    NOT_COLLECTED = 'NOT_COLLECTED'
    MARKER_NOT_STATIC = 'MARKER_NOT_STATIC'
    PROOF_UNAVAILABLE = 'PROOF_UNAVAILABLE'
    PROOF_UNSUPPORTED_RUNNER = 'PROOF_UNSUPPORTED_RUNNER'


class UnavailableReason(StrEnum):
    """Why a proof could not complete (decision 10)."""

    WORKTREE = 'worktree'
    SYNC = 'sync'
    TIMEOUT = 'timeout'
    CRASH = 'crash'
    UNATTRIBUTED_BASE_ERROR = 'unattributed_base_error'


class SideOutcome(StrEnum):
    PASSED = 'passed'
    FAILED = 'failed'
    ERROR_ATTRIBUTED = 'error_attributed'
    ERROR_UNATTRIBUTED = 'error_unattributed'
    SKIPPED = 'skipped'
    NOT_COLLECTED = 'not_collected'


class KeepOutcome(StrEnum):
    KEPT = 'KEPT'
    REMOVED_RETIRED = 'REMOVED_RETIRED'
    REMOVED_UNRETIRED = 'REMOVED_UNRETIRED'
    KEEP_CHECK_INCOMPLETE = 'KEEP_CHECK_INCOMPLETE'
    KEEP_CHECK_UNSUPPORTED = 'KEEP_CHECK_UNSUPPORTED'


class DefectMergeRefusal(StrEnum):
    DEFECT_UNRESOLVED = 'defect_unresolved'
    DEFECT_PROOF_UNAVAILABLE = 'defect_proof_unavailable'
    DEFECT_TEST_REMOVED = 'defect_test_removed'
    KEEP_CHECK_INCOMPLETE = 'keep_check_incomplete'


class PlanCheckOutcome(StrEnum):
    OK = 'ok'
    MISSING = 'missing'
    REPRODUCES_AFTER_FIX = 'reproduces_after_fix'
    UNKNOWN_REF = 'unknown_ref'


class RulingSubject(StrEnum):
    CLOSURE = 'closure'
    RETIREMENT = 'retirement'


class RuledBy(StrEnum):
    REVIEWER = 'reviewer'
    OPERATOR = 'operator'


class DefectEventKind(StrEnum):
    DECLARE = 'declare'
    DECLARE_CLOSURE = 'declare_closure'
    DECLARE_RETIREMENT = 'declare_retirement'
    SIGHTING = 'sighting'
    PROOF = 'proof'
    RULE = 'rule'


# --- The ref grammar (decision 15) -------------------------------------------
#
# Every component pattern is applied with `fullmatch`, which must consume the
# whole component. That is what makes the imported, `$`-anchored
# shared.finding_key.KEY_PATTERN exact too: Python's `$` also matches before a
# trailing newline (the trap shared/src/shared/governed_exceptions.py::_LIST_ID_RE
# documents), but `fullmatch` still refuses the unconsumed newline.

_POSITIVE_INT = re.compile(r'[1-9][0-9]*')
_TOKEN = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]*')
_HUMAN_FILED_AT_FORMAT = '%Y%m%dT%H%M%SZ'


def _is_real_utc_timestamp(text: str) -> bool:
    try:
        datetime.strptime(text, _HUMAN_FILED_AT_FORMAT)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class _RefComponent:
    name: str
    pattern: re.Pattern[str]
    description: str
    semantic_check: Callable[[str], bool] | None = None

    def accepts(self, text: str) -> bool:
        if self.pattern.fullmatch(text) is None:
            return False
        return self.semantic_check is None or self.semantic_check(text)


@dataclass(frozen=True)
class _RefForm:
    prefix: str
    components: tuple[_RefComponent, ...]

    @property
    def shape(self) -> str:
        return ':'.join([self.prefix, *(f'<{c.name}>' for c in self.components)])


def _task_id(name: str) -> _RefComponent:
    return _RefComponent(name, _POSITIVE_INT, 'a task id: decimal, no leading zero')


def _count(name: str) -> _RefComponent:
    return _RefComponent(name, _POSITIVE_INT, 'a count: decimal, at least 1, no leading zero')


def _token(name: str) -> _RefComponent:
    return _RefComponent(
        name, _TOKEN, 'a token: an ASCII letter or digit, then letters, digits, ".", "_" or "-"'
    )


def _hex(name: str, width: int) -> _RefComponent:
    return _RefComponent(name, re.compile(f'[0-9a-f]{{{width}}}'), f'{width} lower-case hex digits')


_REF_GRAMMAR: MappingProxyType[DefectOrigin, _RefForm] = MappingProxyType(
    {
        DefectOrigin.REVIEW: _RefForm('review', (_task_id('task'), _hex('issue_hash', 12))),
        DefectOrigin.REVIEW_INSTRUMENT: _RefForm(
            'finding',
            (_RefComponent('key', re.compile(KEY_PATTERN), 'a finding key (shared.finding_key)'),),
        ),
        DefectOrigin.RED_MAIN: _RefForm('redmain', (_hex('signature', 16),)),
        DefectOrigin.AGENT_FOLLOWUP: _RefForm(
            'followup', (_task_id('spawned_from'), _token('suggestion_hash'))
        ),
        DefectOrigin.HUMAN: _RefForm(
            'human',
            (
                _RefComponent(
                    'filed_at',
                    re.compile(r'[0-9]{8}T[0-9]{6}Z'),
                    'a real UTC time in ISO-8601 basic form, YYYYMMDDTHHMMSSZ',
                    _is_real_utc_timestamp,
                ),
                _hex('hash8', 8),
            ),
        ),
        DefectOrigin.CLIENT: _RefForm('portal', (_token('project'), _token('block'), _count('n'))),
        DefectOrigin.EXPLORATORY: _RefForm('explore', (_token('run'), _count('n'))),
    }
)

_ORIGIN_BY_PREFIX: MappingProxyType[str, DefectOrigin] = MappingProxyType(
    {form.prefix: origin for origin, form in _REF_GRAMMAR.items()}
)


class MalformedDefectRef(ValueError):
    """A defect ref, or a value offered to mint one, is outside the grammar.

    A ``ValueError`` so that a pydantic field validator reports it as a
    ``ValidationError`` carrying this message.

    Attributes:
        origin: The origin whose grammar was violated, or None when the
            prefix names no origin (or the value is not a str at all).
        value: The offending value, verbatim.
    """

    def __init__(self, message: str, *, origin: DefectOrigin | None, value: object) -> None:
        self.origin = origin
        self.value = value
        super().__init__(message)


@dataclass(frozen=True)
class ParsedRef:
    """A ref that parsed, with its components by name in grammar order."""

    origin: DefectOrigin
    ref: str
    components: tuple[tuple[str, str], ...]

    def component(self, name: str) -> str:
        for component_name, text in self.components:
            if component_name == name:
                return text
        valid = [component_name for component_name, _ in self.components]
        raise KeyError(f'a {self.origin} ref has no component {name!r}; its components are {valid}')


def parse_ref(value: object) -> ParsedRef:
    """Parse *value* under the decision-15 grammar; the grammar's one enforcement point.

    Raises:
        MalformedDefectRef: *value* is not a str, has an unknown prefix, has
            the wrong number of components, or a component fails its pattern
            or semantic check.
    """
    if not isinstance(value, str):
        raise MalformedDefectRef(
            f'defect ref {value!r} is a {type(value).__name__}, not a str', origin=None, value=value
        )
    prefix, *parts = value.split(':')
    origin = _ORIGIN_BY_PREFIX.get(prefix)
    if origin is None:
        raise MalformedDefectRef(
            f'defect ref {value!r} has unknown prefix {prefix!r}; the known prefixes are '
            f'{sorted(_ORIGIN_BY_PREFIX)}',
            origin=None,
            value=value,
        )
    form = _REF_GRAMMAR[origin]
    if len(parts) != len(form.components):
        raise MalformedDefectRef(
            f'{origin} ref {value!r} has {len(parts)} component(s) after the prefix, '
            f'expected {len(form.components)}: {form.shape}',
            origin=origin,
            value=value,
        )
    for component, text in zip(form.components, parts, strict=True):
        if not component.accepts(text):
            raise MalformedDefectRef(
                f'{origin} ref {value!r}: component {component.name}={text!r} is not '
                f'{component.description} (shape {form.shape})',
                origin=origin,
                value=value,
            )
    names = (component.name for component in form.components)
    return ParsedRef(origin=origin, ref=value, components=tuple(zip(names, parts, strict=True)))


def _mint(origin: DefectOrigin, *components: str) -> str:
    return parse_ref(':'.join((_REF_GRAMMAR[origin].prefix, *components))).ref


def _positive_int_text(origin: DefectOrigin, name: str, value: int) -> str:
    if type(value) is not int or value < 1:
        raise MalformedDefectRef(
            f'{origin} ref component {name}={value!r} must be an int of at least 1 '
            '(bool is excluded even though it is an int subclass)',
            origin=origin,
            value=value,
        )
    return str(value)


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _normalise(text: str) -> str:
    return ' '.join(text.split()).lower()


def mint_review_ref(task_id: int, location: str, description: str) -> str:
    """``review:<task>:<issue_hash>``, idempotent by the issue's normalised content.

    Each part is whitespace-collapsed and lower-cased separately and joined by
    the unit separator, so ``('a', 'bc')`` and ``('ab', 'c')`` never collide.
    """
    origin = DefectOrigin.REVIEW
    issue_hash = _sha256_hex(f'{_normalise(location)}\x1f{_normalise(description)}')[:12]
    return _mint(origin, _positive_int_text(origin, 'task', task_id), issue_hash)


def mint_finding_ref(key: str) -> str:
    return _mint(DefectOrigin.REVIEW_INSTRUMENT, key)


def mint_redmain_ref(signature: str) -> str:
    """``redmain:<signature>``, the signature RetryLedger.compute_merge_outcome_signature mints."""
    return _mint(DefectOrigin.RED_MAIN, signature)


def mint_followup_ref(spawned_from: int, suggestion_hash: str) -> str:
    origin = DefectOrigin.AGENT_FOLLOWUP
    return _mint(origin, _positive_int_text(origin, 'spawned_from', spawned_from), suggestion_hash)


def mint_human_ref(filed_at: datetime, summary: str) -> str:
    """``human:<filed_at, UTC, seconds>:<hash8 of summary>``; *filed_at* must be tz-aware."""
    origin = DefectOrigin.HUMAN
    if filed_at.utcoffset() is None:
        raise MalformedDefectRef(
            f'{origin} ref filed_at={filed_at!r} is naive; it must carry a timezone so it '
            'can be converted to UTC',
            origin=origin,
            value=filed_at,
        )
    stamp = filed_at.astimezone(UTC).strftime(_HUMAN_FILED_AT_FORMAT)
    return _mint(origin, stamp, _sha256_hex(summary)[:8])


def mint_portal_ref(project: str, block: str, n: int) -> str:
    origin = DefectOrigin.CLIENT
    return _mint(origin, project, block, _positive_int_text(origin, 'n', n))


def mint_explore_ref(run: str, n: int) -> str:
    origin = DefectOrigin.EXPLORATORY
    return _mint(origin, run, _positive_int_text(origin, 'n', n))


# --- Records -----------------------------------------------------------------

_RECORD_CONFIG = ConfigDict(frozen=True, extra='forbid')


def _grammar_checked(value: str) -> str:
    return parse_ref(value).ref


DefectRefStr = Annotated[str, AfterValidator(_grammar_checked)]
"""A str field that must parse under the ref grammar; a violation is a ValidationError."""

_GIT_OBJECT_ID = re.compile(r'[0-9a-f]{40}|[0-9a-f]{64}')


def _git_object_id(value: str) -> str:
    if _GIT_OBJECT_ID.fullmatch(value) is None:
        raise ValueError(
            f'{value!r} is not a git object id: expected 40 (sha1) or 64 (sha256) '
            'lower-case hex digits'
        )
    return value


_GitObjectId = Annotated[str, AfterValidator(_git_object_id)]
_NonEmptyStr = Annotated[str, Field(min_length=1)]


class NodeId(BaseModel):
    """A module-qualified pytest node id, ``<module_root>/<path>::<name>`` (decision 19)."""

    model_config = _RECORD_CONFIG

    module_root: _NonEmptyStr
    path: _NonEmptyStr
    name: _NonEmptyStr

    @field_validator('path')
    @classmethod
    def _path_is_one_file(cls, path: str) -> str:
        if '::' in path:
            raise ValueError(
                f'NodeId path {path!r} holds the node separator "::"; the test name '
                'belongs in name, not in path'
            )
        return path


class MissingSymbol(BaseModel):
    """Decision 9's structured missing name.

    ``module`` is ``e.name``, ``symbol`` is ``e.name_from`` and ``owner_type``
    is the type of ``e.obj``; which are set depends on the exception class.
    """

    model_config = _RECORD_CONFIG

    module: str | None = None
    symbol: str | None = None
    owner_type: str | None = None

    @model_validator(mode='after')
    def _names_something(self) -> Self:
        if self.module is None and self.symbol is None and self.owner_type is None:
            raise ValueError(
                'MissingSymbol names nothing: at least one of module, symbol and '
                'owner_type must be set'
            )
        return self


_ERROR_SIDE_OUTCOMES = frozenset({SideOutcome.ERROR_ATTRIBUTED, SideOutcome.ERROR_UNATTRIBUTED})


class SideResult(BaseModel):
    """One node's outcome on one side (base or tip) of a proof."""

    model_config = _RECORD_CONFIG

    node_id: NodeId
    outcome: SideOutcome
    exc_type: str | None = None
    missing_symbol: MissingSymbol | None = None

    @model_validator(mode='after')
    def _symbol_fits_outcome(self) -> Self:
        if self.outcome is SideOutcome.ERROR_ATTRIBUTED and self.missing_symbol is None:
            raise ValueError(
                f'SideResult for {self.node_id}: an error_attributed outcome needs the '
                'missing_symbol it was attributed to'
            )
        if self.missing_symbol is not None and self.outcome not in _ERROR_SIDE_OUTCOMES:
            raise ValueError(
                f'SideResult for {self.node_id}: outcome {self.outcome} is not an error, so '
                f'it carries no missing_symbol (got {self.missing_symbol})'
            )
        return self


class DefectProof(BaseModel):
    """One proof of one ref against the pre-fix tree (decision 9).

    The validator enforces field coherence only. Whether the per-node sides
    amount to PROVEN (I4) is the proof runner's classification, not this record's.
    """

    model_config = _RECORD_CONFIG

    ref: DefectRefStr
    outcome: ProofOutcome
    base_sha: _GitObjectId
    tip_sha: _GitObjectId
    measured_at: AwareDatetime
    node_ids: tuple[NodeId, ...]
    base: tuple[SideResult, ...]
    tip: tuple[SideResult, ...]
    unavailable_reason: UnavailableReason | None = None
    runner_rc: int | None = None
    duration_s: float = Field(ge=0)

    @model_validator(mode='after')
    def _fields_cohere(self) -> Self:
        unavailable = self.outcome is ProofOutcome.PROOF_UNAVAILABLE
        if unavailable != (self.unavailable_reason is not None):
            raise ValueError(
                f'DefectProof for {self.ref}: unavailable_reason is set iff the outcome is '
                f'PROOF_UNAVAILABLE; got outcome={self.outcome}, '
                f'unavailable_reason={self.unavailable_reason}'
            )
        if self.outcome is ProofOutcome.PROVEN and not self.node_ids:
            raise ValueError(
                f'DefectProof for {self.ref}: a PROVEN proof names at least one node id'
            )
        return self


class Ruling(BaseModel):
    """An accept or reject of a closure or a retirement, by the reviewer or the operator."""

    model_config = _RECORD_CONFIG

    subject: RulingSubject
    accepted: bool
    ruled_by: RuledBy
    verdict_cycle: int | None = Field(default=None, ge=0)

    @model_validator(mode='after')
    def _cycle_fits_ruler(self) -> Self:
        if (self.ruled_by is RuledBy.REVIEWER) != (self.verdict_cycle is not None):
            raise ValueError(
                'Ruling: verdict_cycle is set iff ruled_by is reviewer (an operator ruling '
                f'has no verdict cycle); got ruled_by={self.ruled_by}, '
                f'verdict_cycle={self.verdict_cycle}'
            )
        return self


def _debt_from_wire(value: object) -> Debt:
    if isinstance(value, Debt):
        return value
    try:
        return Debt(owner_from_record(value))
    except MalformedDisposition as exc:
        # MalformedDisposition is deliberately not a ValueError; pydantic wraps
        # only ValueError, and parse_metadata catches only ValidationError.
        raise ValueError(str(exc)) from exc


def _debt_to_wire(debt: Debt) -> dict[str, object]:
    return owner_record(debt.owner)


_DebtOwner = Annotated[Debt, PlainValidator(_debt_from_wire), PlainSerializer(_debt_to_wire)]


class Exemption(BaseModel):
    """Decision 13's ``not_reproducible`` closure: why no proof can exist, and who owes one."""

    model_config = _RECORD_CONFIG

    kind: Literal['exemption'] = 'exemption'
    reason: ExemptionReason
    owner: _DebtOwner


def _require_ref_iff(code: StrEnum, parameterised: StrEnum, ref: str | None) -> None:
    if (code is parameterised) != (ref is not None):
        raise ValueError(
            f'reason {code} {"requires" if code is parameterised else "takes no"} ref '
            f'(only {parameterised} names one); got ref={ref!r}'
        )


def _require_not_self(ref: str, named: str | None, relation: str) -> None:
    if named == ref:
        raise ValueError(f'{ref} cannot be {relation} itself')


class WithdrawalReason(BaseModel):
    model_config = _RECORD_CONFIG

    reason: WithdrawalCode
    ref: DefectRefStr | None = None

    @model_validator(mode='after')
    def _ref_iff_duplicate(self) -> Self:
        _require_ref_iff(self.reason, WithdrawalCode.DUPLICATE_OF, self.ref)
        return self


class Withdrawal(BaseModel):
    """Decision 13's withdrawal closure: the declared defect is not one to prove."""

    model_config = _RECORD_CONFIG

    kind: Literal['withdrawal'] = 'withdrawal'
    reason: WithdrawalReason


Closure = Annotated[Exemption | Withdrawal, Field(discriminator='kind')]


class RetirementReason(BaseModel):
    model_config = _RECORD_CONFIG

    reason: RetirementCode
    ref: DefectRefStr | None = None

    @model_validator(mode='after')
    def _ref_iff_superseded(self) -> Self:
        _require_ref_iff(self.reason, RetirementCode.SUPERSEDED_BY, self.ref)
        return self


class RetirementState(StrEnum):
    PENDING = 'pending'
    ACCEPTED = 'accepted'
    REJECTED = 'rejected'


class Retirement(BaseModel):
    """A ruled record, on the task whose diff removes a proven defect's test (decisions 13, 14).

    Not a state of the defect's own entry: only this record's ruling changes it.
    """

    model_config = _RECORD_CONFIG

    ref: DefectRefStr
    reason: RetirementReason
    ruling: Ruling | None = None

    @model_validator(mode='after')
    def _coheres(self) -> Self:
        if self.ruling is not None and self.ruling.subject is not RulingSubject.RETIREMENT:
            raise ValueError(
                f'Retirement {self.ref}: its ruling must have subject retirement, '
                f'got {self.ruling.subject}'
            )
        _require_not_self(self.ref, self.reason.ref, 'superseded_by')
        return self

    @property
    def state(self) -> RetirementState:
        if self.ruling is None:
            return RetirementState.PENDING
        return RetirementState.ACCEPTED if self.ruling.accepted else RetirementState.REJECTED


class PlanCheck(BaseModel):
    """Whether a plan reproduces the task's declared defects, and which refs the outcome is about."""

    model_config = _RECORD_CONFIG

    outcome: PlanCheckOutcome
    defect_refs: tuple[DefectRefStr, ...] = ()


class KeepCheck(BaseModel):
    """The keep-ratchet's result for one candidate against the tree it lands on (decision 14)."""

    model_config = _RECORD_CONFIG

    outcome: KeepOutcome
    removed_refs: tuple[DefectRefStr, ...] = ()
    retired_refs: tuple[DefectRefStr, ...] = ()
    incomplete_files: tuple[str, ...] = ()
    parent_sha: _GitObjectId
    candidate_sha: _GitObjectId


def _require_closure_not_self(ref: str, closure: Exemption | Withdrawal | None) -> None:
    if isinstance(closure, Withdrawal):
        _require_not_self(ref, closure.reason.ref, 'duplicate_of')


class DefectDeclaration(BaseModel):
    """One entry of ``metadata.defects``: a declared defect and where it stands (decision 7).

    After creation it changes only through :func:`apply_event`, never by edit.
    """

    model_config = _RECORD_CONFIG

    ref: DefectRefStr
    origin: DefectOrigin
    summary: _NonEmptyStr
    declared_at: AwareDatetime
    declared_by: _NonEmptyStr
    status: DefectStatus
    sightings: int = Field(default=1, ge=1)
    proof: DefectProof | None = None
    closure: Closure | None = None
    ruling: Ruling | None = None

    @model_validator(mode='after')
    def _coheres(self) -> Self:
        _require_coherent(self)
        return self


class _ClosureStanding(StrEnum):
    NONE = 'none'
    UNRULED = 'unruled'
    REJECTED = 'rejected'
    ACCEPTED = 'accepted'


_STANDINGS_BY_STATUS: MappingProxyType[DefectStatus, frozenset[_ClosureStanding]] = (
    MappingProxyType(
        {
            DefectStatus.OPEN: frozenset({_ClosureStanding.NONE, _ClosureStanding.REJECTED}),
            DefectStatus.PROVEN: frozenset({_ClosureStanding.NONE, _ClosureStanding.REJECTED}),
            DefectStatus.PENDING: frozenset({_ClosureStanding.UNRULED}),
            DefectStatus.EXEMPTED: frozenset({_ClosureStanding.ACCEPTED}),
            DefectStatus.WITHDRAWN: frozenset({_ClosureStanding.ACCEPTED}),
        }
    )
)

_CLOSURE_TYPE_BY_STATUS: MappingProxyType[DefectStatus, type[Exemption | Withdrawal]] = (
    MappingProxyType({DefectStatus.EXEMPTED: Exemption, DefectStatus.WITHDRAWN: Withdrawal})
)


def _closure_standing(entry: DefectDeclaration) -> _ClosureStanding:
    if entry.closure is None:
        return _ClosureStanding.NONE
    if entry.ruling is None:
        return _ClosureStanding.UNRULED
    return _ClosureStanding.ACCEPTED if entry.ruling.accepted else _ClosureStanding.REJECTED


def _require_coherent(entry: DefectDeclaration) -> None:
    """The one coherence rule for an entry: status, closure, ruling and proof agree.

    Run at construction, and therefore again on every result :func:`apply_event` builds.
    """
    ref_origin = parse_ref(entry.ref).origin
    if entry.origin is not ref_origin:
        raise ValueError(
            f'{entry.ref}: origin {entry.origin} does not match the ref, whose prefix '
            f'names origin {ref_origin}'
        )
    if entry.proof is not None and entry.proof.ref != entry.ref:
        raise ValueError(f'{entry.ref}: its proof is for a different ref, {entry.proof.ref}')
    if entry.status is DefectStatus.PROVEN and entry.proof is None:
        raise ValueError(f'{entry.ref}: a proven entry carries its latest proof')
    if entry.ruling is not None and entry.closure is None:
        raise ValueError(f'{entry.ref}: a ruling rules on a closure, and the entry has none')
    if entry.ruling is not None and entry.ruling.subject is not RulingSubject.CLOSURE:
        raise ValueError(
            f'{entry.ref}: an entry carries only closure rulings, got subject {entry.ruling.subject}'
        )
    standing = _closure_standing(entry)
    allowed = _STANDINGS_BY_STATUS[entry.status]
    if standing not in allowed:
        raise ValueError(
            f'{entry.ref}: status {entry.status} allows a closure standing of '
            f'{" or ".join(sorted(allowed))}, but its closure standing is {standing}'
        )
    closure_type = _CLOSURE_TYPE_BY_STATUS.get(entry.status)
    if closure_type is not None and not isinstance(entry.closure, closure_type):
        raise ValueError(
            f'{entry.ref}: status {entry.status} needs a {closure_type.__name__} closure, '
            f'got {type(entry.closure).__name__}'
        )
    _require_closure_not_self(entry.ref, entry.closure)


# --- Events ------------------------------------------------------------------


def _require_fresh(declaration: DefectDeclaration) -> None:
    stale = [
        name
        for name, is_fresh in (
            ('status', declaration.status is DefectStatus.OPEN),
            ('sightings', declaration.sightings == 1),
            ('proof', declaration.proof is None),
            ('closure', declaration.closure is None),
            ('ruling', declaration.ruling is None),
        )
        if not is_fresh
    ]
    if stale:
        raise ValueError(
            f'{declaration.ref}: a new declaration is fresh (status open, one sighting, no '
            f'proof, closure or ruling); these fields are not: {stale}'
        )


class Declare(BaseModel):
    """Declare a new defect; the creation floor every creating path reuses."""

    model_config = _RECORD_CONFIG

    event: Literal[DefectEventKind.DECLARE] = DefectEventKind.DECLARE
    declaration: DefectDeclaration

    @model_validator(mode='after')
    def _is_fresh(self) -> Self:
        _require_fresh(self.declaration)
        return self

    @property
    def ref(self) -> str:
        return self.declaration.ref


class DeclareClosure(BaseModel):
    model_config = _RECORD_CONFIG

    event: Literal[DefectEventKind.DECLARE_CLOSURE] = DefectEventKind.DECLARE_CLOSURE
    ref: DefectRefStr
    closure: Closure

    @model_validator(mode='after')
    def _not_self(self) -> Self:
        _require_closure_not_self(self.ref, self.closure)
        return self


class DeclareRetirement(BaseModel):
    model_config = _RECORD_CONFIG

    event: Literal[DefectEventKind.DECLARE_RETIREMENT] = DefectEventKind.DECLARE_RETIREMENT
    retirement: Retirement

    @model_validator(mode='after')
    def _is_unruled(self) -> Self:
        if self.retirement.ruling is not None:
            raise ValueError(
                f'{self.retirement.ref}: a retirement is declared unruled; its ruling '
                'arrives later as a Rule'
            )
        return self

    @property
    def ref(self) -> str:
        return self.retirement.ref


class Sighting(BaseModel):
    model_config = _RECORD_CONFIG

    event: Literal[DefectEventKind.SIGHTING] = DefectEventKind.SIGHTING
    ref: DefectRefStr


class Proof(BaseModel):
    model_config = _RECORD_CONFIG

    event: Literal[DefectEventKind.PROOF] = DefectEventKind.PROOF
    proof: DefectProof

    @property
    def ref(self) -> str:
        return self.proof.ref


class Rule(BaseModel):
    model_config = _RECORD_CONFIG

    event: Literal[DefectEventKind.RULE] = DefectEventKind.RULE
    ref: DefectRefStr
    ruling: Ruling


DeclareEvent = Annotated[
    Declare | DeclareClosure | DeclareRetirement | Sighting, Field(discriminator='event')
]
"""What ``declare_defect`` carries; any filer may send one."""

OutcomeEvent = Annotated[Proof | Rule, Field(discriminator='event')]
"""What ``record_defect_outcome`` carries; only the orchestrator and prove_defect.py send one."""

_DECLARE_EVENT_ADAPTER: Final = TypeAdapter(DeclareEvent)
_OUTCOME_EVENT_ADAPTER: Final = TypeAdapter(OutcomeEvent)


def parse_declare_event(raw: object) -> DeclareEvent:
    """Validate a wire dict as a :data:`DeclareEvent`; raises pydantic ``ValidationError``."""
    return _DECLARE_EVENT_ADAPTER.validate_python(raw)


def parse_outcome_event(raw: object) -> OutcomeEvent:
    """Validate a wire dict as an :data:`OutcomeEvent`; raises pydantic ``ValidationError``."""
    return _OUTCOME_EVENT_ADAPTER.validate_python(raw)


# --- I2: the defects-entry transition table ----------------------------------


class DefectTrigger(StrEnum):
    """What an event does to a record, finer than its kind: I2 routes on proof class and closure.

    One vocabulary for both :data:`DEFECT_TRANSITIONS` and :data:`RETIREMENT_TRANSITIONS`.
    """

    DECLARE = 'declare'
    DECLARE_CLOSURE = 'declare_closure'
    DECLARE_RETIREMENT = 'declare_retirement'
    SIGHTING = 'sighting'
    PROOF_PROVEN = 'proof_proven'
    PROOF_FIX_FINDING = 'proof_fix_finding'
    PROOF_NO_VERDICT = 'proof_no_verdict'
    RULE_ACCEPTED_EXEMPTION = 'rule_accepted_exemption'
    RULE_ACCEPTED_WITHDRAWAL = 'rule_accepted_withdrawal'
    RULE_ACCEPTED_RETIREMENT = 'rule_accepted_retirement'
    RULE_REJECTED = 'rule_rejected'


FIX_FINDING_OUTCOMES: frozenset[ProofOutcome] = frozenset(
    {
        ProofOutcome.PASSES_AT_BASE,
        ProofOutcome.FAILS_AT_TIP,
        ProofOutcome.NOT_COLLECTED,
        ProofOutcome.MARKER_NOT_STATIC,
    }
)
"""Outcomes that are findings about the fix (decision 10); in enforce they drive a replan."""

NO_VERDICT_OUTCOMES: frozenset[ProofOutcome] = frozenset(
    {ProofOutcome.PROOF_UNAVAILABLE, ProofOutcome.PROOF_UNSUPPORTED_RUNNER}
)
"""Outcomes recorded as the latest proof with no transition (I2)."""

_PROOF_TRIGGERS: MappingProxyType[ProofOutcome, DefectTrigger] = MappingProxyType(
    {
        ProofOutcome.PROVEN: DefectTrigger.PROOF_PROVEN,
        **dict.fromkeys(FIX_FINDING_OUTCOMES, DefectTrigger.PROOF_FIX_FINDING),
        **dict.fromkeys(NO_VERDICT_OUTCOMES, DefectTrigger.PROOF_NO_VERDICT),
    }
)

DEFECT_TRANSITIONS: Mapping[tuple[DefectStatus | None, DefectTrigger], DefectStatus] = (
    MappingProxyType(
        {
            (None, DefectTrigger.DECLARE): DefectStatus.OPEN,
            (DefectStatus.OPEN, DefectTrigger.DECLARE_CLOSURE): DefectStatus.PENDING,
            (DefectStatus.OPEN, DefectTrigger.PROOF_PROVEN): DefectStatus.PROVEN,
            (DefectStatus.OPEN, DefectTrigger.PROOF_FIX_FINDING): DefectStatus.OPEN,
            (DefectStatus.OPEN, DefectTrigger.PROOF_NO_VERDICT): DefectStatus.OPEN,
            (DefectStatus.PROVEN, DefectTrigger.PROOF_PROVEN): DefectStatus.PROVEN,
            (DefectStatus.PROVEN, DefectTrigger.PROOF_FIX_FINDING): DefectStatus.OPEN,
            (DefectStatus.PROVEN, DefectTrigger.PROOF_NO_VERDICT): DefectStatus.PROVEN,
            (DefectStatus.PENDING, DefectTrigger.RULE_ACCEPTED_EXEMPTION): DefectStatus.EXEMPTED,
            (DefectStatus.PENDING, DefectTrigger.RULE_ACCEPTED_WITHDRAWAL): DefectStatus.WITHDRAWN,
            (DefectStatus.PENDING, DefectTrigger.RULE_REJECTED): DefectStatus.OPEN,
            **{(status, DefectTrigger.SIGHTING): status for status in DefectStatus},
        }
    )
)
"""PRD I2, the single home: (from status, None when absent; trigger) -> to status.

Every pair not listed is refused. The rows that keep the status are explicit:
a sighting in every status, and a proof that is recorded without a transition.
I2's "later proof" is the one rule a (status, trigger) key cannot carry:
:func:`apply_event` also refuses a proof measured no later than the entry's
latest one.
"""


class IllegalDefectTransition(Exception):
    """An event the transition table does not allow from the record's current state.

    Deliberately not a ValueError: the record and the event are each well
    formed, and it is the protocol that refuses the pair.

    Attributes:
        ref: The record's ref (the event's, when no record exists yet).
        from_status: The record's status, or None when it does not exist.
        event: The refused event's kind.
        trigger: The trigger the event classified as, or None when it was
            refused before classification.
    """

    def __init__(
        self,
        message: str,
        *,
        ref: str,
        from_status: DefectStatus | RetirementState | None,
        event: DefectEventKind,
        trigger: DefectTrigger | None,
    ) -> None:
        self.ref = ref
        self.from_status = from_status
        self.event = event
        self.trigger = trigger
        super().__init__(message)


_AnyEvent = Declare | DeclareClosure | DeclareRetirement | Sighting | Proof | Rule
_EntryEvent = Declare | DeclareClosure | Sighting | Proof | Rule


def _refusal(
    table: Iterable[tuple[StrEnum | None, DefectTrigger]],
    *,
    ref: str,
    from_status: DefectStatus | RetirementState | None,
    event: DefectEventKind,
    trigger: DefectTrigger | None,
    why: str,
) -> IllegalDefectTransition:
    state = 'absent' if from_status is None else str(from_status)
    legal = sorted(row_trigger for row_status, row_trigger in table if row_status == from_status)
    via = '' if trigger is None else f' (trigger {trigger})'
    return IllegalDefectTransition(
        f'{ref}: a {event} event{via} is refused from {state}: {why}. '
        f'Legal triggers from {state}: {", ".join(legal) or "none"}',
        ref=ref,
        from_status=from_status,
        event=event,
        trigger=trigger,
    )


def _require_event_ref(
    record_ref: str,
    from_status: DefectStatus | RetirementState,
    event: _AnyEvent,
    table: Iterable[tuple[StrEnum | None, DefectTrigger]],
) -> None:
    if event.ref != record_ref:
        raise _refusal(
            table,
            ref=record_ref,
            from_status=from_status,
            event=event.event,
            trigger=None,
            why=f'the event concerns {event.ref}, not {record_ref}',
        )


def _defect_refusal(
    entry: DefectDeclaration | None, event: _AnyEvent, trigger: DefectTrigger | None, why: str
) -> IllegalDefectTransition:
    return _refusal(
        DEFECT_TRANSITIONS,
        ref=event.ref if entry is None else entry.ref,
        from_status=None if entry is None else entry.status,
        event=event.event,
        trigger=trigger,
        why=why,
    )


def _entry_event(entry: DefectDeclaration | None, event: _AnyEvent) -> _EntryEvent:
    if isinstance(event, DeclareRetirement):
        raise _defect_refusal(
            entry,
            event,
            None,
            'a retirement is a record on the removing task, never a defects-entry event',
        )
    return event


def _rule_trigger(entry: DefectDeclaration | None, event: Rule) -> DefectTrigger:
    ruling = event.ruling
    if ruling.subject is not RulingSubject.CLOSURE:
        raise _defect_refusal(
            entry, event, None, f'a {ruling.subject} ruling does not apply to a defects entry'
        )
    if not ruling.accepted:
        return DefectTrigger.RULE_REJECTED
    closure = None if entry is None else entry.closure
    if isinstance(closure, Exemption):
        return DefectTrigger.RULE_ACCEPTED_EXEMPTION
    if isinstance(closure, Withdrawal):
        return DefectTrigger.RULE_ACCEPTED_WITHDRAWAL
    raise _defect_refusal(entry, event, None, 'an accepted ruling needs a closure to accept')


def _trigger(entry: DefectDeclaration | None, event: _EntryEvent) -> DefectTrigger:
    match event:
        case Declare():
            return DefectTrigger.DECLARE
        case DeclareClosure():
            return DefectTrigger.DECLARE_CLOSURE
        case Sighting():
            return DefectTrigger.SIGHTING
        case Proof():
            return _PROOF_TRIGGERS[event.proof.outcome]
        case Rule():
            return _rule_trigger(entry, event)


def _destination(
    entry: DefectDeclaration | None, event: _EntryEvent, trigger: DefectTrigger
) -> DefectStatus:
    to_status = DEFECT_TRANSITIONS.get((None if entry is None else entry.status, trigger))
    if to_status is None:
        raise _defect_refusal(entry, event, trigger, 'PRD I2 has no such transition')
    return to_status


def _require_later_proof(
    entry: DefectDeclaration | None, event: _EntryEvent, trigger: DefectTrigger
) -> None:
    if not isinstance(event, Proof) or entry is None or entry.proof is None:
        return
    offered, latest = event.proof.measured_at, entry.proof.measured_at
    if offered <= latest:
        raise _defect_refusal(
            entry,
            event,
            trigger,
            f'PRD I2 replaces a proof only with a later one; this one was measured at '
            f'{offered.isoformat()}, not after the latest, measured at {latest.isoformat()}',
        )


def _effect(
    entry: DefectDeclaration | None, event: _EntryEvent, to_status: DefectStatus
) -> DefectDeclaration:
    """The entry after a transition the table allowed, always CONSTRUCTED so coherence re-runs."""
    if isinstance(event, Declare):
        return event.declaration
    if entry is None:
        raise _defect_refusal(entry, event, None, 'only a declare event creates an entry')
    changes: dict[str, object]
    match event:
        case DeclareClosure():
            changes = {'closure': event.closure, 'ruling': None}
        case Sighting():
            changes = {'sightings': entry.sightings + 1}
        case Proof():
            changes = {'proof': event.proof}
        case Rule():
            changes = {'ruling': event.ruling}
    return DefectDeclaration.model_validate({**dict(entry), **changes, 'status': to_status})


def apply_event(
    entry: DefectDeclaration | None, event: DeclareEvent | OutcomeEvent
) -> DefectDeclaration:
    """The defects entry after *event*, per :data:`DEFECT_TRANSITIONS`; pure.

    *entry* is None when no entry for the ref exists yet.

    Raises:
        IllegalDefectTransition: the event concerns another ref, is not a
            defects-entry event, I2 has no transition for it from the
            entry's status, or it is a proof not measured after the entry's
            latest proof.
    """
    if entry is not None:
        _require_event_ref(entry.ref, entry.status, event, DEFECT_TRANSITIONS)
    entry_event = _entry_event(entry, event)
    trigger = _trigger(entry, entry_event)
    to_status = _destination(entry, entry_event, trigger)
    _require_later_proof(entry, entry_event, trigger)
    return _effect(entry, entry_event, to_status)


# --- Retirement lifecycle ----------------------------------------------------

RETIREMENT_TRANSITIONS: Mapping[tuple[RetirementState | None, DefectTrigger], RetirementState] = (
    MappingProxyType(
        {
            (None, DefectTrigger.DECLARE_RETIREMENT): RetirementState.PENDING,
            (RetirementState.PENDING, DefectTrigger.RULE_ACCEPTED_RETIREMENT): (
                RetirementState.ACCEPTED
            ),
            (RetirementState.PENDING, DefectTrigger.RULE_REJECTED): RetirementState.REJECTED,
            (RetirementState.REJECTED, DefectTrigger.DECLARE_RETIREMENT): RetirementState.PENDING,
        }
    )
)
"""A retirement's lifecycle (decision 13): only its own declaration and ruling change it.

A rejected retirement may be declared again with a better reason.
"""

_RetirementEvent = DeclareRetirement | Rule


def _retirement_refusal(
    record: Retirement | None, event: _AnyEvent, trigger: DefectTrigger | None, why: str
) -> IllegalDefectTransition:
    return _refusal(
        RETIREMENT_TRANSITIONS,
        ref=event.ref if record is None else record.ref,
        from_status=None if record is None else record.state,
        event=event.event,
        trigger=trigger,
        why=why,
    )


def _retirement_event(record: Retirement | None, event: _AnyEvent) -> _RetirementEvent:
    if isinstance(event, DeclareRetirement) or (
        isinstance(event, Rule) and event.ruling.subject is RulingSubject.RETIREMENT
    ):
        return event
    raise _retirement_refusal(
        record,
        event,
        None,
        'only a declare_retirement event or a retirement-subject rule applies to a retirement',
    )


def _retirement_trigger(event: _RetirementEvent) -> DefectTrigger:
    if isinstance(event, DeclareRetirement):
        return DefectTrigger.DECLARE_RETIREMENT
    if event.ruling.accepted:
        return DefectTrigger.RULE_ACCEPTED_RETIREMENT
    return DefectTrigger.RULE_REJECTED


def _retirement_effect(record: Retirement | None, event: _RetirementEvent) -> Retirement:
    """The record after a transition the table allowed, always CONSTRUCTED so validation re-runs."""
    if isinstance(event, DeclareRetirement):
        return event.retirement
    if record is None:
        raise _retirement_refusal(record, event, None, 'only declare_retirement creates a record')
    return Retirement(ref=record.ref, reason=record.reason, ruling=event.ruling)


def apply_retirement_event(
    record: Retirement | None, event: DeclareEvent | OutcomeEvent
) -> Retirement:
    """The retirement record after *event*, per :data:`RETIREMENT_TRANSITIONS`; pure.

    *record* is None when the removing task holds no retirement for the ref yet.

    Raises:
        IllegalDefectTransition: the event concerns another ref, is not a
            retirement event, or the table has no transition for it from the
            record's state.
    """
    if record is not None:
        _require_event_ref(record.ref, record.state, event, RETIREMENT_TRANSITIONS)
    retirement_event = _retirement_event(record, event)
    trigger = _retirement_trigger(retirement_event)
    if (None if record is None else record.state, trigger) not in RETIREMENT_TRANSITIONS:
        raise _retirement_refusal(
            record, retirement_event, trigger, 'decision 13 has no such retirement transition'
        )
    return _retirement_effect(record, retirement_event)


# --- Reviewer-issue kind (decision 16) ---------------------------------------


class IssueKindFlag(StrEnum):
    MISSING = 'missing'
    UNRECOGNISED = 'unrecognised'


@dataclass(frozen=True)
class IssueKindReading:
    """A reviewer issue's kind, flagged when it was read fail-closed rather than as written."""

    kind: IssueKind
    flag: IssueKindFlag | None = None

    @property
    def flagged(self) -> bool:
        return self.flag is not None


_ISSUE_KIND_BY_VALUE: MappingProxyType[str, IssueKind] = MappingProxyType(
    {kind.value: kind for kind in IssueKind}
)


def issue_kind_of(issue: Mapping[str, object]) -> IssueKindReading:
    """The kind of a reviewer issue; every reviewer-issue reader calls this, never ``issue['kind']``.

    A missing or unrecognised kind reads as ``defect``, the fail-closed value,
    with a flag the caller records in its own structured output. Pure: no logging.
    """
    raw = issue.get('kind')
    if raw is None:
        return IssueKindReading(IssueKind.DEFECT, IssueKindFlag.MISSING)
    kind = _ISSUE_KIND_BY_VALUE.get(raw) if isinstance(raw, str) else None
    if kind is None:
        return IssueKindReading(IssueKind.DEFECT, IssueKindFlag.UNRECOGNISED)
    return IssueKindReading(kind)


# --- Metadata registration ---------------------------------------------------

DEFECTS_METADATA_KEY = 'defects'
DEFECT_TEST_RETIREMENTS_METADATA_KEY = 'defect_test_retirements'

# An import side effect of THIS module, never of shared.task_metadata; why is
# shared/src/shared/deploy_state.py::DeployState's module docstring.
register_metadata_submodel(DEFECTS_METADATA_KEY, DefectDeclaration, cardinality='list')
register_metadata_submodel(DEFECT_TEST_RETIREMENTS_METADATA_KEY, Retirement, cardinality='list')
