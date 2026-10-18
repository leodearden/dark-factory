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
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from types import MappingProxyType

from shared.finding_key import KEY_PATTERN


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
# Every pattern is applied with `fullmatch` and uses explicit character
# classes: Python's `$` also matches before a trailing newline, the trap
# shared/src/shared/governed_exceptions.py::_LIST_ID_RE documents.

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
