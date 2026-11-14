"""The reviewer issue contract, and the intake check of an issue against it.

An issue is one object in a reviewer verdict's ``issues`` list. The prompt that
states the contract, the verdict-tools intake that normalises against it and
the review gate that reads it all take the vocabulary from here. Stdlib only:
the per-invocation verdict-tools stdio server imports it on its startup path.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, TypeVar


class IssueSeverity(StrEnum):
    BLOCKING = 'blocking'
    SUGGESTION = 'suggestion'


_Member = TypeVar('_Member', bound=StrEnum)


def _parse_member(vocabulary: type[_Member], value: object) -> _Member | None:
    if not isinstance(value, str):
        return None
    folded = value.strip().casefold()
    return next((member for member in vocabulary if member.value == folded), None)


def parse_severity(value: object) -> IssueSeverity | None:
    """*value* read as a severity, ignoring case and surrounding whitespace.

    The one reading of severity: the intake records an issue's severity as
    this, and the review gate decides what blocks by it, so a ``'Blocking'``
    that bypassed intake blocks exactly as the intake would have recorded it.
    """
    return _parse_member(IssueSeverity, value)


class IssueField(StrEnum):
    SEVERITY = 'severity'
    LOCATION = 'location'
    CATEGORY = 'category'
    DESCRIPTION = 'description'
    SUGGESTED_FIX = 'suggested_fix'


class Deviation(StrEnum):
    NOT_AN_OBJECT = 'not_an_object'
    MISSING = 'missing'
    NOT_A_STRING = 'not_a_string'
    OFF_VOCABULARY = 'off_vocabulary'


#: The name an off-contract issue intake is recorded under wherever it is seen:
#: the verdict-tools fact journal's ``fact`` and the review gate's log ``event``.
REVIEW_ISSUE_OFF_CONTRACT_FACT = 'review_issue_off_contract'

#: Longest repr of a value a deviation description quotes, so the description
#: fits the fact journal's per-item cap with its field path intact.
DESCRIBED_VALUE_MAX_CHARS = 60


def _shortened_repr(value: object) -> str:
    text = repr(value)
    if len(text) <= DESCRIBED_VALUE_MAX_CHARS:
        return text
    return text[:DESCRIBED_VALUE_MAX_CHARS - 1] + '…'


def _vocabulary_text(vocabulary: type[StrEnum]) -> str:
    return ' | '.join(repr(member.value) for member in vocabulary)


@dataclass(frozen=True)
class _FieldRule:
    required: bool = False
    vocabulary: type[StrEnum] | None = None
    off_vocabulary_as: str = ''
    when_missing: Callable[[Mapping[str, Any]], str] = lambda _issue: ''


#: Keys reviewers have sent a finding's text under instead of ``description``,
#: body before heading: every such issue in the 2026-11 verdict corpus that
#: carries a ``title`` also carries a ``detail`` or a ``summary``.
_DESCRIPTION_ALTERNATIVES = ('detail', 'summary', 'title')


def _description_from_alternatives(issue: Mapping[str, Any]) -> str:
    return next(
        (
            value for key in _DESCRIPTION_ALTERNATIVES
            if isinstance(value := issue.get(key), str) and value.strip()
        ),
        '',
    )


def _location_from_file_and_line(issue: Mapping[str, Any]) -> str:
    file, line = issue.get('file'), issue.get('line')
    if not isinstance(file, str) or not file.strip():
        return ''
    if isinstance(line, int) and not isinstance(line, bool):
        return f'{file.strip()}:{line}'
    if isinstance(line, str) and line.strip().isdigit():
        return f'{file.strip()}:{line.strip()}'
    return file.strip()


_RULES: dict[IssueField, _FieldRule] = {
    IssueField.SEVERITY: _FieldRule(
        required=True,
        vocabulary=IssueSeverity,
        off_vocabulary_as=IssueSeverity.SUGGESTION.value,
    ),
    IssueField.LOCATION: _FieldRule(when_missing=_location_from_file_and_line),
    IssueField.CATEGORY: _FieldRule(),
    IssueField.DESCRIPTION: _FieldRule(
        required=True, when_missing=_description_from_alternatives,
    ),
    IssueField.SUGGESTED_FIX: _FieldRule(),
}


@dataclass(frozen=True)
class IssueDeviation:
    """One way issue *index* departs from the contract, and what was recorded.

    ``recorded_as`` is None when the issue is unusable and so not recorded.
    A deviation is a field recorded differently from how it arrived, so intake
    output passes check_issues clean, and a deviation found at the review gate
    means the artifact bypassed intake.
    """

    index: int
    field: IssueField | None
    deviation: Deviation
    received: object
    recorded_as: str | None

    @property
    def unusable(self) -> bool:
        return self.recorded_as is None

    def describe(self) -> str:
        """One line led by the field path, short enough for the fact journal."""
        path = f'issues[{self.index}]' if self.field is None else (
            f'issues[{self.index}].{self.field}'
        )
        vocabulary = None if self.field is None else _RULES[self.field].vocabulary
        contract = (
            f' (contract: {_vocabulary_text(vocabulary)})'
            if self.deviation is Deviation.OFF_VOCABULARY and vocabulary else ''
        )
        outcome = (
            'the issue is unusable' if self.recorded_as is None
            else f'recorded as {_shortened_repr(self.recorded_as)}'
        )
        return (
            f'{path}: {_shortened_repr(self.received)} is '
            f'{self.deviation}{contract}; {outcome}'
        )


@dataclass(frozen=True)
class IssueIntake:
    """The usable issues, normalised to the contract, and every deviation found."""

    issues: tuple[dict[str, Any], ...]
    deviations: tuple[IssueDeviation, ...]

    @property
    def unusable(self) -> tuple[IssueDeviation, ...]:
        return tuple(d for d in self.deviations if d.unusable)

    @property
    def normalised(self) -> tuple[IssueDeviation, ...]:
        return tuple(d for d in self.deviations if not d.unusable)


def _is_missing(value: object) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _recorded_form(
    rule: _FieldRule, issue: Mapping[str, Any], value: object,
) -> tuple[str | None, Deviation | None]:
    if _is_missing(value):
        derived = rule.when_missing(issue)
        return (None if rule.required and not derived else derived), Deviation.MISSING
    if not isinstance(value, str):
        return (None if rule.required else str(value)), Deviation.NOT_A_STRING
    if rule.vocabulary is None:
        return value, None
    member = _parse_member(rule.vocabulary, value)
    recorded = rule.off_vocabulary_as if member is None else member.value
    return recorded, (None if recorded == value else Deviation.OFF_VOCABULARY)


def _check_field(
    index: int, field: IssueField, issue: Mapping[str, Any],
) -> tuple[str | None, IssueDeviation | None]:
    value = issue.get(field)
    recorded, deviation = _recorded_form(_RULES[field], issue, value)
    if deviation is None or (recorded is not None and recorded == value):
        return recorded, None
    return recorded, IssueDeviation(index, field, deviation, value, recorded)


def _check_issue(
    index: int, issue: object,
) -> tuple[dict[str, Any] | None, list[IssueDeviation]]:
    if not isinstance(issue, Mapping):
        return None, [IssueDeviation(index, None, Deviation.NOT_AN_OBJECT, issue, None)]
    recorded: dict[str, Any] = {}
    deviations: list[IssueDeviation] = []
    for field in IssueField:
        value, deviation = _check_field(index, field, issue)
        if deviation is not None:
            deviations.append(deviation)
        recorded[field] = value
    if any(d.unusable for d in deviations):
        return None, deviations
    return {**issue, **{str(field): value for field, value in recorded.items()}}, deviations


def check_issues(issues: Sequence[object]) -> IssueIntake:
    """Check each of *issues* against the contract, without mutating them."""
    usable: list[dict[str, Any]] = []
    deviations: list[IssueDeviation] = []
    for index, issue in enumerate(issues):
        normalised, found = _check_issue(index, issue)
        deviations.extend(found)
        if normalised is not None:
            usable.append(normalised)
    return IssueIntake(issues=tuple(usable), deviations=tuple(deviations))
