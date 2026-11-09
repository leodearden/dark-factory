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
from typing import Any


class IssueSeverity(StrEnum):
    BLOCKING = 'blocking'
    SUGGESTION = 'suggestion'


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
    IssueField.DESCRIPTION: _FieldRule(required=True),
    IssueField.SUGGESTED_FIX: _FieldRule(),
}


@dataclass(frozen=True)
class IssueDeviation:
    """One way issue *index* departs from the contract, and what was recorded.

    ``recorded_as`` is None when the issue is unusable and so not recorded.
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


def _in_vocabulary(rule: _FieldRule, value: str) -> str:
    assert rule.vocabulary is not None
    folded = value.strip().casefold()
    for member in rule.vocabulary:
        if member.value == folded:
            return member.value
    return rule.off_vocabulary_as


def _check_field(
    index: int, field: IssueField, issue: Mapping[str, Any],
) -> tuple[str | None, IssueDeviation | None]:
    rule = _RULES[field]
    value = issue.get(field)
    if _is_missing(value):
        recorded = None if rule.required else rule.when_missing(issue)
        return recorded, IssueDeviation(index, field, Deviation.MISSING, value, recorded)
    if not isinstance(value, str):
        recorded = None if rule.required else str(value)
        return recorded, IssueDeviation(index, field, Deviation.NOT_A_STRING, value, recorded)
    if rule.vocabulary is None or value in rule.vocabulary:
        return value, None
    recorded = _in_vocabulary(rule, value)
    return recorded, IssueDeviation(index, field, Deviation.OFF_VOCABULARY, value, recorded)


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
