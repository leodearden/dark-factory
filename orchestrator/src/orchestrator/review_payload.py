"""Normalise a reviewer artifact (verdict-tool envelope or flat payload) to a typed ReviewPayload."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar

from escalation.disposition import ReviewerDisposition

from orchestrator.review_issue_contract import IssueField, IssueSeverity, parse_severity


class Verdict(StrEnum):
    PASS = 'PASS'
    ISSUES_FOUND = 'ISSUES_FOUND'
    ERROR = 'ERROR'


class ReviewPayloadError(ValueError):
    pass


def _is_blocking(issue: Mapping[str, Any]) -> bool:
    return parse_severity(issue.get(IssueField.SEVERITY)) is IssueSeverity.BLOCKING


@dataclass(frozen=True)
class NoteDisposition:
    escalation_id: str
    disposition: ReviewerDisposition
    rationale: str

    def to_record(self) -> dict[str, str]:
        return {
            'escalation_id': self.escalation_id,
            'disposition': self.disposition.value,
            'rationale': self.rationale,
        }


def note_dispositions_record(dispositions: Iterable[NoteDisposition]) -> dict[str, Any]:
    records = [disposition.to_record() for disposition in dispositions]
    return {'note_dispositions': records} if records else {}


def _note_disposition(entry: object, index: int, reviewer: object) -> NoteDisposition:
    where = f'note_dispositions entry {index} of the review from {reviewer!r}'
    if not isinstance(entry, Mapping):
        raise ReviewPayloadError(f'{where} is a {type(entry).__name__}, not a mapping: {entry!r}')
    escalation_id = entry.get('escalation_id')
    if not isinstance(escalation_id, str) or not escalation_id.strip():
        raise ReviewPayloadError(
            f'{where} has escalation_id {escalation_id!r}; expected a non-blank str'
        )
    raw_disposition = entry.get('disposition')
    try:
        disposition = ReviewerDisposition(raw_disposition)
    except ValueError:
        raise ReviewPayloadError(
            f'{where} has disposition {raw_disposition!r}; '
            f'expected one of {[d.value for d in ReviewerDisposition]}'
        ) from None
    rationale = entry.get('rationale')
    if not isinstance(rationale, str):
        raise ReviewPayloadError(f'{where} has rationale {rationale!r}; expected a str')
    return NoteDisposition(escalation_id, disposition, rationale)


def parse_note_dispositions(
    raw: object, *, reviewer: object = None,
) -> tuple[NoteDisposition, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ReviewPayloadError(
            f'review from {reviewer!r} has note_dispositions {raw!r} '
            f'({type(raw).__name__}); expected a list'
        )
    parsed = tuple(_note_disposition(entry, index, reviewer) for index, entry in enumerate(raw))
    first_index: dict[str, int] = {}
    for index, note in enumerate(parsed):
        if note.escalation_id in first_index:
            raise ReviewPayloadError(
                f'note_dispositions entries {first_index[note.escalation_id]} and {index} '
                f'of the review from {reviewer!r} both disposition {note.escalation_id!r}'
            )
        first_index[note.escalation_id] = index
    return parsed


@dataclass(frozen=True)
class ReviewPayload:
    verdict: Verdict
    issues: tuple[dict[str, Any], ...]
    reviewer: str | None
    summary: str = ''
    note_dispositions: tuple[NoteDisposition, ...] = ()

    __hash__: ClassVar[None] = None

    def __post_init__(self) -> None:
        if not isinstance(self.reviewer, str | None):
            raise ReviewPayloadError(
                f'review names its reviewer with a {type(self.reviewer).__name__}, '
                f'not a str or None: {self.reviewer!r}'
            )
        if not isinstance(self.summary, str):
            raise ReviewPayloadError(
                f'summary of the review from {self.reviewer!r} is a '
                f'{type(self.summary).__name__}, not a str: {self.summary!r}'
            )
        if self.verdict is Verdict.ERROR and self.issues:
            raise ReviewPayloadError(
                f'ERROR review from {self.reviewer!r} carries {len(self.issues)} '
                'issue(s); an ERROR review carries none'
            )
        for index, issue in enumerate(self.issues):
            if not isinstance(issue, dict):
                raise ReviewPayloadError(
                    f'issue {index} of the review from {self.reviewer!r} is a '
                    f'{type(issue).__name__}, not a dict: {issue!r}'
                )
        if self.verdict is Verdict.ERROR and self.note_dispositions:
            raise ReviewPayloadError(
                f'ERROR review from {self.reviewer!r} carries '
                f'{len(self.note_dispositions)} note disposition(s); an ERROR review carries none'
            )
        for index, note in enumerate(self.note_dispositions):
            if not isinstance(note, NoteDisposition):
                raise ReviewPayloadError(
                    f'note disposition {index} of the review from {self.reviewer!r} is a '
                    f'{type(note).__name__}, not a NoteDisposition: {note!r}'
                )

    @property
    def blocking_issues(self) -> tuple[dict[str, Any], ...]:
        return tuple(issue for issue in self.issues if _is_blocking(issue))

    @property
    def suggestions(self) -> tuple[dict[str, Any], ...]:
        return tuple(issue for issue in self.issues if not _is_blocking(issue))

    def to_record(self) -> dict[str, Any]:
        return {
            'reviewer': self.reviewer,
            'verdict': self.verdict.value,
            'issues': list(self.issues),
            'summary': self.summary,
            **note_dispositions_record(self.note_dispositions),
        }


def _inner_payload(artifact: Mapping[str, Any]) -> Mapping[str, Any]:
    if 'schema_version' not in artifact or 'role' not in artifact:
        return artifact
    payload = artifact.get('verdict')
    if not isinstance(payload, Mapping):
        raise ReviewPayloadError(
            f'verdict envelope for role {artifact["role"]!r} wraps a '
            f'{type(payload).__name__} verdict, not a payload dict: {payload!r}'
        )
    return payload


def _verdict(payload: Mapping[str, Any]) -> Verdict:
    value = payload.get('verdict')
    try:
        return Verdict(value)
    except ValueError:
        raise ReviewPayloadError(
            f'review from {payload.get("reviewer")!r} has verdict {value!r}; '
            f'expected one of {[v.value for v in Verdict]}'
        ) from None


def _issues(payload: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    raw = payload.get('issues')
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ReviewPayloadError(
            f'review from {payload.get("reviewer")!r} has issues {raw!r} '
            f'({type(raw).__name__}); expected a list'
        )
    return tuple(dict(issue) if isinstance(issue, dict) else issue for issue in raw)


def normalize_review(artifact: object) -> ReviewPayload:
    if not isinstance(artifact, Mapping):
        raise ReviewPayloadError(
            f'review artifact is a {type(artifact).__name__}, not a mapping: {artifact!r}'
        )
    payload = _inner_payload(artifact)
    verdict = _verdict(payload)
    issues = () if verdict is Verdict.ERROR else _issues(payload)
    reviewer = payload.get('reviewer')
    note_dispositions = (
        () if verdict is Verdict.ERROR
        else parse_note_dispositions(payload.get('note_dispositions'), reviewer=reviewer)
    )
    summary = payload.get('summary')
    return ReviewPayload(
        verdict=verdict,
        issues=issues,
        reviewer=reviewer,
        summary='' if summary is None else summary,
        note_dispositions=note_dispositions,
    )
