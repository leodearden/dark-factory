"""Normalise a reviewer artifact (verdict-tool envelope or flat payload) to a typed ReviewPayload."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar, TypeVar

from escalation.disposition import ReviewerDisposition
from shared.defects import MalformedDefectRef, RulingSubject, issue_kind_of, parse_ref

from orchestrator.review_issue_contract import IssueField, IssueSeverity, parse_severity


class Verdict(StrEnum):
    PASS = 'PASS'
    ISSUES_FOUND = 'ISSUES_FOUND'
    ERROR = 'ERROR'


class ReviewPayloadError(ValueError):
    pass


def _is_blocking(issue: Mapping[str, Any]) -> bool:
    return parse_severity(issue.get(IssueField.SEVERITY)) is IssueSeverity.BLOCKING


_Entry = TypeVar('_Entry')


def _parse_entries(
    raw: object,
    *,
    field: str,
    reviewer: object,
    parse_entry: Callable[[Mapping[str, Any], str], _Entry],
    key_of: Callable[[_Entry], Hashable],
    duplicate_claim: Callable[[_Entry], str],
) -> tuple[_Entry, ...]:
    """*raw* read as the review's *field* list: absent is empty, and no two entries share a key.

    *parse_entry* reads one mapping entry, given where it sits for its error
    messages; *duplicate_claim* says what two entries sharing a key both do.
    """
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ReviewPayloadError(
            f'review from {reviewer!r} has {field} {raw!r} '
            f'({type(raw).__name__}); expected a list'
        )
    parsed: list[_Entry] = []
    for index, entry in enumerate(raw):
        where = f'{field} entry {index} of the review from {reviewer!r}'
        if not isinstance(entry, Mapping):
            raise ReviewPayloadError(
                f'{where} is a {type(entry).__name__}, not a mapping: {entry!r}'
            )
        parsed.append(parse_entry(entry, where))
    first_index: dict[Hashable, int] = {}
    for index, entry in enumerate(parsed):
        key = key_of(entry)
        if key in first_index:
            raise ReviewPayloadError(
                f'{field} entries {first_index[key]} and {index} of the review from '
                f'{reviewer!r} both {duplicate_claim(entry)}'
            )
        first_index[key] = index
    return tuple(parsed)


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


def _note_disposition(entry: Mapping[str, Any], where: str) -> NoteDisposition:
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
    return _parse_entries(
        raw,
        field='note_dispositions',
        reviewer=reviewer,
        parse_entry=_note_disposition,
        key_of=lambda note: note.escalation_id,
        duplicate_claim=lambda note: f'disposition {note.escalation_id!r}',
    )


@dataclass(frozen=True)
class DefectRuling:
    ref: str
    subject: RulingSubject
    accepted: bool

    def to_record(self) -> dict[str, Any]:
        return {'ref': self.ref, 'subject': self.subject.value, 'accepted': self.accepted}


def defect_rulings_record(rulings: Iterable[DefectRuling]) -> dict[str, Any]:
    records = [ruling.to_record() for ruling in rulings]
    return {'defect_rulings': records} if records else {}


def _defect_ruling(entry: Mapping[str, Any], where: str) -> DefectRuling:
    try:
        ref = parse_ref(entry.get('ref')).ref
    except MalformedDefectRef as exc:
        raise ReviewPayloadError(f'{where} has a malformed ref: {exc}') from None
    raw_subject = entry.get('subject')
    try:
        subject = RulingSubject(raw_subject)
    except ValueError:
        raise ReviewPayloadError(
            f'{where} has subject {raw_subject!r}; '
            f'expected one of {[s.value for s in RulingSubject]}'
        ) from None
    accepted = entry.get('accepted')
    if type(accepted) is not bool:
        raise ReviewPayloadError(f'{where} has accepted {accepted!r}; expected true or false')
    return DefectRuling(ref, subject, accepted)


def parse_defect_rulings(
    raw: object, *, reviewer: object = None,
) -> tuple[DefectRuling, ...]:
    return _parse_entries(
        raw,
        field='defect_rulings',
        reviewer=reviewer,
        parse_entry=_defect_ruling,
        key_of=lambda ruling: (ruling.ref, ruling.subject),
        duplicate_claim=lambda ruling: f'rule on the {ruling.subject} of {ruling.ref!r}',
    )


@dataclass(frozen=True)
class ReviewPayload:
    verdict: Verdict
    issues: tuple[dict[str, Any], ...]
    reviewer: str | None
    summary: str = ''
    note_dispositions: tuple[NoteDisposition, ...] = ()
    defect_rulings: tuple[DefectRuling, ...] = ()

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
        self._require_members('issue', self.issues, dict)
        self._require_members('note disposition', self.note_dispositions, NoteDisposition)
        self._require_members('defect ruling', self.defect_rulings, DefectRuling)

    def _require_members(
        self, noun: str, members: tuple[object, ...], member_type: type,
    ) -> None:
        """An ERROR review carries no *members*; otherwise each is a *member_type*."""
        if self.verdict is Verdict.ERROR and members:
            raise ReviewPayloadError(
                f'ERROR review from {self.reviewer!r} carries {len(members)} '
                f'{noun}(s); an ERROR review carries none'
            )
        for index, member in enumerate(members):
            if not isinstance(member, member_type):
                raise ReviewPayloadError(
                    f'{noun} {index} of the review from {self.reviewer!r} is a '
                    f'{type(member).__name__}, not a {member_type.__name__}: {member!r}'
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
            **defect_rulings_record(self.defect_rulings),
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


def _with_kind_reading(issue: dict[str, Any]) -> dict[str, Any]:
    """A copy of *issue* recording ``kind_flag`` when its kind reads fail-closed as ``defect``.

    The written ``kind`` stays as received, so every later ``issue_kind_of`` reader
    flags the same issue the same way.
    """
    reading = issue_kind_of(issue)
    return {**issue, 'kind_flag': reading.flag.value} if reading.flag else dict(issue)


def _issues(payload: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    raw = payload.get('issues')
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ReviewPayloadError(
            f'review from {payload.get("reviewer")!r} has issues {raw!r} '
            f'({type(raw).__name__}); expected a list'
        )
    return tuple(_with_kind_reading(issue) if isinstance(issue, dict) else issue for issue in raw)


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
    defect_rulings = (
        () if verdict is Verdict.ERROR
        else parse_defect_rulings(payload.get('defect_rulings'), reviewer=reviewer)
    )
    summary = payload.get('summary')
    return ReviewPayload(
        verdict=verdict,
        issues=issues,
        reviewer=reviewer,
        summary='' if summary is None else summary,
        note_dispositions=note_dispositions,
        defect_rulings=defect_rulings,
    )
