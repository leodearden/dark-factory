"""The defect-record events one review round produces, and their delivery through declare_defect / record_defect_outcome.

``plans/defect-tests-prd.md`` decisions 7, 15 and 16, and the hook-point row
"In-task declarations (δ2)".
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from datetime import datetime

from shared.defects import (
    Declare,
    DefectDeclaration,
    DefectOrigin,
    DefectStatus,
    IssueKind,
    Rule,
    RuledBy,
    Ruling,
    RulingSubject,
    Sighting,
    issue_kind_of,
    mint_review_ref,
)

from orchestrator.review_issue_contract import IssueField
from orchestrator.review_payload import DefectRuling

_UNNAMED_REVIEWER = 'reviewer'


def _text(issue: Mapping[str, object], field: str) -> str:
    value = issue.get(field)
    return value if isinstance(value, str) else ''


def _reviewer_of(issue: Mapping[str, object]) -> str:
    return _text(issue, 'reviewer').strip() or _UNNAMED_REVIEWER


def _summary(location: str, description: str, reviewer: str) -> str:
    return (
        description.strip()
        or location.strip()
        or f'a blocking defect reported by {reviewer} with no description or location'
    )


def review_defect_declarations(
    task_id: int,
    blocking_issues: Sequence[Mapping[str, object]],
    *,
    known_refs: Collection[str],
    declared_at: datetime,
) -> tuple[Declare | Sighting, ...]:
    """One event per blocking ``kind=defect`` issue: a Declare for a new ref, else a Sighting.

    A ref is known when it is in *known_refs* or an earlier issue of this call
    declared it. A missing or unrecognised kind reads as a defect.
    """
    seen = set(known_refs)
    events: list[Declare | Sighting] = []
    for issue in blocking_issues:
        if issue_kind_of(issue).kind is not IssueKind.DEFECT:
            continue
        location = _text(issue, IssueField.LOCATION)
        description = _text(issue, IssueField.DESCRIPTION)
        ref = mint_review_ref(task_id, location, description)
        if ref in seen:
            events.append(Sighting(ref=ref))
            continue
        seen.add(ref)
        reviewer = _reviewer_of(issue)
        events.append(Declare(declaration=DefectDeclaration(
            ref=ref,
            origin=DefectOrigin.REVIEW,
            summary=_summary(location, description, reviewer),
            declared_at=declared_at,
            declared_by=reviewer,
            status=DefectStatus.OPEN,
        )))
    return tuple(events)


def review_ruling_events(
    rulings: Sequence[DefectRuling], *, verdict_cycle: int,
) -> tuple[Rule, ...]:
    """One reviewer Rule per (ref, subject), accepted only if every ruling on it accepted.

    A panel's rulings are folded first so that a split panel rejects rather
    than letting its first accept win. Rules come in order of first appearance.
    """
    accepted_by_subject: dict[tuple[str, RulingSubject], bool] = {}
    for ruling in rulings:
        key = (ruling.ref, ruling.subject)
        accepted_by_subject[key] = accepted_by_subject.get(key, True) and ruling.accepted
    return tuple(
        Rule(ref=ref, ruling=Ruling(
            subject=subject,
            accepted=accepted,
            ruled_by=RuledBy.REVIEWER,
            verdict_cycle=verdict_cycle,
        ))
        for (ref, subject), accepted in accepted_by_subject.items()
    )
