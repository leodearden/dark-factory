"""The defect-record events one review round produces, and their delivery through declare_defect / record_defect_outcome.

``plans/defect-tests-prd.md`` decisions 7, 15 and 16, and the hook-point row
"In-task declarations (δ2)".
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pydantic import ValidationError
from shared.defects import (
    DEFECT_TEST_RETIREMENTS_METADATA_KEY,
    DEFECTS_METADATA_KEY,
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
from shared.mcp_envelope import parse_tool_result, write_failure_reason

from orchestrator.review_issue_contract import IssueField
from orchestrator.review_payload import DefectRuling

logger = logging.getLogger(__name__)

DECLARE_DEFECT_TOOL = 'declare_defect'
RECORD_DEFECT_OUTCOME_TOOL = 'record_defect_outcome'

#: The ``event`` a refused or undeliverable defect-record write is logged under.
DEFECT_WRITE_FAILED_FACT = 'review_defect_write_failed'

_UNNAMED_REVIEWER = 'reviewer'

ReviewDefectEvent = Declare | Sighting | Rule
DefectToolDispatch = Callable[[str, dict], Awaitable[dict]]


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


def _is_defect(issue: Mapping[str, object]) -> bool:
    return issue_kind_of(issue).kind is IssueKind.DEFECT


def reports_a_defect(blocking_issues: Sequence[Mapping[str, object]]) -> bool:
    """Whether any of a round's blocking issues reads as ``kind=defect`` (fail-closed)."""
    return any(_is_defect(issue) for issue in blocking_issues)


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
        if not _is_defect(issue):
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


_TOOL_BY_EVENT: Mapping[type[ReviewDefectEvent], str] = {
    Declare: DECLARE_DEFECT_TOOL,
    Sighting: DECLARE_DEFECT_TOOL,
    Rule: RECORD_DEFECT_OUTCOME_TOOL,
}


@dataclass(frozen=True)
class DefectWrite:
    """One event's delivery: the tool it went to, its ref, and why it did not land (None if it did)."""

    tool: str
    ref: str
    failure: str | None


def _arguments(project_root: str, task_id: str, event: ReviewDefectEvent) -> dict[str, Any]:
    arguments: dict[str, Any] = {'project_root': project_root, 'task_id': task_id}
    if isinstance(event, Rule):
        arguments['ref'] = event.ref
    arguments['event'] = event.model_dump(mode='json')
    return arguments


def _failure_of(body: object) -> str | None:
    reply, envelope_error = parse_tool_result(body, None, dict)
    if envelope_error is not None:
        return f'reply was not a tool result: {envelope_error}'
    return write_failure_reason(reply)


async def _send_one(
    dispatch: DefectToolDispatch, project_root: str, task_id: str, event: ReviewDefectEvent,
) -> DefectWrite:
    tool = _TOOL_BY_EVENT[type(event)]
    try:
        body = await dispatch(tool, _arguments(project_root, task_id, event))
    except Exception as exc:
        failure: str | None = f'{type(exc).__name__}: {exc}'
    else:
        failure = _failure_of(body)
    return DefectWrite(tool=tool, ref=event.ref, failure=failure)


async def send_defect_events(
    dispatch: DefectToolDispatch,
    *,
    project_root: str,
    task_id: str,
    events: Sequence[ReviewDefectEvent],
) -> tuple[DefectWrite, ...]:
    """Send each event through its defect-record tool, in order; never raises.

    A write that fails is logged once and does not stop the events after it.
    """
    writes: list[DefectWrite] = []
    for event in events:
        write = await _send_one(dispatch, project_root, task_id, event)
        if write.failure is not None:
            logger.warning(
                'Task %s: %s for %s did not land: %s',
                task_id, write.tool, write.ref, write.failure,
                extra={
                    'event': DEFECT_WRITE_FAILED_FACT,
                    'task_id': task_id,
                    'tool': write.tool,
                    'ref': write.ref,
                    'failure': write.failure,
                },
            )
        writes.append(write)
    return tuple(writes)


_LIVE_METADATA_KEYS = (DEFECTS_METADATA_KEY, DEFECT_TEST_RETIREMENTS_METADATA_KEY)


def with_live_defect_records(task: Mapping[str, Any], fresh: object) -> dict[str, Any]:
    """*task* with its two defect-record keys as a fresh read of the task holds them now.

    Only ``metadata.defects`` and ``metadata.defect_test_retirements`` come from
    *fresh*; a key it lacks is dropped. Every other field stays as *task* has
    it. An unreadable *fresh* (not a dict, or metadata not a dict) leaves
    *task* as it is. *task* is never mutated.
    """
    fresh_metadata = fresh.get('metadata') if isinstance(fresh, Mapping) else None
    if not isinstance(fresh_metadata, Mapping):
        return dict(task)
    dispatch_metadata = task.get('metadata')
    kept = dispatch_metadata.items() if isinstance(dispatch_metadata, Mapping) else ()
    live_metadata = {key: value for key, value in kept if key not in _LIVE_METADATA_KEYS}
    live_metadata.update(
        (key, fresh_metadata[key]) for key in _LIVE_METADATA_KEYS if key in fresh_metadata
    )
    return {**task, 'metadata': live_metadata}


def declared_refs(metadata: Mapping[str, Any]) -> frozenset[str]:
    """The refs of the readable ``metadata.defects`` entries; malformed entries are skipped."""
    entries = metadata.get(DEFECTS_METADATA_KEY)
    if not isinstance(entries, list):
        return frozenset()
    refs: set[str] = set()
    for entry in entries:
        try:
            refs.add(DefectDeclaration.model_validate(entry).ref)
        except ValidationError:
            continue
    return frozenset(refs)
