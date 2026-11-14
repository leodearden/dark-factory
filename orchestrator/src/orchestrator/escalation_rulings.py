"""Turn a task's resolved escalation records into the rulings an agent is briefed with.

A ruling is an escalation filed on the task that someone resolved with text; a
record closed as it was filed (a note to file, an eval-lane containment, a
terminal-task auto-close) is not one, because no one ruled on it.  A ruling
binds every later agent on the task (task 4764).  A resumed agent learns its
resolutions from ``orchestrator/resume_resolution.py`` instead, which shares
:func:`ruling_resolution` so both paths agree on what counts as one.  Order is imposed
here because ``escalation/src/escalation/queue.py::EscalationQueue.get_by_task``
returns records in unsorted filesystem-glob order.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING

from escalation.models import cascade_parent_id, closed_at_filing

if TYPE_CHECKING:
    from escalation.models import Escalation

_UNKNOWN_TIME = datetime.min.replace(tzinfo=UTC)

RULINGS_TITLE = 'Escalation Rulings'

_DEEPEST_SECTION_LEVEL = 5
"""A section's entries sit one level below it, and markdown stops at six."""

RULINGS_LIMIT = 10
"""Max rulings one briefing lists, the most recent kept; older ones are named by id."""

_RULINGS_INTRO = 'Each ruling below resolved an escalation filed on this task, and binds this task.'


class RulingAudience(StrEnum):
    PLAN = 'plan'
    EXECUTE = 'execute'
    PLAN_AND_EXECUTE = 'plan_and_execute'
    REVIEW = 'review'


RULING_INSTRUCTIONS: Mapping[RulingAudience, str] = MappingProxyType({
    RulingAudience.PLAN: (
        'The plan must carry out every ruling above: add the files or steps a ruling '
        'requires, and never plan its opposite.'
    ),
    RulingAudience.EXECUTE: (
        'Work consistently with every ruling above. Never implement the opposite of a '
        'ruling, or describe its opposite as intentional. If honouring a ruling needs work '
        "outside the plan's files or steps, do not work around it: stop and call "
        "`escalate_blocker` naming the ruling's escalation id."
    ),
    RulingAudience.PLAN_AND_EXECUTE: (
        'Plan and work consistently with every ruling above: add the files or steps a '
        'ruling requires, never plan or implement its opposite, and never describe its '
        "opposite as intentional. If honouring a ruling needs work outside this task's "
        "files, stop and call `escalate_blocker` naming the ruling's escalation id."
    ),
    RulingAudience.REVIEW: (
        'Check the diff against every ruling above. A change that does the opposite of a '
        'ruling, or a comment calling its opposite intentional, is a blocking issue that '
        "names the ruling's escalation id."
    ),
})
"""What each briefed audience is told to do with the rulings it is shown."""


@dataclass(frozen=True, slots=True)
class Ruling:
    escalation_id: str
    level: int
    summary: str
    resolution: str
    resolved_at: str | None
    resolved_by: str | None
    resolution_action: str | None


def ruling_resolution(record: Escalation) -> str | None:
    """The resolution text that makes *record* a ruling, else None; a record closed at filing (escalation/src/escalation/models.py::closed_at_filing) is never one, because no one ruled on it."""
    if record.status != 'resolved' or closed_at_filing(record):
        return None
    return record.resolution


def select_rulings(
    records: Iterable[Escalation], *, since: datetime | None,
) -> tuple[Ruling, ...]:
    """Every ruling resolved strictly after *since* (all when None), oldest first.

    A ruling whose resolution time is missing or unparseable is kept, and a
    cascade member is dropped when the L2 that resolved it is selected too.
    """
    chosen = [
        Ruling(
            escalation_id=record.id,
            level=record.level,
            summary=record.summary,
            resolution=resolution,
            resolved_at=record.resolved_at,
            resolved_by=record.resolved_by,
            resolution_action=record.resolution_action,
        )
        for record in records
        if (resolution := ruling_resolution(record)) is not None
        and _resolved_after(record.resolved_at, since)
    ]
    chosen_ids = {ruling.escalation_id for ruling in chosen}
    kept = [ruling for ruling in chosen if cascade_parent_id(ruling.resolved_by) not in chosen_ids]
    return tuple(sorted(kept, key=_chronological))


def render_rulings_section(
    rulings: Sequence[Ruling], audience: RulingAudience, *, heading_level: int = 1,
) -> str:
    """The briefing block for *rulings*, oldest first; empty for none, else ending in a blank line.

    The block is headed at *heading_level* and each ruling one level below it,
    so it nests under the prompt section it sits in.
    """
    if not 1 <= heading_level <= _DEEPEST_SECTION_LEVEL:
        raise ValueError(
            f'heading_level must be 1..{_DEEPEST_SECTION_LEVEL}, got {heading_level}',
        )
    if not rulings:
        return ''
    marker = '#' * heading_level
    listed = rulings[-RULINGS_LIMIT:]
    omitted = rulings[:-RULINGS_LIMIT]
    entries = '\n\n'.join(_ruling_entry(ruling, marker + '#') for ruling in listed)
    overflow = (
        f'{len(omitted)} older ruling(s) are not listed: '
        + ', '.join(f'`{ruling.escalation_id}`' for ruling in omitted)
        + '.\n\n'
        if omitted else ''
    )
    return (
        f'{marker} {RULINGS_TITLE}\n\n{_RULINGS_INTRO}\n\n{entries}\n\n'
        f'{overflow}{RULING_INSTRUCTIONS[audience]}\n\n'
    )


def _ruling_entry(ruling: Ruling, marker: str) -> str:
    quoted = '\n'.join(f'> {line}' for line in ruling.resolution.splitlines()) or '> (none)'
    return (
        f'{marker} `{ruling.escalation_id}` (level {ruling.level}): {one_line(ruling.summary)}\n'
        f'- resolved by: {one_line(ruling.resolved_by) or "unknown"}\n'
        f'- resolved at: {one_line(ruling.resolved_at) or "unknown"}\n'
        f'- action: {one_line(ruling.resolution_action) or "unknown"}\n'
        f'\n{quoted}'
    )


def one_line(text: str | None) -> str:
    """*text* with its line breaks collapsed, so a field cannot start a heading of its own; null reads as empty."""
    return ' '.join((text or '').split())


def plan_confirmed_at(plan: Mapping[str, object]) -> datetime | None:
    """When an architect last confirmed *plan*, from its ``_finalized_at`` stamp.

    ``orchestrator/mcp/plan_tools.py::confirm_plan`` is the stamp's writer, on
    every confirm and re-confirm.  ``_created_at`` is rewritten every iteration
    and ``_revalidated_at`` is bumped by a revalidation skip that reads no new
    input, so neither dates the plan's authorship.
    """
    return _parse_utc(plan.get('_finalized_at'))


def _chronological(ruling: Ruling) -> tuple[datetime, str]:
    return (_parse_utc(ruling.resolved_at) or _UNKNOWN_TIME, ruling.escalation_id)


def _resolved_after(resolved_at: str | None, since: datetime | None) -> bool:
    if since is None:
        return True
    resolved = _parse_utc(resolved_at)
    return resolved is None or resolved > since


def _parse_utc(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)

