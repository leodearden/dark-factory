"""Turn a task's escalation records into the attributed text a resumed agent reads.

A round is one ESCALATED wait.  A resolution answers this round when its record
was not yet resolved as the wait began; every other resolution comes from an
earlier round and may already have been applied.  Order is imposed here because
``escalation/src/escalation/queue.py::EscalationQueue.get_by_task`` returns
records in unsorted filesystem-glob order.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence, Set
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from escalation.models import Escalation

NO_CARRIED_RESOLUTIONS = 'No resolution text was recorded for any escalation on this task.'
NO_RESOLUTION_THIS_ROUND = "No resolution text was recorded for this round's escalation(s)."
THIS_ROUND_HEADING = "Answering this round's escalation(s):"
EARLIER_ROUNDS_HEADING = 'From earlier rounds, each possibly already applied:'
ANSWERED_HEADING = "This round's escalation(s), answered by the resolution(s) below:"
ANSWERED_EARLIER_HEADING = 'Answered in earlier rounds:'
STILL_PENDING_HEADING = 'Still pending, with no resolution yet:'
MISMATCH_INSTRUCTION = (
    'The resolutions below answer only the escalations listed as answered. Do not act on '
    'resolution text that does not answer your current situation.'
)


@dataclass(frozen=True, slots=True)
class CarriedResolution:
    escalation_id: str
    summary: str
    resolution: str
    resolved_at: str | None
    resolved_by: str | None
    resolution_action: str | None
    answered_this_round: bool


def select_carried_resolutions(
    escalations: Iterable[Escalation], *, resolved_before_round: Set[str],
) -> tuple[CarriedResolution, ...]:
    ordered = sorted(escalations, key=lambda e: (e.resolved_at or e.timestamp or '', e.id))
    return tuple(
        CarriedResolution(
            escalation_id=e.id,
            summary=e.summary,
            resolution=e.resolution,
            resolved_at=e.resolved_at,
            resolved_by=e.resolved_by,
            resolution_action=e.resolution_action,
            answered_this_round=e.id not in resolved_before_round,
        )
        for e in ordered
        if e.status == 'resolved' and e.resolution is not None
    )


def render_carried_resolutions(items: Sequence[CarriedResolution]) -> str:
    return _render_by_round(
        items, _render_resolution_block, separator='\n\n',
        this_round_heading=THIS_ROUND_HEADING, earlier_heading=EARLIER_ROUNDS_HEADING,
    )


def _render_resolution_block(item: CarriedResolution) -> str:
    provenance = ' · '.join(
        f'{label}: {value or "unknown"}'
        for label, value in (
            ('resolved_at', item.resolved_at),
            ('resolved_by', item.resolved_by),
            ('action', item.resolution_action),
        )
    )
    return f'### `{item.escalation_id}`: {item.summary}\n{provenance}\n\n{item.resolution}'


def render_issue_context(
    pending: Sequence[Escalation], carried: Sequence[CarriedResolution],
) -> str:
    sections = [_render_answered(carried)]
    if pending:
        sections += [_render_still_pending(pending), MISMATCH_INSTRUCTION]
    return '\n\n'.join(sections)


def _render_answered(carried: Sequence[CarriedResolution]) -> str:
    return _render_by_round(
        carried, _render_answered_line, separator='\n',
        this_round_heading=ANSWERED_HEADING, earlier_heading=ANSWERED_EARLIER_HEADING,
    )


def _render_answered_line(item: CarriedResolution) -> str:
    return f'- `{item.escalation_id}`: {item.summary}'


def _render_by_round(
    items: Sequence[CarriedResolution],
    render_item: Callable[[CarriedResolution], str],
    *,
    separator: str,
    this_round_heading: str,
    earlier_heading: str,
) -> str:
    if not items:
        return NO_CARRIED_RESOLUTIONS

    def section(heading: str, members: list[CarriedResolution]) -> str:
        return heading + separator + separator.join(render_item(m) for m in members)

    this_round = [item for item in items if item.answered_this_round]
    earlier = [item for item in items if not item.answered_this_round]
    sections = [section(this_round_heading, this_round) if this_round else NO_RESOLUTION_THIS_ROUND]
    if earlier:
        sections.append(section(earlier_heading, earlier))
    return '\n\n'.join(sections)


def _render_still_pending(pending: Sequence[Escalation]) -> str:
    lines = '\n'.join(
        f'- `{e.id}` (severity={e.severity}, level={e.level}): {e.summary}' for e in pending
    )
    return f'{STILL_PENDING_HEADING}\n{lines}'
