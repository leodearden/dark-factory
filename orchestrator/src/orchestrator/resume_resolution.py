"""Turn a task's escalation records into the attributed text a resumed agent reads.

Order is imposed here because
``escalation/src/escalation/queue.py::EscalationQueue.get_by_task`` returns
records in unsorted filesystem-glob order.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from escalation.models import Escalation

NO_CARRIED_RESOLUTIONS = 'No resolution text was recorded for any escalation on this task.'
ANSWERED_HEADING = 'Escalations answered by the resolution(s) below:'
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


def select_carried_resolutions(
    escalations: Iterable[Escalation],
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
        )
        for e in ordered
        if e.status == 'resolved' and e.resolution is not None
    )


def render_carried_resolutions(items: Sequence[CarriedResolution]) -> str:
    if not items:
        return NO_CARRIED_RESOLUTIONS
    return '\n\n'.join(_render_resolution_block(item) for item in items)


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
    if not carried:
        return NO_CARRIED_RESOLUTIONS
    lines = '\n'.join(f'- `{c.escalation_id}`: {c.summary}' for c in carried)
    return f'{ANSWERED_HEADING}\n{lines}'


def _render_still_pending(pending: Sequence[Escalation]) -> str:
    lines = '\n'.join(
        f'- `{e.id}` (severity={e.severity}, level={e.level}): {e.summary}' for e in pending
    )
    return f'{STILL_PENDING_HEADING}\n{lines}'
