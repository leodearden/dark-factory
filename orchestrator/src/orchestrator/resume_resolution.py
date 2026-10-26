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
