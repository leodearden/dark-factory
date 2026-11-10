"""Turn a task's resolved escalation records into the rulings an agent is briefed with.

A ruling is an escalation filed on the task that someone resolved with text; it
binds every later agent on the task (task 4764).  A resumed agent learns its
resolutions from ``orchestrator/resume_resolution.py`` instead, which shares
:func:`ruling_resolution` so both paths agree on what counts as one.  Order is imposed
here because ``escalation/src/escalation/queue.py::EscalationQueue.get_by_task``
returns records in unsorted filesystem-glob order.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from escalation.models import cascade_parent_id

if TYPE_CHECKING:
    from escalation.models import Escalation

_UNKNOWN_TIME = datetime.min.replace(tzinfo=UTC)


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
    """The resolution text that makes *record* a ruling, else None."""
    return record.resolution if record.status == 'resolved' else None


def is_ruling(record: Escalation) -> bool:
    return ruling_resolution(record) is not None


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

