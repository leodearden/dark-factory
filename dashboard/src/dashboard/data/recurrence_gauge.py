"""The R-D5/C-4 chain-state gauge of ``docs/prds/recurring-deterministic-tasks.md``.

Every recurrence chain gets exactly one state, read from its current link.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

from shared.task_metadata import Milestone, Recurrence
from shared.task_statuses import TaskStatus

OVERDUE_GRACE_CAP = timedelta(hours=24)
_OVERDUE_GRACE_CAP_SECS = int(OVERDUE_GRACE_CAP.total_seconds())


class ChainState(StrEnum):
    SCHEDULED = 'scheduled'
    OVERDUE = 'overdue'
    PAUSED = 'paused'
    BROKEN = 'broken'
    ENDED = 'ended'


_STATE_BY_STATUS: Mapping[str, ChainState] = {
    TaskStatus.BLOCKED: ChainState.PAUSED,
    TaskStatus.DONE: ChainState.BROKEN,
    TaskStatus.CANCELLED: ChainState.ENDED,
}


def overdue_grace(metadata: Mapping[str, Any]) -> timedelta:
    """How long past its fire time a live link may stay unfired before it is overdue."""
    try:
        recurrence = Recurrence.model_validate(metadata.get('recurrence'))
    except ValueError:
        return OVERDUE_GRACE_CAP
    return timedelta(seconds=min(recurrence.interval_secs, _OVERDUE_GRACE_CAP_SECS))


def _fire_time(metadata: Mapping[str, Any]) -> datetime | None:
    try:
        milestone = Milestone.model_validate(metadata.get('milestone'))
    except ValueError:
        return None
    if milestone.mode != 'dated' or milestone.at is None:
        return None
    at = datetime.fromisoformat(milestone.at)
    return at if at.tzinfo is not None else at.replace(tzinfo=UTC)


def chain_state(link: Mapping[str, Any], *, now: datetime) -> ChainState:
    """The one state of the chain whose current link is *link* (a decoded task row)."""
    state = _STATE_BY_STATUS.get(link['status'])
    if state is not None:
        return state
    metadata = link['metadata']
    fire_time = _fire_time(metadata)
    if fire_time is None or now - overdue_grace(metadata) > fire_time:
        return ChainState.OVERDUE
    return ChainState.SCHEDULED
