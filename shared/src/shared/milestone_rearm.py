"""A predicate re-arm advances a link's timer and keeps its scheduled fire time.

plans/completion-driven-triggers-prd.md C1/D1: a declared not-yet re-arm
advances ``milestone.at``, the scheduler's one timer, and keeps the link's
scheduled fire time in ``milestone.scheduled_fire_at``. The recurrence mint's
run label and the chain-state gauge read that fire time.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

__all__ = ['rearmed_milestone', 'scheduled_fire_at']

_SCHEDULED_FIRE_AT_KEY = 'scheduled_fire_at'


def scheduled_fire_at(milestone: object) -> str | None:
    """The link's scheduled fire time: ``scheduled_fire_at`` once re-armed, else ``at``; None when unreadable."""
    if not isinstance(milestone, Mapping):
        return None
    fire_at = milestone.get(_SCHEDULED_FIRE_AT_KEY, milestone.get('at'))
    return fire_at if isinstance(fire_at, str) else None


def rearmed_milestone(milestone: Mapping[str, Any], next_check_at: str) -> dict[str, Any]:
    """A copy of ``milestone`` re-armed to ``next_check_at``, keeping its scheduled fire time.

    Raises ``ValueError`` when ``milestone`` has no ISO-8601 scheduled fire
    time, so a re-arm never persists an unreadable one.
    """
    fire_at = scheduled_fire_at(milestone)
    if fire_at is None or not _is_iso_datetime(fire_at):
        raise ValueError(
            'A re-armed milestone must keep an ISO-8601 scheduled fire time '
            f'({_SCHEDULED_FIRE_AT_KEY}, else at); got milestone={milestone!r}'
        )
    return {**milestone, 'at': next_check_at, _SCHEDULED_FIRE_AT_KEY: fire_at}


def _is_iso_datetime(value: str) -> bool:
    try:
        datetime.fromisoformat(value)
    except ValueError:
        return False
    return True
