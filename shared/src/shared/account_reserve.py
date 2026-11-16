"""The reserve rule: which work may use the reserve account, and when.

One account of the shared pool can be held back for urgent work until shortly before its weekly
reset. This module is the rule of plans/allowance-reserve-by-account-prd.md §6.1-6.2 and the event
names of §6.6, as pure functions of a config, a work class and a clock: nothing is stored here,
and nothing here logs or does I/O.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias

__all__ = [
    'ReserveEvent',
    'WorkClass',
    'WorkClassSource',
    'WorkKind',
    'resolve_work_class',
]


class WorkKind(StrEnum):
    TASK = 'task'
    ESSENTIAL = 'essential'
    BACKGROUND = 'background'


@dataclass(frozen=True)
class WorkClass:
    kind: WorkKind
    project_id: str | None = None
    tier: str | None = None
    task_id: str | None = None
    role: str | None = None


WorkClassSource: TypeAlias = WorkClass | Callable[[], WorkClass]


def resolve_work_class(source: WorkClassSource | None) -> WorkClass:
    """The caller's class now: background when it declared none, re-read when it is a function."""
    if source is None:
        return WorkClass(kind=WorkKind.BACKGROUND)
    if isinstance(source, WorkClass):
        return source
    result = source()
    if not isinstance(result, WorkClass):
        raise TypeError(f'work class source returned {result!r}, not a WorkClass')
    return result


class ReserveEvent(StrEnum):
    CONFIG = 'reserve_config'
    WAIT_STARTED = 'reserve_wait_started'
    WAIT_ENDED = 'reserve_wait_ended'
    SERVED = 'reserve_served'
    SCHEDULE_DRIFT = 'reserve_schedule_drift'
    ACCOUNT_UNAVAILABLE = 'reserve_account_unavailable'
