"""The reserve rule: which work may use the reserve account, and when.

One account of the shared pool can be held back for urgent work until shortly before its weekly
reset. This module is the rule of plans/allowance-reserve-by-account-prd.md §6.1-6.2 and the event
names of §6.6, as pure functions of a config, a work class and a clock: nothing is stored here,
and nothing here logs or does I/O.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from enum import StrEnum
from typing import TypeAlias, assert_never

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

__all__ = [
    'DEFAULT_ADMIT_TIERS',
    'DRIFT_HORIZON',
    'DRIFT_TOLERANCE',
    'TIER_WHEN_UNKNOWN',
    'AdmitTiers',
    'ReserveConfig',
    'ReserveEvent',
    'ScheduleDrift',
    'WeeklyReset',
    'WorkClass',
    'WorkClassSource',
    'WorkKind',
    'admits',
    'resolve_work_class',
    'schedule_drift',
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


def _as_utc(instant: datetime) -> datetime:
    if instant.tzinfo is None:
        raise ValueError(f'expected a timezone-aware datetime, got {instant!r}')
    return instant.astimezone(UTC)


_WEEKDAY_NAMES = ('mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun')
_WEEKLY_RESET_TEXT = re.compile(r'([a-z]{3})\s+(\d{1,2}):(\d{2})')


@dataclass(frozen=True)
class WeeklyReset:
    """A weekly instant in UTC; ``weekday`` counts as ``datetime.weekday()`` does (0 is Monday)."""

    weekday: int
    time_utc: time

    def __post_init__(self) -> None:
        if self.weekday not in range(len(_WEEKDAY_NAMES)):
            raise ValueError(f'weekday must be 0 (Monday) to 6 (Sunday), got {self.weekday!r}')
        if self.time_utc.tzinfo is not None:
            raise ValueError(f'time_utc is UTC and carries no tzinfo, got {self.time_utc!r}')

    @classmethod
    def parse(cls, text: str) -> WeeklyReset:
        refusal = (
            f'weekly_reset_utc must be a weekday and HH:MM in UTC, like "mon 21:00"; got {text!r}'
        )
        match = _WEEKLY_RESET_TEXT.fullmatch(text.strip().lower())
        if match is None:
            raise ValueError(refusal)
        day, hour, minute = match.groups()
        try:
            return cls(_WEEKDAY_NAMES.index(day), time(int(hour), int(minute)))
        except ValueError:
            raise ValueError(refusal) from None

    def __str__(self) -> str:
        return f'{_WEEKDAY_NAMES[self.weekday]} {self.time_utc:%H:%M}'

    def next_after(self, instant: datetime) -> datetime:
        """The first reset strictly after ``instant``."""
        now = _as_utc(instant)
        days_ahead = timedelta(days=(self.weekday - now.weekday()) % 7)
        candidate = datetime.combine(now.date() + days_ahead, self.time_utc, tzinfo=UTC)
        return candidate if candidate > now else candidate + timedelta(days=7)


DEFAULT_ADMIT_TIERS: tuple[str, ...] = ('critical', 'high')


class AdmitTiers(BaseModel):
    """The tiers admitted in the protected window: ``default``, or a list keyed by project_id."""

    model_config = ConfigDict(frozen=True, extra='allow')
    __pydantic_extra__: dict[str, tuple[str, ...]] = Field(init=False)

    default: tuple[str, ...] = DEFAULT_ADMIT_TIERS

    def for_project(self, project_id: str | None) -> tuple[str, ...]:
        if project_id is None:
            return self.default
        return self.__pydantic_extra__.get(project_id, self.default)


class ReserveConfig(BaseModel):
    """The accounts file's ``reserve:`` block (PRD §6.2), and the weekly calendar it implies.

    Windows are half-open: the release window is ``[release_at, next_reset)`` and the protected
    window begins at the reset instant itself.
    """

    model_config = ConfigDict(frozen=True, extra='forbid')

    account: str = Field(min_length=1)
    weekly_reset_utc: WeeklyReset
    release_hours: int = Field(gt=0, lt=168, strict=True)
    admit_tiers: AdmitTiers = Field(default_factory=AdmitTiers)

    @field_validator('weekly_reset_utc', mode='plain')
    @classmethod
    def _parse_weekly_reset(cls, value: object) -> WeeklyReset:
        if isinstance(value, WeeklyReset):
            return value
        if isinstance(value, str):
            return WeeklyReset.parse(value)
        raise ValueError(f'weekly_reset_utc must be text like "mon 21:00", got {value!r}')

    @field_serializer('weekly_reset_utc')
    def _write_weekly_reset(self, value: WeeklyReset) -> str:
        return str(value)

    def next_reset(self, now: datetime) -> datetime:
        return self.weekly_reset_utc.next_after(now)

    def release_at(self, now: datetime) -> datetime:
        return self.next_reset(now) - timedelta(hours=self.release_hours)

    def in_release_window(self, now: datetime) -> bool:
        return self.release_at(now) <= _as_utc(now)


TIER_WHEN_UNKNOWN = 'medium'


def admits(config: ReserveConfig, work_class: WorkClass, now: datetime) -> bool:
    """Whether ``work_class`` may use the reserve account at ``now`` (PRD §6.2 invariant 6)."""
    if config.in_release_window(now):
        return True
    match work_class.kind:
        case WorkKind.ESSENTIAL:
            return True
        case WorkKind.BACKGROUND:
            return False
        case WorkKind.TASK:
            tier = work_class.tier or TIER_WHEN_UNKNOWN
            return tier in config.admit_tiers.for_project(work_class.project_id)
    assert_never(work_class.kind)


DRIFT_TOLERANCE = timedelta(minutes=30)
DRIFT_HORIZON = timedelta(hours=3)


@dataclass(frozen=True)
class ScheduleDrift:
    configured_reset: datetime
    observed_reset: datetime


def schedule_drift(config: ReserveConfig, observed_reset: datetime) -> ScheduleDrift | None:
    """The configured reset a banner's reset time drifted from, if it did (PRD §6.2 invariant 5)."""
    observed = _as_utc(observed_reset)
    following = config.next_reset(observed)
    preceding = following - timedelta(days=7)
    nearest = min(following, preceding, key=lambda reset: abs(observed - reset))
    if DRIFT_TOLERANCE < abs(observed - nearest) <= DRIFT_HORIZON:
        return ScheduleDrift(configured_reset=nearest, observed_reset=observed)
    return None
