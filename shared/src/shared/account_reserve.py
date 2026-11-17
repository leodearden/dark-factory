"""The reserve rule: which work may use the reserve account, and when.

One account of the shared pool can be held back for urgent work until shortly before its weekly
reset. This module is the rule of plans/allowance-reserve-by-account-prd.md §6.1-6.2, the
reserve-refused definition of §6.3, and the event names and detail shapes of §6.6, as pure
functions of a config, a work class and a clock: nothing is stored here, and nothing here logs or
does I/O.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Collection
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from enum import StrEnum
from typing import ClassVar, TypeAlias, assert_never

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_serializer,
    field_validator,
)

__all__ = [
    'ActiveReserve',
    'DEFAULT_ADMIT_TIERS',
    'DRIFT_HORIZON',
    'DRIFT_TOLERANCE',
    'TIER_WHEN_UNKNOWN',
    'AdmitTiers',
    'InvalidReserve',
    'Reserve',
    'ReserveConfig',
    'ReserveEvent',
    'ReserveRefusal',
    'ReserveState',
    'ReserveWaitEnd',
    'ScheduleDrift',
    'WeeklyReset',
    'WorkClass',
    'WorkClassSource',
    'WorkKind',
    'admits',
    'config_event_details',
    'first_choice',
    'parse_reserve',
    'refusal',
    'refuses_account',
    'resolve_work_class',
    'schedule_drift',
    'wait_end_for_lease',
    'work_class_details',
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
    return source()


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
        if self.time_utc.second or self.time_utc.microsecond:
            raise ValueError(f'time_utc is a whole minute, got {self.time_utc!r}')

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


def refuses_account(
    config: ReserveConfig, account_name: str, work_class: WorkClass, now: datetime
) -> bool:
    """Whether selection must skip ``account_name`` for ``work_class`` at ``now``."""
    now = _as_utc(now)
    return account_name == config.account and not admits(config, work_class, now)


@dataclass(frozen=True)
class ReserveRefusal:
    account: str
    release_at: datetime


def refusal(
    config: ReserveConfig, open_accounts: Collection[str], work_class: WorkClass, now: datetime
) -> ReserveRefusal | None:
    """The refusal when every open account is the reserve and it is refused (PRD §6.3).

    ``open_accounts`` names the accounts that are neither capped, probe-in-flight nor failing
    authentication. With none open the caller is not reserve-refused: nothing at all is open.
    """
    now = _as_utc(now)
    if not open_accounts or any(name != config.account for name in open_accounts):
        return None
    if not refuses_account(config, config.account, work_class, now):
        return None
    return ReserveRefusal(account=config.account, release_at=config.release_at(now))


def first_choice(config: ReserveConfig, now: datetime) -> str | None:
    """The account selection tries before the roster order: the reserve, once it is released."""
    return config.account if config.in_release_window(now) else None


class ReserveWaitEnd(StrEnum):
    """How a reserve wait ended: PRD §6.6's five, and ``cancelled`` for a caller that gave up."""

    ACCOUNT_OPENED = 'account_opened'
    RELEASED = 'released'
    ADMITTED = 'admitted'
    RESERVE_UNUSABLE = 'reserve_unusable'
    DEADLINE = 'deadline'
    CANCELLED = 'cancelled'


def wait_end_for_lease(
    config: ReserveConfig, leased_account: str, now: datetime
) -> ReserveWaitEnd:
    """Why a waiting caller was served by ``leased_account``."""
    now = _as_utc(now)
    if leased_account != config.account:
        return ReserveWaitEnd.ACCOUNT_OPENED
    if config.in_release_window(now):
        return ReserveWaitEnd.RELEASED
    return ReserveWaitEnd.ADMITTED


def work_class_details(work_class: WorkClass, *, unclassified: bool) -> dict[str, object]:
    """A work class as the JSON details of the reserve's wait and served events."""
    return {
        'kind': work_class.kind.value,
        'role': work_class.role,
        'project_id': work_class.project_id,
        'tier': work_class.tier,
        'task_id': work_class.task_id,
        'unclassified': unclassified,
    }


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


class ReserveState(StrEnum):
    ACTIVE = 'active'
    INVALID = 'invalid'


@dataclass(frozen=True)
class ActiveReserve:
    config: ReserveConfig
    state: ClassVar[ReserveState] = ReserveState.ACTIVE


@dataclass(frozen=True)
class InvalidReserve:
    """A block that was present but unusable: inert, and always saying why."""

    reason: str
    state: ClassVar[ReserveState] = ReserveState.INVALID

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError(f'an invalid reserve needs a reason, got {self.reason!r}')


Reserve: TypeAlias = ActiveReserve | InvalidReserve


def parse_reserve(block: object, account_names: Collection[str]) -> Reserve:
    """The accounts file's ``reserve:`` block as a reserve; every failure is an InvalidReserve."""
    try:
        config = ReserveConfig.model_validate(block)
    except ValidationError as error:
        return InvalidReserve(_validation_reason(error))
    if config.account not in account_names:
        listed = ', '.join(account_names) or 'none'
        return InvalidReserve(
            f'reserve.account {config.account!r} names no listed account (listed: {listed})'
        )
    return ActiveReserve(config)


def config_event_details(reserve: Reserve, now: datetime) -> dict[str, object]:
    """The JSON details of the ``reserve_config`` event for ``reserve`` as seen at ``now``."""
    now = _as_utc(now)
    match reserve:
        case ActiveReserve(config=config):
            return {
                'state': reserve.state.value,
                'reason': None,
                'account': config.account,
                'next_reset': config.next_reset(now).isoformat(),
                'release_at': config.release_at(now).isoformat(),
                'admit_tiers': config.admit_tiers.model_dump(mode='json'),
            }
        case InvalidReserve(reason=reason):
            return {
                'state': reserve.state.value,
                'reason': reason,
                'account': None,
                'next_reset': None,
                'release_at': None,
                'admit_tiers': None,
            }
    assert_never(reserve)


def _validation_reason(error: ValidationError) -> str:
    entries = []
    for detail in error.errors():
        location = '.'.join(['reserve', *(str(part) for part in detail['loc'])])
        entries.append(f'{location}: {detail["msg"]}')
    return '; '.join(entries)
