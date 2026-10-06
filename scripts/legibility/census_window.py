"""The session window one census run mines.

A run mines the sessions dated in the last ``census.ledger_retention_days``
UTC days before today, minus those already ledgered as coded and those with
no confusion signal. Until the ledger holds a census row, the window starts
no earlier than the last census, so the first ledgered run does not re-mine
history that predates the ledger. Contract: plans/census-incremental-prd.md
§4.2 (C2, decision D2).

Imports only downward; it never imports census.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta


def _utc_midnight(day: date) -> datetime:
    return datetime.combine(day, time(), UTC)


@dataclass(frozen=True)
class MiningWindow:
    """The half-open UTC interval ``[start, end)`` of session dates mined."""

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        for name, value in (('start', self.start), ('end', self.end)):
            if not isinstance(value, datetime) or value.utcoffset() is None:
                raise ValueError(f'MiningWindow.{name} must be timezone-aware: {value!r}')
        if self.start > self.end:
            raise ValueError(
                f'MiningWindow.start {self.start.isoformat()} is after '
                f'end {self.end.isoformat()}',
            )

    @property
    def first_day(self) -> date:
        return self.start.date()

    @property
    def last_day(self) -> date:
        return (self.end - timedelta(days=1)).date()

    @property
    def is_empty(self) -> bool:
        return self.start == self.end

    def contains(self, day: date) -> bool:
        return self.first_day <= day <= self.last_day

    def to_record(self) -> list[str]:
        return [self.start.isoformat(), self.end.isoformat()]


def retention_window(now: datetime, retention_days: int) -> MiningWindow:
    end = _utc_midnight(now.astimezone(UTC).date())
    return MiningWindow(start=end - timedelta(days=retention_days), end=end)


def transition_window(window: MiningWindow, *, last_census_at: date | None) -> MiningWindow:
    """*window* with its start raised to the last census's day, never past
    its end."""
    if last_census_at is None:
        return window
    floor = min(max(window.start, _utc_midnight(last_census_at)), window.end)
    return MiningWindow(start=floor, end=window.end)
