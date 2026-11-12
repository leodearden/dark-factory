"""One ISO week's figures over per-landing rows: proportions over landings and event rates."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from shared.quality_series.rows import LandingRow, ReviewRoundsSource, unit_heads

__all__ = ['EventsPerLanding', 'KOfN', 'WeekSummary', 'week_bounds', 'week_summary']


@dataclass(frozen=True)
class KOfN:
    """A proportion: k of n landings have the property, so 0 <= k <= n."""

    k: int
    n: int

    def __post_init__(self) -> None:
        if not 0 <= self.k <= self.n:
            raise ValueError(f'a proportion needs 0 <= k <= n (k={self.k!r}, n={self.n!r})')

    def __str__(self) -> str:
        return f'{self.k}/{self.n}'


@dataclass(frozen=True)
class EventsPerLanding:
    """An event rate: events over landings; one landing can carry several events."""

    events: int
    landings: int

    def __post_init__(self) -> None:
        if self.events < 0 or self.landings < 0:
            raise ValueError(
                f'an event rate needs non-negative counts (events={self.events!r}, '
                f'landings={self.landings!r})'
            )

    def per(self, scale: int, digits: int) -> str:
        if self.landings == 0:
            return 'none'
        return f'{scale * self.events / self.landings:.{digits}f}'


@dataclass(frozen=True)
class WeekSummary:
    """One ISO week's figures; a landing whose value is unknown (None) drops out of that figure.

    Event counts (rejection events, re-verifies) are summed once per unit over unit heads, since a
    unit's value is copied to every member; every rate's denominator is the week's landings whose
    value is known.
    """

    week: str
    units: int
    landings: int
    rejection_events: EventsPerLanding
    rejected_landings: KOfN
    flake_confirmed: KOfN
    flake_real_red: KOfN
    review_rounds_ge2: KOfN
    review_rounds_sources: tuple[tuple[ReviewRoundsSource, int], ...]
    reverifies: EventsPerLanding

    def lines(self) -> list[str]:
        sources = ', '.join(f'{source.value}={count}' for source, count in self.review_rounds_sources)
        return [
            f'# week {self.week}: {self.units} units, {self.landings} landings',
            '#   rejections per 100 landings: '
            f'{self.rejection_events.per(100, 1)} '
            f'({self.rejection_events.events} events over {self.rejection_events.landings} landings)',
            f'#   landings with >= 1 rejection: {self.rejected_landings}',
            f'#   confirmed-flake landings: {self.flake_confirmed}',
            f'#   real-red landings: {self.flake_real_red}',
            f'#   review rounds >= 2: {self.review_rounds_ge2} ({sources})',
            f'#   re-verifies per landing: {self.reverifies.per(1, 2)} '
            f'({self.reverifies.events} over {self.reverifies.landings} landings)',
        ]


def week_bounds(iso_week: str) -> tuple[datetime, datetime]:
    """[Monday 00:00 UTC, next Monday) of an ISO week like 2026-W44; ValueError otherwise."""
    monday = datetime.strptime(f'{iso_week}-1', '%G-W%V-%u').replace(tzinfo=UTC)
    return monday, monday + timedelta(days=7)


def _count(values: Sequence[bool | None]) -> KOfN:
    known = [value for value in values if value is not None]
    return KOfN(sum(1 for value in known if value), len(known))


def _unit_events_over_landings(
    week: Sequence[LandingRow], value: Callable[[LandingRow], int | None]
) -> EventsPerLanding:
    events = [value(head) for head in unit_heads(week)]
    known_landings = sum(1 for row in week if value(row) is not None)
    return EventsPerLanding(sum(event for event in events if event is not None), known_landings)


def week_summary(rows: Sequence[LandingRow], iso_week: str) -> WeekSummary:
    start, end = week_bounds(iso_week)
    week = [row for row in rows if start <= row.landing.landed_at < end]
    heads = unit_heads(week)
    rounds = [row.cost for row in week if row.cost.review_rounds is not None]
    return WeekSummary(
        week=iso_week,
        units=len(heads),
        landings=len(week),
        rejection_events=_unit_events_over_landings(week, lambda r: r.outcomes.rejection_events),
        rejected_landings=_count([row.outcomes.rejected for row in week]),
        flake_confirmed=_count([row.outcomes.flake_confirmed for row in week]),
        flake_real_red=_count([row.outcomes.flake_real_red for row in week]),
        review_rounds_ge2=KOfN(sum(1 for cost in rounds if (cost.review_rounds or 0) >= 2),
                               len(rounds)),
        review_rounds_sources=tuple(
            (source, sum(1 for cost in rounds if cost.review_rounds_source is source))
            for source in ReviewRoundsSource
        ),
        reverifies=_unit_events_over_landings(week, lambda r: r.outcomes.reverify_count),
    )
