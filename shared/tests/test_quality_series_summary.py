"""shared.quality_series.summary: the week's proportions and event rates."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from shared.quality_series.summary import EventsPerLanding, KOfN, week_bounds


def test_an_event_rate_may_exceed_one_per_landing() -> None:
    rate = EventsPerLanding(events=3, landings=1)
    assert rate.per(100, 1) == '300.0'
    assert EventsPerLanding(0, 0).per(100, 1) == 'none'


@pytest.mark.parametrize(('k', 'n'), [(3, 1), (-1, 2)])
def test_a_proportion_refuses_k_outside_0_to_n(k: int, n: int) -> None:
    with pytest.raises(ValueError, match=f'k={k}, n={n}'):
        KOfN(k, n)


def test_week_bounds_are_monday_to_monday_utc() -> None:
    assert week_bounds('2026-W44') == (
        datetime(2026, 10, 26, tzinfo=UTC), datetime(2026, 11, 2, tzinfo=UTC),
    )


def test_a_malformed_week_is_refused() -> None:
    with pytest.raises(ValueError):
        week_bounds('2026-44')
