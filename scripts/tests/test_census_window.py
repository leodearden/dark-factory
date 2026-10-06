"""Tests for scripts/legibility/census_window.py — the session window one
census run mines (plans/census-incremental-prd.md §4.2)."""
from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from legibility import census_window
from legibility.census_window import MiningWindow

_NOW = datetime(2026, 10, 6, 15, 0, tzinfo=UTC)
_START = datetime(2026, 9, 6, tzinfo=UTC)
_END = datetime(2026, 10, 6, tzinfo=UTC)


# ---------------------------------------------------------------------------
# window maths
# ---------------------------------------------------------------------------

def test_retention_window_is_the_utc_days_before_today():
    window = census_window.retention_window(_NOW, 30)
    assert window == MiningWindow(start=_START, end=_END)
    assert window.first_day == date(2026, 9, 6)
    assert window.last_day == date(2026, 10, 5)
    assert window.is_empty is False


def test_window_contains_exactly_its_days():
    window = census_window.retention_window(_NOW, 30)
    assert window.contains(date(2026, 9, 6))
    assert window.contains(date(2026, 10, 5))
    assert not window.contains(date(2026, 10, 6))
    assert not window.contains(date(2026, 9, 5))


def test_window_record_is_its_half_open_iso_bounds():
    window = census_window.retention_window(_NOW, 30)
    assert window.to_record() == ['2026-09-06T00:00:00+00:00', '2026-10-06T00:00:00+00:00']


def test_transition_raises_the_start_to_the_last_census_day():
    window = census_window.retention_window(_NOW, 30)
    moved = census_window.transition_window(window, last_census_at=date(2026, 10, 1))
    assert moved == MiningWindow(start=datetime(2026, 10, 1, tzinfo=UTC), end=_END)


@pytest.mark.parametrize('last_census_at', [None, date(2026, 8, 1), date(2026, 9, 6)])
def test_transition_leaves_the_window_when_the_last_census_is_older(last_census_at):
    window = census_window.retention_window(_NOW, 30)
    assert census_window.transition_window(window, last_census_at=last_census_at) == window


def test_a_census_already_run_today_leaves_an_empty_window():
    window = census_window.retention_window(_NOW, 30)
    moved = census_window.transition_window(window, last_census_at=date(2026, 10, 6))
    assert moved.start == moved.end == _END
    assert moved.is_empty is True
    assert moved.last_day < moved.first_day
    assert not moved.contains(date(2026, 10, 5))


def test_mining_window_rejects_naive_bounds():
    naive = datetime(2026, 9, 6)
    with pytest.raises(ValueError) as excinfo:
        MiningWindow(start=naive, end=_END)
    assert str(naive) in str(excinfo.value) or repr(naive) in str(excinfo.value)


def test_mining_window_rejects_start_after_end():
    with pytest.raises(ValueError) as excinfo:
        MiningWindow(start=_END, end=_START)
    message = str(excinfo.value)
    assert _END.isoformat() in message
    assert _START.isoformat() in message
