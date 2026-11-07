"""Dashboard routes read the present only through their app's clock.

The seam is ``dashboard/src/dashboard/api/clock.py``: lifespan installs the
live clock, and a test installs a fixed one to pin every instant a route serves.
"""

from __future__ import annotations

from datetime import UTC, datetime

from _dt_helpers import fixed_clock, installed_clock

from dashboard.api.clock import app_clock, live_clock

# A PAST instant no live read can equal; a stale read is the worst it can leak.
FIXED = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


def test_the_app_serves_the_live_clock_by_default(client):
    assert app_clock(client.app) is live_clock


def test_an_installed_clock_is_read_until_it_is_restored(client):
    with installed_clock(client.app, fixed_clock(FIXED)):
        assert app_clock(client.app)() == FIXED
    assert app_clock(client.app) is live_clock
