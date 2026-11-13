"""Dashboard routes read the present only through their app's clock.

The seam is ``dashboard/src/dashboard/api/clock.py``: lifespan installs the
live clock, and a test installs a fixed one to pin every instant a route serves.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from _dt_helpers import fixed_clock, installed_clock
from _served_reads import forget_served_reads

from dashboard.api.clock import app_clock, live_clock

# A PAST instant no live read can equal; a stale read is the worst it can leak.
FIXED = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)

API_ROUTES = [
    ('/api/v2/dashboard/tasks', {}),
    ('/api/v2/dashboard/tasks', {'projection': 'census'}),
    ('/api/v2/dashboard/merge-queue', {}),
    ('/api/v2/dashboard/burndown', {}),
    ('/api/v2/dashboard/memory', {}),
    ('/api/v2/dashboard/orchestrators', {}),
    ('/api/v2/dashboard/escalations', {}),
    ('/api/v2/dashboard/escalation-analytics', {}),
    ('/api/v2/dashboard/invariants', {}),
    ('/api/v2/dashboard/prds', {}),
]

APP_ROUTES = [
    ('/api/v2/dashboard/memory-graphs', {}),
    ('/api/v2/dashboard/performance', {}),
]


@pytest.fixture(autouse=True)
def _no_served_read_outlives_its_test():
    forget_served_reads()
    yield
    forget_served_reads()


def _get_at_the_installed_clock(client, path: str, params: dict[str, str]) -> dict:
    with installed_clock(client.app, fixed_clock(FIXED)):
        resp = client.get(path, params=params)
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_the_app_serves_the_live_clock_by_default(client):
    assert app_clock(client.app) is live_clock


def test_an_installed_clock_is_read_until_it_is_restored(client):
    with installed_clock(client.app, fixed_clock(FIXED)):
        assert app_clock(client.app)() == FIXED
    assert app_clock(client.app) is live_clock


@pytest.mark.parametrize(('path', 'params'), API_ROUTES + APP_ROUTES)
def test_a_route_serves_at_the_installed_clock(client, path, params):
    body = _get_at_the_installed_clock(client, path, params)
    assert body['served_at'] == FIXED.isoformat()


def test_the_terminal_answer_serves_at_the_installed_clock(client):
    label = client.app.state.config.project_root.name
    body = _get_at_the_installed_clock(client, '/api/v2/dashboard/tasks', {'terminal': label})
    assert body['served_at'] == FIXED.isoformat()


def test_memory_evals_are_generated_at_the_installed_clock(client):
    body = _get_at_the_installed_clock(client, '/api/v2/dashboard/memory-evals', {})
    assert body['MEMORY_EVALS']['generated_at'] == FIXED.isoformat()
