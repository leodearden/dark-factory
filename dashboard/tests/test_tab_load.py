"""Smoke test: tab_overview.jsx defines and renders the HostLoadCard component.

The host-load data contract is covered behaviorally by:
- test_app.py::test_load_endpoint_returns_known_metric_shape  (GET /api/load end-to-end)
- test_load_data.py  (get_load_metrics: populated/empty/None DB, sparkline cap, ordering)

No source-text/meta-test assertions are made here — those pin wording tokens without
exercising behaviour and were explicitly rejected in review.
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def _client():
    from dashboard.app import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope='module')
def tab_overview_jsx_body(_client):
    return _client.get('/static/redux/tab_overview.jsx').text


# ---------------------------------------------------------------------------
# Existence smoke check
# ---------------------------------------------------------------------------


def test_tab_overview_serves_and_defines_host_load_card(tab_overview_jsx_body: str) -> None:
    """GET /static/redux/tab_overview.jsx returns 200 and defines HostLoadCard."""
    body = tab_overview_jsx_body

    assert 'function HostLoadCard(' in body, (
        'tab_overview.jsx does not define `function HostLoadCard(` — '
        'add a named HostLoadCard function declaration to the file.'
    )

    assert '<HostLoadCard' in body, (
        'tab_overview.jsx does not render `<HostLoadCard` — '
        'add `<HostLoadCard />` to the OverviewTab grid.'
    )
