"""Asset-serve smoke tests for MergeTab in tabs.jsx.

Fetches the served .jsx asset and asserts it is served successfully.
Follows the idiom established in test_tab_overview.py / test_tab_curator.py /
test_tab_orchestrators.py.

Note: source-introspection tests (positive-anchor, row-count expression) were
removed — they pin cosmetic source tokens without exercising runtime behaviour
and break on harmless refactors (rename, arrow-function conversion, local-var
extraction).  The {d.recent.length} matching presentation span (tabs.jsx step-4)
is verified by manual / e2e testing; the JS project has no test harness
(no package.json / jest / vitest / babel) so a jsdom rendering test is out of
scope.
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient


@pytest.fixture(scope='module')
def _client():
    from dashboard.app import app

    with TestClient(app) as c:
        yield c


class TestMergeTabAssetServed:
    """Smoke-tests that the tabs.jsx static asset is served correctly."""

    def test_tabs_jsx_served(self, _client):
        resp = _client.get('/static/redux/tabs.jsx')
        assert resp.status_code == 200
