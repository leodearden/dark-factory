"""Rationale document for the absence of MergeTab rendering tests.

The /static/redux/tabs.jsx 200 asset-serve check is already covered by
test_tab_orchestrators.TestOrchTabCurrentFocusRemoved.test_tabs_jsx_served —
this file therefore adds no new behavioral coverage and exists solely to record
why fuller tests are absent:

  * Source-introspection tests (positive-anchor, row-count expression) were
    removed — they pin cosmetic source tokens without exercising runtime
    behaviour and break on harmless refactors (rename, arrow-function
    conversion, local-var extraction).

  * The {d.recent.length} matching presentation span (tabs.jsx step-4) is
    verified by manual / e2e testing; the JS project has no test harness
    (no package.json / jest / vitest / babel) so a jsdom rendering test is
    out of scope.

The single test below is kept rather than deleted so that the file, and its
explanatory docstring, remain discoverable by future maintainers.  If a JS
test harness is ever added, rendering-contract tests should live here.
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
