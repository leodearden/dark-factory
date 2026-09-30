"""Rationale document for the absence of MergeTab rendering tests.

The /static/redux/tabs.jsx 200 asset-serve check is already covered by
test_tab_orchestrators.TestOrchTabCurrentFocusRemoved.test_tabs_jsx_served —
this file therefore adds no new behavioral coverage and exists solely to record
why fuller tests are absent:

  * Source-introspection tests (positive-anchor, row-count expression) were
    removed — they pin cosmetic source tokens without exercising runtime
    behaviour and break on harmless refactors (rename, arrow-function
    conversion, local-var extraction).

  * The Recent-merges caption ("showing N of M in <window>") is formatted by
    window_chip.js::recentMergesCaption, whose behaviour is covered by
    dashboard/tests/js/window_chip.test.mjs; tabs.jsx only passes it the row
    count, the payload's recent_total and the /merge-queue WINDOW echo. There
    is no JSX render harness, so the wiring itself is verified by manual / e2e
    testing.

The single test below is kept rather than deleted so that the file, and its
explanatory docstring, remain discoverable by future maintainers.  If a JS
test harness is ever added, rendering-contract tests should live here.
"""

from __future__ import annotations


class TestMergeTabAssetServed:
    """Smoke-tests that the tabs.jsx static asset is served correctly."""

    def test_tabs_jsx_served(self, _client):
        resp = _client.get('/static/redux/tabs.jsx')
        assert resp.status_code == 200
