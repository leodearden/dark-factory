"""Asset-wiring tests for the Tasks tab's honest offline/degraded banner.

The banner DECISION is unit-tested as a pure function in
``dashboard/tests/js/tasks_offline_banner.test.mjs``, and the RENDER contract
belongs in that same node ``--test`` suite, against rendered output. This
module covers only the third thing neither of those can see: that the module
is actually reachable from the browser — served over the app, tagged in
``index.html`` exactly once, with a cache-buster matching its siblings, and
evaluated before the JSX that calls its global.

A pure module nobody loads is worse than no module: it passes its own suite
while the tab keeps rendering the old lie.

DELIBERATELY NOT HERE (removed under esc-3857-9): assertions that grep the
SOURCE TEXT of ``tab_tasks.jsx`` for identifiers, testids and exact JSX
spellings. Those test another file's code text rather than any behaviour:
they go vacuously green on a rename or a reflow while the regression they
name ships, and several could not fail at all. Every assertion below is a
structural fact about HTML this app serves. Do not reintroduce source greps
here — cover render behaviour in the node suite instead.
"""

from __future__ import annotations

import re

import pytest
from starlette.testclient import TestClient

_BANNER_SRC = '/static/redux/tasks_offline_banner.js'


@pytest.fixture(scope='module')
def _client():
    from dashboard.app import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope='module')
def index_html_body(_client):
    return _client.get('/static/redux/index.html').text


def _script_srcs(body: str) -> list[str]:
    """Every ``<script src=...>`` value in document order."""
    return re.findall(r'<script[^>]*\bsrc="([^"]+)"', body)


def _cache_buster(src: str) -> str | None:
    match = re.search(r'\?v=(\d+)', src)
    return match.group(1) if match else None


def test_banner_module_is_served(_client):
    """(a) the file is actually reachable, not merely present in the tree."""
    resp = _client.get(_BANNER_SRC)

    assert resp.status_code == 200
    assert 'tasksBannerNotices' in resp.text


def test_index_html_loads_the_banner_module_with_the_sibling_cache_buster(
    index_html_body,
):
    """(b) the script tag exists and shares its siblings' ``?v=`` value.

    Asserted against a sibling's PARSED value rather than a hard-coded number,
    so the next cache-bust bump cannot leave this one stale — a stale buster
    serves the browser a cached module without ``tasksBannerNotices`` while the
    freshly-bumped JSX calls it, which is a blank tab, not a degraded one.
    """
    srcs = _script_srcs(index_html_body)

    banner = [s for s in srcs if s.startswith(_BANNER_SRC)]
    assert len(banner) == 1, f'expected exactly one {_BANNER_SRC} tag, got {banner}'

    siblings = [
        s for s in srcs
        if s.startswith('/static/redux/task_status_counts.js')
        or s.startswith('/static/redux/prd_grouping.js')
        or s.startswith('/static/redux/runtime_format.js')
    ]
    assert len(siblings) == 3, f'expected the three plain-JS siblings, got {siblings}'

    sibling_busters = {_cache_buster(s) for s in siblings}
    assert len(sibling_busters) == 1, (
        f'the plain-JS siblings disagree on ?v= ({sibling_busters}) — pick one '
        'and compare against it'
    )
    expected = next(iter(sibling_busters))
    assert expected is not None, 'siblings carry no ?v= cache-buster to match'
    assert _cache_buster(banner[0]) == expected, (
        f'{_BANNER_SRC} carries ?v={_cache_buster(banner[0])} but its plain-JS '
        f'siblings carry ?v={expected}'
    )


def test_banner_module_is_loaded_before_tab_tasks_jsx(index_html_body):
    """(c) a classic script must be evaluated before the JSX that calls it."""
    srcs = _script_srcs(index_html_body)

    banner_at = next(i for i, s in enumerate(srcs) if s.startswith(_BANNER_SRC))
    jsx_at = next(
        i for i, s in enumerate(srcs)
        if s.startswith('/static/redux/tab_tasks.jsx')
    )

    assert banner_at < jsx_at, (
        f'{_BANNER_SRC} is loaded at position {banner_at}, after tab_tasks.jsx '
        f'at {jsx_at} — its global would be undefined when the JSX runs'
    )

