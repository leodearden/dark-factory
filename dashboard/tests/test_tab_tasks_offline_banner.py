"""Wiring tests for the Tasks tab's honest offline/degraded banner.

The banner DECISION is unit-tested as a pure function in
``dashboard/tests/js/tasks_offline_banner.test.mjs``. This module tests the
other half — that the decision is actually reachable from the browser: the
module is served, the script tag exists with the right cache-buster and load
order, and ``tab_tasks.jsx`` calls it instead of re-deriving the copy inline.

A pure module nobody loads is worse than no module: it passes its own suite
while the tab keeps rendering the old lie.

Follows the ``test_tab_escalations.py`` / ``test_tab_curator.py`` idiom:
module-scoped ``TestClient``, assets fetched over the app rather than read off
disk, and JSX comments stripped before asserting so a MENTION in prose cannot
satisfy a render-site assertion (this file's header comments name every
identifier the render code names).
"""

from __future__ import annotations

import re

import pytest
from starlette.testclient import TestClient

_BANNER_SRC = '/static/redux/tasks_offline_banner.js'

# The verbatim global-outage copy. Asserted ABSENT from tab_tasks.jsx: it now
# lives in the banner module, and a surviving inline copy would mean the JSX
# can still emit a total-outage claim without going through the decision.
_GLOBAL_COPY = 'fused-memory offline — task data unavailable'


@pytest.fixture(scope='module')
def _client():
    from dashboard.app import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope='module')
def index_html_body(_client):
    return _client.get('/static/redux/index.html').text


@pytest.fixture(scope='module')
def tab_tasks_jsx_code(_client):
    """``tab_tasks.jsx`` with every comment stripped.

    Same rationale as ``tab_escalations.py``'s equivalent fixture: the file
    carries explanatory prose naming the same identifiers the render code
    names, so a whole-file grep is satisfied by a comment — delete the render
    site, leave the comment, and the assertion stays green.
    """
    body = _client.get('/static/redux/tab_tasks.jsx').text
    return re.sub(r'/\*[\s\S]*?\*/|//[^\n]*', '', body)


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


def test_tab_tasks_renders_through_the_banner_decision(tab_tasks_jsx_code):
    """(d) the JSX calls the decision and no longer emits the copy itself."""
    assert 'tasksBannerNotices' in tab_tasks_jsx_code, (
        'tab_tasks.jsx must render over tasksBannerNotices() — otherwise the '
        'tested decision is dead code and the tab keeps its own copy logic'
    )
    assert _GLOBAL_COPY not in tab_tasks_jsx_code, (
        'the global-outage copy must live only in tasks_offline_banner.js — an '
        'inline copy in the JSX is a render site that can still claim a total '
        'outage without consulting the decision'
    )


def test_tab_tasks_reads_the_degraded_projects_key(tab_tasks_jsx_code):
    """(e) the new server-side fact is actually consumed by the render."""
    assert 'TASKS_DEGRADED_PROJECTS' in tab_tasks_jsx_code, (
        'TASKS_DEGRADED_PROJECTS is on the wire but unread — a project the '
        'handler ran out of budget for would be invisible on the tab'
    )


def test_banner_testids_distinguish_the_three_notice_kinds(tab_tasks_jsx_code):
    """(f) the existing hook survives and the new notices get their own.

    ``tasks-offline-banner`` is an existing testid other tests key on, so it
    must stay attached to the GLOBAL notice (the state it always described).
    Partial and degraded need distinct hooks, or a test asserting "the outage
    banner is showing" would pass on a partial degradation.
    """
    assert 'data-testid="tasks-offline-banner"' in tab_tasks_jsx_code

    for testid in (
        'tasks-partial-banner',
        'tasks-degraded-banner',
    ):
        assert f'data-testid="{testid}"' in tab_tasks_jsx_code, (
            f'missing distinct testid {testid!r} — without it a partial '
            'degradation is indistinguishable from a total outage in tests'
        )
