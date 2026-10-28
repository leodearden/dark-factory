"""Contract: a render throw in the active tab is contained to that tab.

Without an error boundary, one throw anywhere in a tab's render (datum.js's
``assertDatum`` throws on purpose) unmounts React's whole root and blanks the
page. The BEHAVIOUR of the fallback text, including that it never throws itself,
is executed in ``dashboard/tests/js/render_failure.test.mjs``. What is left
here is the wiring that source text can settle, since app.jsx is
``type="text/babel"`` and cannot run under node (see test_datum_components.py's
docstring). Every probe runs over comment-stripped code.
"""

from __future__ import annotations

import re

from _app_jsx_source import app_body, app_code, assert_app_destructures_without_fallback
from _dashboard_helpers import walk_balanced

_KEYED_BODY_TAG = '<div className="body" key={tab}>'


def _boundary_class_body(app_jsx_body: str) -> str:
    code = app_code(app_jsx_body)
    match = re.search(r'class\s+TabErrorBoundary\s+extends\s+React\.Component\s*\{', code)
    assert match, (
        'app.jsx declares no `class TabErrorBoundary extends React.Component`. React '
        'only accepts a class component as an error boundary; without one a tab '
        'that throws while rendering unmounts the whole page.'
    )
    body = walk_balanced(code, match.end() - 1)
    assert body, 'the TabErrorBoundary class body is never closed'
    return body


def test_the_active_tab_renders_inside_the_error_boundary(app_jsx_body: str) -> None:
    body = app_body(app_jsx_body)

    calls = re.findall(r'(?<!function )\brenderTab\s*\(', body)
    assert len(calls) == 1, f'App() must call renderTab() exactly once; found {len(calls)}'
    assert re.search(
        r'<TabErrorBoundary\s+tabLabel=\{\s*tabLabel\s*\}\s*>\s*\{\s*renderTab\(\)\s*\}\s*</TabErrorBoundary>',
        body,
    ), (
        'renderTab() must be the sole child of <TabErrorBoundary tabLabel={tabLabel}>. '
        'A tab body rendered outside the boundary takes the whole page down when '
        'its render throws.'
    )


def test_the_notices_and_the_tab_retry_sit_outside_the_boundary(app_jsx_body: str) -> None:
    body = app_body(app_jsx_body)
    keyed_body = body.find(_KEYED_BODY_TAG)
    assert keyed_body >= 0, f'App() no longer renders {_KEYED_BODY_TAG}'
    boundary = body.find('<TabErrorBoundary')
    assert boundary > keyed_body, (
        f'<TabErrorBoundary must sit inside {_KEYED_BODY_TAG}: the key remounts a '
        'fresh boundary on every tab switch, which is how a failed tab is retried.'
    )
    for testid in ('endpoint-stale-banner', 'endpoint-loading-banner'):
        position = body.find(testid, keyed_body)
        assert keyed_body < position < boundary, (
            f'data-testid="{testid}" must render inside .body but OUTSIDE the '
            'boundary, so a throwing tab cannot take the staleness signals down '
            'with it.'
        )


def test_tab_error_boundary_is_a_react_error_boundary_rendering_through_the_decision_module(
    app_jsx_body: str,
) -> None:
    body = _boundary_class_body(app_jsx_body)

    assert 'static getDerivedStateFromError(' in body, (
        'TabErrorBoundary declares no `static getDerivedStateFromError`, so React '
        'does not treat it as an error boundary and a render throw still blanks '
        'the page.'
    )
    assert 'tabRenderFailureNotice(' in body, (
        'TabErrorBoundary must render the text render_failure.js decides, '
        'via tabRenderFailureNotice.'
    )
    assert not re.search(r'\.\s*(message|name)\b', body), (
        'TabErrorBoundary reads .message or .name itself: describing the throw '
        'belongs to tabRenderFailureNotice, which is total over hostile values.'
    )
    assert not re.search(r'\bString\s*\(', body), (
        'TabErrorBoundary converts a value with String(): describing the throw '
        'belongs to tabRenderFailureNotice, which is total over hostile values.'
    )


def test_app_destructures_the_failure_notice_without_fallback(app_jsx_body: str) -> None:
    assert_app_destructures_without_fallback(
        app_jsx_body, 'DF_RENDER_FAILURE', 'tabRenderFailureNotice',
        fallback_consequence='a silently uncontained page',
    )
