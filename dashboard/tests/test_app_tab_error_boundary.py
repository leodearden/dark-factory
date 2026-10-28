"""Contract: a render throw in the active tab is contained to that tab.

Without an error boundary, one throw anywhere in a tab's render (datum.js's
``assertDatum`` throws on purpose) unmounts React's whole root and blanks the
page. The BEHAVIOUR is executed under node: the notice text, including that it
never throws itself, in ``dashboard/tests/js/render_failure.test.mjs``; the
boundary catching a throw and rendering that notice in
``dashboard/tests/js/jsx_render.test.mjs``. What is left here is the tab site's
wiring that no behavioural test covers: renderTab() is the boundary's sole
child, the boundary sits inside the keyed ``.body``, and the stale and loading
notices sit outside it. Every probe runs over comment-stripped code.
"""

from __future__ import annotations

import re

import pytest
from _app_jsx_source import app_body, assert_app_destructures_without_fallback

_KEYED_BODY_TAG = '<div className="body" key={tab}>'
_BOUNDARY_OPEN = '<RenderFailureBoundary'
_RENDER_TAB_AS_SOLE_CHILD = re.compile(r'>\s*\{\s*renderTab\(\)\s*\}\s*</RenderFailureBoundary>')


def _tab_boundary_opening_tag(body: str) -> tuple[int, str]:
    """Offset and text of the ``<RenderFailureBoundary ...>`` tag whose sole child is renderTab()."""
    child = _RENDER_TAB_AS_SOLE_CHILD.search(body)
    assert child, (
        'renderTab() must be the sole child of a <RenderFailureBoundary>. A tab '
        'body rendered outside the boundary takes the whole page down when its '
        'render throws.'
    )
    start = body.rfind(_BOUNDARY_OPEN, 0, child.start())
    assert start >= 0, 'no <RenderFailureBoundary opens before {renderTab()}'
    tag = body[start:child.start() + 1]
    assert not re.search(r'<\s*[A-Za-z/]', tag[len(_BOUNDARY_OPEN):]), (
        f'{{renderTab()}} is not the direct child of a <RenderFailureBoundary> tag: {tag!r}'
    )
    return start, tag


def test_the_active_tab_renders_inside_the_error_boundary(app_jsx_body: str) -> None:
    body = app_body(app_jsx_body)

    calls = re.findall(r'(?<!function )\brenderTab\s*\(', body)
    assert len(calls) == 1, f'App() must call renderTab() exactly once; found {len(calls)}'
    _, tag = _tab_boundary_opening_tag(body)
    assert re.search(r'\btabRenderFailureNotice\(\s*tabLabel\s*,', tag), (
        'the boundary around renderTab() must describe its throw with '
        f'tabRenderFailureNotice(tabLabel, ...), so the notice names the tab: {tag!r}'
    )


def test_the_notices_and_the_tab_retry_sit_outside_the_boundary(app_jsx_body: str) -> None:
    body = app_body(app_jsx_body)
    keyed_body = body.find(_KEYED_BODY_TAG)
    assert keyed_body >= 0, f'App() no longer renders {_KEYED_BODY_TAG}'
    boundary, _ = _tab_boundary_opening_tag(body)
    assert boundary > keyed_body, (
        f'the tab boundary must sit inside {_KEYED_BODY_TAG}: the key remounts a '
        'fresh boundary on every tab switch, which is how a failed tab is retried.'
    )
    for testid in ('endpoint-stale-banner', 'endpoint-loading-banner'):
        position = body.find(testid, keyed_body)
        assert keyed_body < position < boundary, (
            f'data-testid="{testid}" must render inside .body but OUTSIDE the '
            'boundary, so a throwing tab cannot take the staleness signals down '
            'with it.'
        )


@pytest.mark.parametrize(
    'describer', ['tabRenderFailureNotice', 'chromeRenderFailureNotice', 'pageRenderFailureNotice'],
)
def test_app_destructures_the_failure_notice_without_fallback(app_jsx_body: str, describer: str) -> None:
    assert_app_destructures_without_fallback(
        app_jsx_body, 'DF_RENDER_FAILURE', describer,
        fallback_consequence='a silently uncontained page',
    )
