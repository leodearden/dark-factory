"""Wiring tests for the ζ lifecycle-flow diagram (mini-Sankey) component.

Follows the source-assertion idiom established in test_tab_escalations.py /
test_tab_escalation_analytics.py: static text checks against the served .jsx
(no JS runtime in this project — .jsx needs Babel, so behavioral geometry
correctness lives in esc_flow_layout.js's node:test suite instead; see
dashboard/tests/js/esc_flow_layout.test.mjs). This file only checks wiring:
that esc_flow_diagram.jsx exists, consumes the layout module correctly,
renders the expected SVG shape, and (in later steps) that index.html loads
it in the right order and tab_escalation_analytics.jsx mounts it.

Helpers below (_client, _extract_function_body) are copied — not imported —
from test_tab_escalation_analytics.py, per that suite's established
copy-not-import convention for these cross-file test helpers.
"""

from __future__ import annotations

import re

import pytest
from starlette.testclient import TestClient


@pytest.fixture(scope='module')
def _client():
    from dashboard.app import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope='module')
def esc_flow_diagram_jsx_response(_client):
    return _client.get('/static/redux/esc_flow_diagram.jsx')


@pytest.fixture(scope='module')
def esc_flow_diagram_jsx_body(esc_flow_diagram_jsx_response) -> str:
    return esc_flow_diagram_jsx_response.text


# ---------------------------------------------------------------------------
# Helper: extract a named JS/JSX function body (brace-aware).
# Copied from test_tab_escalation_analytics.py (itself copied from
# test_tab_escalations.py) — scopes token-presence checks to a specific
# function body rather than searching the entire file.
# ---------------------------------------------------------------------------


def _extract_function_body(src: str, fn_name: str) -> str:
    """Return the body block of a ``function <fn_name>(`` declaration, braces included.

    Paren-depth walks past the parameter list before looking for the body's
    opening ``{`` — a destructured parameter (``function Foo({ a, b }) {``)
    contains its own ``{``/``}`` pair *inside* the parameter list, so naively
    taking the first ``{`` after the opening ``(`` would return just the
    destructuring pattern instead of the function body.
    """
    m = re.search(rf'\bfunction\s+{re.escape(fn_name)}\s*\(', src)
    if m is None:
        return ''
    paren_depth = 1
    i = m.end()
    while i < len(src) and paren_depth > 0:
        if src[i] == '(':
            paren_depth += 1
        elif src[i] == ')':
            paren_depth -= 1
        i += 1
    if paren_depth != 0:
        return ''
    start = src.find('{', i)
    if start == -1:
        return ''
    depth = 0
    for j in range(start, len(src)):
        c = src[j]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return src[start : j + 1]
    return ''


# ---------------------------------------------------------------------------
# step-5: esc_flow_diagram.jsx exists and wires up correctly
# ---------------------------------------------------------------------------


def test_esc_flow_diagram_jsx_is_served(esc_flow_diagram_jsx_response) -> None:
    assert esc_flow_diagram_jsx_response.status_code == 200, (
        'GET /static/redux/esc_flow_diagram.jsx did not return 200 — '
        'the file must exist under dashboard/src/dashboard/static/redux/.'
    )


def test_lifecycle_flow_diagram_consumes_layout_module(esc_flow_diagram_jsx_body: str) -> None:
    """LifecycleFlowDiagram must consume esc_flow_layout.js's aggregateFlow +
    layoutFlow — not reimplement any aggregation/geometry logic inline.
    """
    body = esc_flow_diagram_jsx_body
    assert 'window.DF_ESC_FLOW_LAYOUT' in body, (
        'esc_flow_diagram.jsx does not reference window.DF_ESC_FLOW_LAYOUT — '
        'it must consume esc_flow_layout.js rather than reimplementing '
        'aggregation/geometry logic inline.'
    )
    assert 'function LifecycleFlowDiagram(' in body, (
        'esc_flow_diagram.jsx does not define `function LifecycleFlowDiagram(` — '
        'add the component.'
    )
    component_body = _extract_function_body(body, 'LifecycleFlowDiagram')
    assert component_body, 'Could not locate the LifecycleFlowDiagram( function body.'

    assert re.search(r'\baggregateFlow\s*\(', component_body), (
        'LifecycleFlowDiagram does not call aggregateFlow(...) — it must build '
        'the count model via the layout module, not inline.'
    )
    assert re.search(r'\blayoutFlow\s*\(', component_body), (
        'LifecycleFlowDiagram does not call layoutFlow(...) — it must compute '
        'pixel geometry via the layout module, not inline.'
    )


def test_lifecycle_flow_diagram_reads_flow_daily_prop(esc_flow_diagram_jsx_body: str) -> None:
    body = esc_flow_diagram_jsx_body
    component_body = _extract_function_body(body, 'LifecycleFlowDiagram')
    assert component_body, 'Could not locate the LifecycleFlowDiagram( function body.'
    assert 'flowDaily' in component_body, (
        'LifecycleFlowDiagram does not reference `flowDaily` — it must accept '
        'the already-windowed flow rows as a prop.'
    )


def test_lifecycle_flow_diagram_renders_svg_nodes_and_ribbons(esc_flow_diagram_jsx_body: str) -> None:
    """Renders an SVG with a <rect> per node and a <path> per bezier ribbon,
    each ribbon's `d` attribute fed from its computed `.d` string.
    """
    body = esc_flow_diagram_jsx_body
    component_body = _extract_function_body(body, 'LifecycleFlowDiagram')
    assert component_body, 'Could not locate the LifecycleFlowDiagram( function body.'

    assert '<svg' in component_body, 'LifecycleFlowDiagram does not render an <svg> element.'
    assert '<rect' in component_body, (
        'LifecycleFlowDiagram does not render <rect — one rect per Sankey node is required.'
    )
    assert '<path' in component_body, (
        'LifecycleFlowDiagram does not render <path — one path per bezier ribbon is required.'
    )
    assert re.search(r'\bd=\{\s*[\w.]*\.d\s*\}', component_body), (
        'LifecycleFlowDiagram does not map a ribbon\'s `.d` string into a <path d={...}> '
        'attribute — expected something like `d={rb.d}`.'
    )


def test_lifecycle_flow_diagram_uses_shared_palette(esc_flow_diagram_jsx_body: str) -> None:
    body = esc_flow_diagram_jsx_body
    assert 'window.DF_CHARTS' in body, (
        'esc_flow_diagram.jsx does not reference window.DF_CHARTS — colors must '
        'come from the shared chart palette, not ad-hoc hex/oklch literals.'
    )
    component_body = _extract_function_body(body, 'LifecycleFlowDiagram')
    assert component_body, 'Could not locate the LifecycleFlowDiagram( function body.'
    assert re.search(r'\bC\.PALETTE\b', component_body), (
        'LifecycleFlowDiagram does not reference `C.PALETTE` — node/ribbon colors '
        'must be drawn from the shared palette.'
    )


def test_lifecycle_flow_diagram_hover_shows_ribbon_count(esc_flow_diagram_jsx_body: str) -> None:
    """Hover state (React.useState) highlights a ribbon and surfaces its count."""
    body = esc_flow_diagram_jsx_body
    component_body = _extract_function_body(body, 'LifecycleFlowDiagram')
    assert component_body, 'Could not locate the LifecycleFlowDiagram( function body.'

    assert re.search(r'React\.useState\s*\(', component_body), (
        'LifecycleFlowDiagram does not call React.useState — hover state must be '
        'tracked via a React state hook (no external state library).'
    )
    assert re.search(r'\bonMouse(Enter|Over|Leave)\b', component_body), (
        'LifecycleFlowDiagram does not wire an onMouseEnter/onMouseOver/onMouseLeave '
        'handler — hover interaction is required.'
    )

    mouse_handler_positions = [
        m.start() for m in re.finditer(r'\bonMouse(Enter|Over|Leave)\b', component_body)
    ]
    assert any(
        '.count' in component_body[max(0, i - 300) : i + 300] or '<title' in component_body[max(0, i - 300) : i + 300]
        for i in mouse_handler_positions
    ), (
        'No onMouse handler in LifecycleFlowDiagram appears near a `.count` reference '
        'or a <title> element — hovering a ribbon must surface its count.'
    )


def test_lifecycle_flow_diagram_empty_state(esc_flow_diagram_jsx_body: str) -> None:
    body = esc_flow_diagram_jsx_body
    component_body = _extract_function_body(body, 'LifecycleFlowDiagram')
    assert component_body, 'Could not locate the LifecycleFlowDiagram( function body.'
    assert re.search(r'flowDaily\.length\s*===?\s*0|!flowDaily\.length|!flowDaily\b', component_body), (
        'LifecycleFlowDiagram does not guard on an empty flowDaily — an empty-state '
        'message must render when there are no flow rows in the window.'
    )


def test_window_df_esc_flow_export_is_additive_and_not_clobbered(esc_flow_diagram_jsx_body: str) -> None:
    """window.DF_ESC_FLOW = { ... LifecycleFlowDiagram ... } is assigned exactly
    once (no later reassignment silently dropping it — the scheduler_heatmap.jsx
    -> window.DF_SCHED_HEATMAP precedent this mirrors is likewise a single
    assignment).
    """
    body = esc_flow_diagram_jsx_body
    assignments = re.findall(r'window\.DF_ESC_FLOW\s*=\s*\{', body)
    assert assignments, (
        'esc_flow_diagram.jsx does not assign `window.DF_ESC_FLOW = { ... }` — '
        'add the additive export.'
    )
    assert len(assignments) == 1, (
        f'expected exactly one `window.DF_ESC_FLOW = {{` assignment, found {len(assignments)} — '
        'a later reassignment would clobber the first.'
    )
    export_block_start = body.index('window.DF_ESC_FLOW')
    export_block = body[export_block_start : export_block_start + 200]
    assert 'LifecycleFlowDiagram' in export_block, (
        'window.DF_ESC_FLOW export does not include LifecycleFlowDiagram.'
    )
