"""Wiring tests for the ζ lifecycle-flow diagram (mini-Sankey) component.

Follows the source-assertion idiom established in test_tab_escalations.py /
test_tab_escalation_analytics.py: static text checks against the served .jsx
(no JS runtime in this project — .jsx needs Babel, so behavioral geometry
correctness lives in esc_flow_layout.js's node:test suite instead; see
dashboard/tests/js/esc_flow_layout.test.mjs). This file checks wiring only:
that esc_flow_diagram.jsx exists, consumes the layout module correctly,
renders the expected SVG shape, that index.html loads everything in the
right order, and that tab_escalation_analytics.jsx mounts it.

Helpers below (_client, _extract_function_body, _ScriptTagCollector,
_find_script_position, _assert_script_loads_before) are copied — not
imported — from test_tab_escalation_analytics.py, per that suite's
established copy-not-import convention for these cross-file test helpers.
"""

from __future__ import annotations

import html.parser
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


@pytest.fixture(scope='module')
def index_html_body(_client) -> str:
    return _client.get('/static/redux/index.html').text


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
# Load-order helpers (copied from test_tab_escalation_analytics.py, itself
# copied from test_tab_escalations.py / test_index_html.py)
# ---------------------------------------------------------------------------


class _ScriptTagCollector(html.parser.HTMLParser):
    """Collects the attribute dicts for every <script> start-tag encountered."""

    def __init__(self) -> None:
        super().__init__()
        self.script_attrs: list[dict[str, str | None]] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag == 'script':
            self.script_attrs.append(dict(attrs))


def _find_script_position(
    body: str, src_prefix: str
) -> tuple[int, dict[str, str | None]] | None:
    """Return ``(index, attrs)`` for the first <script> tag whose ``src``
    starts with ``src_prefix``, or ``None`` if no such tag exists.
    """
    collector = _ScriptTagCollector()
    collector.feed(body)
    for i, attrs in enumerate(collector.script_attrs):
        if (attrs.get('src') or '').startswith(src_prefix):
            return i, attrs
    return None


def _assert_script_loads_before(
    body: str,
    before_src_prefix: str,
    after_src_prefix: str,
    before_label: str,
    after_label: str,
    consumer_note: str = '',
) -> None:
    """Assert that the script for ``before_src_prefix`` loads BEFORE the
    script for ``after_src_prefix`` in ``body``.  Combines a
    defer/async/type=module false-pass guard with the document-order
    position comparison.
    """
    before_result = _find_script_position(body, before_src_prefix)
    assert before_result is not None, (
        f'No <script src="{before_src_prefix}..."> tag found in index.html. '
        f'{consumer_note}'
    )
    before_pos, before_attrs = before_result
    before_src = before_attrs.get('src')

    after_result = _find_script_position(body, after_src_prefix)
    assert after_result is not None, (
        f'<script src="{after_src_prefix}..."> not found in index.html — '
        f'cannot verify load-order invariant for {before_label}.'
    )
    after_pos, after_attrs = after_result

    # Both tags must be classic synchronous scripts — otherwise document order
    # diverges from execution order and the position comparison below is moot.
    for _label, _attrs in [
        (before_label, before_attrs),
        (after_label, after_attrs),
    ]:
        assert 'defer' not in _attrs, (
            f'{_label} has a defer attribute; document order no longer implies '
            f'execution order, so the load-order check below may give a false pass.'
        )
        assert 'async' not in _attrs, (
            f'{_label} has an async attribute; document order no longer implies '
            f'execution order, so the load-order check below may give a false pass.'
        )
        assert (_attrs.get('type') or '').lower() != 'module', (
            f'{_label} has type="module"; ES modules are deferred by default, '
            f'so document order no longer implies execution order.'
        )

    assert before_pos < after_pos, (
        f'{before_label} (position {before_pos}, src={before_src!r}) must load '
        f'BEFORE {after_label} (position {after_pos}). '
        f'{consumer_note}'
    )


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


# ---------------------------------------------------------------------------
# step-7: index.html registers esc_flow_layout.js + esc_flow_diagram.jsx in
# the correct load order, and the shared cache-buster version is bumped.
# ---------------------------------------------------------------------------


def test_index_html_registers_esc_flow_layout_load_order(index_html_body: str) -> None:
    """esc_flow_layout.js is a classic script that must load BEFORE
    esc_flow_diagram.jsx (whose top-level destructure reads
    window.DF_ESC_FLOW_LAYOUT at parse time) and BEFORE
    tab_escalation_analytics.jsx (which mounts LifecycleFlowDiagram).
    """
    _LAYOUT_PREFIX = '/static/redux/esc_flow_layout.js'
    _DIAGRAM_PREFIX = '/static/redux/esc_flow_diagram.jsx'
    _TAB_ANALYTICS_PREFIX = '/static/redux/tab_escalation_analytics.jsx'

    result = _find_script_position(index_html_body, _LAYOUT_PREFIX)
    assert result is not None, (
        f'No <script src="{_LAYOUT_PREFIX}..."> tag found in index.html — '
        'add it as a classic script (e.g. after runtime_format.js).'
    )

    # (a) esc_flow_layout.js loads before esc_flow_diagram.jsx
    _assert_script_loads_before(
        index_html_body,
        _LAYOUT_PREFIX,
        _DIAGRAM_PREFIX,
        'esc_flow_layout.js',
        'esc_flow_diagram.jsx',
        'esc_flow_layout.js must load before esc_flow_diagram.jsx so '
        'window.DF_ESC_FLOW_LAYOUT is defined before the destructure runs.',
    )

    # (c) esc_flow_layout.js loads before tab_escalation_analytics.jsx too
    _assert_script_loads_before(
        index_html_body,
        _LAYOUT_PREFIX,
        _TAB_ANALYTICS_PREFIX,
        'esc_flow_layout.js',
        'tab_escalation_analytics.jsx',
        'esc_flow_layout.js must load before tab_escalation_analytics.jsx.',
    )


def test_index_html_registers_esc_flow_diagram_load_order(index_html_body: str) -> None:
    """esc_flow_diagram.jsx must be a classic Babel script that loads AFTER
    charts.jsx (needs C.PALETTE) and BEFORE tab_escalation_analytics.jsx
    (the consumer that mounts LifecycleFlowDiagram in the esc-flow-slot).
    """
    _CHARTS_PREFIX = '/static/redux/charts.jsx'
    _DIAGRAM_PREFIX = '/static/redux/esc_flow_diagram.jsx'
    _TAB_ANALYTICS_PREFIX = '/static/redux/tab_escalation_analytics.jsx'

    result = _find_script_position(index_html_body, _DIAGRAM_PREFIX)
    assert result is not None, (
        f'No <script src="{_DIAGRAM_PREFIX}..."> tag found in index.html — '
        'add it as a Babel script before tab_escalation_analytics.jsx.'
    )
    _, diagram_attrs = result

    # Must be a classic (non-deferred, non-module) Babel script.
    assert 'defer' not in diagram_attrs, (
        'esc_flow_diagram.jsx script tag has defer= — remove it; classic '
        'synchronous scripts are required for Babel-standalone transpilation.'
    )
    assert 'async' not in diagram_attrs, (
        'esc_flow_diagram.jsx script tag has async= — remove it.'
    )
    assert (diagram_attrs.get('type') or '').lower() in ('text/babel', ''), (
        'esc_flow_diagram.jsx script must have type="text/babel" (or no type) — '
        f'got {diagram_attrs.get("type")!r}.'
    )

    # (b) Loads after charts.jsx
    _assert_script_loads_before(
        index_html_body,
        _CHARTS_PREFIX,
        _DIAGRAM_PREFIX,
        'charts.jsx',
        'esc_flow_diagram.jsx',
        'charts.jsx must load before esc_flow_diagram.jsx so window.DF_CHARTS/C.PALETTE is available.',
    )

    # (b) Loads before tab_escalation_analytics.jsx
    _assert_script_loads_before(
        index_html_body,
        _DIAGRAM_PREFIX,
        _TAB_ANALYTICS_PREFIX,
        'esc_flow_diagram.jsx',
        'tab_escalation_analytics.jsx',
        'esc_flow_diagram.jsx must load before tab_escalation_analytics.jsx so window.DF_ESC_FLOW is available to mount.',
    )


def test_index_html_cache_buster_bumped_for_esc_flow(index_html_body: str) -> None:
    """All /static/redux/*?v= cache-busters must still share a single
    version, and that version must be >= 33 (the uniform bump that
    accompanies registering the two new esc_flow_* files).
    """
    versions = set(re.findall(r'/static/redux/[^"?]+\?v=(\d+)', index_html_body))
    assert len(versions) == 1, (
        f'index.html has mixed /static/redux/?v= cache-buster versions: {sorted(versions)} — '
        'bump all of them uniformly to the same value.'
    )
    v = int(next(iter(versions)))
    assert v >= 33, (
        f'index.html cache-buster version is {v}, expected >= 33 (uniform bump for esc_flow_* registration).'
    )
