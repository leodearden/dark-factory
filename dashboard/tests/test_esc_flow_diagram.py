"""Wiring tests for the ζ lifecycle-flow diagram (mini-Sankey) component.

Follows the source-assertion idiom established in test_tab_escalations.py /
test_tab_escalation_analytics.py: static text checks against the served .jsx
(no JS runtime in this project — .jsx needs Babel, so behavioral geometry
correctness lives in esc_flow_layout.js's node:test suite instead; see
dashboard/tests/js/esc_flow_layout.test.mjs). This file checks wiring only:
that esc_flow_diagram.jsx is served and exports window.DF_ESC_FLOW, that
index.html loads everything in the right order (with the shared cache-buster
bumped), and that tab_escalation_analytics.jsx mounts it in the esc-flow-slot
seam. It deliberately does NOT assert on esc_flow_diagram.jsx's internal
implementation tokens (helper-call presence, hover-handler wiring,
empty-state guard regex, etc.) — those lock implementation detail rather
than behavior; a harmless refactor would break them while a real
geometry/wiring regression that happens to keep the tokens would pass
silently. That behavioral correctness is already covered by the .mjs suite
above.

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


@pytest.fixture(scope='module')
def tab_analytics_jsx_body(_client) -> str:
    return _client.get('/static/redux/tab_escalation_analytics.jsx').text


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
    """Every /static/redux/* asset must be at or past the floor that
    accompanies registering the two new esc_flow_* files.

    FLOOR only: whether the versions are UNIFORM is asserted once,
    canonically, in test_index_html.py. It was replicated byte-identically
    across five test modules, each with its own stale monotonic floor, so a
    partial bump failed five tests with five different floors in the
    message. `min(...)` is the strictly stronger floor claim under mixed
    versions anyway — the OLDEST asset is the one that serves stale code.
    """
    versions = {int(v) for v in re.findall(r'/static/redux/[^"?]+\?v=(\d+)', index_html_body)}
    assert versions, (
        'index.html carries no /static/redux/*?v=<n> asset tags at all — the '
        'cache-buster convention has been dropped or the URLs were rewritten.'
    )
    assert min(versions) >= 33, (
        f'the oldest index.html cache-buster version is {min(versions)}, '
        'expected >= 33 (the floor esc_flow_* registration landed at).'
    )


# ---------------------------------------------------------------------------
# step-9: tab_escalation_analytics.jsx mounts LifecycleFlowDiagram inside the
# esc-flow-slot seam.
# ---------------------------------------------------------------------------


def test_tab_analytics_destructures_lifecycle_flow_diagram(tab_analytics_jsx_body: str) -> None:
    """tab_escalation_analytics.jsx must destructure LifecycleFlowDiagram from
    window.DF_ESC_FLOW at the top level, alongside its other window.DF_*
    destructures.
    """
    assert re.search(
        r'const\s*\{[^}]*LifecycleFlowDiagram[^}]*\}\s*=\s*window\.DF_ESC_FLOW\b',
        tab_analytics_jsx_body,
    ), (
        'tab_escalation_analytics.jsx does not destructure LifecycleFlowDiagram '
        'from window.DF_ESC_FLOW — add `const { LifecycleFlowDiagram } = '
        'window.DF_ESC_FLOW || {};` near the other window.DF_* destructures.'
    )


def test_workflow_panel_mounts_lifecycle_flow_diagram_in_slot(tab_analytics_jsx_body: str) -> None:
    """WorkflowPanel must render <LifecycleFlowDiagram flowDaily={flowDaily} />
    inside the esc-flow-slot seam, and must still preserve the esc-flow-slot
    class + flow_daily reference (δ's test_tab_analytics_workflow_panel
    invariants).
    """
    body = tab_analytics_jsx_body
    panel_body = _extract_function_body(body, 'WorkflowPanel')
    assert panel_body, 'Could not locate the WorkflowPanel( function body.'

    # (c) regression guard — δ's seam markers must survive this edit.
    assert 'esc-flow-slot' in panel_body, (
        'WorkflowPanel no longer contains the esc-flow-slot class — this '
        'seam must be preserved so δ\'s test_tab_analytics_workflow_panel stays green.'
    )
    assert 'flow_daily' in panel_body, (
        'WorkflowPanel no longer references flow_daily — the windowed '
        'flowDaily derivation from workflow.flow_daily must be preserved.'
    )

    # (b) LifecycleFlowDiagram is rendered, fed the already-windowed flowDaily.
    assert re.search(r'<LifecycleFlowDiagram\b', panel_body), (
        'WorkflowPanel does not render <LifecycleFlowDiagram — mount it '
        'inside the esc-flow-slot div.'
    )
    assert re.search(r'<LifecycleFlowDiagram[^>]*\bflowDaily=\{\s*flowDaily\s*\}', panel_body), (
        'WorkflowPanel does not pass flowDaily={flowDaily} to LifecycleFlowDiagram.'
    )

    # LifecycleFlowDiagram must be mounted INSIDE the esc-flow-slot div, not
    # merely somewhere else in WorkflowPanel.
    slot_start = panel_body.index('esc-flow-slot')
    slot_region = panel_body[slot_start : slot_start + 400]
    assert '<LifecycleFlowDiagram' in slot_region, (
        'LifecycleFlowDiagram is not rendered inside the esc-flow-slot container.'
    )
