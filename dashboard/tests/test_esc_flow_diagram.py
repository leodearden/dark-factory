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

This file used to carry its own copies of `_client` and
`_extract_function_body`, per what was then this suite's established
copy-not-import convention for cross-file test helpers.  Task 3549 retired
that convention: nine modules held a copy, under two names covering four
distinct implementations, so a fix to any of them had to be applied nine
times or not at all — and the one worth making (scoping to a NESTED
declaration) went unmade for exactly that reason.  `_client` is now the
module-scoped fixture in conftest.py and `extract_function_body` is imported
from `_dashboard_helpers`; test_jsx_source_helpers.py owns their contract and
guards against the copies returning.

That retirement stopped at the two JSX-slicing helpers, DELIBERATELY, and the
remaining copies are tracked rather than forgotten — read the paragraph above
as scoped to those two, not as a claim that this file holds no copies.  The
script-order helpers below (ScriptTagCollector, find_script_position,
assert_script_loads_before) are still local copies from
test_tab_escalation_analytics.py, and are copied again in test_index_html.py,
test_tab_escalations.py and test_tab_memory_evals.py; `_extract_df_data_block`
stands in three modules; test_charts_axis_labels.py's `_extract_signature`
re-derives the paren-depth walk `extract_function_body` now owns.  Moving them
needs test_index_html.py, which task 3549 held no lock on, so it filed
ticket tkt_0RSN5VVGAVK7BQ8K9GX4PM2YBZ for the rest.
"""

from __future__ import annotations

import re

import pytest
from _dashboard_helpers import (
    assert_script_loads_before,
    extract_function_body,
    find_script_position,
)


@pytest.fixture(scope='module')
def esc_flow_diagram_jsx_response(_client):
    return _client.get('/static/redux/esc_flow_diagram.jsx')


@pytest.fixture(scope='module')
def esc_flow_diagram_jsx_body(esc_flow_diagram_jsx_response) -> str:
    return esc_flow_diagram_jsx_response.text


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

    result = find_script_position(index_html_body, _LAYOUT_PREFIX)
    assert result is not None, (
        f'No <script src="{_LAYOUT_PREFIX}..."> tag found in index.html — '
        'add it as a classic script (e.g. after runtime_format.js).'
    )

    # (a) esc_flow_layout.js loads before esc_flow_diagram.jsx
    assert_script_loads_before(
        index_html_body,
        _LAYOUT_PREFIX,
        _DIAGRAM_PREFIX,
        'esc_flow_layout.js',
        'esc_flow_diagram.jsx',
        'esc_flow_layout.js must load before esc_flow_diagram.jsx so '
        'window.DF_ESC_FLOW_LAYOUT is defined before the destructure runs.',
    )

    # (c) esc_flow_layout.js loads before tab_escalation_analytics.jsx too
    assert_script_loads_before(
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

    result = find_script_position(index_html_body, _DIAGRAM_PREFIX)
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
    assert_script_loads_before(
        index_html_body,
        _CHARTS_PREFIX,
        _DIAGRAM_PREFIX,
        'charts.jsx',
        'esc_flow_diagram.jsx',
        'charts.jsx must load before esc_flow_diagram.jsx so window.DF_CHARTS/C.PALETTE is available.',
    )

    # (b) Loads before tab_escalation_analytics.jsx
    assert_script_loads_before(
        index_html_body,
        _DIAGRAM_PREFIX,
        _TAB_ANALYTICS_PREFIX,
        'esc_flow_diagram.jsx',
        'tab_escalation_analytics.jsx',
        'esc_flow_diagram.jsx must load before tab_escalation_analytics.jsx so window.DF_ESC_FLOW is available to mount.',
    )


def test_index_html_cache_buster_not_reverted_below_esc_flow_floor(
    index_html_body: str,
) -> None:
    """Every /static/redux/* asset must stay at or past the floor that
    accompanied registering the two new esc_flow_* files.

    This is an ANTI-REVERT PIN, not a live bump check: index.html is far past
    33 today, so it fails only if someone rolls the cache-busters back below
    what esc_flow_* registration needed. Whether the versions are UNIFORM, and
    whether the newest bump landed, are both asserted in test_index_html.py.
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
    panel_body = extract_function_body(body, 'WorkflowPanel')

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
