"""Wiring tests for the Curator tab UI.

Tests parse JSX/JS source files as text and assert structural contracts
(export names, endpoint registration, rail entry, tab registration, charts
export). Follows the idiom established in test_index_html.py.
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient


@pytest.fixture(scope='module')
def _client():
    from dashboard.app import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope='module')
def charts_jsx_body(_client):
    return _client.get('/static/redux/charts.jsx').text


@pytest.fixture(scope='module')
def data_js_body(_client):
    return _client.get('/static/redux/data.js').text


@pytest.fixture(scope='module')
def tab_curator_jsx_body(_client):
    return _client.get('/static/redux/tab_curator.jsx').text


@pytest.fixture(scope='module')
def tabs_jsx_body(_client):
    return _client.get('/static/redux/tabs.jsx').text


@pytest.fixture(scope='module')
def shell_jsx_body(_client):
    return _client.get('/static/redux/shell.jsx').text


@pytest.fixture(scope='module')
def app_jsx_body(_client):
    return _client.get('/static/redux/app.jsx').text


# ---------------------------------------------------------------------------
# step-1 test: charts.jsx exports StepSpark
# ---------------------------------------------------------------------------

def test_charts_jsx_exports_step_spark(charts_jsx_body: str) -> None:
    """charts.jsx must define a StepSpark function and register it on window.DF_CHARTS.

    StepSpark uses horizontal-then-vertical step interpolation (distinct from
    Sparkline's smooth diagonal lines) and is consumed by CuratorTab to render
    the capped_spark time-series.
    """
    assert 'function StepSpark(' in charts_jsx_body, (
        'charts.jsx does not define a `function StepSpark(` — add StepSpark '
        'next to Sparkline with horizontal-then-vertical SVG path segments.'
    )
    assert 'StepSpark' in charts_jsx_body.split('window.DF_CHARTS')[1], (
        'StepSpark is not registered in the `window.DF_CHARTS = { ... }` export '
        '— add StepSpark to the export object at the bottom of charts.jsx.'
    )


# ---------------------------------------------------------------------------
# step-3 test: data.js registers the curator endpoint
# ---------------------------------------------------------------------------

def test_data_js_registers_curator_endpoint(data_js_body: str) -> None:
    """data.js must register /api/v2/dashboard/curator mapped to ['CURATOR_STATE'].

    The entry must be present in the static (unwindowed) section of endpointsFor,
    and the empty-defaults block must initialise CURATOR_STATE with the keys
    expected by the shape_curator envelope from redux_api.py.
    """
    assert '/api/v2/dashboard/curator' in data_js_body, (
        "data.js does not contain the literal URL '/api/v2/dashboard/curator' — "
        "add it to the unwindowed entries in endpointsFor."
    )
    assert "'CURATOR_STATE'" in data_js_body or '"CURATOR_STATE"' in data_js_body, (
        "data.js does not reference 'CURATOR_STATE' — add it as the mapped key "
        "for '/api/v2/dashboard/curator' in endpointsFor."
    )
    for key in ('pending', 'latency_spark', 'capped_spark', 'state'):
        assert key in data_js_body, (
            f"data.js CURATOR_STATE default does not include key '{key}' — "
            f"add it to the CURATOR_STATE seed in the window.DF_DATA initializer."
        )


# ---------------------------------------------------------------------------
# step-5 test: tab_curator.jsx is served and exports CuratorTab
# ---------------------------------------------------------------------------

def test_tab_curator_jsx_served_and_exports_component(_client) -> None:
    """GET /static/redux/tab_curator.jsx returns 200 with the expected wiring."""
    resp = _client.get('/static/redux/tab_curator.jsx')
    assert resp.status_code == 200, (
        f'Expected 200 for /static/redux/tab_curator.jsx, got {resp.status_code}'
    )
    body = resp.text
    assert 'function CuratorTab(' in body, (
        'tab_curator.jsx does not define `function CuratorTab(` — the component '
        'must be declared as a named function for the export to work.'
    )
    assert 'window.DF_CURATOR' in body, (
        'tab_curator.jsx does not set window.DF_CURATOR — add '
        '`window.DF_CURATOR = { CuratorTab }` at the bottom of the file.'
    )
    assert '/api/v2/dashboard/curator/cancel' in body, (
        "tab_curator.jsx does not reference '/api/v2/dashboard/curator/cancel' "
        "inside a fetch() call — add the cancel POST handler."
    )


# ---------------------------------------------------------------------------
# step-7 test: tabs.jsx exports ChipList for cross-tab reuse
# ---------------------------------------------------------------------------

def test_tabs_jsx_exports_chiplist_for_reuse(tabs_jsx_body: str) -> None:
    """tabs.jsx must include ChipList in its window.DF_TABS export.

    tab_curator.jsx reuses ChipList (accessed as window.DF_TABS.ChipList)
    to render per-row file arrays as collapsible chips. Without this export,
    tab_curator.jsx cannot access ChipList without reimplementing it.
    """
    # Find the window.DF_TABS = { ... } line
    assert 'window.DF_TABS' in tabs_jsx_body, (
        'tabs.jsx does not contain window.DF_TABS — the export line is missing.'
    )
    export_section = tabs_jsx_body.split('window.DF_TABS')[1]
    assert 'ChipList' in export_section, (
        'ChipList is not exported via window.DF_TABS — add ChipList to the '
        'window.DF_TABS = { ... } export at the bottom of tabs.jsx so that '
        'tab_curator.jsx can access it via window.DF_TABS.ChipList.'
    )


# ---------------------------------------------------------------------------
# step-9 test: shell.jsx registers curator glyph and rail entry
# ---------------------------------------------------------------------------

def test_shell_jsx_registers_curator_glyph_and_rail_entry(shell_jsx_body: str) -> None:
    """shell.jsx must include a Rail item with id 'curator'.

    Asserts only the routing-relevant id field.  Cosmetic fields (label,
    glyph key) are intentionally omitted — they can be renamed without
    breaking the wiring.
    """
    assert "id: 'curator'" in shell_jsx_body, (
        "shell.jsx Rail items array does not contain an entry with `id: 'curator'` "
        "— add the curator item to the Rail items array."
    )


# ---------------------------------------------------------------------------
# step-11 test: app.jsx wires the curator tab
# ---------------------------------------------------------------------------

def test_app_jsx_wires_curator_tab(app_jsx_body: str) -> None:
    """app.jsx must destructure CuratorTab, add it to tabs[], and render it.

    (a) CuratorTab must be destructured from window.DF_CURATOR.
    (b) The tabs array must contain an entry with id 'curator' and label 'Curator'.
    (c) The renderTab switch must have a `case 'curator':` returning <CuratorTab .../>.
    """
    import re

    assert re.search(r'const\s*\{[^}]*CuratorTab[^}]*\}\s*=\s*window\.DF_CURATOR', app_jsx_body), (
        "app.jsx does not destructure CuratorTab from window.DF_CURATOR — add "
        "`const { CuratorTab } = window.DF_CURATOR;` near the other destructures."
    )
    assert "id: 'curator'" in app_jsx_body, (
        "app.jsx tabs array does not contain an entry with `id: 'curator'` — "
        "add `{ id: 'curator', label: 'Curator' }` to the tabs array."
    )
    assert "label: 'Curator'" in app_jsx_body, (
        "app.jsx tabs array entry for curator must specify `label: 'Curator'`."
    )
    assert "case 'curator':" in app_jsx_body, (
        "app.jsx renderTab switch does not have `case 'curator':` — add the "
        "case branch to render <CuratorTab projectFilter={projects} />."
    )
    assert 'CuratorTab' in app_jsx_body.split("case 'curator':")[1].split('\n')[0] + \
           app_jsx_body.split("case 'curator':")[1].split('\n')[1], (
        "app.jsx `case 'curator':` does not render <CuratorTab ...> — add the "
        "return statement to the case branch."
    )
