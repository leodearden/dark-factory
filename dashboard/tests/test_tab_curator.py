"""Wiring tests for the Curator tab UI.

Tests parse JSX/JS source files as text and assert structural contracts
(export names, endpoint registration, rail entry, tab registration, charts
export). Follows the idiom established in test_index_html.py.
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
# Helper: extract the CURATOR_STATE: { ... } seed block from data.js
# ---------------------------------------------------------------------------


def _extract_curator_state_block(data_js: str) -> str:
    """Return the body of the ``CURATOR_STATE: { ... }`` seed object, braces included.

    Locates ``CURATOR_STATE:`` followed by ``{`` (allowing arbitrary whitespace),
    then walks forward counting ``{``/``}`` to find the matching close brace.
    This is brace-aware: a simple regex ``[^}]*`` would stop at the first nested
    ``}`` (e.g. the one closing ``latency_spark: { ... }``) and miss later keys.
    Returns the empty string if no CURATOR_STATE block is found.

    Note: the brace-depth walk does not skip ``{``/``}`` inside JS string
    literals.  This is acceptable because the data.js seed block uses simple
    numeric/array values and does not embed brace characters inside quoted
    strings.
    """
    m = re.search(r'CURATOR_STATE\s*:\s*\{', data_js)
    if m is None:
        return ''
    start = m.end() - 1  # index of the opening `{`
    depth = 0
    for i in range(start, len(data_js)):
        c = data_js[i]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return data_js[start : i + 1]
    return ''


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
        'add it to the unwindowed entries in endpointsFor.'
    )
    assert "'CURATOR_STATE'" in data_js_body or '"CURATOR_STATE"' in data_js_body, (
        "data.js does not reference 'CURATOR_STATE' — add it as the mapped key "
        "for '/api/v2/dashboard/curator' in endpointsFor."
    )
    seed_block = _extract_curator_state_block(data_js_body)
    assert seed_block, (
        'data.js does not contain a `CURATOR_STATE: { ... }` seed block — '
        'add the initializer to the window.DF_DATA assignment so applyKey has '
        'something to replace on each poll.'
    )
    # Scoped check: assert each key is present as a property name inside the
    # CURATOR_STATE block only.  A bare-substring check against the full data.js
    # body would pass even if this block were deleted (e.g. 'pending' appears in
    # MEMORY_STATUS and BURNDOWN; 'state' appears in MEMORY_STATUS.burst_state).
    for key in ('pending', 'latency_spark', 'pending_spark', 'capped_spark', 'state'):
        assert re.search(rf'\b{key}\s*:', seed_block), (
            f"CURATOR_STATE seed missing key '{key}:' — "
            f'add it to the window.DF_DATA initializer in data.js.'
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
        'inside a fetch() call — add the cancel POST handler.'
    )


# ---------------------------------------------------------------------------
# step-1300-6 test: tab_curator.jsx references pending_spark
# ---------------------------------------------------------------------------


def test_tab_curator_jsx_renders_pending_spark(tab_curator_jsx_body: str) -> None:
    """tab_curator.jsx must satisfy two independent contracts for pending_spark.

    1. **Destructure binding**: ``pending_spark`` appears as a destructured
       binding (followed by ',' or '}') in CuratorTab, confirming the metric is
       bound into local scope.  A scoped regex avoids false positives from
       comments or unused identifiers.  Mirrors the rf'\\b{key}\\s*:' pattern
       used in test_data_js_registers_curator_endpoint.

    2. **JSX rendering**: ``<PendingPanel pending_spark=`` appears in the body,
       confirming PendingPanel is actually invoked with the pending_spark prop.
       Mutation criterion (reviewer esc-1300-40 #1): deleting
       ``<PendingPanel pending_spark={pending_spark}/>`` from tab_curator.jsx
       causes the second assertion to fail even though the destructure binding
       remains in place.
    """
    assert re.search(r'\bpending_spark\b\s*[,}]', tab_curator_jsx_body), (
        "tab_curator.jsx does not destructure 'pending_spark' in CuratorTab — "
        'add pending_spark to the destructuring of cs and pass it to PendingPanel.'
    )
    assert re.search(r'<PendingPanel\b[^>]*\bpending_spark\s*=', tab_curator_jsx_body), (
        'tab_curator.jsx does not render `<PendingPanel pending_spark={...}/>` — '
        'even though pending_spark is destructured, PendingPanel is not being '
        'invoked with it. Add or restore the `<PendingPanel pending_spark={pending_spark} />` '
        "element inside CuratorTab's JSX."
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
        '— add the curator item to the Rail items array.'
    )


# ---------------------------------------------------------------------------
# step-11 test: app.jsx wires the curator tab
# ---------------------------------------------------------------------------


def test_app_jsx_wires_curator_tab(app_jsx_body: str) -> None:
    """app.jsx must destructure CuratorTab, add it to tabs[], and handle it in renderTab.

    Asserts the three structural wiring contracts:
    (a) CuratorTab destructured from window.DF_CURATOR.
    (b) tabs[] contains an entry with id 'curator'.
    (c) renderTab switch has a `case 'curator':` branch.

    Cosmetic fields (display label) are omitted — they can be renamed without
    breaking the routing.
    """
    assert re.search(r'const\s*\{[^}]*CuratorTab[^}]*\}\s*=\s*window\.DF_CURATOR', app_jsx_body), (
        'app.jsx does not destructure CuratorTab from window.DF_CURATOR — add '
        '`const { CuratorTab } = window.DF_CURATOR;` near the other destructures.'
    )
    assert "id: 'curator'" in app_jsx_body, (
        "app.jsx tabs array does not contain an entry with `id: 'curator'` — "
        "add `{ id: 'curator', ... }` to the tabs array."
    )
    assert "case 'curator':" in app_jsx_body, (
        "app.jsx renderTab switch does not have `case 'curator':` — add the "
        'case branch to render <CuratorTab projectFilter={projects} />.'
    )


# ---------------------------------------------------------------------------
# meta-test: _extract_curator_state_block scoping
# ---------------------------------------------------------------------------


def test_curator_state_key_check_is_scoped_to_seed_block() -> None:
    """The scoped key check must detect a missing CURATOR_STATE block.

    This meta-test exercises _extract_curator_state_block with two synthetic
    data.js bodies:

    (a) fake_without_curator_state — contains all four key names in *other*
        top-level objects but NO CURATOR_STATE block.  The bare-substring check
        that the original test used would pass on this input; the scoped check
        must fail.

    (b) fake_with_curator_state — includes a proper CURATOR_STATE: { ... }
        block with nested objects so the brace-counted extractor is exercised
        past the first inner `}`.
    """
    fake_without_curator_state = (
        'window.DF_DATA = {\n'
        '  MEMORY_STATUS: { queue: { counts: { pending: 0 } }, burst_state: [] },\n'
        "  BURNDOWN: { pending: [], state: 'idle' },\n"
        '  LATENCY: { latency_spark: [], capped_spark: [] },\n'
        '};\n'
    )

    fake_with_curator_state = (
        'window.DF_DATA = {\n'
        '  MEMORY_STATUS: { queue: { counts: { pending: 0 } }, burst_state: [] },\n'
        "  BURNDOWN: { pending: [], state: 'idle' },\n"
        '  LATENCY: { latency_spark: [], capped_spark: [] },\n'
        '  CURATOR_STATE: {\n'
        '    pending: [],\n'
        '    latency_spark: { labels: [] },\n'
        '    capped_spark: { labels: [] },\n'
        '    state: { capped_now: 0 },\n'
        '  },\n'
        '};\n'
    )

    # Helper must return '' when CURATOR_STATE block is absent.
    assert _extract_curator_state_block(fake_without_curator_state) == '', (
        "_extract_curator_state_block should return '' when no CURATOR_STATE "
        'block is present — the bare-substring match is defeated by other keys.'
    )

    # Helper must return the full block (including nested braces) when present.
    block = _extract_curator_state_block(fake_with_curator_state)
    assert block, (
        "_extract_curator_state_block returned '' for input that contains a "
        'CURATOR_STATE block — check the regex anchor and brace-depth walk.'
    )
    for key in ('pending', 'latency_spark', 'capped_spark', 'state'):
        assert re.search(rf'\b{key}\s*:', block), (
            f"Key '{key}:' not found inside the extracted CURATOR_STATE block — "
            f"the brace-depth walk may have stopped at a nested '}}' before "
            f'reaching this key.'
        )


# ---------------------------------------------------------------------------
# step-9 (task-1510): data.js seed and tab_curator.jsx accounts_summary
# ---------------------------------------------------------------------------


def test_data_js_seed_carries_accounts_summary(data_js_body: str) -> None:
    """CURATOR_STATE seed in data.js must include accounts_summary.

    Uses the brace-aware _extract_curator_state_block helper to scope the
    check to the CURATOR_STATE block only.

    RED baseline: accounts_summary not yet added to the seed block.
    Msg: CURATOR_STATE.state seed missing accounts_summary — React seed falls
    back to undefined and the curator pill throws at render.
    """
    seed_block = _extract_curator_state_block(data_js_body)
    assert seed_block, (
        'data.js does not contain a CURATOR_STATE: { ... } seed block — '
        'prerequisite for this test (see test_data_js_registers_curator_endpoint).'
    )
    assert re.search(r'\baccounts_summary\s*:', seed_block), (
        'CURATOR_STATE.state seed missing accounts_summary: — '
        'React seed falls back to undefined and the curator pill throws at render. '
        "Add `accounts_summary: { total: 0, capped: 0, available: 0, capped_accounts: [] }` "
        'to the state object inside the CURATOR_STATE seed in data.js.'
    )


def test_tab_curator_jsx_renders_accounts_summary(tab_curator_jsx_body: str) -> None:
    """tab_curator.jsx must reference accounts_summary.available in the state pill.

    Two independent contracts:
    (1) accounts_summary appears in the state destructure (followed by comma or brace).
    (2) The word 'available' appears within ~400 chars after 'accounts_summary'
        (scoped proximity check guards against a stale reference in a different part
        of the file).

    RED baseline: data.js and tab_curator.jsx not yet updated.
    Msg: tab_curator.jsx state pill does not reference accounts_summary.available —
    UI still shows only Capped/Open without a count.
    """
    assert re.search(r'\baccounts_summary\b\s*[,}=]', tab_curator_jsx_body), (
        'tab_curator.jsx state pill does not destructure or reference accounts_summary — '
        'add accounts_summary to the const { ... } = state destructure in CuratorTab.'
    )
    assert re.search(r'accounts_summary[\s\S]{0,400}available', tab_curator_jsx_body), (
        'tab_curator.jsx does not reference available within 400 chars of accounts_summary — '
        'UI still shows only Capped/Open without a count. '
        'Add the `${accounts_summary.available}/${accounts_summary.total} available` '
        'label to the badge ternary in the state pill.'
    )
