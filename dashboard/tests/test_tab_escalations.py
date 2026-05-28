"""Wiring tests for the Escalations tab UI.

Tests parse JSX/JS/HTML source files as text and assert structural contracts
(endpoint registration, export names, rail entry, tab registration, load-order).
Follows the idiom established in test_tab_curator.py and test_index_html.py.
"""

from __future__ import annotations

import html.parser
import re

import pytest
from starlette.testclient import TestClient


# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def _client():
    from dashboard.app import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope='module')
def data_js_body(_client):
    return _client.get('/static/redux/data.js').text


@pytest.fixture(scope='module')
def tab_escalations_jsx_body(_client):
    return _client.get('/static/redux/tab_escalations.jsx').text


@pytest.fixture(scope='module')
def app_jsx_body(_client):
    return _client.get('/static/redux/app.jsx').text


@pytest.fixture(scope='module')
def shell_jsx_body(_client):
    return _client.get('/static/redux/shell.jsx').text


@pytest.fixture(scope='module')
def index_html_body(_client):
    return _client.get('/static/redux/index.html').text


# ---------------------------------------------------------------------------
# Helper: extract a named seed block from window.DF_DATA (brace-aware)
# ---------------------------------------------------------------------------


def _extract_df_data_block(src: str, key: str) -> str:
    """Return the body of the ``<key>: { ... }`` seed object, braces included.

    Locates ``<key>:`` followed by ``{`` (allowing arbitrary whitespace), then
    walks forward counting ``{``/``}`` to find the matching close brace.
    This is brace-aware: a simple regex ``[^}]*`` would stop at the first
    nested ``}`` and miss later keys.
    Returns the empty string if no matching block is found.

    Note: the brace-depth walk does not skip ``{``/``}`` inside JS string
    literals.  This is acceptable because the data.js seed block uses simple
    numeric/array values and does not embed brace characters inside quoted
    strings.
    """
    m = re.search(rf'{re.escape(key)}\s*:\s*\{{', src)
    if m is None:
        return ''
    start = m.end() - 1  # index of the opening `{`
    depth = 0
    for i in range(start, len(src)):
        c = src[i]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return src[start : i + 1]
    return ''


# ---------------------------------------------------------------------------
# Load-order helpers (copied from test_index_html.py)
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
# step-1 test: data.js registers the escalations endpoint
# ---------------------------------------------------------------------------


def test_data_js_registers_escalations_endpoint(data_js_body: str) -> None:
    """data.js must register /api/v2/dashboard/escalations mapped to ['ESCALATIONS'].

    The entry must be present in the static (unwindowed) section of endpointsFor,
    and the empty-defaults block must initialise ESCALATIONS with the shape
    expected by redux_api.shape_escalations: {subsections, summary:{by_level,by_status}}.
    """
    assert '/api/v2/dashboard/escalations' in data_js_body, (
        "data.js does not contain the literal URL '/api/v2/dashboard/escalations' — "
        'add it to the unwindowed entries in endpointsFor.'
    )
    assert "'ESCALATIONS'" in data_js_body or '"ESCALATIONS"' in data_js_body, (
        "data.js does not reference 'ESCALATIONS' — add it as the mapped key "
        "for '/api/v2/dashboard/escalations' in endpointsFor."
    )
    seed_block = _extract_df_data_block(data_js_body, 'ESCALATIONS')
    assert seed_block, (
        'data.js does not contain an `ESCALATIONS: { ... }` seed block — '
        'add the initializer to the window.DF_DATA assignment so applyKey has '
        'something to replace on each poll.'
    )
    # Scoped checks: assert top-level keys are present inside the ESCALATIONS block.
    assert re.search(r'\bsubsections\s*:', seed_block), (
        "ESCALATIONS seed missing key 'subsections:' — "
        'add it to the window.DF_DATA ESCALATIONS initializer in data.js.'
    )
    assert re.search(r'\bsummary\s*:', seed_block), (
        "ESCALATIONS seed missing key 'summary:' — "
        'add it to the window.DF_DATA ESCALATIONS initializer in data.js.'
    )
    # summary sub-block: check by_level and by_status are nested under summary.
    summary_block = _extract_df_data_block(seed_block, 'summary')
    assert summary_block, (
        'ESCALATIONS seed summary block not found via _extract_df_data_block — '
        'ensure summary is an object, not a scalar.'
    )
    assert re.search(r'\bby_level\s*:', summary_block), (
        "ESCALATIONS.summary seed missing key 'by_level:' — "
        'add it nested under summary in the ESCALATIONS seed block.'
    )
    assert re.search(r'\bby_status\s*:', summary_block), (
        "ESCALATIONS.summary seed missing key 'by_status:' — "
        'add it nested under summary in the ESCALATIONS seed block.'
    )


# ---------------------------------------------------------------------------
# step-3 test: tab_escalations.jsx is served and exports EscalationsTab
# ---------------------------------------------------------------------------


def test_tab_escalations_jsx_served_and_exports_component(_client) -> None:
    """GET /static/redux/tab_escalations.jsx returns 200 with the expected wiring.

    Asserts:
    (a) 200 HTTP status.
    (b) function EscalationsTab( is declared.
    (c) exports ADDITIVELY via window.DF_TABS.EscalationsTab = (not window.DF_TABS = {).
    (d) reads window.DF_DATA.ESCALATIONS (or an aliased DF.ESCALATIONS).
    (e) renders <ProjectGroup.
    (f) fold state is persisted via useOpenSet( referencing 'df.open.esc'.
    """
    resp = _client.get('/static/redux/tab_escalations.jsx')
    assert resp.status_code == 200, (
        f'Expected 200 for /static/redux/tab_escalations.jsx, got {resp.status_code}'
    )
    body = resp.text
    assert 'function EscalationsTab(' in body, (
        'tab_escalations.jsx does not define `function EscalationsTab(` — the component '
        'must be declared as a named function for the export to work.'
    )
    # Additive export — must NOT clobber window.DF_TABS = {...} and must assign EscalationsTab
    assert 'window.DF_TABS.EscalationsTab' in body, (
        'tab_escalations.jsx does not set window.DF_TABS.EscalationsTab — add '
        '`window.DF_TABS.EscalationsTab = EscalationsTab;` at the bottom of the file '
        'to export additively without clobbering the existing window.DF_TABS object.'
    )
    # Reads ESCALATIONS data
    assert 'ESCALATIONS' in body, (
        'tab_escalations.jsx does not reference ESCALATIONS — it should read '
        'window.DF_DATA.ESCALATIONS (or an alias like DF.ESCALATIONS) for its data source.'
    )
    # Renders ProjectGroup for subsection folding
    assert '<ProjectGroup' in body, (
        'tab_escalations.jsx does not render <ProjectGroup — each subsection must be '
        'wrapped in a ProjectGroup from window.DF_SHELL for foldable sections.'
    )
    # Fold state persisted with the correct key
    assert "useOpenSet(" in body, (
        "tab_escalations.jsx does not call useOpenSet( — add the local copy of "
        "useOpenSet from tabs.jsx and call it with subsection ids and 'df.open.esc'."
    )
    assert "'df.open.esc'" in body, (
        "tab_escalations.jsx does not reference the localStorage key 'df.open.esc' — "
        "pass it as the storageKey argument to useOpenSet so fold state is persisted."
    )


# ---------------------------------------------------------------------------
# step-5 test: app.jsx wires the escalations tab
# ---------------------------------------------------------------------------


def test_app_jsx_wires_escalations_tab(app_jsx_body: str) -> None:
    """app.jsx must destructure EscalationsTab from window.DF_TABS, add an 'esc'
    tab entry, handle it in renderTab, include it in railCounts, and configure toolbarConfig.

    Asserts five structural wiring contracts:
    (a) EscalationsTab is destructured from window.DF_TABS.
    (b) tabs[] contains an entry with id 'esc'.
    (c) renderTab switch has a `case 'esc':` branch.
    (d) railCounts includes an `esc:` key that references ESCALATIONS.pending.
    (e) toolbarConfig has an `esc:` entry.
    """
    # (a) EscalationsTab destructured from window.DF_TABS
    assert re.search(
        r'const\s*\{[^}]*EscalationsTab[^}]*\}\s*=\s*window\.DF_TABS', app_jsx_body
    ) or 'EscalationsTab' in (app_jsx_body.split('window.DF_TABS')[1] if 'window.DF_TABS' in app_jsx_body else ''), (
        'app.jsx does not destructure EscalationsTab from window.DF_TABS — add '
        '`EscalationsTab` to the `const { ... } = window.DF_TABS;` destructure.'
    )
    # (b) tabs[] entry
    assert "id: 'esc'" in app_jsx_body, (
        "app.jsx tabs array does not contain an entry with `id: 'esc'` — "
        "add `{ id: 'esc', ... }` to the tabs array."
    )
    # (c) renderTab switch case
    assert "case 'esc':" in app_jsx_body, (
        "app.jsx renderTab switch does not have `case 'esc':` — add the case "
        'branch to render <EscalationsTab projectFilter={projects} />.'
    )
    # (d) railCounts esc key references ESCALATIONS and pending
    assert re.search(r'esc\s*:', app_jsx_body), (
        "app.jsx railCounts does not contain an `esc:` key — add "
        "`esc: DD.ESCALATIONS?.summary?.by_status?.pending ?? 0` to railCounts."
    )
    assert 'ESCALATIONS' in app_jsx_body, (
        'app.jsx does not reference ESCALATIONS — add `esc: DD.ESCALATIONS?.summary?.by_status?.pending ?? 0` '
        'to railCounts so the escalations count shows in the rail.'
    )
    assert 'pending' in app_jsx_body.split('ESCALATIONS')[1][:200], (
        "app.jsx railCounts esc entry does not reference 'pending' near ESCALATIONS — "
        'ensure esc count reads by_status.pending from ESCALATIONS.'
    )
    # (e) toolbarConfig esc entry
    # Check that 'esc:' appears in a config-like context (toolbarConfig block)
    esc_in_toolbar = False
    parts = app_jsx_body.split('toolbarConfig')
    if len(parts) > 1:
        esc_in_toolbar = "esc:" in parts[1] or "'esc'" in parts[1]
    assert esc_in_toolbar, (
        "app.jsx toolbarConfig does not have an 'esc:' entry — add "
        "`esc: { showWindow: false, showAgents: false, search: false }` to toolbarConfig."
    )


# ---------------------------------------------------------------------------
# step-7 test: shell.jsx registers escalations glyph and rail entry
# ---------------------------------------------------------------------------


def test_shell_jsx_registers_escalations_glyph_and_rail_entry(shell_jsx_body: str) -> None:
    """shell.jsx must include a Rail item with id 'esc' and a Glyph case 'esc'.

    Asserts only the routing-relevant id and glyph-key wiring.  Cosmetic
    fields (label, SVG path data) are intentionally omitted — they can be
    renamed without breaking the routing.
    """
    assert "id: 'esc'" in shell_jsx_body, (
        "shell.jsx Rail items array does not contain an entry with `id: 'esc'` "
        '— add the escalations item to the Rail items array.'
    )
    assert "case 'esc':" in shell_jsx_body, (
        "shell.jsx Glyph switch does not have a `case 'esc':` branch — "
        'add an esc case returning a simple stroke SVG.'
    )


# ---------------------------------------------------------------------------
# step-9 test: index.html registers tab_escalations.jsx in correct load order
# ---------------------------------------------------------------------------


def test_index_html_registers_tab_escalations_load_order(index_html_body: str) -> None:
    """index.html must include tab_escalations.jsx, loaded AFTER data.js, shell.jsx,
    and tabs.jsx and BEFORE app.jsx; must be a classic synchronous script (no
    defer/async/type=module); and all /static/redux/*?v= busters must share a
    single version >= 10.

    Checks:
    (a) tab_escalations.jsx script tag exists.
    (b) Loads after data.js.
    (c) Loads after shell.jsx.
    (d) Loads after tabs.jsx.
    (e) Loads before app.jsx.
    (f) Not deferred/async/module.
    (g) All /static/redux/ v= cache-busters share one version >= 10.
    """
    _TAB_ESC_PREFIX = '/static/redux/tab_escalations.jsx'

    # (a) tab_escalations.jsx script tag must exist
    result = _find_script_position(index_html_body, _TAB_ESC_PREFIX)
    assert result is not None, (
        f'No <script src="{_TAB_ESC_PREFIX}..."> tag found in index.html — '
        'add it after tabs.jsx and before app.jsx.'
    )
    _, esc_attrs = result

    # (f) Must be a classic synchronous script
    assert 'defer' not in esc_attrs, (
        'tab_escalations.jsx script tag has defer= — remove it; classic synchronous '
        'scripts are required for Babel-standalone transpilation.'
    )
    assert 'async' not in esc_attrs, (
        'tab_escalations.jsx script tag has async= — remove it.'
    )
    assert (esc_attrs.get('type') or '').lower() in ('text/babel', ''), (
        'tab_escalations.jsx script must have type="text/babel" (or no type) — '
        f'got {esc_attrs.get("type")!r}.'
    )

    # (b) Loads after data.js
    _assert_script_loads_before(
        index_html_body,
        '/static/redux/data.js',
        _TAB_ESC_PREFIX,
        'data.js',
        'tab_escalations.jsx',
        'data.js must load before tab_escalations.jsx so window.DF_DATA is seeded.',
    )

    # (c) Loads after shell.jsx
    _assert_script_loads_before(
        index_html_body,
        '/static/redux/shell.jsx',
        _TAB_ESC_PREFIX,
        'shell.jsx',
        'tab_escalations.jsx',
        'shell.jsx must load before tab_escalations.jsx so window.DF_SHELL is available.',
    )

    # (d) Loads after tabs.jsx
    _assert_script_loads_before(
        index_html_body,
        '/static/redux/tabs.jsx',
        _TAB_ESC_PREFIX,
        'tabs.jsx',
        'tab_escalations.jsx',
        'tabs.jsx must load before tab_escalations.jsx so window.DF_TABS is available to mutate.',
    )

    # (e) Loads before app.jsx
    _assert_script_loads_before(
        index_html_body,
        _TAB_ESC_PREFIX,
        '/static/redux/app.jsx',
        'tab_escalations.jsx',
        'app.jsx',
        'tab_escalations.jsx must load before app.jsx so EscalationsTab is set on window.DF_TABS.',
    )

    # (g) All /static/redux/ v= cache-busters share one version >= 10
    import re as _re
    versions = set(_re.findall(r'/static/redux/[^"?]+\?v=(\d+)', index_html_body))
    assert len(versions) == 1, (
        f'index.html has mixed /static/redux/?v= cache-buster versions: {sorted(versions)} — '
        'bump all of them uniformly to the same value.'
    )
    v = int(next(iter(versions)))
    assert v >= 10, (
        f'index.html cache-buster version is {v}, expected >= 10.'
    )


# ---------------------------------------------------------------------------
# step-11 test: tab_escalations.jsx has sort state and expand/collapse-all
# ---------------------------------------------------------------------------


def test_tab_escalations_task_sort_and_expand_collapse(tab_escalations_jsx_body: str) -> None:
    """tab_escalations.jsx must have sort state persisted with 'df.esc.sort',
    a numeric-aware task_id comparator using Number() with timestamp as secondary,
    a direction toggle that flips between 'asc'/'desc', and an expand/collapse-all
    control wired to useOpenSet's setAll.
    """
    # Sort state persisted with the correct key
    assert "'df.esc.sort'" in tab_escalations_jsx_body, (
        "tab_escalations.jsx does not call usePersistedState with 'df.esc.sort' — "
        "add `const [sort, setSort] = usePersistedState('df.esc.sort', ...)` for persisted sort."
    )
    # Numeric-aware comparator uses Number()
    assert 'Number(' in tab_escalations_jsx_body, (
        "tab_escalations.jsx sort comparator does not use Number( for numeric task_id — "
        'add `Number(a.task_id)` / `Number(b.task_id)` for numeric-aware sort.'
    )
    # Secondary sort key: timestamp
    assert 'timestamp' in tab_escalations_jsx_body, (
        "tab_escalations.jsx does not reference 'timestamp' in a sort/compare context — "
        'add timestamp as a secondary sort key in the comparator.'
    )
    # Direction toggle flips between 'asc' and 'desc'
    assert "'asc'" in tab_escalations_jsx_body and "'desc'" in tab_escalations_jsx_body, (
        "tab_escalations.jsx does not reference both 'asc' and 'desc' — "
        "add a direction toggle that flips sort.dir between 'asc' and 'desc'."
    )
    # Expand/collapse-all via setAll
    assert 'setAll' in tab_escalations_jsx_body, (
        "tab_escalations.jsx does not use setAll (from useOpenSet) for expand/collapse-all — "
        'wire GroupAllToggle or a button to the setAll function returned by useOpenSet.'
    )
