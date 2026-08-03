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
def tab_escalations_jsx_code(tab_escalations_jsx_body):
    """`tab_escalations.jsx` with every comment stripped.

    Same rationale as the `tab_memory_evals_jsx_code` fixture in
    test_tab_memory_evals.py: this file carries explanatory prose naming most
    of the identifiers the render code also names, so a whole-file substring
    grep is satisfied by a MENTION — delete the render site, leave the comment,
    and the assertion stays green.

    Safe to strip naively: the source contains no `//` inside a string literal
    (no URLs) and no regex literals, so no `/`-bearing code is eaten.
    """
    return re.sub(r'/\*[\s\S]*?\*/|//[^\n]*', '', tab_escalations_jsx_body)


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
# Helper: extract a named JS/JSX function body (brace-aware)
# ---------------------------------------------------------------------------


def _extract_function_body(src: str, fn_name: str) -> str:
    """Return the body block of a ``function <fn_name>(`` declaration, braces included.

    Uses the same brace-depth walk as ``_extract_df_data_block``.  Only matches
    named ``function`` declarations — not arrow functions or class methods.
    Returns the empty string if the function is not found.

    Paren-depth walks past the parameter list before looking for the body's
    opening ``{`` — a destructured parameter (``function Foo({ a, b }) {``)
    contains its own ``{``/``}`` pair *inside* the parameter list, so naively
    taking the first ``{`` after the opening ``(`` would return just the
    destructuring pattern (e.g. ``{ a, b }``) instead of the function body.
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
    # Numeric-aware comparator uses Number() — scoped to the sortRows function body
    # so we don't get a false pass from Number() appearing in an unrelated context.
    sort_fn = _extract_function_body(tab_escalations_jsx_body, 'sortRows')
    assert sort_fn, (
        'tab_escalations.jsx does not define a `function sortRows(` — '
        'add a named sortRows function to sort escalation rows by numeric task_id.'
    )
    assert 'Number(' in sort_fn, (
        "sortRows function does not use Number( for numeric task_id conversion — "
        'add `Number(a.task_id)` / `Number(b.task_id)` for numeric-aware sort.'
    )
    # Secondary sort key: timestamp — also scoped to sortRows body
    assert 'timestamp' in sort_fn, (
        "sortRows function does not reference 'timestamp' — "
        'add timestamp as a secondary sort key in the comparator.'
    )
    # Direction toggle: 'asc'/'desc' must appear in a ternary flip expression,
    # e.g. `s.dir === 'asc' ? 'desc' : 'asc'` — asserts co-occurrence not just presence.
    assert re.search(r"'asc'\s*[?:]\s*'desc'|'desc'\s*[?:]\s*'asc'", tab_escalations_jsx_body), (
        "tab_escalations.jsx does not flip between 'asc' and 'desc' in a ternary — "
        "add a direction toggle like `s.dir === 'asc' ? 'desc' : 'asc'`."
    )
    # Expand/collapse-all via setAll
    assert 'setAll' in tab_escalations_jsx_body, (
        "tab_escalations.jsx does not use setAll (from useOpenSet) for expand/collapse-all — "
        'wire GroupAllToggle or a button to the setAll function returned by useOpenSet.'
    )


# ---------------------------------------------------------------------------
# step-13 test: tab_escalations.jsx has global level+status filter chips
# ---------------------------------------------------------------------------


def test_tab_escalations_global_filter_chips(tab_escalations_jsx_body: str) -> None:
    """tab_escalations.jsx must persist filter state with 'df.esc.filter', render
    level chips for 0/1/2 and status chips for pending/resolved/dismissed, and
    apply a filter predicate to rows of every subsection.
    """
    # Filter state persisted with the correct key
    assert "'df.esc.filter'" in tab_escalations_jsx_body, (
        "tab_escalations.jsx does not call usePersistedState with 'df.esc.filter' — "
        "add `const [filter, setFilter] = usePersistedState('df.esc.filter', ...)` for persisted filter."
    )
    # Level chip values 0/1/2 appear together as a mapped array, not just lone
    # digits elsewhere in the file.  The `[0, 1, 2].map(` idiom is the expected
    # pattern; a bare `0` in a timeout or index gives a false pass with a lone check.
    assert re.search(r'\[0,\s*1,\s*2\]', tab_escalations_jsx_body), (
        "tab_escalations.jsx does not render level chips as a mapped [0, 1, 2] array — "
        "add `[0, 1, 2].map(lv => ...)` for the level filter chips."
    )
    # Status chip values: 'pending', 'resolved', 'dismissed' appear in sequence
    # (i.e. in a single array literal) rather than scattered through the file.
    assert re.search(
        r"'pending'[^']*'resolved'[^']*'dismissed'",
        tab_escalations_jsx_body,
        re.DOTALL,
    ), (
        "tab_escalations.jsx does not list 'pending', 'resolved', 'dismissed' consecutively "
        "in an array — add `['pending', 'resolved', 'dismissed'].map(st => ...)` for the "
        "status filter chips."
    )
    # Filter predicate function references the filter state
    assert 'matchesFilter' in tab_escalations_jsx_body or (
        'filter.levels' in tab_escalations_jsx_body and 'filter.statuses' in tab_escalations_jsx_body
    ), (
        "tab_escalations.jsx does not apply a filter predicate referencing filter state to rows — "
        'add a matchesFilter(row) function using filter.levels[row.level] && filter.statuses[row.status].'
    )


# ---------------------------------------------------------------------------
# step-15 test: tab_escalations.jsx has a detail sidebar for selected rows
# ---------------------------------------------------------------------------


def test_tab_escalations_detail_sidebar(tab_escalations_jsx_body: str) -> None:
    """tab_escalations.jsx must maintain a selected-row state, render a detail
    sidebar with role="dialog" and a close button (onClose), and the sidebar
    must reference all required escalation record fields plus the linked task card.

    Asserts:
    (a) selected-row state maintained (useState for selected row, set from row onClick).
    (b) sidebar rendered with role="dialog" and an onClose handler / close button.
    (c) sidebar references escalation record fields: detail, suggested_action,
        level, category, severity, agent_role, workflow_state, worktree, resolution.
    (d) sidebar references linked task card fields: task.title, task.status, task.description.
    (e) renders the resolved project label.
    (f) task_unresolved fallback branch present.
    """
    body = tab_escalations_jsx_body

    # (a) Selected-row state set from a row onClick
    # useState for selected (uS(null) or useState(null)) and setSelected used in onClick
    assert 'setSelected' in body, (
        'tab_escalations.jsx does not maintain a selected-row state with `setSelected` — '
        'add `const [selected, setSelected] = uS(null)` and wire `onClick={() => setSelected(row)}`.'
    )
    assert 'onClick' in body and 'setSelected' in body, (
        'tab_escalations.jsx does not set selected row from row onClick — '
        'add `onClick={() => setSelected(row)}` on each row <tr>.'
    )

    # (b) Sidebar with role="dialog" and close button calling onClose
    assert 'role="dialog"' in body, (
        'tab_escalations.jsx detail sidebar does not have `role="dialog"` — '
        'add role="dialog" to the sidebar container element to match the SchedulerDrawer pattern.'
    )
    assert 'onClose' in body, (
        'tab_escalations.jsx detail sidebar does not reference onClose — '
        'add an `onClose` prop/handler for the sidebar close button.'
    )

    # (c) Escalation record fields referenced in the sidebar
    for field in ('detail', 'suggested_action', 'level', 'category', 'severity',
                  'agent_role', 'workflow_state', 'worktree', 'resolution'):
        assert field in body, (
            f'tab_escalations.jsx detail sidebar does not reference escalation field '
            f'`{field}` — add it to the sidebar content (section or label).'
        )

    # (d) Linked task card fields
    for field in ('task.title', 'task.status', 'task.description'):
        assert field in body, (
            f'tab_escalations.jsx detail sidebar does not reference linked task field '
            f'`{field}` — add it to the sidebar linked-task section.'
        )

    # (e) Resolved project label
    assert 'row.project' in body, (
        'tab_escalations.jsx sidebar does not render the resolved `project` label — '
        'add `{row.project}` display in the sidebar.'
    )

    # (f) task_unresolved fallback branch
    assert 'task_unresolved' in body, (
        'tab_escalations.jsx sidebar does not have a `task_unresolved` fallback branch — '
        'add `{row.task_unresolved ? <unresolved note> : <task card>}` to handle '
        'escalations whose linked task could not be resolved.'
    )


# ---------------------------------------------------------------------------
# step-1 test: EscalationStatStrip is declared, wired to DF_CHARTS/analytics,
# and mounted at the top of EscalationsTab
# ---------------------------------------------------------------------------


def test_tab_escalations_strip_mounted_and_wired(tab_escalations_jsx_body: str) -> None:
    """EscalationStatStrip must be declared, use DF_CHARTS, read the analytics
    payload, render StatTiles, and mount at the top of EscalationsTab (above
    the level-filter controls).

    Asserts:
    (1) file declares `function EscalationStatStrip(`.
    (2) file declares `const C = window.DF_CHARTS` (charts primitives available).
    (3) scoped to the EscalationStatStrip body: references ESCALATION_ANALYTICS
        (reads the analytics payload) and renders <C.StatTile.
    (4) scoped to the EscalationsTab body: renders <EscalationStatStrip, and
        that render's index is BEFORE the "Level:" filter-chip controls — the
        strip must sit at the top of the tab.
    """
    body = tab_escalations_jsx_body

    # (1) Component declared
    assert 'function EscalationStatStrip(' in body, (
        'tab_escalations.jsx does not define `function EscalationStatStrip(` — '
        'add the summary-strip component as a named function.'
    )

    # (2) Charts primitives available
    assert re.search(r'const\s+C\s*=\s*window\.DF_CHARTS', body), (
        'tab_escalations.jsx does not alias `const C = window.DF_CHARTS` — add it '
        'near the top alongside the existing `const DF = window.DF_DATA;` so '
        'EscalationStatStrip can render <C.StatTile.'
    )

    # (3) Scoped to the EscalationStatStrip body: reads the analytics payload,
    # renders StatTile.
    strip_fn = _extract_function_body(body, 'EscalationStatStrip')
    assert strip_fn, (
        'tab_escalations.jsx does not define a `function EscalationStatStrip(` body — '
        'add the component definition.'
    )
    assert 'ESCALATION_ANALYTICS' in strip_fn, (
        'EscalationStatStrip does not reference ESCALATION_ANALYTICS — it should read '
        'the analytics payload (e.g. `analytics || DF.ESCALATION_ANALYTICS`).'
    )
    assert '<C.StatTile' in strip_fn, (
        'EscalationStatStrip does not render <C.StatTile — add the tile container.'
    )

    # (4) Scoped to the EscalationsTab body: strip mounts before the level-filter
    # controls (top-of-tab placement).
    tab_fn = _extract_function_body(body, 'EscalationsTab')
    assert tab_fn, (
        'tab_escalations.jsx does not define a `function EscalationsTab(` body.'
    )
    strip_idx = tab_fn.find('<EscalationStatStrip')
    assert strip_idx != -1, (
        'EscalationsTab does not render <EscalationStatStrip — mount it as the first '
        'child of the returned JSX, above the controls header.'
    )
    level_idx = tab_fn.find('Level:')
    assert level_idx != -1, (
        'EscalationsTab does not render the "Level:" filter-chip label — cannot '
        'verify strip placement relative to the controls header.'
    )
    assert strip_idx < level_idx, (
        f'EscalationsTab renders <EscalationStatStrip AFTER the level-filter controls '
        f'(index {strip_idx} vs {level_idx}) — the strip must sit at the TOP of the '
        f'tab, above the controls header.'
    )


# ---------------------------------------------------------------------------
# step-3 test: EscalationStatStrip computes all four metrics from the
# payload's existing daily series (no duplicated computation)
# ---------------------------------------------------------------------------


def test_tab_escalations_strip_four_metrics(tab_escalations_jsx_body: str) -> None:
    """EscalationStatStrip must compute all four metrics from series the
    backend aggregator already emits, and render four labeled tiles.

    Asserts, scoped to the EscalationStatStrip body:
    (a) benign-rate — references flow_daily, the 'benign'/'actionable' class
        literals, and stamped_share (the hint).
    (b) 6h-breach — references open_items and breach_6h.
    (c) esc-per-done — references esc_per_done_daily.
    (d) churn — references churn_daily.
    (e) at least four <C.StatTile tiles are rendered.
    """
    strip_fn = _extract_function_body(tab_escalations_jsx_body, 'EscalationStatStrip')
    assert strip_fn, (
        'tab_escalations.jsx does not define a `function EscalationStatStrip(` body — '
        'add the component definition.'
    )

    # (a) benign-rate substrate
    assert 'flow_daily' in strip_fn, (
        'EscalationStatStrip does not reference flow_daily — the benign-rate tile '
        'must be computed from workflow.flow_daily (the only per-day benign/'
        'actionable series in the payload).'
    )
    assert "'benign'" in strip_fn, (
        "EscalationStatStrip does not reference the 'benign' class literal — "
        "sum flow_daily rows where class == 'benign'."
    )
    assert "'actionable'" in strip_fn, (
        "EscalationStatStrip does not reference the 'actionable' class literal — "
        "sum flow_daily rows where class == 'actionable'."
    )
    assert 'stamped_share' in strip_fn, (
        'EscalationStatStrip does not reference stamped_share — add the all-time '
        'stamped-share hint computed from origin.sources[].'
    )

    # (b) 6h-breach substrate
    assert 'open_items' in strip_fn, (
        'EscalationStatStrip does not reference lifespan.open_items — the '
        '6h-breach tile must count the live pending queue.'
    )
    assert 'breach_6h' in strip_fn, (
        'EscalationStatStrip does not reference breach_6h — count open_items '
        'where breach_6h is truthy.'
    )

    # (c) esc-per-done substrate
    assert 'esc_per_done_daily' in strip_fn, (
        'EscalationStatStrip does not reference workflow.esc_per_done_daily — '
        'the esc-per-done tile must be computed from it.'
    )

    # (d) churn substrate
    assert 'churn_daily' in strip_fn, (
        'EscalationStatStrip does not reference workflow.churn_daily — the '
        'churn-24h tile must be computed from it.'
    )

    # (e) at least four StatTile occurrences
    tile_count = len(re.findall(r'<C\.StatTile', strip_fn))
    assert tile_count >= 4, (
        f'EscalationStatStrip renders {tile_count} <C.StatTile tiles, expected >= 4 '
        '(benign rate, 6h breaches, esc/done, churn).'
    )


# ---------------------------------------------------------------------------
# step-5 test: strip windows its series to a trailing 7d anchored to
# analytics.generated_at (never Date.now())
# ---------------------------------------------------------------------------


def test_tab_escalations_strip_window_anchored_7d(tab_escalations_jsx_body: str) -> None:
    """The strip's series must be windowed to a trailing 7d anchored to the
    payload's own generated_at clock, never Date.now() — immune to
    browser-clock skew (matches the analytics tab's clock discipline).

    Asserts:
    (1) a `function windowCutoffDate(` helper is defined whose body
        references `generatedAt` and does NOT call `Date.now(`.
    (2) the EscalationStatStrip body references `generated_at` (reads the
        payload clock).
    (3) `Date.now(` does not appear anywhere in the EscalationStatStrip body.
    (4) a 7-day window is expressed near the cutoff/window logic.
    """
    body = tab_escalations_jsx_body

    # (1) windowCutoffDate helper, generatedAt-anchored, no Date.now()
    cutoff_fn = _extract_function_body(body, 'windowCutoffDate')
    assert cutoff_fn, (
        'tab_escalations.jsx does not define `function windowCutoffDate(` — add a '
        'local generated_at-anchored cutoff helper (copy the pattern from '
        'tab_escalation_analytics.jsx).'
    )
    assert 'generatedAt' in cutoff_fn, (
        'windowCutoffDate does not reference `generatedAt` in its body — it must '
        'anchor the cutoff to the payload clock, not the browser clock.'
    )
    assert 'Date.now(' not in cutoff_fn, (
        'windowCutoffDate calls Date.now( — the cutoff must be anchored exclusively '
        'to generatedAt (immune to browser-clock skew), never Date.now().'
    )

    # (2)/(3) EscalationStatStrip anchors to generated_at, never Date.now()
    strip_fn = _extract_function_body(body, 'EscalationStatStrip')
    assert strip_fn, (
        'tab_escalations.jsx does not define a `function EscalationStatStrip(` body.'
    )
    assert 'generated_at' in strip_fn, (
        'EscalationStatStrip does not reference generated_at — anchor the window '
        'cutoff to analytics.generated_at.'
    )
    assert 'Date.now(' not in strip_fn, (
        'EscalationStatStrip calls Date.now( — the window must be anchored '
        'exclusively to analytics.generated_at, never the browser clock.'
    )

    # (4) 7-day window literal expressed near the cutoff/window logic — either
    # a `windowCutoffDate(..., 7)` call site in the strip, or a `days * 7`
    # (or `7 * days`) expression inside the helper itself.
    assert (
        re.search(r'windowCutoffDate\([^)]*,\s*7\s*\)', strip_fn)
        or re.search(r'days\s*\*\s*7\b|\b7\s*\*\s*days', cutoff_fn)
    ), (
        'No 7-day window literal found — express the trailing-7d window via a '
        '`windowCutoffDate(..., 7)` call in EscalationStatStrip (or an equivalent '
        '`days * 7` expression in the windowCutoffDate helper itself).'
    )


# ---------------------------------------------------------------------------
# step-7 test: strip feeds trend sparklines and retains churn-24h
# ---------------------------------------------------------------------------


def test_tab_escalations_strip_sparklines_and_churn_retained(tab_escalations_jsx_body: str) -> None:
    """The strip must feed trend sparklines for the three series-backed tiles
    (benign-rate, esc-per-done, churn) via StatTile's spark prop, and
    churn-24h must be RETAINED — not the first tile dropped — per the
    open-question-4 decision (all four tiles kept; responsive grid instead).

    Asserts:
    (1) at least three `spark=` props are passed to <C.StatTile within the
        EscalationStatStrip body (one each for the series-backed tiles).
    (2) churn-24h is retained: a `spark=` occurs within ~200 chars of a churn
        tile label / churn_daily reference (co-occurrence, not bare
        presence — a stray spark= elsewhere wouldn't prove churn has one).
    """
    strip_fn = _extract_function_body(tab_escalations_jsx_body, 'EscalationStatStrip')
    assert strip_fn, (
        'tab_escalations.jsx does not define a `function EscalationStatStrip(` body.'
    )

    # (1) at least three spark= props within <C.StatTile tiles
    spark_count = len(re.findall(r'<C\.StatTile[^>]*\bspark=', strip_fn))
    assert spark_count >= 3, (
        f'EscalationStatStrip passes spark= to only {spark_count} <C.StatTile tiles, '
        'expected >= 3 (benign rate, esc/done, and churn are series-backed).'
    )

    # (2) churn-24h retained: spark= co-occurs near a churn reference
    found_churn_spark = False
    for m in re.finditer(r'churn', strip_fn, re.IGNORECASE):
        window = strip_fn[max(0, m.start() - 200): m.end() + 200]
        if 'spark=' in window:
            found_churn_spark = True
            break
    assert found_churn_spark, (
        'No `spark=` prop found within ~200 chars of a churn tile label / '
        'churn_daily reference — churn-24h must be RETAINED with its own '
        'sparkline per the open-question-4 decision (keep all four tiles).'
    )


# ---------------------------------------------------------------------------
# task 3470: cross-tab focus must retry until the payload lands, then report
# ---------------------------------------------------------------------------


def test_focus_handoff_retries_then_reports_a_miss(
    tab_escalations_jsx_body: str,
    tab_escalations_jsx_code: str,
) -> None:
    """A cross-tab focus that finds no row must retry, then SAY it missed.

    Two halves of one defect in the focus effect:

    (1) It consumes `focusId` UNCONDITIONALLY.  On a miss the operator — who
        just clicked an escalation link over in the memory-evals section —
        lands on the Escalations tab with nothing selected and no explanation.
        A dead click with zero feedback is exactly the silent degradation this
        repo's loud-over-silent norm exists to prevent.

    (2) It is keyed only on `[focusId]`, so the lookup runs ONCE.  `DF.ESCALATIONS`
        starts as data.js's seed (`subsections: []`) and `applyKey` replaces the
        reference wholesale on the first successful poll — ESCALATIONS is
        deliberately not in STABLE_ARRAY_KEYS (data.js:96-105).  So on a cold
        load, or with the escalations endpoint sitting in data.js's exponential
        backoff, the single mount-time pass searches an EMPTY payload, drops the
        focus, and never retries when the rows arrive.

    Asserted structurally and on `data-testid` values, never on copy.
    """
    body = tab_escalations_jsx_body
    code = tab_escalations_jsx_code

    # (a) module-scope helpers exist AND are called. Naming a helper and then
    #     never calling it is the dead-code failure the `trendGaps` precedent
    #     in test_tab_memory_evals.py:1311-1318 guards against.
    for helper in ('findEscalationRow', 'escalationsLoaded'):
        assert re.search(rf'\bfunction\s+{helper}\s*\(', code), (
            f'tab_escalations.jsx must define `function {helper}(`. Lifting the '
            'row search and the payload-arrival check to module scope is what '
            'keeps the focus effect small enough to stay under the 900-char cap '
            'the cross-tab contract test (test_tab_memory_evals.py) matches on.'
        )
        assert len(re.findall(rf'\b{helper}\s*\(', code)) >= 2, (
            f'`{helper}` is defined but never called.'
        )

    # Extract the focus effect the same way the cross-tab contract test does,
    # so the two cannot drift apart on what "the focus effect" means.
    effect = re.search(
        r'uE\(\(\)\s*=>\s*\{([\s\S]{0,900}?)\n\s*\},\s*\[([^\]]*focusId[^\]]*)\]\)',
        code,
    )
    assert effect is not None, (
        'no `uE` effect keyed on `focusId` found within the 900-char body cap. '
        'If the effect body outgrew the cap, the cross-tab handoff contract '
        'test in test_tab_memory_evals.py has silently stopped matching too.'
    )
    eff, deps = effect.group(1), effect.group(2)

    # (b) the effect DECLINES to decide before the payload arrives.
    assert re.search(r'if\s*\(\s*!\s*escalationsLoaded\s*\([^)]*\)\s*\)\s*\{?\s*return', eff), (
        'the focus effect must bail out early while the escalations payload is '
        'still data.js\'s seed. Searching an empty payload and calling it a '
        'miss asserts "no longer in the queue" on evidence that cannot support '
        'it — the endpoint may simply be in backoff.'
    )

    # (c) the dep array names the escalations payload local, so the lookup
    #     RETRIES on each 3s poll instead of firing once at mount.
    esc_local = re.search(r'const\s+(\w+)\s*=\s*DF\.ESCALATIONS', code)
    assert esc_local is not None, (
        'EscalationsTab must read `DF.ESCALATIONS` into a local.'
    )
    assert re.search(r'\b' + re.escape(esc_local.group(1)) + r'\b', deps), (
        f'the focus effect\'s dep array is `[{deps.strip()}]` — it does not '
        f'name `{esc_local.group(1)}`, so the lookup never re-runs when the '
        'payload lands. A cold load drops the focus permanently.'
    )

    # (d) a miss is RECORDED and REACHES RENDER, not merely stored.
    miss_state = None
    for decl in re.finditer(r'const\s*\[\s*(\w+)\s*,\s*(\w+)\s*\]\s*=\s*uS\(', code):
        state, setter = decl.group(1), decl.group(2)
        if re.search(r'\b' + re.escape(setter) + r'\s*\(', eff):
            if re.search(r'\{\s*' + re.escape(state) + r'\b', code) or re.search(
                r'\b' + re.escape(state) + r'\s*&&', code
            ):
                miss_state = state
                break
    assert miss_state is not None, (
        'no `uS` state whose setter the focus effect calls also reaches a JSX '
        'render position. A miss must be DISPLAYED, not swallowed — the '
        'operator clicked a link and is owed an answer either way.'
    )

    # (e) both no-selection outcomes are visible, pinned by testid not by copy.
    #     `pending` needs its own state because with the endpoint in backoff the
    #     payload never leaves the seed: a miss-only fix leaves precisely the
    #     cold-load case silent, which is half the defect.
    for testid in ('esc-focus-miss', 'esc-focus-pending'):
        assert f'data-testid="{testid}"' in code, (
            f'the tab must render a `data-testid="{testid}"` notice.'
        )

    # (f) the stale-drawer invariant survives: the focus is still consumed.
    assert 'onFocusConsumed' in eff, (
        'the focus effect must still call onFocusConsumed() once a decision is '
        'reachable — leaving the focus set would reopen a stale drawer on every '
        'later visit to this tab.'
    )
