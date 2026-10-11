"""Wiring tests for the Escalations tab UI.

Tests parse JSX/JS/HTML source files as text and assert structural contracts
(endpoint registration, export names, rail entry, tab registration, load-order).
Follows the idiom established in test_tab_curator.py and test_index_html.py.
"""

from __future__ import annotations

import re

import pytest
from _dashboard_helpers import (
    assert_script_loads_before,
    extract_df_data_block,
    extract_function_body,
    find_script_position,
    strip_js_comments,
    walk_balanced,
)

# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def tab_escalations_jsx_code(tab_escalations_jsx_body):
    """`tab_escalations.jsx` with every comment stripped.

    Same rationale as the `tab_memory_evals_jsx_code` fixture in
    test_tab_memory_evals.py: this file carries explanatory prose naming most
    of the identifiers the render code also names, so a whole-file substring
    grep is satisfied by a MENTION — delete the render site, leave the comment,
    and the assertion stays green.
    """
    return strip_js_comments(tab_escalations_jsx_body)


# ---------------------------------------------------------------------------
# Helper: extract a named JS/JSX function body (brace-aware)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# step-1 test: data.js registers the escalations endpoint
# ---------------------------------------------------------------------------


def test_data_js_registers_escalations_endpoint(data_js_body: str) -> None:
    """data.js must register /api/v2/dashboard/escalations mapped to ['ESCALATIONS'].

    The entry must be present in the static (unwindowed) section of endpointsFor,
    and the empty-defaults block must initialise ESCALATIONS with the shape
    expected by redux_api.shape_escalations:
    {subsections, summary:{by_level, by_status, skipped_count}}.
    """
    assert '/api/v2/dashboard/escalations' in data_js_body, (
        "data.js does not contain the literal URL '/api/v2/dashboard/escalations' — "
        'add it to the unwindowed entries in endpointsFor.'
    )
    assert "'ESCALATIONS'" in data_js_body or '"ESCALATIONS"' in data_js_body, (
        "data.js does not reference 'ESCALATIONS' — add it as the mapped key "
        "for '/api/v2/dashboard/escalations' in endpointsFor."
    )
    seed_block = extract_df_data_block(data_js_body, 'ESCALATIONS')
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
    summary_block = extract_df_data_block(seed_block, 'summary')
    assert summary_block, (
        'ESCALATIONS seed summary block not found via extract_df_data_block — '
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
    assert re.search(r'\bskipped_count\s*:', summary_block), (
        "ESCALATIONS.summary seed missing key 'skipped_count:' — "
        'add it nested under summary in the ESCALATIONS seed block.  The seed '
        "declares the server's true payload shape before the first poll resolves; "
        'omitting the key leaves the global "N unreadable" pill reading `undefined` '
        'rather than a real zero.'
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
        "tab_escalations.jsx does not call useOpenSet( — call persisted_state.js's "
        "useOpenSet with subsection ids and 'df.open.esc'."
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
    result = find_script_position(index_html_body, _TAB_ESC_PREFIX)
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
    assert_script_loads_before(
        index_html_body,
        '/static/redux/data.js',
        _TAB_ESC_PREFIX,
        'data.js',
        'tab_escalations.jsx',
        'data.js must load before tab_escalations.jsx so window.DF_DATA is seeded.',
    )

    # (c) Loads after shell.jsx
    assert_script_loads_before(
        index_html_body,
        '/static/redux/shell.jsx',
        _TAB_ESC_PREFIX,
        'shell.jsx',
        'tab_escalations.jsx',
        'shell.jsx must load before tab_escalations.jsx so window.DF_SHELL is available.',
    )

    # (d) Loads after tabs.jsx
    assert_script_loads_before(
        index_html_body,
        '/static/redux/tabs.jsx',
        _TAB_ESC_PREFIX,
        'tabs.jsx',
        'tab_escalations.jsx',
        'tabs.jsx must load before tab_escalations.jsx so window.DF_TABS is available to mutate.',
    )

    # (e) Loads before app.jsx
    assert_script_loads_before(
        index_html_body,
        _TAB_ESC_PREFIX,
        '/static/redux/app.jsx',
        'tab_escalations.jsx',
        'app.jsx',
        'tab_escalations.jsx must load before app.jsx so EscalationsTab is set on window.DF_TABS.',
    )

    # (g) every /static/redux/ cache-buster is at or past this tab's floor.
    #     FLOOR only: whether the versions are UNIFORM is asserted once,
    #     canonically, in test_index_html.py. It was replicated byte-identically
    #     across five test modules, each with its own stale monotonic floor, so a
    #     partial bump failed five tests with five different floors in the
    #     message. `min(...)` is the strictly stronger floor claim under mixed
    #     versions anyway — the OLDEST asset is the one that serves stale code.
    versions = {int(v) for v in re.findall(r'/static/redux/[^"?]+\?v=(\d+)', index_html_body)}
    assert versions, (
        'index.html carries no /static/redux/*?v=<n> asset tags at all — the '
        'cache-buster convention has been dropped or the URLs were rewritten.'
    )
    assert min(versions) >= 10, (
        f'the oldest index.html cache-buster version is {min(versions)}, '
        'expected >= 10 (the floor tab_escalations.jsx landed at).'
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
    sort_fn = extract_function_body(tab_escalations_jsx_body, 'sortRows')
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
    (f) the linked task is the row's served task Datum, read through
        escalation_views.js::taskCard, and a card with no measurement renders
        the server's reason rather than a local guess at one.
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

    # (f) The card is the row's task Datum, read through taskCard, and a hole
    # renders the reason the server gave for it.
    sidebar = strip_js_comments(extract_function_body(body, 'EscalationSidebar'))
    assert re.search(r'\btaskCard\(\s*row\b', sidebar), (
        'EscalationSidebar does not read its linked task through '
        '`taskCard(row, ...)` (escalation_views.js) — the card is the row\'s served '
        'task Datum, and taskCard is its one reader.'
    )
    assert 'isHole' in sidebar and re.search(r'\.reason\b', sidebar), (
        'EscalationSidebar does not branch on the card\'s `isHole` and render its '
        '`.reason` — a task the server could not read must say why, not show a '
        'blank card or a locally invented note.'
    )
    assert 'task_unresolved' not in strip_js_comments(body), (
        'tab_escalations.jsx still reads `task_unresolved` — the payload no longer '
        'carries it: an unknown task card is a Datum with its own reason.'
    )


def test_tab_escalations_renders_skipped_queue_files(tab_escalations_jsx_body: str) -> None:
    """tab_escalations.jsx must surface the queue files the reader could not parse.

    The ESCALATIONS payload carries per-subsection ``skipped``
    (``[{path, error}, ...]``) and ``summary.skipped_count``.  A tab that
    renders one escalation short with nothing saying so is the exact silent
    degradation INV-2 (``structured-facts-at-failure``) forbids.

    Asserts:
    (a) a ``data-testid="escalation-skipped"`` notice element exists.
    (b) the notice names BOTH ``.path`` and ``.error`` — the operator must be
        told WHICH file and WHY, not just a count.
    (c) the notice is driven by ``sec.skipped`` and gated on a non-zero length.
    (d) the notice sits OUTSIDE the ``filteredRows.length === 0`` ternary, so a
        group whose every row is filtered out still shows it.
    (e) the notice is NOT gated on the level/status filter chips — a file that
        could not be parsed has neither field to filter on.
    (f) the per-subsection group summary carries a skipped badge, beside the
        existing L1/L2 pips, so it is visible while the group is collapsed.
    (g) the global controls header renders a ``skipped_count`` pill, gated on > 0.
    (h) the notice caps how many rows it lists while still counting the true
        total, so a whole-directory fault does not bury the escalation table.
    """
    body = tab_escalations_jsx_body

    # (a) The notice element exists.
    assert 'data-testid="escalation-skipped"' in body, (
        'tab_escalations.jsx has no `data-testid="escalation-skipped"` notice — '
        'add a SkippedNotice component (modelled on tab_memory_evals.jsx::IssuesNotice) '
        'that renders the subsection\'s unreadable queue files.'
    )

    # (b) The notice names each file and its cause, not just a count.
    #
    # The window is the SkippedNotice component body — sliced from its `function`
    # declaration to the next top-level one — not a fixed character count.  A
    # char-count window makes the negative assertions in (e) below depend on how
    # much unrelated source happens to sit within N characters, so an edit
    # nowhere near this notice could fail them.
    comp_m = re.search(r'^function SkippedNotice\(', body, re.MULTILINE)
    assert comp_m is not None, (
        'tab_escalations.jsx declares no top-level `function SkippedNotice(` — the '
        'notice must be a named component, not inlined into the subsection map.'
    )
    next_fn = re.search(r'^function ', body[comp_m.end():], re.MULTILINE)
    notice = body[comp_m.start():comp_m.end() + next_fn.start()] if next_fn else body[comp_m.start():]
    assert 'data-testid="escalation-skipped"' in notice, (
        'the `escalation-skipped` testid is not rendered by SkippedNotice — keep the '
        'notice and its testid in one component so this test window covers it.'
    )
    for field in ('.path', '.error'):
        assert field in notice, (
            f'the escalation-skipped notice does not reference `{field}` — a bare count '
            'tells the operator something is wrong but not what; render one line per '
            'entry naming the path and the parse error.'
        )

    # (c) Driven by the subsection field, gated on a non-zero length.
    assert 'sec.skipped' in body, (
        'tab_escalations.jsx never reads `sec.skipped` — the notice must be driven by '
        'the per-subsection payload field, not re-derived or hardcoded.'
    )
    assert re.search(r'skipped\.length|skipped\s*\.\s*length', body), (
        'the escalation-skipped notice is not gated on the skipped list length — '
        'render nothing when the queue read cleanly.'
    )

    # (d) The notice sits OUTSIDE the all-rows-filtered empty-state branch.
    map_m = re.search(r'subsections\.map\(sec\s*=>', body)
    assert map_m is not None, (
        'could not locate the `subsections.map(sec =>` body in tab_escalations.jsx'
    )
    map_body = body[map_m.start():]
    skipped_at = map_body.find('sec.skipped')
    mount_at = map_body.find('<SkippedNotice')
    empty_at = map_body.find('filteredRows.length === 0')
    assert skipped_at != -1 and empty_at != -1
    # Anchor on the MOUNT site, not the `const skipped = sec.skipped` read: an
    # implementation that hoisted the read but moved the mount into the
    # empty-state branch would satisfy a read-position check while silently
    # reintroducing the exact looks-empty-but-isn't case this asserts against.
    assert mount_at != -1, (
        'the subsection map never mounts `<SkippedNotice …>` — the component must be '
        'rendered inside the group, not merely declared.'
    )
    assert mount_at < empty_at, (
        'the skipped notice is mounted inside or after the '
        '`filteredRows.length === 0` ternary — hoist it above that branch so a group '
        'rendering "No escalations match current filters" while holding an unreadable '
        'file still says so.'
    )

    # (e) Not routed through the level/status filter chips.
    for gate in ('matchesFilter', 'filter.levels', 'filter.statuses'):
        assert gate not in notice, (
            f'the escalation-skipped notice references `{gate}` — a file that could not '
            'be parsed has no level or status, so routing it through the chips would let '
            'an arbitrary default decide whether the operator is told about corruption.'
        )
    skipped_stmt = map_body[skipped_at:map_body.find('\n', skipped_at)]
    for gate in ('matchesFilter', 'filter.levels', 'filter.statuses'):
        assert gate not in skipped_stmt, (
            f'`sec.skipped` is filtered through `{gate}` before reaching the notice — '
            'pass the list through unfiltered.'
        )

    # (f) Per-subsection badge beside the existing L1/L2 pips.
    #
    # Bounded structurally — from the `const summary = (` fragment to the group's
    # own `return (` — rather than by a character count, and anchored on the
    # `secByLevel[2]` identifier rather than the rendered `L2 ·` label, so a
    # reformat or a label tweak does not fail this test with a misdirecting
    # "widen the window" message.
    frag_start = map_body.find('const summary = (')
    assert frag_start != -1, (
        'could not locate the per-subsection `const summary = (` fragment'
    )
    frag_end = map_body.find('return (', frag_start)
    frag = map_body[frag_start:frag_end if frag_end != -1 else len(map_body)]
    assert 'secByLevel[2]' in frag, (
        'the per-subsection summary fragment no longer renders the L2 pip from '
        '`secByLevel[2]` — this test anchors the skipped badge beside it.'
    )
    assert re.search(r'skipped\.length\s*>\s*0', frag), (
        'the per-subsection group summary carries no skipped badge — add a '
        '`{skipped.length > 0 && <span className="pip">…</span>}` badge after the L2 pip '
        'so the count is visible while the group is collapsed.'
    )
    assert 'badge' in frag, (
        'the skipped badge does not use the `badge` span idiom the L1/L2 pips use'
    )

    # (g) Global controls-header pill from the top-level summary.
    assert 'skipped_count' in body, (
        'tab_escalations.jsx never reads `skipped_count` — add a global pill beside the '
        'existing `N pending · N L1 · N L2` summary pills.'
    )
    # Anchored on the `levelCount(DF, 2)` call (the global L2 pill; the
    # subsection pips read `secByLevel[2]`, which does not match) and bounded at
    # the next structural landmark — the subsection map — rather than by a
    # character count.  The stated contract is "in the controls header, after
    # the pending/L1/L2 pills and before the subsection groups", which survives
    # a reformat or a label change; pinning the pill's surrounding text would
    # pin exact whitespace and an interpunct.
    pills_at = body.find('levelCount(DF, 2)')
    assert pills_at != -1, 'could not locate the global summary-pill span'
    pills_end = body.find('subsections.map(', pills_at)
    pills_window = body[pills_at:pills_end if pills_end != -1 else len(body)]
    assert 'skipped_count' in pills_window, (
        'the `skipped_count` pill is not rendered beside the existing pending/L1/L2 '
        'pills — append it to that same summary span.'
    )
    assert re.search(r'skipped_count[^\n]*>\s*0', pills_window), (
        'the global `skipped_count` pill is not gated on `> 0` — a clean fleet must not '
        'render a permanent "0 unreadable" pill.'
    )

    # (h) The notice bounds how many rows it renders, without understating the
    # loss.  A whole-directory fault (truncated write, permission fault, a
    # half-synced mount) yields hundreds of records; rendering them all,
    # always-expanded, pushes the escalation table below the fold and degrades
    # the same operator view the notice exists to serve.
    assert '.slice(' in notice, (
        'SkippedNotice renders every entry unbounded — slice the list to a cap and add '
        'an overflow line, so a whole-directory fault does not bury the escalation table.'
    )
    assert 'rows.length' in notice, (
        'the SkippedNotice headline no longer counts from the full `rows.length` — the '
        'headline must state the TRUE total even when the listing is capped, or the '
        'notice understates the loss it exists to report.'
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
    strip_fn = extract_function_body(body, 'EscalationStatStrip')
    assert 'ESCALATION_ANALYTICS' in strip_fn, (
        'EscalationStatStrip does not reference ESCALATION_ANALYTICS — it should read '
        'the analytics payload (e.g. `analytics || DF.ESCALATION_ANALYTICS`).'
    )
    assert '<C.StatTile' in strip_fn, (
        'EscalationStatStrip does not render <C.StatTile — add the tile container.'
    )

    # (4) Scoped to the EscalationsTab body: strip mounts before the level-filter
    # controls (top-of-tab placement).
    tab_fn = extract_function_body(body, 'EscalationsTab')
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
    (a) benign-rate — reads flow_daily through escalation_views.js's
        windowedClassSplit, whose denominator is every resolution class, and
        stamped_share (the hint).
    (b) 6h-breach — references open_items and breach_6h.
    (c) esc-per-done — references esc_per_done_daily.
    (d) churn — references churn_daily.
    (e) at least four <C.StatTile tiles are rendered.
    """
    strip_fn = extract_function_body(tab_escalations_jsx_body, 'EscalationStatStrip')

    # (a) benign-rate substrate
    assert 'flow_daily' in strip_fn, (
        'EscalationStatStrip does not reference flow_daily — the benign-rate tile '
        'must be computed from workflow.flow_daily (the only per-day benign/'
        'actionable series in the payload).'
    )
    assert re.search(r'\bwindowedClassSplit\(', strip_fn), (
        'EscalationStatStrip does not compute the benign rate through '
        '`windowedClassSplit(` (escalation_views.js) — summing only the benign and '
        'actionable rows divides by a narrower whole than origin\'s class split.'
    )
    assert "'actionable'" not in strip_js_comments(strip_fn), (
        "EscalationStatStrip still names the 'actionable' class — the benign "
        'rate\'s denominator is every class the payload serves, not a fixed pair.'
    )
    assert 'stamped_share' in strip_fn, (
        'EscalationStatStrip does not reference stamped_share — add the all-time '
        'stamped-share hint computed from origin.sources[].'
    )
    assert re.search(r'\.classified\b', strip_fn) and not re.search(r'\bs\.(benign|actionable)\b', strip_fn), (
        'the stamped-share hint must weight each source by its served `classified` '
        'count — summing `s.benign + s.actionable` drops every other class.'
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
    cutoff_fn = extract_function_body(body, 'windowCutoffDate')
    assert 'generatedAt' in cutoff_fn, (
        'windowCutoffDate does not reference `generatedAt` in its body — it must '
        'anchor the cutoff to the payload clock, not the browser clock.'
    )
    assert 'Date.now(' not in cutoff_fn, (
        'windowCutoffDate calls Date.now( — the cutoff must be anchored exclusively '
        'to generatedAt (immune to browser-clock skew), never Date.now().'
    )

    # (2)/(3) EscalationStatStrip anchors to generated_at, never Date.now()
    strip_fn = extract_function_body(body, 'EscalationStatStrip')
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
    (benign-rate, esc-per-done, churn) via StatTile's history prop, and
    churn-24h must be RETAINED — not the first tile dropped — per the
    open-question-4 decision (all four tiles kept; responsive grid instead).

    The prop was named `spark` until task 5588 renamed it `history`: the series
    is the tile's PAST, and `spark` named the drawing rather than the data,
    which reads badly beside the `datum` carrying the tile's present value.

    Asserts:
    (1) at least three `history=` props are passed to <C.StatTile within the
        EscalationStatStrip body (one each for the series-backed tiles).
    (2) churn-24h is retained: a `history=` occurs within ~200 chars of a churn
        tile label / churn_daily reference (co-occurrence, not bare
        presence — a stray history= elsewhere wouldn't prove churn has one).
    """
    strip_fn = extract_function_body(tab_escalations_jsx_body, 'EscalationStatStrip')

    # (1) at least three history= props within <C.StatTile tiles.
    #
    # Split at each tile and read only as far as that tile's `/>`, rather than
    # the `<C\.StatTile[^>]*` this used to be: every tile now carries a `format`
    # callback, and an arrow function puts a `>` inside the tag, which truncated
    # the old class mid-prop and read every tile as series-less.
    tiles = re.split(r'(?=<C\.StatTile\b)', strip_fn)[1:]
    spark_count = sum(1 for tile in tiles if 'history=' in tile.split('/>')[0])
    assert spark_count >= 3, (
        f'EscalationStatStrip passes history= to only {spark_count} <C.StatTile '
        'tiles, expected >= 3 (benign rate, esc/done, and churn are series-backed).'
    )

    # (2) churn-24h retained: history= co-occurs near a churn reference
    found_churn_spark = False
    for m in re.finditer(r'churn', strip_fn, re.IGNORECASE):
        window = strip_fn[max(0, m.start() - 200): m.end() + 200]
        if 'history=' in window:
            found_churn_spark = True
            break
    assert found_churn_spark, (
        'No `history=` prop found within ~200 chars of a churn tile label / '
        'churn_daily reference — churn-24h must be RETAINED with its own '
        'sparkline per the open-question-4 decision (keep all four tiles).'
    )


# ---------------------------------------------------------------------------
# task 3470: cross-tab focus must retry until the payload lands, then report
# ---------------------------------------------------------------------------


def test_focus_handoff_retries_then_reports_a_miss(
    tab_escalations_jsx_code: str,
) -> None:
    """A cross-tab focus that finds no row must retry, then SAY it missed.

    Two halves of one defect in the focus effect:

    (1) It consumes `focus` UNCONDITIONALLY.  On a miss the operator — who
        just clicked an escalation link over in the memory-evals section —
        lands on the Escalations tab with nothing selected and no explanation.
        A dead click with zero feedback is exactly the silent degradation this
        repo's loud-over-silent norm exists to prevent.

    (2) It is keyed only on `[focus]`, so the lookup runs ONCE.  `DF.ESCALATIONS`
        starts as data.js's seed (`subsections: []`) and `applyKey` replaces the
        reference wholesale on the first successful poll — ESCALATIONS is
        deliberately not in STABLE_ARRAY_KEYS (data.js:96-105).  So on a cold
        load, or with the escalations endpoint sitting in data.js's exponential
        backoff, the single mount-time pass searches an EMPTY payload, drops the
        focus, and never retries when the rows arrive.

    Asserted structurally and on `data-testid` values, never on copy.
    """
    code = tab_escalations_jsx_code

    # (a) the payload-arrival check is a module-scope helper AND is called.
    #     Naming a helper and then never calling it is the dead-code failure the
    #     `trendGaps` precedent in test_tab_memory_evals.py guards against.
    #     Lifting it out is what keeps the focus effect under the 900-char cap
    #     the cross-tab contract test (test_tab_memory_evals.py) matches on.
    assert re.search(r'\bfunction\s+escalationsLoaded\s*\(', code), (
        'tab_escalations.jsx must define `function escalationsLoaded(`.'
    )
    assert len(re.findall(r'\bescalationsLoaded\s*\(', code)) >= 2, (
        '`escalationsLoaded` is defined but never called.'
    )

    # (a') the row lookup is escalation_focus.js's, never a local copy: that
    #      module is the one home of the (queue, id) rule and its node tests.
    assert not re.search(r'\bfunction\s+findEscalationRow\s*\(', code), (
        'tab_escalations.jsx defines its own `function findEscalationRow(`, which '
        'would shadow the (queue, id) lookup in escalation_focus.js.'
    )
    assert re.search(
        r'const\s*\{[^}]*\bfindEscalationRow\b[^}]*\}\s*=\s*window\.DF_ESCALATION_FOCUS\s*;',
        code,
    ), 'tab_escalations.jsx must destructure findEscalationRow from window.DF_ESCALATION_FOCUS.'

    # Extract the focus effect the same way the cross-tab contract test does,
    # so the two cannot drift apart on what "the focus effect" means.
    effect = re.search(
        r'uE\(\(\)\s*=>\s*\{([\s\S]{0,900}?)\n\s*\},\s*\[([^\]]*\bfocus\b[^\]]*)\]\)',
        code,
    )
    assert effect is not None, (
        'no `uE` effect keyed on `focus` found within the 900-char body cap. '
        'If the effect body outgrew the cap, the cross-tab handoff contract '
        'test in test_tab_memory_evals.py has silently stopped matching too.'
    )
    eff, deps = effect.group(1), effect.group(2)

    # (a'') the effect resolves the focus through the module and reads BOTH
    #       outcomes it reports: the elected row and every candidate.
    assert re.search(r'\bfindEscalationRow\s*\(', eff), (
        'the focus effect must resolve the focus with findEscalationRow.'
    )
    for member in ('row', 'candidates'):
        assert re.search(rf'\.{member}\b', eff), (
            f'the focus effect must read the lookup\'s `.{member}`.'
        )

    # (b) the effect DECLINES to decide before the payload arrives.
    assert re.search(r'if\s*\(\s*!\s*escalationsLoaded\s*\([^)]*\)\s*\)\s*\{?\s*return', eff), (
        'the focus effect must bail out early while the escalations payload is '
        'still data.js\'s seed. Searching an empty payload and calling it a '
        'miss asserts "no longer in the queue" on evidence that cannot support '
        'it — the endpoint may simply be in backoff.'
    )

    # (c) the dep array names the escalations payload local, so the lookup
    #     RETRIES on each 3s poll instead of firing once at mount.
    #     Derived from inside the COMPONENT body, not the whole file: the
    #     module-scope seed capture reads `DF.ESCALATIONS` too, and it is a
    #     different thing — the frozen pre-fetch reference, which correctly
    #     must NOT appear in the deps.
    tab_body = extract_function_body(code, 'EscalationsTab')
    esc_local = re.search(r'const\s+(\w+)\s*=\s*DF\.ESCALATIONS', tab_body)
    assert esc_local is not None, (
        'EscalationsTab must read `DF.ESCALATIONS` into a local.'
    )
    assert re.search(r'\b' + re.escape(esc_local.group(1)) + r'\b', deps), (
        f'the focus effect\'s dep array is `[{deps.strip()}]` — it does not '
        f'name `{esc_local.group(1)}`, so the lookup never re-runs when the '
        'payload lands. A cold load drops the focus permanently.'
    )

    # (d) a miss is RECORDED BY THE NO-ROW BRANCH into the state the miss
    #     NOTICE renders — not merely stored somewhere.
    #
    #     Both anchors are load-bearing.  A loop over every `uS` declaration
    #     asking only "is this setter called anywhere in the effect, and does
    #     this state reach any JSX position?" is VACUOUS here: `setSelected` is
    #     also called in this effect and `selected` also reaches render, it is
    #     declared first, so such a loop binds to `selected` and passes — as it
    #     did on the pre-fix code, which recorded no miss at all.
    no_row_branch = re.search(r'\belse\b([\s\S]*)$', eff)
    assert no_row_branch is not None, (
        'the focus effect has no `else` branch: a lookup that found NO row must '
        'take a distinct path, not fall through to the one a hit takes.'
    )
    branch = no_row_branch.group(1)
    miss_state = miss_setter = None
    for decl in re.finditer(r'const\s*\[\s*(\w+)\s*,\s*(\w+)\s*\]\s*=\s*uS\(', code):
        state, setter = decl.group(1), decl.group(2)
        if not re.search(r'\b' + re.escape(setter) + r'\s*\(', branch):
            continue
        # ...and that same state must GATE the miss notice's subtree. Derived
        # from the testid, not from the state's spelling, so a rename is free.
        if not re.search(
            r'\{\s*' + re.escape(state) + r'\s*&&[\s\S]{0,400}?data-testid="esc-focus-miss"',
            code,
        ):
            continue
        miss_state, miss_setter = state, setter
        break
    assert miss_state is not None and miss_setter is not None, (
        'no `uS` state is both written by the focus effect\'s no-row branch and '
        'the gate on the `data-testid="esc-focus-miss"` subtree. A miss must be '
        'recorded by the branch that observed it AND displayed — the operator '
        'clicked a link and is owed an answer either way.'
    )

    # (e) both no-selection outcomes are visible, pinned by testid not by copy.
    #     `pending` needs its own state because with the endpoint in backoff the
    #     payload never leaves the seed: a miss-only fix leaves precisely the
    #     cold-load case silent, which is half the defect.
    for testid in ('esc-focus-miss', 'esc-focus-pending', 'esc-focus-ambiguous'):
        assert f'data-testid="{testid}"' in code, (
            f'the tab must render a `data-testid="{testid}"` notice.'
        )

    # The pending notice is shown only while a focus is held.
    assert re.search(r'\{\s*focus\s*&&[\s\S]{0,200}?data-testid="esc-focus-pending"', code), (
        'the pending notice must be gated on `focus`.'
    )

    # (f) the stale-drawer invariant survives: the focus is consumed on EVERY
    #     path, so the call sits outside the branches, at the effect's top level.
    consumed = re.search(r'\bonFocusConsumed\s*\(', eff)
    assert consumed is not None, (
        'the focus effect must still call onFocusConsumed() once a decision is '
        'reachable — leaving the focus set would reopen a stale drawer on every '
        'later visit to this tab.'
    )
    prefix = eff[:consumed.start()]
    assert prefix.count('{') == prefix.count('}'), (
        'onFocusConsumed() is called inside a branch of the focus effect, so some '
        'outcome leaves the focus set. Call it once, after the branches.'
    )

    # (g) a TIE is its own outcome. The ambiguity notice is gated on a state
    #     that only the `candidates.length > 1` branch sets to a value: the
    #     lookup never elects a row, and nothing else may claim ambiguity.
    many = re.search(r'candidates\.length\s*>\s*1\s*\)\s*\{', eff)
    assert many is not None, (
        'the focus effect has no `candidates.length > 1` branch for a tie.'
    )
    many_block = walk_balanced(eff, many.end() - 1)
    many_span = (many.end() - 1, many.end() - 1 + len(many_block))
    ambiguous = None
    for decl in re.finditer(r'const\s*\[\s*(\w+)\s*,\s*(\w+)\s*\]\s*=\s*uS\(', code):
        gate = re.search(
            r'\{\s*' + re.escape(decl.group(1)) + r'\s*&&[\s\S]{0,400}?data-testid="esc-focus-ambiguous"',
            code,
        )
        if gate:
            ambiguous = (decl.group(1), decl.group(2), gate.start())
            break
    assert ambiguous is not None, (
        'no `uS` state gates the `data-testid="esc-focus-ambiguous"` notice.'
    )
    _state, amb_setter, gate_at = ambiguous
    valued = [
        call for call in re.finditer(re.escape(amb_setter) + r'\(\s*([^)]*?)\s*\)', eff)
        if call.group(1) != 'null'
    ]
    assert valued, f'the focus effect never sets `{amb_setter}` to a value.'
    for call in valued:
        assert many_span[0] <= call.start() < many_span[1], (
            f'`{amb_setter}({call.group(1)})` is reached outside the '
            '`candidates.length > 1` branch, so the notice could claim a tie '
            'that the lookup did not report.'
        )
    notice = code[gate_at:gate_at + 1200]
    assert re.search(
        r'<button[\s\S]{0,300}?' + re.escape(amb_setter) + r'\(\s*null\s*\)', notice
    ), 'the ambiguity notice must carry a dismiss control that clears it.'


def test_payload_arrival_read_from_a_first_success_marker_not_object_identity(
    data_js_body: str,
    tab_escalations_jsx_code: str,
) -> None:
    """"Has the payload arrived?" must come from data.js, not from a captured ref.

    The seed-identity form — capture `DF.ESCALATIONS` at module-eval time, then
    test `payload !== SEED` — races Babel and can WEDGE.  data.js is a classic
    script whose `startPolling()` fires its first fetch at load time
    (`refreshDFData` before the interval); tab_escalations.jsx is
    `type="text/babel"`, transpiled and evaluated after DOMContentLoaded.  So
    the first escalations response can land BEFORE the capture runs, making the
    captured "seed" a REAL payload.  Nothing then clears it while either
    (a) `tw.pauseLive` was already on at load — `startPolling` bypasses
    `__DF_PAUSE` for that first fetch, but every later `pollTick` is skipped —
    or (b) the endpoint enters backoff (up to 60s) right after that first
    success.  In those windows the tab renders a "still loading" notice directly
    above a fully-populated escalation table and never consumes the cross-tab
    focus: the same dead-link outcome the focus fix exists to remove, in a new
    form.

    The sound signal is a per-key FIRST-SUCCESS marker recorded where the apply
    actually happens.  Asserted structurally, deriving the registry expression
    and the key from the source rather than pinning either spelling.
    """
    code = tab_escalations_jsx_code

    # (a) data.js records the marker inside applyKey — the one place that knows
    #     a real server value was applied.
    apply_body = extract_function_body(data_js_body, 'applyKey')
    marker = re.search(r'([\w.$]+)\[\s*key\s*\]\s*=\s*true', apply_body)
    assert marker is not None, (
        'data.js\'s applyKey records no per-key first-success marker '
        '(`<registry>[key] = true`). Without one, a consumer cannot tell a '
        'pre-fetch seed from a loaded-but-empty payload: the two are '
        'structurally identical by design, so nothing about a payload\'s '
        'contents can distinguish them.'
    )
    registry = marker.group(1)
    leaf = registry.split('.')[-1]

    # (b) the marker means "a real value LANDED", so the null/undefined guard
    #     must come first — a response that omits the key must not mark it.
    guard = re.search(
        r'if\s*\([^)]*value\s*===\s*(?:undefined|null)[\s\S]{0,120}?return', apply_body
    )
    assert guard is not None and guard.end() <= marker.start(), (
        f'applyKey marks `{registry}[key]` before (or without) its '
        'null/undefined early return, so the marker would claim arrival for a '
        'response that omitted the key entirely.'
    )

    # (c) the registry is initialised at module scope, so the first applyKey
    #     call has a target — the same reason every DF_DATA key is seeded.
    assert re.search(rf'\b{re.escape(leaf)}\s*[:=]\s*\{{\s*\}}', data_js_body), (
        f'data.js never initialises the `{leaf}` marker registry to an empty '
        'object, so the first applyKey call would throw on a missing target.'
    )

    # (d) tab_escalations.jsx reads THAT marker, keyed on its own payload key.
    loaded_body = extract_function_body(code, 'escalationsLoaded')
    assert leaf in loaded_body, (
        f'escalationsLoaded does not read data.js\'s `{leaf}` marker registry: '
        f'{loaded_body.strip()!r}'
    )
    assert 'ESCALATIONS' in loaded_body, (
        'escalationsLoaded must key the marker lookup on its own payload key, '
        f'not on whatever landed last: {loaded_body.strip()!r}'
    )

    # (e) and the racy form is GONE: no module-scope (column-0) capture of
    #     `DF.ESCALATIONS`. The component's own per-render read is indented and
    #     is a different thing entirely.
    seed_capture = re.search(r'^const\s+\w+\s*=\s*DF\.ESCALATIONS', code, re.M)
    assert seed_capture is None, (
        f'tab_escalations.jsx still captures `DF.ESCALATIONS` at module scope: '
        f'{seed_capture.group(0)!r}. A first poll that resolves before this '
        'module is evaluated freezes a real payload as the "seed", and paused '
        'polling or endpoint backoff makes that wedge permanent.'
    )


# ---------------------------------------------------------------------------
# task 5596 (PRD leaf eta): the tab reads the escalation corpus' named views
#
# "Pending in the live queue" and "open in history" are two named populations
# of ONE walk, served as Datums; escalation_views.js is their one client
# reader and dashboard/tests/js/escalation_views.test.mjs executes it. What is
# pinned here is only the wiring a .jsx body cannot run under node.
# ---------------------------------------------------------------------------

_VIEWS_DESTRUCTURE_RE = re.compile(
    r'const\s*\{([^}]*)\}\s*=\s*window\.DF_ESCALATION_VIEWS\s*;'
)


def test_tab_escalations_reads_escalation_views_at_module_scope(
    tab_escalations_jsx_code: str,
) -> None:
    """DF_ESCALATION_VIEWS is destructured at module scope, with no fallback.

    The CANONICAL note in datum.js's header: a missing or mis-ordered
    dependency must throw at load, not degrade silently inside a render.
    """
    m = _VIEWS_DESTRUCTURE_RE.search(tab_escalations_jsx_code)
    assert m is not None, (
        'tab_escalations.jsx does not destructure `window.DF_ESCALATION_VIEWS` at '
        'module scope (`const { … } = window.DF_ESCALATION_VIEWS;`, no `|| {}`).'
    )
    names = {n.split(':')[-1].strip() for n in m.group(1).split(',') if n.strip()}
    for name in ('queuePending', 'subsectionQueuePending', 'openInHistoryOver',
                 'corpusAgeCaption', 'windowedClassSplit', 'taskCard'):
        assert name in names, (
            f'tab_escalations.jsx does not take `{name}` from DF_ESCALATION_VIEWS.'
        )


def test_tab_escalations_pill_reads_the_queue_pending_view(
    tab_escalations_jsx_code: str,
) -> None:
    """The header pill renders the served queue_pending view, labelled as such.

    It used to print `byStatus.pending || 0` — a root-only count, request-fresh,
    beside a strip counting the archive too at a 60s TTL. Now both are views of
    one walk and each says which population it is.
    """
    tab_fn = extract_function_body(tab_escalations_jsx_code, 'EscalationsTab')
    m = re.search(r'<DatumReading\s+datum=\{([^}]+)\}\s*/>\s*queue pending', tab_fn)
    assert m is not None, (
        'EscalationsTab renders no `<DatumReading datum={…} /> queue pending` pill — '
        'the header count must be the served queue_pending Datum, labelled with the '
        'population it counts.'
    )
    datum_expr = m.group(1).strip()
    if datum_expr != 'queuePending(DF)':
        assert re.search(rf'const\s+{re.escape(datum_expr)}\s*=\s*queuePending\(DF\)', tab_fn), (
            f'the header pill renders `{datum_expr}`, which is not `queuePending(DF)`.'
        )
    assert 'byStatus.pending' not in tab_escalations_jsx_code, (
        'tab_escalations.jsx still counts `byStatus.pending` itself — the pill reads '
        'the served view, never a client re-count.'
    )


def test_tab_escalations_subsection_pip_reads_the_queue_pending_view(
    tab_escalations_jsx_code: str,
) -> None:
    """Each subsection's pending pip is that queue's served view."""
    tab_fn = extract_function_body(tab_escalations_jsx_code, 'EscalationsTab')
    assert re.search(
        r'<Pip\s+datum=\{subsectionQueuePending\(sec\b[^}]*\}\s+label="queue pending"', tab_fn,
    ), (
        'the per-subsection pip does not render '
        '`<Pip datum={subsectionQueuePending(sec, …)} label="queue pending" />`.'
    )
    assert 'secByStatus.pending' not in tab_fn, (
        'the subsection pip still reads `secByStatus.pending` — read the served view.'
    )


def test_tab_escalations_strip_renders_open_in_history(tab_escalations_jsx_code: str) -> None:
    """The strip carries an 'open in history' tile over the project filter.

    lifespan.open_items holds every pending record, root or archive, so the 6h
    breaches it counts are not "of N pending" in the pill's sense; the hint stops
    saying 'pending' and the open-in-history tile states that population.
    """
    strip_fn = extract_function_body(tab_escalations_jsx_code, 'EscalationStatStrip')
    tiles = re.split(r'(?=<C\.StatTile\b)', strip_fn)[1:]
    open_tiles = [t.split('/>')[0] for t in tiles if 'label="open in history"' in t.split('/>')[0]]
    assert len(open_tiles) == 1, (
        'EscalationStatStrip renders no `<C.StatTile label="open in history" …/>` tile.'
    )
    assert re.search(r'datum=\{openInHistoryOver\(DF,\s*projectFilter\)\}', open_tiles[0]), (
        'the open-in-history tile does not render `openInHistoryOver(DF, projectFilter)`.'
    )
    breach_tiles = [t.split('/>')[0] for t in tiles if 'label="6h breaches"' in t.split('/>')[0]]
    assert len(breach_tiles) == 1, 'could not locate the "6h breaches" tile'
    assert 'pending' not in breach_tiles[0], (
        'the 6h-breaches hint still says "pending" — open_items spans the archive '
        'too, which is the open-in-history population, not the queue\'s pending one.'
    )


def test_tab_escalations_states_the_corpus_age(tab_escalations_jsx_code: str) -> None:
    """The tab renders how old the corpus walk it shows is, even when fresh."""
    tab_fn = extract_function_body(tab_escalations_jsx_code, 'EscalationsTab')
    assert re.search(r'\{\s*corpusAgeCaption\(', tab_fn), (
        'EscalationsTab renders no `{corpusAgeCaption(…)}` — a count read from a '
        'cached walk must say when the walk was.'
    )
