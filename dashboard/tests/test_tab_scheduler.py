"""Wiring tests for the Scheduler tab UI (frontend).

Tests parse JSX/CSS source files as text and assert structural contracts
(CSS width values, export names, component patterns). Follows the idiom
established in test_tab_curator.py and test_index_html.py.

Each RED test is added before its corresponding GREEN implementation step.
Actual rendering must be visually verified — these are source-structure
assertions only.
"""

from __future__ import annotations

import re

import pytest
from _dashboard_helpers import extract_function_body, strip_js_comments


@pytest.fixture(scope='module')
def styles_css_body(_client):
    return _client.get('/static/redux/styles.css').text


@pytest.fixture(scope='module')
def tab_scheduler_jsx_body(_client):
    return _client.get('/static/redux/tab_scheduler.jsx').text


# ---------------------------------------------------------------------------
# step-1: CSS widens scheduler title column
# ---------------------------------------------------------------------------


def _extract_css_rule_block(css: str, selector: str) -> str:
    """Return the body of the first matching CSS rule block (braces included).

    Walks forward from the opening ``{`` counting brace depth to find the
    matching close brace.  Returns the empty string if the selector is not
    found.  Does not skip ``{``/``}`` inside string literals — acceptable
    because the CSS here does not embed brace characters in quoted values.
    """
    m = re.search(re.escape(selector) + r'\s*\{', css)
    if m is None:
        return ''
    start = m.end() - 1
    depth = 0
    for i in range(start, len(css)):
        c = css[i]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return css[start:i + 1]
    return ''


def _parse_min_width(block: str) -> int | None:
    """Return the numeric pixel value of the first min-width declaration."""
    m = re.search(r'min-width\s*:\s*(\d+)px', block)
    return int(m.group(1)) if m else None


def _parse_max_width(block: str) -> int | None:
    """Return the numeric pixel value of the first max-width declaration, or None."""
    m = re.search(r'max-width\s*:\s*(\d+)px', block)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# step-13: visibleModules uses same chip-selection predicate as visibleRows
# ---------------------------------------------------------------------------


def test_scheduler_module_filter_is_consistent_with_rows(tab_scheduler_jsx_body):
    """tab_scheduler.jsx visibleModules must use the chip-selection predicate
    (module visible iff !m.project or selection includes m.project) and must
    NOT contain the old holder_project widening keep-branch.

    Asserts:
    - The old `m.holder_project && ...includes(m.holder_project)` widening
      clause is absent from the modules filter
    - visibleModules filter uses the same project predicate as visibleRows
    """
    # The permissive holder_project OR-branch must be gone
    assert not re.search(
        r'm\.holder_project\s*&&\s*[^;]+includes\s*\(\s*m\.holder_project\s*\)',
        tab_scheduler_jsx_body,
    ), (
        'tab_scheduler.jsx visibleModules must not contain the holder_project widening '
        'keep-branch (m.holder_project && ...includes(m.holder_project))'
    )
    # visibleModules must use the specific new strict predicate:
    #   m => !m.project || effectiveSelected.includes(m.project)
    # The discriminating part is `effectiveSelected.includes(m.project)` —
    # the old implementation had `m.holder_project && ...includes(m.holder_project)`
    # which the negative assertion above already excludes. Asserting the literal
    # new predicate guards against generic `modules.filter(...m.project...)` forms
    # that would also have passed the old (holder_project-widened) implementation.
    assert re.search(
        r'effectiveSelected\.includes\s*\(\s*m\.project\s*\)',
        tab_scheduler_jsx_body,
    ), (
        'tab_scheduler.jsx visibleModules must use the strict per-project predicate '
        'effectiveSelected.includes(m.project), not a holder_project widening branch'
    )


# ---------------------------------------------------------------------------
# step-11: tab_scheduler.jsx renders ProjectChips with data-derived project list
# ---------------------------------------------------------------------------


def test_scheduler_tab_renders_project_chips(tab_scheduler_jsx_body):
    """tab_scheduler.jsx must:
    - Destructure ProjectChips from window.DF_SHELL
    - Render <ProjectChips in JSX
    - Derive its options from SCHEDULER data by mapping .project over BOTH
      rows AND modules (both contribute to the project list).
    """
    # Must destructure ProjectChips from window.DF_SHELL
    assert re.search(
        r'const\s*\{[^}]*ProjectChips[^}]*\}\s*=\s*window\.DF_SHELL',
        tab_scheduler_jsx_body,
    ), (
        'tab_scheduler.jsx must destructure ProjectChips from window.DF_SHELL'
    )
    # Must render <ProjectChips
    assert re.search(r'<ProjectChips', tab_scheduler_jsx_body), (
        'tab_scheduler.jsx must render <ProjectChips'
    )
    # Derives project list from rows .project
    assert re.search(r'rows[^\n]*\.project', tab_scheduler_jsx_body), (
        'tab_scheduler.jsx must map .project over rows for the chip options'
    )
    # Derives project list from modules .project
    assert re.search(r'modules[^\n]*\.project', tab_scheduler_jsx_body), (
        'tab_scheduler.jsx must map .project over modules for the chip options'
    )


# ---------------------------------------------------------------------------
# step-9: app.jsx hides global project dropdown on scheduler tab
# ---------------------------------------------------------------------------


def test_app_hides_global_project_dropdown_on_scheduler(app_jsx_body):
    """app.jsx must set showProjects: false for the scheduler toolbarConfig entry
    and pass a showProjects value derived from toolbarConfig to <Toolbar>.

    Asserts:
    - The scheduler entry in toolbarConfig includes `showProjects: false`
    - The <Toolbar ...> invocation passes showProjects derived from toolbarConfig
    """
    # scheduler entry in toolbarConfig must set showProjects: false
    assert re.search(
        r'scheduler\s*:\s*\{[^}]*showProjects\s*:\s*false',
        app_jsx_body,
    ), (
        'app.jsx toolbarConfig scheduler entry must include showProjects: false'
    )
    # Toolbar invocation must pass showProjects sourced from toolbarConfig
    assert re.search(
        r'showProjects\s*=\s*\{',
        app_jsx_body,
    ), (
        'app.jsx <Toolbar> must pass showProjects={...} derived from toolbarConfig'
    )


# ---------------------------------------------------------------------------
# step-7: Toolbar gates project MultiSelect via showProjects prop
# ---------------------------------------------------------------------------


def test_toolbar_gates_project_control(shell_jsx_body):
    """shell.jsx Toolbar must accept a showProjects prop (default true) and
    render the project MultiSelect only when showProjects is truthy.

    Asserts:
    - The Toolbar destructured props include `showProjects`
    - The project MultiSelect render is guarded by a showProjects conditional
    """
    # Toolbar props must include showProjects (with a default of true)
    assert re.search(
        r'function\s+Toolbar\s*\([^)]*showProjects',
        shell_jsx_body,
    ), (
        'Toolbar must destructure showProjects from its props'
    )
    # The project MultiSelect must be wrapped in a {showProjects && ...} guard
    assert re.search(
        r'\{showProjects\s*&&',
        shell_jsx_body,
    ), (
        'Toolbar must guard the project MultiSelect with {showProjects && ...}'
    )


# ---------------------------------------------------------------------------
# step-5: shell.jsx exports ProjectChips
# ---------------------------------------------------------------------------


def test_shell_exports_project_chips(shell_jsx_body):
    """shell.jsx must define ProjectChips and export it via window.DF_SHELL.

    Asserts:
    - `function ProjectChips(` is defined
    - `ProjectChips` appears in the `window.DF_SHELL = { ... }` export
    - the none toggle calls `onChange([])` (empty selection = show nothing)
    - the all toggle calls `onChange(` with the full options (non-empty array)
    """
    # Function must be defined
    assert re.search(r'function\s+ProjectChips\s*\(', shell_jsx_body), (
        'shell.jsx must define function ProjectChips(...)'
    )
    # Must be in the DF_SHELL export
    assert re.search(r'window\.DF_SHELL\s*=\s*\{[^}]*ProjectChips', shell_jsx_body), (
        'ProjectChips must be exported in window.DF_SHELL = { ... }'
    )
    # none toggle calls onChange([])
    assert re.search(r'onChange\s*\(\s*\[\s*\]\s*\)', shell_jsx_body), (
        'ProjectChips must have a none toggle calling onChange([])'
    )
    # all toggle calls onChange(options) or similar non-empty arg
    assert re.search(r'onChange\s*\(\s*options\s*\)', shell_jsx_body), (
        'ProjectChips must have an all toggle calling onChange(options)'
    )


# ---------------------------------------------------------------------------
# step-17: MultiSelect select-none keeps real single-project selection
# ---------------------------------------------------------------------------


def test_multiselect_select_none_keeps_real_selection(shell_jsx_body):
    """MultiSelect select-none/select-all toggle must use onChange(allSelected ? [options[0]] : []).

    Background: throughout the dashboard [] means "show all" — downstream
    project filters are `projectFilter.length === 0 || projectFilter.includes(...)`,
    the global projects state initialises to [] (// [] = all), and MultiSelect's
    own allSelected is `selected.length === 0 || selected.length === options.length`.

    Consequence: onChange([]) for the select-none branch is a no-op — clicking
    "select none" when allSelected is true produces [] which is still allSelected,
    so the label never flips to "select all" and no filtering occurs.

    The correct behaviour: onChange(allSelected ? [options[0]] : [])
      - allSelected=true  → click → [options[0]] (one project, allSelected=false, label="select all")
      - allSelected=false → click → [] (allSelected=true, label="select none")

    Asserts:
    (a) the no-op select-none regression 'allSelected ? [] :' is absent
    (b) the reverted real-selection pattern [options[0]] is present
    (c) the empty-equals-all invariant 'selected.length === 0' is present
        (this is the allSelected definition that [] triggers)
    """
    # (a) The no-op select-none regression must be gone
    assert 'allSelected ? [] :' not in shell_jsx_body, (
        'shell.jsx MultiSelect select-none branch must not use onChange([]) — '
        '[] means show-all so onChange([]) is a no-op when allSelected is true'
    )
    # (b) The reverted real-selection pattern must be restored
    assert '[options[0]]' in shell_jsx_body, (
        'shell.jsx MultiSelect must use [options[0]] for the select-none branch '
        'so clicking select-none produces a real single-project view'
    )
    # (c) Pin the empty-equals-all invariant (the allSelected definition)
    assert 'selected.length === 0' in shell_jsx_body, (
        'shell.jsx MultiSelect allSelected definition must include selected.length === 0 '
        '(empty array = show all convention)'
    )


# ---------------------------------------------------------------------------
# step-15: index.html cache-buster stays at or above the scheduler floor (19)
# ---------------------------------------------------------------------------


def test_index_html_cache_buster_not_reverted_below_scheduler_floor(index_html_body):
    """Every /static/redux/* asset in index.html must carry a ?v= cache-buster
    that is >= 19.

    This is an ANTI-REVERT PIN, not a live bump check: index.html is far past
    19 today, so it fails only if someone rolls the cache-busters back below
    what the scheduler tab needed. Whether the versions are UNIFORM, and
    whether the newest bump landed, are both asserted in test_index_html.py —
    do not restate either claim here.

    Asserts:
    - All ?v= values found on /static/redux/* paths are integers >= 19
    - At least one versioned asset is present (sanity guard)
    """
    versions = [
        int(m.group(1))
        for m in re.finditer(r'/static/redux/[^\s"\']+\?v=(\d+)', index_html_body)
    ]
    assert versions, 'No versioned /static/redux/* assets found in index.html'
    assert all(v >= 19 for v in versions), (
        f'All ?v= values must be >= 19; found: {versions}'
    )


# ---------------------------------------------------------------------------
# step-3 (task 1873): evict-park guarded button — JSX source-structure tests
# ---------------------------------------------------------------------------


def test_handle_evict_and_on_evict_wired(tab_scheduler_jsx_body):
    """handleEvict callback must be defined and onEvict must be passed to ParkStacksSection."""
    assert re.search(r'\bhandleEvict\b', tab_scheduler_jsx_body), (
        'tab_scheduler.jsx must define a handleEvict callback'
    )
    assert re.search(r'onEvict\s*=\s*\{handleEvict\}', tab_scheduler_jsx_body), (
        'tab_scheduler.jsx must pass onEvict={handleEvict} to <ParkStacksSection'
    )


def test_evict_button_guard_covers_liveness_and_unresolvable_root(tab_scheduler_jsx_body):
    """The evict button's disabled guard covers both liveness and unresolvable project_root.

    disabled={entry.live || !ownerProjectRoot} pins two defense-in-depth conditions:
    - entry.live: owner is live → button disabled (live owner must not be evicted)
    - !ownerProjectRoot: project_root absent from row index → suppresses guaranteed-400 request
      (a dead park owner without a synthetic stranded row would send project_root:'' and
      receive a confusing 'Evict failed (400)' toast instead of a meaningful signal)
    """
    assert re.search(
        r'disabled\s*=\s*\{\s*entry\.live\s*\|\|\s*!ownerProjectRoot\s*\}',
        tab_scheduler_jsx_body,
    ), (
        'tab_scheduler.jsx evict button must be guarded with '
        'disabled={entry.live || !ownerProjectRoot} '
        '(defense-in-depth: disabled when owner is live OR project_root is unresolvable)'
    )


def test_evict_button_calls_on_evict(tab_scheduler_jsx_body):
    """The evict button must invoke onEvict(entry.owner, ...) on click and render >evict</button>.

    Replaces the non-discriminating `re.search(r'evict', body, re.IGNORECASE)` test which
    matched comments, identifiers (handleEvict/onEvict), and the endpoint string — staying
    green even if the <button> element itself were deleted.  These two assertions pin the
    actual button element and its click wiring, both unique to the evict <button> in
    tab_scheduler.jsx (lines ~219, ~222), so they fail iff the button is removed or rewired.
    """
    body = tab_scheduler_jsx_body
    assert re.search(r'onClick=\{\s*\(\)\s*=>\s*onEvict\(\s*entry\.owner', body), (
        'tab_scheduler.jsx evict button must wire onClick to onEvict(entry.owner, ...)'
    )
    assert re.search(r'>\s*evict\s*</button>', body), (
        'tab_scheduler.jsx must render a <button>...>evict</button> element in ParkStacksSection'
    )


def test_styles_css_widens_scheduler_title_column(styles_css_body):
    """styles.css must widen the scheduler title column to readable widths.

    .sched-row-label min-width must be >= 320px (currently 220 — too narrow).
    .sched-row-title max-width must be >= 300px or absent (currently 240px —
    truncates titles prematurely).
    """
    label_block = _extract_css_rule_block(styles_css_body, '.sched-row-label')
    assert label_block, '.sched-row-label rule not found in styles.css'

    min_w = _parse_min_width(label_block)
    assert min_w is not None, 'min-width not found in .sched-row-label'
    assert min_w >= 320, (
        f'.sched-row-label min-width is {min_w}px; expected >= 320px for readable titles'
    )

    title_block = _extract_css_rule_block(styles_css_body, '.sched-row-title')
    assert title_block, '.sched-row-title rule not found in styles.css'

    max_w = _parse_max_width(title_block)
    # max-width may be absent (good) or >= 300px
    assert max_w is None or max_w >= 300, (
        f'.sched-row-title max-width is {max_w}px; expected >= 300px or absent'
    )


# ---------------------------------------------------------------------------
# task 5705: the heatmap renders BOUNDED axes, not the raw rows x modules
# cross-product
# ---------------------------------------------------------------------------
#
# WHY SOURCE TEXT. scheduler_heatmap.jsx is served as type="text/babel" and
# transpiled in the browser; nothing in this project can import it. The bound
# ITSELF is proven executably against a 2,991 x 4,302 fixture in
# dashboard/tests/js/scheduler_heatmap_bounds.test.mjs — what that suite
# structurally cannot see is whether the component actually CONSUMES it. That
# is the whole difference between a structural cap and an advisory one, and it
# is what these probes pin.
#
# Every probe is scoped with extract_function_body and comment-stripped. Both
# properties are load-bearing: an unscoped substring probe would be satisfied
# by prose or by a sibling function, and strip_js_comments is what stops a
# comment describing the OLD expression from answering an ABSENCE assertion.
# extract_function_body RAISES on a miss by design, so a rename fails loudly
# instead of yielding an empty body over which every absence check passes
# vacuously.


@pytest.fixture(scope='module')
def scheduler_heatmap_jsx_body(_client):
    return _client.get('/static/redux/scheduler_heatmap.jsx').text


def _heatmap_body(scheduler_heatmap_jsx_body: str) -> str:
    """`SchedulerHeatmap`'s comment-stripped body."""
    return strip_js_comments(
        extract_function_body(scheduler_heatmap_jsx_body, 'SchedulerHeatmap')
    )


def test_scheduler_heatmap_consumes_the_shared_bound(scheduler_heatmap_jsx_body):
    """SchedulerHeatmap must call boundHeatmapAxes to choose its axes.

    Destructured at module scope from window.DF_SCHED_HEATMAP_BOUNDS — the
    same contract tab_scheduler.jsx:15 relies on for window.DF_SCHED_HEATMAP,
    and enforced by
    test_index_html.py::test_scheduler_heatmap_bounds_js_loads_before_scheduler_heatmap.
    """
    src = strip_js_comments(scheduler_heatmap_jsx_body)
    assert re.search(
        r'const\s*\{[^}]*boundHeatmapAxes[^}]*\}\s*=\s*window\.DF_SCHED_HEATMAP_BOUNDS',
        src,
    ), (
        'scheduler_heatmap.jsx must destructure boundHeatmapAxes from '
        'window.DF_SCHED_HEATMAP_BOUNDS at module scope'
    )
    assert re.search(r'boundHeatmapAxes\s*\(', _heatmap_body(scheduler_heatmap_jsx_body)), (
        'SchedulerHeatmap must CALL boundHeatmapAxes — importing the bound '
        'without applying it leaves the cap advisory, which is the exact '
        'failure mode task 5705 closes'
    )


def test_scheduler_heatmap_no_longer_iterates_the_raw_props(scheduler_heatmap_jsx_body):
    """The two unbounded render iterations must be GONE.

    This is the assertion that actually fails against the pre-5705 source and
    the one that catches a regression. `rows.map(` built one <tr> per composed
    task row and `enriched.map(` one <td><div> per module, in both the <thead>
    and the <tbody> — 2,991 x 4,302 = 12,867,282 cells on the 2026-09-20 live
    snapshot, which kills the browser renderer.

    Asserted as an ABSENCE rather than "the bounded arrays are also iterated",
    because adding a bounded iteration while LEAVING an unbounded one is
    exactly the half-fix that would otherwise pass.
    """
    body = _heatmap_body(scheduler_heatmap_jsx_body)

    # The leading `(?<![.\w])` is what makes these probes mean what they say:
    # a bare substring test for `rows.map(` also matches `bounded.rows.map(`,
    # i.e. it would reject the fix itself. Only an iteration over the RAW
    # identifier is the defect; one reached through a qualifier is not.
    assert not re.search(r'(?<![.\w])rows\.map\s*\(', body), (
        'SchedulerHeatmap still iterates the raw `rows` prop — the row axis is '
        'unbounded. Iterate the bounded selection instead.'
    )
    assert not re.search(r'(?<![.\w])enriched\.map\s*\(', body), (
        'SchedulerHeatmap still iterates the unbounded `enriched` module list '
        '— the column axis is unbounded. Enrich the BOUNDED modules instead.'
    )


def test_scheduler_heatmap_iterates_the_bounded_axes(scheduler_heatmap_jsx_body):
    """Both axes of the grid come off the bounded selection.

    The companion to the absence check above: together they pin that the grid
    is built from `bounded.*` and from nothing else. Without this half, simply
    deleting the render would pass.
    """
    body = _heatmap_body(scheduler_heatmap_jsx_body)

    assert re.search(r'bounded\.rows\b', body), (
        'the <tbody> must iterate the bounded rows'
    )
    assert re.search(r'bounded\.modules\b', body), (
        'the <thead>/<tbody> columns must come from the bounded modules'
    )


def test_scheduler_heatmap_discloses_what_it_is_not_showing(scheduler_heatmap_jsx_body):
    """A truncated grid must say so, naming the totals.

    A silently-truncated heatmap is worse than a slow one: an operator reading
    60 rows has no way to tell whether that is the whole picture or the top
    2% of it. The affordance reads rowsTotal/modulesTotal — the INPUT sizes —
    so it stays honest whether the shrinkage came from the contention filter
    or from the hard cap.
    """
    body = _heatmap_body(scheduler_heatmap_jsx_body)

    assert 'rowsTotal' in body and 'modulesTotal' in body, (
        'SchedulerHeatmap must render the input totals so a truncated grid '
        'discloses what it is hiding'
    )
    assert 'rowsTruncated' in body or 'modulesTruncated' in body, (
        'the disclosure must be conditioned on the truncation flags rather '
        'than shown unconditionally'
    )


def test_cell_state_delegates_membership_to_the_shared_predicate(scheduler_heatmap_jsx_body):
    """cellStateFor must DELEGATE its membership check, not restate it.

    SPOT (heuristic 11). Row selection and the cell renderer must answer "is
    this cell non-blank?" identically; two copies of the project-scope +
    lock_set pair would let the axis filter drift from the renderer, and the
    filter could then drop a row whose cells the renderer would have coloured.
    The cross-project branch is the subtle one — a path match is not a lock
    match, because modules are keyed by (project, path) on the server.
    """
    body = strip_js_comments(extract_function_body(scheduler_heatmap_jsx_body, 'cellStateFor'))

    assert re.search(r'rowTouchesModule\s*\(', body), (
        'cellStateFor must call rowTouchesModule for its project-scope + '
        'lock_set membership check'
    )
    assert not re.search(r'module\.project\s*!==\s*row\.project', body), (
        'cellStateFor still restates the cross-project rule that '
        'rowTouchesModule owns — the two copies will drift'
    )
    assert not re.search(r'lockSet\.includes\s*\(', body), (
        'cellStateFor still restates the lock_set membership check that '
        'rowTouchesModule owns'
    )


# ---------------------------------------------------------------------------
# task 5705: memoisation — what turns a one-off expensive render into a
# skipped one
# ---------------------------------------------------------------------------


def test_scheduler_heatmap_export_is_memoised(scheduler_heatmap_jsx_body):
    """The value exported as SchedulerHeatmap must be a React.memo result.

    app.jsx runs `setInterval(() => setNow(new Date()), 1000)`, so the active
    tab's subtree re-renders once a second whether or not any data changed.
    Bounding the grid makes each render cheap; memoising it makes the 1 Hz tick
    free, because `rows`/`modules` are referentially identical between ticks.

    Asserted on the BINDING, not on the mere presence of the string
    `React.memo` somewhere in the file — a memo applied to HeatmapCell while
    SchedulerHeatmap stayed bare would otherwise pass while fixing nothing.
    """
    src = strip_js_comments(scheduler_heatmap_jsx_body)

    assert re.search(r'const\s+SchedulerHeatmap\s*=\s*React\.memo\s*\(', src), (
        'the SchedulerHeatmap binding must be a React.memo(...) result — '
        'without it the 1 Hz clock tick re-renders the whole grid'
    )
    assert re.search(
        r'window\.DF_SCHED_HEATMAP\s*=\s*\{[^}]*\bSchedulerHeatmap\b[^}]*\}', src
    ), 'the memoised SchedulerHeatmap must be the value exported on window.DF_SCHED_HEATMAP'


def test_scheduler_tab_memoises_the_props_it_feeds_the_heatmap(tab_scheduler_jsx_body):
    """visibleRows and visibleModules must come from stUseMemo.

    React.memo on the heatmap is inert without this half: whenever a chip
    filter is active both are `.filter(...)` results computed inline, so they
    are fresh array identities on EVERY render and the memo never hits. The two
    changes only pay off together.
    """
    src = strip_js_comments(tab_scheduler_jsx_body)

    for name in ('visibleRows', 'visibleModules'):
        assert re.search(rf'const\s+{name}\s*=\s*stUseMemo\s*\(', src), (
            f'{name} must be wrapped in stUseMemo — recomputed inline it is a '
            'fresh array identity every render, and React.memo on '
            'SchedulerHeatmap would never hit'
        )


def test_memoising_the_module_filter_preserved_its_predicate(tab_scheduler_jsx_body):
    """Wrapping the modules filter in a memo must not paraphrase it.

    Deliberately redundant with
    test_scheduler_module_filter_is_consistent_with_rows above: that test pins
    the predicate generally, this one pins it AT THE POINT OF CHANGE, where a
    "while I'm in here" rewrite is most likely. The expressions move inside the
    memo callback unchanged, or the chip filter silently widens.
    """
    assert not re.search(
        r'm\.holder_project\s*&&\s*[^;]+includes\s*\(\s*m\.holder_project\s*\)',
        tab_scheduler_jsx_body,
    ), 'the holder_project widening keep-branch must stay absent after memoisation'
    assert re.search(
        r'effectiveSelected\.includes\s*\(\s*m\.project\s*\)', tab_scheduler_jsx_body
    ), 'the strict per-project predicate must survive the memo wrap verbatim'
