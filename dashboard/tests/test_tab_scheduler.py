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
from starlette.testclient import TestClient


@pytest.fixture(scope='module')
def _client():
    from dashboard.app import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope='module')
def styles_css_body(_client):
    return _client.get('/static/redux/styles.css').text


@pytest.fixture(scope='module')
def shell_jsx_body(_client):
    return _client.get('/static/redux/shell.jsx').text


@pytest.fixture(scope='module')
def tab_scheduler_jsx_body(_client):
    return _client.get('/static/redux/tab_scheduler.jsx').text


@pytest.fixture(scope='module')
def app_jsx_body(_client):
    return _client.get('/static/redux/app.jsx').text


@pytest.fixture(scope='module')
def index_html_body(_client):
    return _client.get('/static/redux/index.html').text


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
# step-15: index.html cache-buster is bumped to >= 19
# ---------------------------------------------------------------------------


def test_index_html_cache_buster_bumped(index_html_body):
    """Every /static/redux/* asset in index.html must carry a ?v= cache-buster
    that is equal and >= 19.

    The current value is 18.  Any change to styles.css, shell.jsx,
    tab_scheduler.jsx, or app.jsx requires bumping the version so browsers
    reload the changed assets.

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


def test_evict_park_endpoint_referenced_in_jsx(tab_scheduler_jsx_body):
    """tab_scheduler.jsx must reference the evict-park POST endpoint URL."""
    assert '/api/v2/dashboard/scheduler/evict-park' in tab_scheduler_jsx_body, (
        'tab_scheduler.jsx must reference /api/v2/dashboard/scheduler/evict-park '
        '(the evict POST target)'
    )


def test_handle_evict_and_on_evict_wired(tab_scheduler_jsx_body):
    """handleEvict callback must be defined and onEvict must be passed to ParkStacksSection."""
    assert re.search(r'\bhandleEvict\b', tab_scheduler_jsx_body), (
        'tab_scheduler.jsx must define a handleEvict callback'
    )
    assert re.search(r'onEvict\s*=\s*\{handleEvict\}', tab_scheduler_jsx_body), (
        'tab_scheduler.jsx must pass onEvict={handleEvict} to <ParkStacksSection'
    )


def test_evict_button_guard_is_per_entry_live(tab_scheduler_jsx_body):
    """The evict button's disabled guard must be per-entry liveness: disabled={entry.live}.

    A bare `entry.live` already appears at line ~162 for the isDead variable;
    the discriminating assertion pins the `disabled={entry.live}` attribute form,
    which is absent until the evict button is implemented.
    """
    assert re.search(r'disabled\s*=\s*\{\s*entry\.live\s*\}', tab_scheduler_jsx_body), (
        'tab_scheduler.jsx evict button must be guarded with disabled={entry.live} '
        '(enabled only for non-live owners; defense-in-depth UI guard)'
    )


def test_evict_button_label_present(tab_scheduler_jsx_body):
    """An evict button label or title must be present in ParkStacksSection."""
    assert re.search(r'evict', tab_scheduler_jsx_body, re.IGNORECASE), (
        'tab_scheduler.jsx must render an evict button/label in ParkStacksSection'
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
