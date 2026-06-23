"""Structural contract tests for the orange 'parked' lock-chip state.

Tests B6 and B7 for both:
  - LockChip in tabs.jsx  (Orchestrators view)
  - Inline Module-locks chip in tab_tasks.jsx  (Tasks card)

All assertions are source-contract tests: they fetch the JSX/CSS/HTML source
via the Starlette TestClient and check structural properties using
normalized-whitespace regex — not exact-string pins.  This is the established
idiom in test_tab_orchestrators.py / test_tab_scheduler.py and is explicitly
sanctioned by the render-wiring note in test_chip_label_disambiguation.py.
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
def tabs_jsx_body(_client):
    resp = _client.get('/static/redux/tabs.jsx')
    assert resp.status_code == 200
    return resp.text


@pytest.fixture(scope='module')
def tab_tasks_jsx_body(_client):
    resp = _client.get('/static/redux/tab_tasks.jsx')
    assert resp.status_code == 200
    return resp.text


@pytest.fixture(scope='module')
def styles_css_body(_client):
    resp = _client.get('/static/redux/styles.css')
    assert resp.status_code == 200
    return resp.text


@pytest.fixture(scope='module')
def index_html_body(_client):
    resp = _client.get('/static/redux/index.html')
    assert resp.status_code == 200
    return resp.text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_ws(s: str) -> str:
    """Collapse all whitespace (spaces, tabs, newlines) to single spaces."""
    return re.sub(r'\s+', ' ', s)


def _extract_function_body(source: str, fn_name: str) -> str:
    """Return the body of the named JS function (opening '{' to matching '}').

    Skips the parameter list first (which may contain destructured {}/[] patterns)
    by matching parens before counting braces for the function body.
    Returns empty string if the function is not found.
    """
    pattern = re.compile(r'function\s+' + re.escape(fn_name) + r'\s*\(')
    m = pattern.search(source)
    if not m:
        return ''
    fn_start = m.start()

    # Skip past the closing ')' of the parameter list, counting nested parens.
    i = m.end() - 1  # position of the opening '(' of the params
    paren_depth = 0
    while i < len(source):
        if source[i] == '(':
            paren_depth += 1
        elif source[i] == ')':
            paren_depth -= 1
            if paren_depth == 0:
                break
        i += 1

    # Find the opening '{' of the function body (after the params close).
    body_open = source.find('{', i)
    if body_open == -1:
        return ''

    # Walk from the body opening brace to its matching closing brace.
    depth = 0
    j = body_open
    while j < len(source):
        if source[j] == '{':
            depth += 1
        elif source[j] == '}':
            depth -= 1
            if depth == 0:
                return source[fn_start:j + 1]
        j += 1
    return source[fn_start:]


# ---------------------------------------------------------------------------
# Step-1 tests: RED — tabs.jsx LockChip (Orchestrators view)
# ---------------------------------------------------------------------------

class TestB6TabsLockChipParkedUnheldIsOrange:
    """B6 (tabs.jsx): LockChip assigns 'lock-parked' for a parked-but-unheld
    module and LocksCell threads m.parked_by through as a parkedBy prop."""

    def test_lockchip_has_lock_parked_class_assignment(self, tabs_jsx_body):
        """LockChip body must contain an assignment to class 'lock-parked'."""
        fn_body = _normalize_ws(_extract_function_body(tabs_jsx_body, 'LockChip'))
        assert fn_body, 'LockChip function not found in tabs.jsx'
        assert 'lock-parked' in fn_body, (
            "LockChip must assign class 'lock-parked' for the parked state"
        )

    def test_lockchip_accepts_parkedby_prop(self, tabs_jsx_body):
        """LockChip must destructure a parkedBy prop (or similar) so the branch
        can read the parked-owner information."""
        fn_body = _normalize_ws(_extract_function_body(tabs_jsx_body, 'LockChip'))
        assert fn_body, 'LockChip function not found in tabs.jsx'
        assert 'parkedBy' in fn_body, (
            "LockChip must accept a parkedBy prop"
        )

    def test_lockchip_renders_pause_icon_for_parked(self, tabs_jsx_body):
        """LockChip must render the ⏸ (U+23F8) pause icon + 'T-' owner prefix
        for the parked display span."""
        fn_body = _normalize_ws(_extract_function_body(tabs_jsx_body, 'LockChip'))
        assert fn_body, 'LockChip function not found in tabs.jsx'
        # Accept either the literal Unicode character or its escape sequence.
        has_pause = '⏸' in fn_body or '\\u23f8' in fn_body.lower()
        assert has_pause, "LockChip must render ⏸ (U+23F8) for the parked owner span"
        # The display string format must include 'T-' prefix.
        assert 'T-' in fn_body, "LockChip parked owner span must include 'T-' prefix"

    def test_lockscell_threads_parked_by_into_lockchip(self, tabs_jsx_body):
        """LocksCell must thread m.parked_by into the <LockChip ... /> call so
        the parked branch in LockChip is reachable (not a dead branch)."""
        fn_body = _normalize_ws(_extract_function_body(tabs_jsx_body, 'LocksCell'))
        assert fn_body, 'LocksCell function not found in tabs.jsx'
        # Accept parkedBy={...} or parkedBy = m && m.parked_by forms.
        assert 'parked_by' in fn_body, (
            "LocksCell must thread m.parked_by into the LockChip call as parkedBy prop"
        )
        assert 'parkedBy' in fn_body, (
            "LocksCell must pass parkedBy to LockChip to wire up the parked state"
        )


class TestB7TabsLockChipHeldAndParkedStaysRed:
    """B7 (tabs.jsx): When a module IS held, LockChip must NOT render
    lock-parked — holder (red lock-taken/green lock-mine) takes precedence."""

    def test_lock_parked_is_gated_on_no_holder(self, tabs_jsx_body):
        """Every path that assigns 'lock-parked' in LockChip must be
        exclusively under a !holder guard, never on the holder-present path."""
        fn_body = _normalize_ws(_extract_function_body(tabs_jsx_body, 'LockChip'))
        assert fn_body, 'LockChip function not found in tabs.jsx'
        # The parked branch must include a !holder guard (no-holder condition).
        # Pattern: any combination of if/ternary/&& with !holder preceding lock-parked.
        # We require that '!holder' appears somewhere before 'lock-parked' in the fn.
        assert '!holder' in fn_body, (
            "LockChip must guard the 'lock-parked' assignment with a !holder check"
        )

    def test_lock_parked_preceded_by_no_holder_check(self, tabs_jsx_body):
        """In LockChip's body, the !holder condition must appear BEFORE the
        lock-parked string, ensuring precedence: holder beats parked."""
        fn_body = _normalize_ws(_extract_function_body(tabs_jsx_body, 'LockChip'))
        assert fn_body, 'LockChip function not found in tabs.jsx'
        pos_no_holder = fn_body.find('!holder')
        pos_lock_parked = fn_body.find('lock-parked')
        assert pos_no_holder != -1, "LockChip must contain '!holder' guard"
        assert pos_lock_parked != -1, "LockChip must contain 'lock-parked' assignment"
        assert pos_no_holder < pos_lock_parked, (
            "'!holder' guard must appear before 'lock-parked' assignment in LockChip"
        )

    def test_lock_parked_not_reachable_when_holder_set(self, tabs_jsx_body):
        """There must be NO code path in LockChip where cls is set to
        'lock-parked' when holder is truthy (held+parked => lock-taken)."""
        fn_body = _normalize_ws(_extract_function_body(tabs_jsx_body, 'LockChip'))
        assert fn_body, 'LockChip function not found in tabs.jsx'
        # The holder path assigns lock-mine or lock-taken.
        # Verify that lock-taken and lock-mine are both still present (regression
        # guard: we haven't accidentally removed the holder-present paths).
        assert 'lock-taken' in fn_body, (
            "LockChip must still handle the lock-taken (held-by-other) path"
        )
        assert 'lock-mine' in fn_body, (
            "LockChip must still handle the lock-mine (held-by-self) path"
        )
        # The parked assignment must be in an if(!holder...) block: ensure it's
        # NOT adjacent to the holder-truthy path by checking structure.
        # A sufficient proxy: parkedBy appears in the same conditional block as !holder.
        assert re.search(r'!holder\s*&&\s*parkedBy|if\s*\(\s*!holder\s*&&\s*parkedBy', fn_body), (
            "The 'lock-parked' branch must be guarded by both !holder and parkedBy"
        )


# ---------------------------------------------------------------------------
# Step-3 tests: RED — tab_tasks.jsx inline Module-locks chip (Tasks card)
# ---------------------------------------------------------------------------

class TestB6TasksInlineChipParkedUnheldIsOrange:
    """B6 (tab_tasks.jsx): inline chip assigns 'lock-parked' for a parked-but-
    unheld module and reads parkedBy from m.parked_by."""

    def test_inline_chip_has_lock_parked_class(self, tab_tasks_jsx_body):
        """The inline chip cls expression must include 'lock-parked' for the
        parked-but-unheld case."""
        body = _normalize_ws(tab_tasks_jsx_body)
        assert 'lock-parked' in body, (
            "tab_tasks.jsx inline chip must assign 'lock-parked' for parked state"
        )

    def test_inline_chip_reads_parked_by_from_m(self, tab_tasks_jsx_body):
        """The inline chip must read m.parked_by to determine the parked state."""
        body = _normalize_ws(tab_tasks_jsx_body)
        assert 'm.parked_by' in body or 'm && m.parked_by' in body, (
            "tab_tasks.jsx must read m.parked_by for the inline module-locks chip"
        )

    def test_inline_chip_cls_ternary_includes_parked(self, tab_tasks_jsx_body):
        """The cls ternary must route to 'lock-parked' when !holder && parkedBy."""
        body = _normalize_ws(tab_tasks_jsx_body)
        # Match the pattern: !holder ? (parkedBy ? 'lock-parked' : 'lock-free') : ...
        # Allow flexible whitespace and quote styles.
        pattern = re.compile(
            r"!holder\s*\?\s*\(\s*parkedBy\s*\?\s*['\"]lock-parked['\"]"
        )
        assert pattern.search(body), (
            "tab_tasks.jsx inline chip cls must use: "
            "!holder ? (parkedBy ? 'lock-parked' : 'lock-free') : ..."
        )

    def test_inline_chip_renders_pause_icon_for_parked(self, tab_tasks_jsx_body):
        """The inline chip must render ⏸ T-{parkedBy} for the parked owner."""
        body = _normalize_ws(tab_tasks_jsx_body)
        has_pause = '⏸' in body or '\\u23f8' in body.lower()
        assert has_pause, (
            "tab_tasks.jsx inline chip must render ⏸ (U+23F8) for parked owner"
        )


class TestB7TasksInlineChipHeldAndParkedStaysRed:
    """B7 (tab_tasks.jsx): When a module IS held, inline chip must NOT render
    lock-parked — holder path still goes to lock-mine/lock-taken."""

    def test_inline_chip_preserves_lock_taken_path(self, tab_tasks_jsx_body):
        """The inline chip must still assign lock-mine and lock-taken for the
        holder-present path."""
        body = _normalize_ws(tab_tasks_jsx_body)
        assert 'lock-taken' in body, (
            "tab_tasks.jsx inline chip must keep the lock-taken (held-by-other) path"
        )
        assert 'lock-mine' in body, (
            "tab_tasks.jsx inline chip must keep the lock-mine (held-by-self) path"
        )

    def test_inline_chip_lock_parked_gated_by_no_holder(self, tab_tasks_jsx_body):
        """The inline chip must only assign lock-parked under the !holder branch."""
        body = _normalize_ws(tab_tasks_jsx_body)
        # The ternary: !holder ? (parkedBy ? 'lock-parked' : ...) : <holder path>
        # means lock-parked can ONLY appear when !holder is true.
        pos_no_holder = body.find('!holder')
        pos_lock_parked = body.find('lock-parked')
        assert pos_no_holder != -1, "tab_tasks.jsx must have a !holder check"
        assert pos_lock_parked != -1, "tab_tasks.jsx must contain 'lock-parked'"
        assert pos_no_holder < pos_lock_parked, (
            "'!holder' must come before 'lock-parked' in the cls ternary"
        )


# ---------------------------------------------------------------------------
# Step-5 tests: RED — styles.css .chip.lock-parked orange rule
# ---------------------------------------------------------------------------

def _extract_css_rule_block(css: str, selector: str) -> str:
    """Return the body of the first CSS rule matching selector (braces included).

    Walks forward from opening '{' counting brace depth to find the matching
    close brace.  Returns '' if the selector is not found.
    """
    idx = css.find(selector)
    if idx == -1:
        return ''
    brace_start = css.find('{', idx)
    if brace_start == -1:
        return ''
    depth = 0
    i = brace_start
    while i < len(css):
        if css[i] == '{':
            depth += 1
        elif css[i] == '}':
            depth -= 1
            if depth == 0:
                return css[brace_start:i + 1]
        i += 1
    return css[brace_start:]


class TestLockParkedCssIsOrangeAndDistinct:
    """Step-5: .chip.lock-parked must exist and use the amber/orange --warn
    palette, distinct from lock-free (grey) and lock-taken (red)."""

    def test_lock_parked_rule_exists(self, styles_css_body):
        """styles.css must contain a .chip.lock-parked rule."""
        assert '.chip.lock-parked' in styles_css_body, (
            "styles.css must define a .chip.lock-parked rule"
        )

    def test_lock_parked_uses_warn_color(self, styles_css_body):
        """The .chip.lock-parked rule must use var(--warn) for its color."""
        block = _extract_css_rule_block(styles_css_body, '.chip.lock-parked')
        assert block, ".chip.lock-parked rule block not found in styles.css"
        assert 'var(--warn)' in block, (
            ".chip.lock-parked must use color: var(--warn) for amber/orange"
        )

    def test_lock_parked_uses_amber_oklch_hue(self, styles_css_body):
        """The .chip.lock-parked rule must use an oklch hue in the ~70-85 range
        (amber/orange, mirroring .chip.dep-pending hue ~75)."""
        block = _extract_css_rule_block(styles_css_body, '.chip.lock-parked')
        assert block, ".chip.lock-parked rule block not found in styles.css"
        # Find all oklch(...) calls in the block and check for an orange hue.
        oklch_matches = re.findall(
            r'oklch\(\s*[\d.]+\s+[\d.]+\s+([\d.]+)', block
        )
        assert oklch_matches, (
            ".chip.lock-parked must have at least one oklch(...) declaration"
        )
        hues = [float(h) for h in oklch_matches]
        assert any(65 <= h <= 90 for h in hues), (
            f".chip.lock-parked hue(s) {hues} must include an amber/orange value "
            f"in the 65-90 range (like .chip.dep-pending at hue ~75)"
        )

    def test_lock_parked_distinct_from_lock_free(self, styles_css_body):
        """.chip.lock-parked must NOT use the lock-free grey palette
        (var(--fg-2) with transparent background)."""
        block = _extract_css_rule_block(styles_css_body, '.chip.lock-parked')
        free_block = _extract_css_rule_block(styles_css_body, '.chip.lock-free')
        assert block, ".chip.lock-parked rule block not found"
        assert free_block, ".chip.lock-free rule block not found (reference check)"
        # lock-free uses var(--fg-2) and transparent; lock-parked must not.
        assert 'var(--fg-2)' not in block, (
            ".chip.lock-parked must not use the grey var(--fg-2) color"
        )

    def test_lock_parked_distinct_from_lock_taken(self, styles_css_body):
        """.chip.lock-parked must NOT use the red lock-taken palette
        (var(--bad) / oklch hue ~25)."""
        block = _extract_css_rule_block(styles_css_body, '.chip.lock-parked')
        assert block, ".chip.lock-parked rule block not found"
        assert 'var(--bad)' not in block, (
            ".chip.lock-parked must not use the red var(--bad) color"
        )


# ---------------------------------------------------------------------------
# Step-7 tests: RED — index.html cache-buster v24 => v25
# ---------------------------------------------------------------------------

class TestIndexHtmlCacheBusterBumpedForParkChip:
    """Step-7: All /static/redux/* ?v=N asset references in index.html must
    have N >= 25 so browsers reload the changed tabs.jsx/tab_tasks.jsx/styles.css."""

    def test_cache_buster_assets_are_present(self, index_html_body):
        """/static/redux/ assets with ?v=N must be present in index.html."""
        matches = re.findall(
            r'/static/redux/[^\s"\']+\?v=(\d+)', index_html_body
        )
        assert matches, (
            "index.html must contain /static/redux/..?v=N asset references"
        )

    def test_all_cache_busters_at_least_v25(self, index_html_body):
        """Every ?v=N query string on a /static/redux/ asset must have N >= 25.

        Current baseline is v=24 (set by task 1870 / γ).  Bumping to 25 forces
        browsers to reload the changed tabs.jsx, tab_tasks.jsx, and styles.css.
        """
        versions = [
            int(v)
            for v in re.findall(
                r'/static/redux/[^\s"\']+\?v=(\d+)', index_html_body
            )
        ]
        assert versions, "No versioned /static/redux/ assets found in index.html"
        below_25 = [v for v in versions if v < 25]
        assert not below_25, (
            f"All /static/redux/ cache-busters must be >= v25 (found v<25: "
            f"{below_25}); bump from v24 to v25 to deliver the parked-chip changes"
        )
