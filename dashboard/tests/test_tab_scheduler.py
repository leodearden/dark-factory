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
# step-3: MultiSelect select-none clears to empty array
# ---------------------------------------------------------------------------


def test_multiselect_select_none_clears_to_empty(shell_jsx_body):
    """MultiSelect select-none handler must clear selection to [] (empty array).

    The current bug: `onChange(allSelected ? [options[0]] : [])` leaves ONE
    project selected when the user clicks "select none" from an all-selected
    state.  The fix: the select-none branch should call `onChange([])`.

    Assert the phantom-single-selection pattern `[options[0]]` is gone and
    the select-none/select-all toggle calls `onChange([])` for the none path.
    """
    # The broken pattern must NOT be present
    assert '[options[0]]' not in shell_jsx_body, (
        'shell.jsx still contains [options[0]] phantom-single-selection pattern'
    )
    # The select-none branch must call onChange([]) — regex that captures it
    # within the allSelected ternary context
    assert re.search(r'onChange\s*\(\s*allSelected\s*\?.*\[\s*\]', shell_jsx_body), (
        'shell.jsx MultiSelect select-none/select-all toggle must call onChange([]) '
        'for the allSelected (select-none) branch'
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
